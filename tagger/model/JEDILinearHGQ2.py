import json
import os
from schema import Schema, And, Use, Optional
from math import log2

import numpy.typing as npt
import keras
import numpy as np
from keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D, AveragePooling1D, Flatten
from hgq.layers import QConv1D, QDense, QMeanPow2,QBatchNormalization, QSoftmax,QLayerBaseSingleInput,QLayerBaseMultiInputs,  QEinsumDenseBatchnorm, QGlobalAveragePooling1D, QAdd,QSum
from hgq.config import LayerConfigScope, QuantizerConfigScope, QuantizerConfig
from hgq.regularizers import MonoL1
from hgq.constraints import MinMax
from hgq.utils.sugar import FreeEBOPs, BetaScheduler,PieceWiseSchedule

from keras.models import load_model
import hls4ml
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tagger.data.tools import load_data, to_ML
from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tagger.model.common import initialise_tensorflow

@JetModelFactory.register('JEDILinearHGQ2')
class JEDILinearHGQ2(JetTagModel):

    schema = Schema(
            {
                "model": str,
                ## generic run config coniguration
                "run_config" : JetTagModel.run_schema,
                "model_config" : {"name" : str,
                                  "conv1d_layers" : list,
                                  "classification_layers" : list,
                                  "regression_layers" : list,
                                  "beta": And(float, lambda s: 1.0 >= s >= 0.0),
                                  },

                "quantization_config" : {'pt_output_quantization' : list},

                "training_config" :     {"weight_method" : And(str, lambda s: s in  ["none", "ptref", "onlyclass"]),
                                         "validation_split" : And(float, lambda s: s > 0.0),
                                         "epochs" : And(int, lambda s: s >= 1),
                                         "batch_size" : And(int, lambda s: s >= 1),
                                         "learning_rate": And(float, lambda s: s > 0.0),
                                         "loss_weights" : And(list, lambda s: len(s) == 2),
                                         "EarlyStopping_patience" : And(int, lambda s: s > 0),
                                         "ReduceLROnPlateau_factor" : And(float, lambda s: 1.0 >= s >= 0.0),
                                         "ReduceLROnPlateau_patience" : int,
                                         "ReduceLROnPlateau_min_lr" : And(float, lambda s: s >= 0.0)},
                
                "firmware_config" : {"input_precision" : str,
                                    "class_precision" : str,
                                    "reg_precision": str,
                                    "clock_period" : And(float, lambda s: 0.0 < s <= 10),
                                    "fpga_part" : str,
                                    "project_name" : str}
            }
    )

    def build_model(self, inputs_shape, outputs_shape):
        
        initialise_tensorflow(self.run_config['num_threads'])

        scope0 = QuantizerConfigScope(default_q_type='kbi',
                                      b0=7,
                                      overflow_mode='wrap',
                                      i0=0,
                                      fr=MonoL1(1.e-8),
                                      ir=MonoL1(1.e-8),
                                    )

        scope1 = QuantizerConfigScope(default_q_type='kif',
                                      place='datalane',
                                      overflow_mode='wrap',
                                      f0=7,
                                      fr=MonoL1(1.e-8),
                                      ic=MinMax(0, 12),
                                    )
        heterogeneous_axis = None

        scope2 = LayerConfigScope(enable_ebops=True, heterogeneous_axis=heterogeneous_axis,beta0=1e-8)
        
        with scope0, scope1, scope2:

            iq_conf = QuantizerConfig(place='datalane', round_mode='RND')

            N = 16
            n_inputs = 20
            #n = 3 if conf.pt_eta_phi else 16
        
            with (
                QuantizerConfigScope(place=('weight', 'bias'), overflow_mode='SAT_SYM'),
                QuantizerConfigScope(place='datalane', heterogeneous_axis=heterogeneous_axis)):
                inp_b = keras.layers.Input((N, n_inputs),name='model_input')
                #inp_b = QBatchNormalization()(inp)
                pool_scale = 2.**-round(log2(N))
                
                x = QEinsumDenseBatchnorm('bnc,cC->bnC', (N, n_inputs), bias_axes='C', activation='relu'  )(inp_b)
                s = QEinsumDenseBatchnorm('bnc,cC->bnC', (N, n_inputs), bias_axes='C', activation='relu', )(x)
                
                d = QEinsumDenseBatchnorm( 'bnc,cC->bnC', (1, n_inputs), bias_axes='C', activation='relu')(QSum(axes=1, scale=pool_scale, keepdims=True)(x))
                x = QAdd()([s, d])

                x = QEinsumDenseBatchnorm('bnc,cC->bnC',
                                          (N, n_inputs),
                                          bias_axes='C',
                                          activation='relu',
                                        )(x)
                x = QSum(axes=1, scale=1 / 16, keepdims=False)(x)
                
                jet_id = QEinsumDenseBatchnorm('bc,cC->bC',n_inputs, bias_axes='C', activation='relu', )(x)
                jet_id = QEinsumDenseBatchnorm('bc,cC->bC', n_inputs, bias_axes='C', activation='relu', )(jet_id)
                jet_id = QEinsumDenseBatchnorm('bc,cC->bC', n_inputs, bias_axes='C', activation='relu', )(jet_id)
                jet_id = QEinsumDenseBatchnorm('bc,cC->bC', 8, bias_axes='C')(jet_id)
                jet_id = Activation('softmax', name='jet_id_output')(jet_id)

                pt_regress = QEinsumDenseBatchnorm('bc,cC->bC', n_inputs, bias_axes='C', activation='relu', )(x)
                pt_regress = QEinsumDenseBatchnorm('bc,cC->bC', n_inputs, bias_axes='C', activation='relu', )(pt_regress)
                pt_regress = QEinsumDenseBatchnorm('bc,cC->bC', n_inputs, bias_axes='C', activation='relu', )(pt_regress)
                pt_regress = QEinsumDenseBatchnorm('bc,cC->bC', 1,name='pT_output', bias_axes='C' )(pt_regress)

                #Define the model using both branches
                self.jet_model = keras.Model(inputs = inp_b, outputs = [jet_id, pt_regress])
                print(self.jet_model.summary())

    # Redefine save and load for HGQ due to needing h5 format
    @JetTagModel.save_decorator
    def save(self, out_dir):
        # Export the model
        #model_export = tfmot.sparsity.keras.strip_pruning(self.jet_model)
        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        export_path = os.path.join(out_dir, "model/saved_model.h5")
        self.jet_model.save(export_path)
        print(f"Model saved to {export_path}")

    @JetTagModel.load_decorator
    def load(self, out_dir=None):
        # Load model

        self.jet_model = load_model(f"{out_dir}/model/saved_model.h5")

    def firmware_convert(self, firmware_dir: str, build: bool = False):
            """Run the hls4ml model conversion

            Args:
                firmware_dir (str): Where to save the firmware
                build (bool, optional): Run the full hls4ml build? Or just create the project. Defaults to False.
            """

            # Remove the old directory if it exists
            hls4ml_outdir = firmware_dir + '/' + self.firmware_config['project_name']
            os.system(f'rm -rf {hls4ml_outdir}')

            # Create default config
            config = hls4ml.utils.config_from_keras_model(self.jet_model, granularity='name')
            config["Model"]["Strategy"] = "distributed_arithmetic"
            config["Model"]["ReuseFactor"]=1
            config['IOType'] = 'io_parallel'
            config['namespace']=self.firmware_config['project_name']+'_emu_v2'
            config['write_weights_txt']=False
            config['write_emulation_constants']=True
           

            # Configuration for conv1d layers
            # hls4ml automatically figures out the paralellization factor
            # config['LayerName']['Conv1D_1']['ParallelizationFactor'] = 8
            # config['LayerName']['Conv1D_2']['ParallelizationFactor'] = 8

            # Additional config
            

            # Write HLS
            self.hls_jet_model = hls4ml.converters.convert_from_keras_model(
                self.jet_model,
                backend='Vitis',
                project_name=self.firmware_config['project_name'],
                clock_period=self.firmware_config['clock_period'],
                hls_config=config,
                output_dir=f'{hls4ml_outdir}',
                part= self.firmware_config['fpga_part'],
            )

            # Compile the project
            self.hls_jet_model.compile()

            # Save config  as json file
            print("Saving default config as config.json ...")
            with open(hls4ml_outdir + '/config.json', 'w') as fp:
                json.dump(config, fp)

            if build:
                # build the project
                self.hls_jet_model.build(csim=False, reset=True)
    
    
    def compile_model(self, num_samples: int):
        
        """compile the model generating callbacks and loss function
        Args:
            num_samples (int): Number of samples in the training set used for scheduling
        """
        def log_beta_schedule(epoch, max_epochs=100):
            log_beta_start = np.log10(1e-7)
            log_beta_end = np.log10(1e-4)
            log_beta = log_beta_start + (log_beta_end - log_beta_start) * (epoch / max_epochs)
            return 10 ** log_beta
        
        def cosine_decay_restarts(global_step):
            from math import cos, pi

            n_cycle = 1
            cycle_step = global_step
            cycle_len = 100
            while cycle_step >= cycle_len:
                cycle_step -= cycle_len
                cycle_len *= 1
                n_cycle += 1

            cycle_t = min(cycle_step / (cycle_len - 10), 1)
            lr = 1.e-6 + 0.5 * (3.e-3 - 1.e-6) * (
                1 + cos(pi * cycle_t)
            ) * 1 ** max(n_cycle - 1, 0)
            return lr

        scheduler = keras.callbacks.LearningRateScheduler(cosine_decay_restarts)
        terminate_on_nan = keras.callbacks.TerminateOnNaN()

    
        #beta_scheduler = BetaScheduler(beta_fn=lambda epoch: log_beta_schedule(epoch, max_epochs=100))
        beta_scheduler = BetaScheduler(PieceWiseSchedule([(0, 0.2e-7, 'linear'), (20, 3e-7, 'log'), (100, 3e-6, 'constant')] ))
        # Define the callbacks using hyperparameters in the config
        self.callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.training_config['EarlyStopping_patience']),
            scheduler,
            terminate_on_nan,
            FreeEBOPs(),
            beta_scheduler

        ]


        # compile the tensorflow model setting the loss and metrics
        self.jet_model.compile(
            optimizer='adam',
            loss={
                self.loss_name + self.output_id_name: 'categorical_crossentropy',
                self.loss_name + self.output_pt_name: keras.losses.Huber(),
            },
            loss_weights=self.training_config['loss_weights'],
            metrics={
                self.loss_name + self.output_id_name: 'categorical_accuracy',
                self.loss_name + self.output_pt_name: ['mae', 'mean_squared_error'],
            },
            weighted_metrics={
                self.loss_name + self.output_id_name: 'categorical_accuracy',
                self.loss_name + self.output_pt_name: ['mae', 'mean_squared_error'],
            },
        )
    def fit(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        pt_target_train: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64],
    ):
        """Fit the model to the training dataset

        Args:
            X_train (npt.NDArray[np.float64]): X train dataset
            y_train (npt.NDArray[np.float64]): y train classification targets
            pt_target_train (npt.NDArray[np.float64]): y train pt regression targets
            sample_weight (npt.NDArray[np.float64]): sample weighting
        """
        keras.config.disable_traceback_filtering()
        sample_weight_dict = {
                            "jet_id_output": sample_weight,
                            "pT_output": sample_weight,
        }
        # Train the model using hyperparameters in yaml config
        history = self.jet_model.fit(
            {'model_input': X_train},
            [y_train,pt_target_train],
            sample_weight = [sample_weight, sample_weight],
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            verbose=self.run_config['verbose'],
            validation_split=self.training_config['validation_split'],
            callbacks=self.callbacks,
            shuffle=True,
        )
        
        self.history = history.history