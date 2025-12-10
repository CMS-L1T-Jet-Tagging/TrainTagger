import json
import os
from schema import Schema, And, Use, Optional
from math import log2

import numpy.typing as npt
import keras
import numpy as np
from keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D, AveragePooling1D, Flatten,Rescaling
from hgq.layers import QConv1D, QDense, QMeanPow2,QBatchNormalization, QSoftmax,QLayerBaseSingleInput,QLayerBaseMultiInputs,  QEinsumDenseBatchnorm, QGlobalAveragePooling1D, QAdd,QSum, QMultiply
from hgq.layers.activation import QUnaryFunctionLUT
from hgq.config import LayerConfigScope, QuantizerConfigScope, QuantizerConfig
from hgq.regularizers import MonoL1
from hgq.constraints import MinMax
from hgq.utils.sugar import FreeEBOPs, BetaScheduler,PieceWiseSchedule

from keras.models import load_model
import hls4ml
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tagger.data.tools import load_data, to_ML
from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tagger.model.common import initialise_tensorflow,cosine_decay_restarts

from da4ml.converter.hgq2.parser import trace_model
from da4ml.trace import comb_trace, HWConfig
from da4ml.codegen import HLSModel, VHDLModel

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
                                        },
                
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

            iq_conf = QuantizerConfig(k0=1, i0=11, f0=12, trainable=False,round_mode='RND',overflow_mode='SAT')
            oq_conf_jetid = QuantizerConfig(k0=0, i0=12, f0=12, trainable=False,round_mode='RND',overflow_mode='SAT')
            oq_conf_pt = QuantizerConfig(k0=1, i0=9, f0=6, trainable=False,round_mode='RND',overflow_mode='SAT')

            N_constituents = inputs_shape[0]
            n_features = inputs_shape[1]
        
            with (
                QuantizerConfigScope(place=('weight', 'bias'), overflow_mode='SAT_SYM'),
                QuantizerConfigScope(place='datalane', heterogeneous_axis=heterogeneous_axis)):
                inp_b = keras.layers.Input((N_constituents, n_features),name='model_input')
                #inp_b = QBatchNormalization()(inp)
                pool_scale = 2.**-round(log2(N_constituents))
                
                x = QEinsumDenseBatchnorm('bnc,cC->bnC', (N_constituents, n_features), bias_axes='C', activation='relu',iq_conf=iq_conf )(inp_b)
                s = QEinsumDenseBatchnorm('bnc,cC->bnC', (N_constituents, n_features), bias_axes='C', activation='relu', )(x)
                
                s2 = AveragePooling1D(N_constituents)(x)
                #s2 = Rescaling(pool_scale)(s2)
                
                d = QEinsumDenseBatchnorm( 'bnc,cC->bnC', (1, n_features), bias_axes='C', activation='relu')(s2)
                
                x = QAdd()([s, d])

                x = QEinsumDenseBatchnorm('bnc,cC->bnC',
                                          (N_constituents, n_features),
                                          bias_axes='C',
                                          activation='relu',
                                        )(x)
                #x = QSum(axes=1, scale=1 / 16, keepdims=False)(x)
                x = AveragePooling1D(N_constituents)(x)
                x = Flatten()(x)
                #x = Rescaling(1/16)(x)
                
                jet_id = QEinsumDenseBatchnorm('bc,cC->bC',n_features, bias_axes='C', activation='relu', )(x)
                jet_id = QEinsumDenseBatchnorm('bc,cC->bC', n_features, bias_axes='C', activation='relu', )(jet_id)
                jet_id = QEinsumDenseBatchnorm('bc,cC->bC', n_features, bias_axes='C', activation='relu', )(jet_id)
                jet_id = QEinsumDenseBatchnorm('bc,cC->bC', outputs_shape[0], bias_axes='C')(jet_id)
                jet_id = Activation('softmax', name='jet_id_output')(jet_id)

                pt_regress = QEinsumDenseBatchnorm('bc,cC->bC', n_features, bias_axes='C', activation='relu', )(x)
                pt_regress = QEinsumDenseBatchnorm('bc,cC->bC', n_features, bias_axes='C', activation='relu', )(pt_regress)
                pt_regress = QEinsumDenseBatchnorm('bc,cC->bC', n_features, bias_axes='C', activation='relu', )(pt_regress)
                pt_regress = QEinsumDenseBatchnorm('bc,cC->bC', 1,name='pT_output', bias_axes='C', enable_oq=True,oq_conf=oq_conf_pt,iq_conf=oq_conf_pt)(pt_regress)

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
            config["Model"]["Strategy"]="distributed_arithmetic"
            config["Model"]["ReuseFactor"]=1
            config['IOType'] = 'io_parallel'
            
            # config['LayerName']['model_input']['Precision']['result'] = self.firmware_config['input_precision']            
            # config["LayerName"]["jet_id_output"]["Precision"]["result"] = self.firmware_config['class_precision']
            # config["LayerName"]["pT_output"]["Precision"]["result"] = self.firmware_config['reg_precision']


            # Write HLS
            self.hls_jet_model = hls4ml.converters.convert_from_keras_model(
                self.jet_model,
                backend='Vitis',
                project_name=self.firmware_config['project_name'],
                clock_period=self.firmware_config['clock_period'],
                hls_config=config,
                output_dir=f'{hls4ml_outdir}',
                part= self.firmware_config['fpga_part'],
                # namespace='hls4ml_'+self.firmware_config['project_name'],
                # write_weights_txt=False,
                # write_emulation_constants=True,
            )

            # Compile the project
            self.hls_jet_model.compile()

            # Save config  as json file
            print("Saving default config as config.json ...")
            with open(hls4ml_outdir + '/config.json', 'w') as fp:
                json.dump(config, fp)
                
                
            old_text = 'nnet::add<quantizer_t, quantizer_1_t, q_add_t, config14>(layer12_out, layer13_out, layer14_out); // q_add'
            new_text = """for (int ii = 0; ii < 16 * 20; ii++) {
                    auto layer13_index = ii % 20;
                    layer14_out[ii] = layer12_out[ii] + layer13_out[layer13_index];
                }"""

            with open(hls4ml_outdir+'/firmware/'+self.firmware_config['project_name']+'.cpp', 'r') as f:
                content = f.read()

            content = content.replace(old_text, new_text)

            with open(hls4ml_outdir+'/firmware/'+self.firmware_config['project_name']+'.cpp', 'w') as f:
                f.write(content)

            print("cpp replacement complete")
            
            old_text = '#pragma HLS ARRAY_PARTITION variable = out_tpose complete'
            new_text = """#pragma HLS ARRAY_PARTITION variable = out_tpose complete
                          #pragma HLS inline recursive
                        """
            
            with open(hls4ml_outdir+'/firmware/nnet_utils/nnet_einsum_dense.h', 'r') as f:
                content = f.read()
            
            content = content.replace(old_text, new_text)

            with open(hls4ml_outdir+'/firmware/nnet_utils/nnet_einsum_dense.h', 'w') as f:
                f.write(content)

            print("einsum dense replacement complete.")

            if build:
                # build the project
                self.hls_jet_model.build(csim=False, reset=True)

    
    
    def compile_model(self, num_samples: int):
        
        """compile the model generating callbacks and loss function
        Args:
            num_samples (int): Number of samples in the training set used for scheduling
        """

        scheduler = keras.callbacks.LearningRateScheduler(schedule = lambda epoch : cosine_decay_restarts(epoch, 
                                                                                                          initial_learning_rate=self.training_config['learning_rate'],
                                                                                                          max_epochs=self.training_config['epochs']))
        terminate_on_nan = keras.callbacks.TerminateOnNaN()

        beta_scheduler = BetaScheduler(PieceWiseSchedule([(0, 0.2e-7, 'linear'), (20, 3e-7, 'log'), (100, 3e-6, 'constant')] ))
        # Define the callbacks using hyperparameters in the config
        self.callbacks = [
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