import json
import os
from schema import Schema, And, Use, Optional


import tensorflow as tf


import keras
import numpy as np
import numpy.typing as npt
from keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D, AveragePooling1D, Flatten
from hgq.layers import QConv1D, QDense, QMeanPow2,QBatchNormalization, QSoftmax,QLayerBaseSingleInput,QLayerBaseMultiInputs,  QEinsumDenseBatchnorm, QGlobalAveragePooling1D
from hgq.config import LayerConfigScope, QuantizerConfigScope, QuantizerConfig
from hgq.regularizers import MonoL1
from hgq.utils.sugar import FreeEBOPs, BetaScheduler
# Qkeras

from tensorflow.keras.models import load_model
import hls4ml
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tagger.data.tools import load_data, to_ML
from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tagger.model.TorchDeepSetModel import JetTagDataset, TorchDeepSetNetwork,TorchDeepSetModel
#from tagger.model.QKerasModel import QKerasModel
from tagger.model.common import initialise_tensorflow

@JetModelFactory.register('DeepSetModelHGQ2')
class DeepSetModelHGQ2(TorchDeepSetModel):

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
                                         "ReduceLROnPlateau_min_lr" : And(float, lambda s: s >= 0.0)}
            }
    )

    def build_model(self, inputs_shape, outputs_shape):

        initialise_tensorflow(self.run_config['num_threads'])

        with QuantizerConfigScope(place='all', k0=1, b0=3, i0=0, default_q_type='kbi',homogeneous_axis=(), overflow_mode='sat_sym'):

            with QuantizerConfigScope(place='datalane', k0=0, default_q_type='kif',homogeneous_axis=(0,1),overflow_mode='wrap', f0=3, i0=3):
            #with LayerConfigScope(enable_ebops=True, homogeneous_axis=(0,)):
                
                L, C = (16,20)
                inputs = Input(shape=(16,20), name='model_input')
                main = QBatchNormalization(name='norm_input')(inputs)
                main = QConv1D(filters=10, parallelization_factor=16,kernel_size=1,activation='relu',name='Conv1D_1')(main) #1.1e-7
                main = QConv1D(filters=10,  parallelization_factor=16,kernel_size=1,activation='relu',name='Conv1D_2')(main)#1.1e-7

                main = GlobalAveragePooling1D(name='avgpool')(main)
             
                
                #jetID branch, 3 layer MLP
                jet_id = QDense(32,parallelization_factor=32, activation='relu',name='Dense_1_jetID')(main)
                jet_id = QDense(16,parallelization_factor=16, activation='relu',name='Dense_2_jetID')(jet_id)
                jet_id = QDense(8, parallelization_factor=8,name='dense_3')(jet_id)
                jet_id = keras.layers.Activation('softmax', name='jet_id_output')(jet_id)

                #pT regression branch
                pt_regress = QDense(10,parallelization_factor=10,activation='relu', name='Dense_1_pT')(main)
                pt_regress = QDense(1,parallelization_factor=1,name='pT_output')(pt_regress)#1.1e-7
    
                #Define the model using both branches
                self.jet_model = keras.Model(inputs = inputs, outputs = [jet_id, pt_regress])

        # Define the model using both branches


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
        beta_scheduler = BetaScheduler(beta_fn=lambda epoch: log_beta_schedule(epoch, max_epochs=100))
        # Define the callbacks using hyperparameters in the config
        self.callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.training_config['EarlyStopping_patience']),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.training_config['ReduceLROnPlateau_factor'],
                patience=self.training_config['ReduceLROnPlateau_patience'],
                min_lr=self.training_config['ReduceLROnPlateau_min_lr'],
            ),
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