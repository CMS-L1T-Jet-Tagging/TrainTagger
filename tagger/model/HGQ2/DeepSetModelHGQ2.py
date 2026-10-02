import json
import os
from schema import Schema, And, Use, Optional
import scipy

import keras
import numpy as np
import numpy.typing as npt
from keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D, AveragePooling1D, Flatten
from hgq.layers import QConv1D, QDense, QMeanPow2,QBatchNormalization, QSoftmax,QLayerBaseSingleInput,QLayerBaseMultiInputs,  QEinsumDenseBatchnorm, QGlobalAveragePooling1D
from hgq.config import LayerConfigScope, QuantizerConfigScope, QuantizerConfig
from hgq.regularizers import MonoL1
from hgq.constraints import MinMax
from hgq.utils.sugar import FreeEBOPs, BetaScheduler, PieceWiseSchedule,EarlyStoppingWithEbopsThres,BetaPID
# Qkeras

from keras.models import load_model
import hls4ml
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tagger.data.tools import load_data, to_ML
from tagger.model.HGQ2.HGQ2Model import HGQ2Model
from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tagger.model.common import log_beta_schedule,cosine_decay_restarts
from tagger.model.common_tensorflow import initialise_tensorflow, huber_loss

@JetModelFactory.register('DeepSetModelHGQ2')
class DeepSetModelHGQ2(HGQ2Model):

    schema = Schema(
            {
                "model": str,
                ## generic run config coniguration
                "run_config" : JetTagModel.run_schema,
                "model_config" : {"name" : str,
                                  "conv1d_layers" : list,
                                  "conv1d_parallelisation_factor" : list,
                                  "classification_layers" : list,
                                  "classification_parallelisation_factor" : list,
                                  "regression_layers" : list,
                                  "regression_parallelisation_factor" : list,
                                  "beta": And(float, lambda s: 1.0 >= s >= 0.0),
                                  },
                
                "input_config" : HGQ2Model.input_config,
                
                "training_config" :  HGQ2Model.training_config,

                "firmware_config" : HGQ2Model.firmware_config,
                
                "quantization_config" : None
            }
    )

    def build_model(self, inputs_shape, outputs_shape):

        initialise_tensorflow(self.run_config['num_threads'])
        #initialise_jax()
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

                N_constituents = inputs_shape['basic_input'][0]
                n_features = inputs_shape['basic_input'][1]

                inputs = Input(shape=(N_constituents,n_features), name='basic_input')
                main = QBatchNormalization(name='norm_input')(inputs)

                for iconv1d, depthconv1d in enumerate(self.model_config['conv1d_layers']):
                    main = QConv1D(filters=depthconv1d, parallelization_factor=self.model_config['conv1d_parallelisation_factor'][iconv1d], kernel_size=1, name='Conv1D_' + str(iconv1d + 1),activation='relu')(main)                
                main = GlobalAveragePooling1D(name='avgpool')(main)

                #jetID branch, 3 layer MLP

                for iclass, depthclass in enumerate(self.model_config['classification_layers']):
                    if iclass == 0:
                        jet_id = QDense(depthclass, parallelization_factor=self.model_config['classification_parallelisation_factor'][iclass], name='Dense_' + str(iclass + 1) + '_jetID',activation='relu')(main)
                    else:
                        jet_id = QDense(depthclass, parallelization_factor=self.model_config['classification_parallelisation_factor'][iclass], name='Dense_' + str(iclass + 1) + '_jetID',activation='relu')(jet_id)                
                jet_id = QDense(outputs_shape[0], parallelization_factor=outputs_shape[0], activation='relu')(jet_id)
                jet_id = Activation('linear', name='jet_id_output')(jet_id)
                #pT regression branch
                for ireg, depthreg in enumerate(self.model_config['regression_layers']):
                    if ireg == 0:
                        pt_regress = QDense(depthreg, parallelization_factor=self.model_config['regression_parallelisation_factor'][ireg], name='Dense_' + str(ireg + 1) + '_pT',activation='relu')(main)
                    else:
                        pt_regress = QDense(depthreg, parallelization_factor=self.model_config['regression_parallelisation_factor'][ireg], name='Dense_' + str(ireg + 1) + '_pT',activation='relu')(pt_regress)      
                pt_regress = QDense(1,name='pT_output')(pt_regress)#1.1e-7

                #Define the model using both branches
                self.jet_model = keras.Model(inputs = inputs, outputs = [jet_id, pt_regress])

                # Define the model using both branches

                print(self.jet_model.summary())
                
                self.results_dict['num_parameters'] = self.jet_model.count_params()

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
            config['LayerName']['Conv1D_1']['ParallelizationFactor'] = 16
            config['LayerName']['Conv1D_2']['ParallelizationFactor'] = 16

            #config['LayerName']['model_input']['Precision']['result'] = self.firmware_config['input_precision']            
            #config["LayerName"]["jet_id_output"]["Precision"]["result"] = self.firmware_config['class_precision']
            #config["LayerName"]["pT_output"]["Precision"]["result"] = self.firmware_config['reg_precision']

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

            if build:
                # build the project
                self.hls_jet_model.build(csim=False, reset=True)
