import json
import os
from schema import Schema, And, Use, Optional
from math import log2
import scipy

import numpy.typing as npt
import keras
import numpy as np
from keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D, AveragePooling1D, Flatten
from hgq.layers import QConv1D, QDense,QEinsumDense, QMeanPow2,QBatchNormalization, QSoftmax,QLayerBaseSingleInput,QLayerBaseMultiInputs,  QEinsumDenseBatchnorm, QGlobalAveragePooling1D, QAdd,QSum
from hgq.config import LayerConfigScope, QuantizerConfigScope, QuantizerConfig
from hgq.regularizers import MonoL1
from hgq.constraints import MinMax
from hgq.utils.sugar import FreeEBOPs, BetaScheduler,PieceWiseSchedule,EarlyStoppingWithEbopsThres,BetaPID

from keras.models import load_model
import hls4ml
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tagger.data.tools import load_data, to_ML
from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tagger.model.HGQ2.HGQ2Model import HGQ2Model
from tagger.model.common import initialise_tensorflow

@JetModelFactory.register('MLPMixerHGQ2')
class MLPMixerHGQ2(HGQ2Model):

    schema = Schema(
            {
                "model": str,
                ## generic run config coniguration
                "run_config" : JetTagModel.run_schema,
                "model_config" : {"name" : str,
                                  "embedding_layers" : list,
                                  "classification_layers" : list,
                                  "regression_layers" : list,
                                  "beta": And(float, lambda s: 1.0 >= s >= 0.0),
                                  },
                
                "training_config" : HGQ2Model.training_config,
                
                "input_config" : HGQ2Model.input_config,

                "firmware_config" : HGQ2Model.firmware_config,
                
                "quantization_config" : None
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

            iq_conf_inputs = QuantizerConfig(k0=1, i0=11, f0=11, trainable=False,round_mode='RND',overflow_mode='SAT')
            oq_conf_jetid = QuantizerConfig(k0=0, i0=12, f0=12, trainable=False,round_mode='RND',overflow_mode='SAT')
            oq_conf_pt = QuantizerConfig(k0=1, i0=9, f0=6, trainable=False,round_mode='RND',overflow_mode='SAT')

            iq_default = QuantizerConfig(place='datalane')
            N_constituents = inputs_shape['basic_input'][0]
            n_features = inputs_shape['basic_input'][1]

            with (QuantizerConfigScope(place='datalane', heterogeneous_axis=heterogeneous_axis)):
                inp_b = keras.layers.Input((N_constituents, n_features),name='basic_input')
                #inp_b = QBatchNormalization()(inp)

                x1 = QEinsumDenseBatchnorm('bnc,cC->bnC', (N_constituents, self.model_config['embedding_layers'][0]), bias_axes='C', activation='relu',iq_conf=iq_conf_inputs)(inp_b)
                x1 = QEinsumDenseBatchnorm('bnc,cC->bnC', (N_constituents, n_features), bias_axes='C', activation='relu', )(x1)
                x2 = QEinsumDenseBatchnorm('bnc,nN->bNc', (N_constituents, n_features), bias_axes='N')(x1)
                x = QAdd(iq_confs=(iq_conf_inputs, iq_default))([inp_b, x2])
                x = QEinsumDenseBatchnorm('bnc,cC->bnC', (N_constituents, self.model_config['embedding_layers'][1]), bias_axes='C', activation='relu', )(x)
                x = QEinsumDenseBatchnorm('bnc,cC->bnC', (N_constituents, self.model_config['embedding_layers'][2]), bias_axes='C', activation='relu', )(x)
                x = QEinsumDense('bnc,n->bc', self.model_config['embedding_layers'][3])(x)

                jet_id = QEinsumDenseBatchnorm('bc,cC->bC', self.model_config['classification_layers'][0], bias_axes='C', activation='relu', )(x)
                jet_id = QEinsumDenseBatchnorm('bc,cC->bC', self.model_config['classification_layers'][1], bias_axes='C', activation='relu', )(jet_id)
                jet_id = QEinsumDenseBatchnorm('bc,cC->bC', self.model_config['classification_layers'][2], bias_axes='C', activation='relu', )(jet_id)
                jet_id = QEinsumDenseBatchnorm('bc,cC->bC', outputs_shape[0], bias_axes='C',activation='relu')(jet_id)
                jet_id = Activation('softmax', name='jet_id_output')(jet_id)

                pt_regress = QEinsumDenseBatchnorm('bc,cC->bC', self.model_config['regression_layers'][0], bias_axes='C', activation='relu', )(x)
                pt_regress = QEinsumDenseBatchnorm('bc,cC->bC', self.model_config['regression_layers'][1], bias_axes='C', activation='relu', )(pt_regress)
                pt_regress = QEinsumDenseBatchnorm('bc,cC->bC', self.model_config['regression_layers'][2], bias_axes='C', activation='relu', )(pt_regress)
                pt_regress = QEinsumDenseBatchnorm('bc,cC->bC', 1, bias_axes='C')(pt_regress)
                pt_regress = Activation('linear', name='pT_output')(pt_regress)
                #Define the model using both branches
                self.jet_model = keras.Model(inputs = inp_b, outputs = [jet_id, pt_regress])
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

            old_text = 'nnet::add<quantizer_t, quantizer_1_t, q_add_t, config12>(layer10_out, layer11_out, layer12_out); // q_add'
            new_text = """for (int ii = 0; ii < 16 * 20; ii++) {
                    auto layer11_index = ii % 20;
                    layer12_out[ii] = layer10_out[ii] + layer11_out[layer11_index];
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
