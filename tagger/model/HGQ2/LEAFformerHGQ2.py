import json
import os
from schema import Schema, And, Use, Optional
from math import log2
import scipy
import numpy.typing as npt
import keras
import numpy as np

import inspect
import string

from keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D, AveragePooling1D, Flatten,Rescaling
from hgq.layers import QAdd, QDenseT, QDense,QGlobalAveragePooling1D,QMultiHeadAttentionT, QSoftmax,QLayerBaseSingleInput,QLayerBaseMultiInputs,  QEinsumDenseBatchnorm, QGlobalAveragePooling1D, QAdd,QSum, QMultiply,QLinformerAttention
from hgq.layers.activation import QUnaryFunctionLUT
from hgq.config import LayerConfigScope, QuantizerConfigScope, QuantizerConfig
from hgq.regularizers import MonoL1
from hgq.constraints import MinMax, Min
from hgq.utils.sugar import FreeEBOPs, BetaScheduler,PieceWiseSchedule,EarlyStoppingWithEbopsThres,BetaPID

from keras.models import load_model
#import hls4ml
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tagger.data.tools import load_data, to_ML
from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tagger.model.HGQ2.HGQ2Model import HGQ2Model
from tagger.model.common import initialise_tensorflow

from hgq.utils import trace_minmax
import inspect
import string

import hgq.layers.table.dense as _table_dense

def _fixed_build_module(self, presum_idxs, presum_shape, weight_idxs):
    layers = []
    labels = iter(sorted(set(string.ascii_letters) - set(presum_idxs) - set(weight_idxs)))
    hidden_in = next(labels)
    hidden_out = next(labels)
    for _ in range(self.n_hl):
        bias_axes = f'{weight_idxs}{hidden_out}' if self.bias_axes is not None and not self.enable_bn else None
        eq = f'{presum_idxs}{hidden_in},{weight_idxs}{hidden_in}{hidden_out}->{presum_idxs}{hidden_out}'
        layers.append(
            keras.layers.EinsumDense(
                eq,
                output_shape=presum_shape[1:] + (self.d_hl,),
                activation=self.subnn_activation,
                bias_axes=bias_axes,
            )
        )
        hidden_in, hidden_out = hidden_out, hidden_in

    bias_axes = weight_idxs if self.bias_axes is not None and not self.enable_bn else None
    layers.append(
        keras.layers.EinsumDense(
            f'{presum_idxs}{hidden_in},{weight_idxs}{hidden_in}->{presum_idxs}',
            output_shape=presum_shape[1:],
            activation='linear',
            bias_axes=bias_axes,
        )
    )
    module = keras.models.Sequential(layers)
    module.build(presum_shape + (1,))  # the fix: the input is always a scalar
    return module


# Patch every class in the table module whose _build_module has the einsum signature
_patched = []
for _name, _cls in inspect.getmembers(_table_dense, inspect.isclass):
    _bm = _cls.__dict__.get('_build_module')
    if _bm is not None and 'presum_idxs' in inspect.signature(_bm).parameters:
        _cls._build_module = _fixed_build_module
        _patched.append(_name)

assert _patched, 'No table einsum-dense class found to patch'
print('Patched HGQ2 table classes:', _patched)

@JetModelFactory.register('LEAFformerHGQ2')
class LEAFformerHGQ2(JetTagModel):

    schema = Schema(
            {
                "model": str,
                ## generic run config coniguration
                "run_config" : JetTagModel.run_schema,
                "model_config" : {"name" : str,
                                  "feedforward_dim" : int,
                                  "num_heads" : int,
                                  "classification_layers" : list,
                                  "classification_parallelisation_factor" : list,
                                  "regression_layers" : list,
                                  "regression_parallelisation_factor" : list,
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
        
        scope0 = QuantizerConfigScope(k0=1, b0=8, i0=1, br=MonoL1(1e-8), overflow_mode='WRAP')

        scope1 = QuantizerConfigScope(place='datalane', k0=1, f0=6, fr=MonoL1(1e-8), ir=MonoL1(1e-8))
        betascope = LayerConfigScope(beta0=self.model_config['beta'])
        
        iq_conf = QuantizerConfig(k0=1, i0=11, f0=12, trainable=False,round_mode='RND',overflow_mode='SAT')
        oq_conf_jetid = QuantizerConfig(k0=0, i0=12, f0=12, trainable=False,round_mode='RND',overflow_mode='SAT')
        oq_conf_pt = QuantizerConfig(k0=1, i0=9, f0=6, trainable=False,round_mode='RND',overflow_mode='SAT')

        # Linformer Attention Config
        mhaconfig = QuantizerConfigScope(place='table', overflow_mode='SAT_SYM', homogeneous_axis=(0,1), bc=Min(0))

        FF_DIM = self.model_config['feedforward_dim']
        NUM_PARTICLES = inputs_shape[0]
        NUM_FEATURES = inputs_shape[1]
        numheads = self.model_config['num_heads']
        
        bn_mom = 0.75
        
        with betascope, scope0, scope1:
                inp = keras.layers.Input((NUM_PARTICLES, NUM_FEATURES),name='model_input')
                emb = QDenseT(FF_DIM, batch_norm=True, bn_momentum=bn_mom)(inp)

                with mhaconfig:
                    xa = QMultiHeadAttentionT(numheads, FF_DIM, dropout=0.0)(emb, emb)

                xa = xf = QAdd()([emb, xa])
                xf = QDenseT(FF_DIM, batch_norm=True, bn_momentum=bn_mom)(xf)
                xf = QDenseT(FF_DIM, batch_norm=True, bn_momentum=bn_mom)(xf)
                x = QAdd()([xa, xf])

                x = QGlobalAveragePooling1D()(x)
                #jetID branch, 3 layer MLP
                
                for iclass, depthclass in enumerate(self.model_config['classification_layers']):
                    if iclass == 0:
                        jet_id = QDense(depthclass, parallelization_factor=self.model_config['classification_parallelisation_factor'][iclass], name='Dense_' + str(iclass + 1) + '_jetID',activation='relu')(x)
                    else:
                        jet_id = QDense(depthclass, parallelization_factor=self.model_config['classification_parallelisation_factor'][iclass], name='Dense_' + str(iclass + 1) + '_jetID',activation='relu')(jet_id)                
                jet_id = QDense(outputs_shape[0], parallelization_factor=outputs_shape[0], activation='relu')(jet_id)
                jet_id = Activation('softmax', name='jet_id_output')(jet_id)
                #pT regression branch
                for ireg, depthreg in enumerate(self.model_config['regression_layers']):
                    if ireg == 0:
                        pt_regress = QDense(depthreg, parallelization_factor=self.model_config['regression_parallelisation_factor'][ireg], name='Dense_' + str(ireg + 1) + '_pT',activation='relu')(x)
                    else:
                        pt_regress = QDense(depthreg, parallelization_factor=self.model_config['regression_parallelisation_factor'][ireg], name='Dense_' + str(ireg + 1) + '_pT',activation='relu')(pt_regress)      
                pt_regress = QDense(1,name='pT_output', enable_oq=True,oq_conf=oq_conf_pt,iq_conf=oq_conf_pt)(pt_regress)#1.1e-7


                #Define the model using both branches
                self.jet_model = keras.Model(inputs = inp, outputs = [jet_id, pt_regress])
                print(self.jet_model.summary())
                self.results_dict['num_parameters'] = self.jet_model.count_params()

    # Redefine save and load for HGQ due to needing h5 format
    @JetTagModel.save_decorator
    def save(self, out_dir):
        # Export the model
        #model_export = tfmot.sparsity.keras.strip_pruning(self.jet_model)
        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        export_path = os.path.join(out_dir, "model/saved_model.keras")
        self.jet_model.save(export_path)
        print(f"Model saved to {export_path}")

    @JetTagModel.load_decorator
    def load(self, out_dir=None):
        # Load model
        self.jet_model = load_model(f"{out_dir}/model/saved_model.keras")

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
                
            
            with open(hls4ml_outdir+'/firmware/'+self.firmware_config['project_name']+'.cpp', 'r') as f:
                content = f.read()
                
            old_text = 'nnet::add<quantizer_t, quantizer_1_t, q_add_t, config30>(layer28_out, layer29_out, layer30_out); // q_add'
            new_text = """for (int ii = 0; ii < 16 * 16; ii++) {
                    auto layer29_index = ii % 16;
                    layer30_out[ii] = layer28_out[ii] + layer29_out[layer29_index];
                }"""

            content = content.replace(old_text, new_text)
            
            old_text = 'nnet::add<quantizer_2_t, quantizer_3_t, q_add_1_t, config39>(layer37_out, layer38_out, layer39_out); // q_add_1'
            new_text = """for (int ii = 0; ii < 16 * 16; ii++) {
                    auto layer38_index = ii % 16;
                    layer39_out[ii] = layer37_out[ii] + layer38_out[layer38_index];
                }"""

            content = content.replace(old_text, new_text)
            
            old_text = 'nnet::add<quantizer_4_t, quantizer_5_t, q_add_2_t, config48>(layer46_out, layer47_out, layer48_out); // q_add_2'
            new_text = """for (int ii = 0; ii < 16 * 16; ii++) {
                    auto layer47_index = ii % 16;
                    layer48_out[ii] = layer46_out[ii] + layer47_out[layer47_index];
                }"""

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