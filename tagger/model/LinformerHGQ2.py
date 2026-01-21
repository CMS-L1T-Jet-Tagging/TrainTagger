import json
import os
from schema import Schema, And, Use, Optional
from math import log2

import numpy.typing as npt
import keras
import numpy as np
from keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D, AveragePooling1D, Flatten,Rescaling
from hgq.layers import QConv1D, QDense, QMeanPow2,QBatchNormalization, QSoftmax,QLayerBaseSingleInput,QLayerBaseMultiInputs,  QEinsumDenseBatchnorm, QGlobalAveragePooling1D, QAdd,QSum, QMultiply,QLinformerAttention
from hgq.layers.activation import QUnaryFunctionLUT
from hgq.config import LayerConfigScope, QuantizerConfigScope, QuantizerConfig
from hgq.regularizers import MonoL1
from hgq.constraints import MinMax
from hgq.utils.sugar import FreeEBOPs, BetaScheduler,PieceWiseSchedule,EarlyStoppingWithEbopsThres,BetaPID

from keras.models import load_model
import hls4ml
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tagger.data.tools import load_data, to_ML
from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tagger.model.common import initialise_tensorflow,cosine_decay_restarts

from hgq.utils import trace_minmax

@JetModelFactory.register('LinformerHGQ2')
class LinformerHGQ2(JetTagModel):

    schema = Schema(
            {
                "model": str,
                ## generic run config coniguration
                "run_config" : JetTagModel.run_schema,
                "model_config" : {"name" : str,
                                  "feedforward_dim" : int,
                                  "projection_k" : int,
                                  "num_heads" : int,
                                  "classification_layers" : list,
                                  "classification_parallelisation_factor" : list,
                                  "regression_layers" : list,
                                  "regression_parallelisation_factor" : list,
                                  "beta": And(float, lambda s: 1.0 >= s >= 0.0),
                                  },

                "quantization_config" : {'pt_output_quantization' : list},

                "training_config" :     {"weight_method" : And(str, lambda s: s in  ["none", "ptref", "onlyclass"]),
                                         "validation_split" : And(float, lambda s: s > 0.0),
                                         "target_ebops" : int,
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

        scope0 = QuantizerConfigScope(k0=1, b0=8, i0=1, br=MonoL1(1e-8), overflow_mode='WRAP')

        scope1 = QuantizerConfigScope(place='datalane', k0=1, f0=6, fr=MonoL1(1e-8), ir=MonoL1(1e-8))
        betascope = LayerConfigScope(beta0=self.model_config['beta'])
        
        iq_conf = QuantizerConfig(k0=1, i0=11, f0=12, trainable=False,round_mode='RND',overflow_mode='SAT')
        oq_conf_jetid = QuantizerConfig(k0=0, i0=12, f0=12, trainable=False,round_mode='RND',overflow_mode='SAT')
        oq_conf_pt = QuantizerConfig(k0=1, i0=9, f0=6, trainable=False,round_mode='RND',overflow_mode='SAT')

        # Linformer Attention Config
        mhaconfig = QuantizerConfigScope(
            k0=1, i0=1, f0=6, round_mode='RND', overflow_mode='SAT',
            bc=MinMax(1, 8)
        )

        FF_DIM = self.model_config['feedforward_dim']
        NUM_PARTICLES = inputs_shape[0]
        NUM_FEATURES = inputs_shape[1]
        PROJ_K = self.model_config['projection_k']
        numheads = self.model_config['num_heads']
        
        with betascope, scope0, scope1:
                qkv = keras.layers.Input((NUM_PARTICLES, NUM_FEATURES),name='model_input')
                emb = QBatchNormalization(name='norm_input')(qkv)
                emb = QDense(FF_DIM, activation='relu')(qkv)
                emb = QDense(FF_DIM, activation='relu')(emb)

                with mhaconfig:
                    x = QLinformerAttention(numheads, key_dim=FF_DIM // numheads,
                            lin_kv_proj_dim=PROJ_K, name='attention1', dropout=0)(emb, emb, emb)

                res = QAdd()([x, emb])
                x = QDense(FF_DIM * 2, activation='relu')(res)
                x = QDense(FF_DIM, activation='relu')(x)
                res = QAdd()([x, res])

                x = QDense(FF_DIM * 2, activation='relu')(res)
                x = QDense(FF_DIM, activation='relu')(x)
                lo2 = QAdd()([x, res])

                x = GlobalAveragePooling1D(data_format='channels_last')(lo2)
                
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
                self.jet_model = keras.Model(inputs = qkv, outputs = [jet_id, pt_regress])
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
            #config["Model"]["Strategy"]="distributed_arithmetic"
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

    
    
    def compile_model(self, num_samples: int):
        
        """compile the model generating callbacks and loss function
        Args:
            num_samples (int): Number of samples in the training set used for scheduling
        """

        reduce_lr = ReduceLROnPlateau(
                                        monitor='val_loss', factor=0.8, patience=50,
                                        min_lr=1e-5, cooldown=200, min_delta=0.05
                                    )
        
        es = EarlyStoppingWithEbopsThres(monitor="val_loss",
                                         patience=150,
                                         verbose=1,
                                         mode="min",
                                         restore_best_weights=True,
                                         start_from_epoch=75,
                                         ebops_threshold=self.training_config['target_ebops'] + 100000
                                        )
        terminate_on_nan = keras.callbacks.TerminateOnNaN()

        ebops_tracker = FreeEBOPs()
        ebops_scheduler = BetaPID(
            p=1, i=0.1, d=0,
            target_ebops=self.training_config['target_ebops'],
            init_beta=1e-10, warmup=10,
            max_beta=5e-6, damp_beta_on_target=0.5
        )
        # Define the callbacks using hyperparameters in the config        
        self.callbacks = [ebops_tracker, terminate_on_nan, ebops_scheduler, reduce_lr, es]

        # compile the tensorflow model setting the loss and metrics
        self.jet_model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=self.training_config['learning_rate']),
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