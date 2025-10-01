import json
import os
from schema import Schema, And, Use, Optional


import tensorflow as tf
import tensorflow_model_optimization as tfmot
from HGQ import FreeBOPs, ResetMinMax, to_proxy_model, trace_minmax
from HGQ.layers import HConv1D, HConv1DBatchNorm, HDense, HQuantize, PAveragePooling1D, PFlatten, Signature

# Qkeras
from qkeras.utils import load_qmodel
from tensorflow.keras.layers import Activation

import hls4ml

from tagger.data.tools import load_data, to_ML
from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tagger.model.QKerasModel import QKerasModel
from tagger.model.common import initialise_tensorflow


@JetModelFactory.register('DeepSetModelHGQ')
class DeepSetModelHGQ(QKerasModel):
    
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

        beta = self.model_config["beta"]

        # Initialize inputs
        inputs = tf.keras.layers.Input(shape=inputs_shape, name='model_input')
        L, C = inputs_shape
        # Main branch
        main = HQuantize(name='quant1', beta=beta)(inputs)
        main = HConv1DBatchNorm(filters=10, activation='relu', kernel_size=1, beta=beta, parallel_factor=1, name='Conv1D_1')(
            main
        )
        main = HConv1D(filters=10, activation='relu', kernel_size=1, beta=beta, parallel_factor=1, name='Conv1D_2')(main)

        # Make Conv1D layers

        main = PAveragePooling1D(L, name='avgpool')(main)
        main = PFlatten()(main)

        # Now split into jet ID and pt regression
        jet_id = HDense(32, beta=beta, activation='relu', parallel_factor=1, name='Dense_1_jetID')(main)
        jet_id = HDense(16, name='Dense_2_jetID', beta=beta, activation='relu', parallel_factor=1)(jet_id)

        jet_id = HDense(outputs_shape[0], beta=beta, name='Dense_3_jetID')(jet_id)
        jet_id = Activation('softmax', name='act_jet')(jet_id)
        jet_id = Signature(bits=18, int_bits=8, keep_negative=0, name='jet_id_output')(jet_id)
        # pT regression branch
        pt_regress = HDense(10, name='Dense_1_pT', parallel_factor=1, beta=beta, activation='relu')(main)

        pt_regress = HDense(1, beta=beta, parallel_factor=1, name='pT_out')(pt_regress)
        pt_regress = Signature(
            bits=self.quantization_config['pt_output_quantization'][0],
            int_bits=self.quantization_config['pt_output_quantization'][1],
            keep_negative=0,
            name='pT_output',
        )(pt_regress)

        # Define the model using both branches
        self.jet_model = tf.keras.Model(inputs=inputs, outputs=[jet_id, pt_regress])

        print(self.jet_model.summary())

    # Redefine save and load for HGQ due to needing h5 format
    @JetTagModel.save_decorator
    def save(self, out_dir):
        # Export the model
        model_export = tfmot.sparsity.keras.strip_pruning(self.jet_model)
        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        export_path = os.path.join(out_dir, "model/saved_model.h5")
        model_export.save(export_path)
        print(f"Model saved to {export_path}")

    @JetTagModel.load_decorator
    def load(self, out_dir=None):
        # Load model

        self.jet_model = load_qmodel(f"{out_dir}/model/saved_model.h5")
