import json
import os

import hls4ml
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from schema import Schema, And, Use, Optional

from tagger.model.common_tensorflow import choose_aggregator, initialise_tensorflow
from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tagger.model.QKerasModel import QKerasModel

from qkeras import QConv1D
from qkeras.qlayers import QActivation, QDense
from qkeras.quantizers import quantized_bits, quantized_relu
from tensorflow.keras.layers import Activation, BatchNormalization

# Register the model in the factory with the string name corresponding to what is in the yaml config
@JetModelFactory.register('WeightedAverageSimpleModel')
class WeightedAverageSimpleModel(QKerasModel):
    """WeightedAverageSimpleModel class

    Args:
        JetTagModel (_type_): Base class of a JetTagModel
    """

    schema = Schema(
            {
                "model": str,
                ## generic run config coniguration
                "run_config" : JetTagModel.run_schema,
                "model_config" : {"name" : str,
                                  "conv1d_layers" : list,
                                  "classification_layers" : list,
                                  "regression_layers" : list,
                                  "kernel_initializer" : str,
                                  "aggregator" : And(str, lambda s: s in  ["mean", "max", "attention"])},
                "quantization_config" : QKerasModel.quantization_schema,
                "training_config" : QKerasModel.training_config_schema,
                ## generic hls4ml configuration
                "firmware_config" : {"input_precision" : dict,
                                    "class_precision" : str,
                                    "reg_precision": str,
                                    "clock_period" : And(float, lambda s: 0.0 < s <= 10),
                                    "fpga_part" : str,
                                    "project_name" : str},
                "inputs" : {
                    "basic_input_config": list,
                    "basic_features": list,
                    "jet_features": list}
            }
    )

    def build_model(self, inputs_shape: tuple, outputs_shape: tuple):
        """build model override, makes the model layer by layer

        Args:
            inputs_shape (tuple): Shape of the input
            outputs_shape (tuple): Shape of the output

        Additional hyperparameters in the config
            conv1d_layers: List of number of nodes for each layer of the conv1d layers.
            classifier_layers: List of number of nodes for each layer of the classifier MLP.
            regression_layers: List of number of nodes for each layer of the regression MLP
            aggregator: String that specifies the type of aggregator to use after the conv1D net.
        """

        initialise_tensorflow(self.run_config['num_threads'])

        self.common_args = {
            'kernel_quantizer': quantized_bits(
                self.quantization_config['quantizer_bits'],
                self.quantization_config['quantizer_bits_int'],
                alpha=self.quantization_config['quantizer_alpha_val'],
            ),
            'bias_quantizer': quantized_bits(
                self.quantization_config['quantizer_bits'],
                self.quantization_config['quantizer_bits_int'],
                alpha=self.quantization_config['quantizer_alpha_val'],
            ),
            'kernel_initializer': self.model_config['kernel_initializer'],
        }

        self.pt_args = {
            'kernel_quantizer': quantized_bits(
                self.quantization_config['pt_output_quantization'][0],
                self.quantization_config['pt_output_quantization'][1],
                alpha=self.quantization_config['quantizer_alpha_val'],
            ),
            'bias_quantizer' : quantized_bits(
                self.quantization_config['pt_output_quantization'][0],
                self.quantization_config['pt_output_quantization'][1],
                alpha=self.quantization_config['quantizer_alpha_val'],
            )
        }

        # Initialize inputs
        inputs = tf.keras.layers.Input(shape=inputs_shape['basic_input'], name='basic_input')
        pt = tf.keras.layers.Input(shape=inputs_shape['constituent_fraction'], name='constituent_fraction')
        jet_features = tf.keras.layers.Input(shape=inputs_shape['jet_features'], name='jet_features')
        pt_mask = tf.keras.layers.Input(shape=inputs_shape['pt_mask'], name='pt_mask')

        # Main branch
        main = BatchNormalization(name='norm_basic_input')(inputs)
        jet_features_norm = BatchNormalization(name='norm_jet_features')(jet_features)

        # Make Conv1D layers
        for iconv1d, depthconv1d in enumerate(self.model_config['conv1d_layers']):
            main = QConv1D(filters=depthconv1d, kernel_size=1, name='Conv1D_' + str(iconv1d + 1), **self.common_args)(main)
            main = QActivation(
                activation=quantized_relu(self.quantization_config['quantizer_bits'], 0), name='relu_' + str(iconv1d + 1)
            )(main)
            # ToDo: fix the bits_int part later, ie use the default not 0

        # Make the pT weights and corrections
        pt_weights = QConv1D(filters=1, kernel_size=1, name='Conv1D_pt_weights', **self.common_args)(main)
        pt_weights = tf.keras.layers.Flatten(name='Conv1D_pt_weights_flat')(pt_weights)  # shape: (batch, timesteps)
        pt_weights = QActivation(activation=quantized_relu(self.quantization_config['quantizer_bits'] +2 , 3), name='Conv1D_pt_weights_relu')(pt_weights)  # Ensure positive weights
        pt_weights = tf.keras.layers.Multiply(name='apply_pt_mask_weights')([pt_weights, pt_mask])

        # Weighted Global Average Pooling
        main = QActivation(activation='quantized_bits(18,8)', name='act_pool')(main)
        main = tf.keras.layers.GlobalAveragePooling1D(name='avg_pooling')(main)
        main = tf.keras.layers.Concatenate(name='concat_jet_features')([main, jet_features_norm])

        # Now split into jet ID and pt regression
        # Make fully connected dense layers for classification task
        for iclass, depthclass in enumerate(self.model_config['classification_layers']):
            if iclass == 0:
                jet_id = QDense(depthclass, name='Dense_' + str(iclass + 1) + '_jetID', **self.common_args)(main)
            else:
                jet_id = QDense(depthclass, name='Dense_' + str(iclass + 1) + '_jetID', **self.common_args)(jet_id)
            jet_id = QActivation(
                activation=quantized_relu(self.quantization_config['quantizer_bits'], 0),
                name='relu_' + str(iclass + 1) + '_jetID',
            )(jet_id)
            # ToDo: fix the bits_int part later, ie use the default not 0

        # Make output layer for classification task
        jet_id = QDense(outputs_shape[0], name='Dense_' + str(iclass + 2) + '_jetID', **self.common_args)(jet_id)
        jet_id = Activation('softmax', name='jet_id_output')(jet_id)

        # concat jet features to pt weights
        pt_weights = tf.keras.layers.Concatenate(name='concat_jet_features_pt_weights')([pt_weights, jet_features_norm])

        # Make fully connected dense layers for regression task
        pt_weights = QDense(16, name='Dense_pt_weights_output_0', **self.common_args)(pt_weights)
        pt_weights = QActivation(
            activation=quantized_relu(self.quantization_config['reg_quantizer_bits'] , self.quantization_config['reg_quantizer_bits_int']),
            name='pt_weights_output_0')(pt_weights)
        pt_weights = QDense(16, name='Dense_pt_weights_output', **self.common_args)(pt_weights)
        pt_weights = QActivation(
            activation=quantized_relu(self.quantization_config['reg_quantizer_bits'] , self.quantization_config['reg_quantizer_bits_int']),
            name='pt_weights_output')(pt_weights)

        weighted_pt = tf.keras.layers.Multiply(name='apply_pt_weights')([pt_weights, pt])
        pt_output_dense = QDense(1, name='pT_output_dense',
            kernel_initializer=tf.keras.initializers.Ones(), # all weights set to 1 to perform a sum
            use_bias = False,
            trainable=False,
            **self.pt_args)(weighted_pt) # fix weights at 1 to perform sum, not updated during training
        pt_output = QActivation(
            activation=quantized_relu(16 , 2),
            name='pT_output')(pt_output_dense)  # Ensure positive output, and cap

        # Define the model using both branches
        self.jet_model = tf.keras.Model(inputs=[inputs, pt, jet_features, pt_mask], outputs=[jet_id, pt_output])

        print(self.jet_model.summary())

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
        config['IOType'] = 'io_parallel'
        for layer in self.firmware_config['input_precision']:
            config['LayerName'][layer]['Precision']['result'] = self.firmware_config['input_precision'][layer]

        # Check that the input layers in the model match the input_precision keys in the firmware config
        input_layer_names = [layer.name for layer in self.jet_model.layers if isinstance(layer, tf.keras.layers.InputLayer)]
        assert set(input_layer_names) == set(self.firmware_config['input_precision'].keys())

        # Configuration for conv1d layers
        # hls4ml does not !!! automatically figure out the paralellization factor, this leads to csim, hdl sim errors
        config['LayerName']['Conv1D_1']['ParallelizationFactor'] = 16
        config['LayerName']['Conv1D_2']['ParallelizationFactor'] = 16
        config['LayerName']['Conv1D_pt_weights']['ParallelizationFactor'] = 16
        config['LayerName']['apply_pt_weights']['ParallelizationFactor'] = 16

        # Additional config
        for layer in self.jet_model.layers:
            layer_name = layer.__class__.__name__
            if layer_name in ["BatchNormalization", "InputLayer"]:
                for k in self.firmware_config['input_precision'].keys():
                    if k in layer.name:
                        precision = self.firmware_config['input_precision'][k]
                        break  # stop once we found a match
                config["LayerName"][layer.name]["Precision"] = precision
                config["LayerName"][layer.name]["result"] = precision
                config["LayerName"][layer.name]["Trace"] = not build

            elif layer_name in ["Permute", "Concatenate", "Flatten", "Reshape", "UpSampling1D", "Add"]:
                print("Skipping trace for:", layer.name)
            else:
                config["LayerName"][layer.name]["Trace"] = not build

        config["LayerName"]["jet_id_output"]["Precision"]["result"] = self.firmware_config['class_precision']
        config["LayerName"]["jet_id_output"]["Implementation"] = "stable"
        config["LayerName"]["pT_output"]["Precision"]["result"] = self.firmware_config['reg_precision']
        config["LayerName"]["pT_output"]["Implementation"] = "latency"

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

        old_text = '#include <tuple>'
        new_text = ""

        # manually remove the #include <tuple>
        path = hls4ml_outdir + '/firmware/defines.h'
        with open(path, 'r') as f:
            content = f.read()

        content = content.replace('#include <tuple>', '')

        with open(path, 'w') as f:
            f.write(content)

        if build:
            # build the project
            self.hls_jet_model.build(csim=False, reset=True)
