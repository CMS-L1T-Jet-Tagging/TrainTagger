import json
import os

import hls4ml
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from schema import Schema, And, Use, Optional
import tensorflow_model_optimization as tfmot

from tagger.model.common import WeightedGlobalAverage1D, WeightedPtResponse, choose_aggregator, initialise_tensorflow
from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tagger.model.QKerasModel import QKerasModel

from qkeras import QConv1D
from qkeras.utils import load_qmodel
from qkeras.qlayers import QActivation, QDense
from qkeras.quantizers import quantized_bits, quantized_relu
from tensorflow.keras.layers import Activation, BatchNormalization
from tagger.model.DeepSetModel import DeepSetModel

# Register the model in the factory with the string name corresponding to what is in the yaml config
@JetModelFactory.register('WeightedAverageModel')
class WeightedAverageModel(DeepSetModel):
    """WeightedAverageModel class

    Args:
        JetTagModel (_type_): Base class of a JetTagModel
    """

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

        # Initialize inputs
        inputs = tf.keras.layers.Input(shape=inputs_shape[0], name='model_input')
        mask = tf.keras.layers.Input(shape=inputs_shape[1], name='masking_input')
        pt = tf.keras.layers.Input(shape=inputs_shape[2], name='pt_input')
        mask_permuted = tf.keras.layers.Permute((2,1), name='pt_weights_permute')(mask)
        pt_mask = tf.keras.layers.Cropping1D(cropping=((9,0)), name='pt_mask_crop')(mask_permuted)
        pt_mask = tf.keras.layers.Flatten(name='pt_mask')(pt_mask)

        # Main branch
        main = BatchNormalization(name='norm_input')(inputs)

        # Make Conv1D layers
        for iconv1d, depthconv1d in enumerate(self.model_config['conv1d_layers']):
            main = QConv1D(filters=depthconv1d, kernel_size=1, name='Conv1D_' + str(iconv1d + 1), **self.common_args)(main)
            main = QActivation(
                activation=quantized_relu(self.quantization_config['quantizer_bits'], 0), name='relu_' + str(iconv1d + 1)
            )(main)
            # ToDo: fix the bits_int part later, ie use the default not 0

        # Linear activation to change HLS bitwidth to fix overflow in AveragePooling
        main = QActivation(activation='quantized_bits(18,8)', name='act_pool')(main)

        # Apply the constituents mask
        main = tf.keras.layers.Multiply(name='apply_mask')([main, mask])

        # Make the pT weights
        pt_weights = QConv1D(filters=1, kernel_size=1, name='Conv1D_weights', **self.common_args)(main)
        pt_weights = tf.keras.layers.Flatten(name='pt_weights_flat')(pt_weights)  # shape: (batch, timesteps)
        pt_weights = QActivation('softplus', name='pt_weights_softplus_1')(pt_weights)  # Ensure weights are positive
        pt_weights = tf.keras.layers.Multiply(name='apply_pt_mask_1')([pt_weights, pt_mask])
        pt_correction = QConv1D(filters=1, kernel_size=1, name='Conv1D_correction', **self.common_args)(main)
        pt_correction = tf.keras.layers.Flatten(name='pt_correction_flat')(pt_correction)  # shape: (batch, timesteps)
        pt_correction = tf.keras.layers.Multiply(name='apply_pt_mask_2')([pt_correction, pt_mask])

        # Weighted Global Average Pooling
        main = WeightedGlobalAverage1D(name='weighted_avg_pool')([main, pt_weights])  # Apply the learned pT weights before pooling

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

        # Make fully connected dense layers for regression task
        pt_weights = QDense(
            16,
            name='weights_output',
            kernel_quantizer=quantized_bits(
                self.quantization_config['pt_output_quantization'][0],
                self.quantization_config['pt_output_quantization'][1],
                alpha=self.quantization_config['quantizer_alpha_val'],
            ),
            bias_quantizer=quantized_bits(
                self.quantization_config['pt_output_quantization'][0],
                self.quantization_config['pt_output_quantization'][1],
                alpha=self.quantization_config['quantizer_alpha_val'],
            ),
            kernel_initializer='lecun_uniform',
            )(pt_weights)
        pt_weights = QActivation('softplus', name='pt_weights_softplus_2')(pt_weights)

        pt_correction = QDense(
            16,
            name='corrections_output',
            kernel_quantizer=quantized_bits(
                self.quantization_config['pt_output_quantization'][0],
                self.quantization_config['pt_output_quantization'][1],
                alpha=self.quantization_config['quantizer_alpha_val'],
            ),
            bias_quantizer=quantized_bits(
                self.quantization_config['pt_output_quantization'][0],
                self.quantization_config['pt_output_quantization'][1],
                alpha=self.quantization_config['quantizer_alpha_val'],
            ),
            kernel_initializer='lecun_uniform',
            )(pt_correction)
        pt_output = WeightedPtResponse(name="pT_output")([pt_weights, pt_correction, pt])

        # Define the model using both branches
        self.jet_model = tf.keras.Model(inputs=[inputs, mask, pt], outputs=[jet_id, pt_output])

        print(self.jet_model.summary())

    def fit(
        self,
        X_train: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
        y_train: npt.NDArray[np.float64],
        pt_target_train: npt.NDArray[np.float64],
        sample_weight: [npt.NDArray[np.float64], npt.NDArray[np.float64]],
    ):
        """Fit the model to the training dataset

        Args:
            X_train (npt.NDArray[np.float64]): X train dataset, containts inputs and pt
            y_train (npt.NDArray[np.float64]): y train classification targets
            pt_target_train (npt.NDArray[np.float64]): y train pt regression targets
            sample_weight (npt.NDArray[np.float64]): sample weighting
        """

        # Train the model using hyperparameters in yaml config
        inputs, mask, pt = X_train
        self.history = self.jet_model.fit(
            {'model_input': inputs, 'masking_input': mask, 'pt_input': pt},
            {self.loss_name + self.output_id_name: y_train, self.loss_name + self.output_pt_name: pt_target_train},
            sample_weight={
                self.loss_name + self.output_id_name: sample_weight[0],
                self.loss_name + self.output_pt_name: sample_weight[1],
            },
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            verbose=self.run_config['verbose'],
            validation_split=self.training_config['validation_split'],
            callbacks=self.callbacks,
            shuffle=True,
        )

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
        config['LayerName']['model_input']['Precision']['result'] = self.firmware_config['input_precision']
        config['LayerName']['masking_input']['Precision']['result'] = self.firmware_config['mask_precision']
        config['LayerName']['pt_input']['Precision']['result'] = self.firmware_config['input_precision']

        # Configuration for conv1d layers
        # hls4ml automatically figures out the paralellization factor
        # config['LayerName']['Conv1D_1']['ParallelizationFactor'] = 8
        # config['LayerName']['Conv1D_2']['ParallelizationFactor'] = 8

        # Additional config
        for layer in self.jet_model.layers:
            layer_name = layer.__class__.__name__

            if layer_name in ["BatchNormalization", "InputLayer"]:
                config["LayerName"][layer.name]["Precision"] = self.firmware_config['input_precision']
                config["LayerName"][layer.name]["result"] = self.firmware_config['input_precision']
                config["LayerName"][layer.name]["Trace"] = not build

            elif layer_name in ["Permute", "Concatenate", "Flatten", "Reshape", "UpSampling1D", "Add"]:
                print("Skipping trace for:", layer.name)
            else:
                config["LayerName"][layer.name]["Trace"] = not build

        config["LayerName"]["jet_id_output"]["Precision"]["result"] = self.firmware_config['class_precision']
        config["LayerName"]["jet_id_output"]["Implementation"] = "latency"
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

        if build:
            # build the project
            self.hls_jet_model.build(csim=False, reset=True)

    # Override load to allow node edge projection to also be loaded
    @JetTagModel.load_decorator
    def load(self, out_dir: str = "None"):
        """Load the model file

        Args:
            out_dir (str, optional): Where to load it if not in the output_directory. Defaults to "None".
        """

        # Additional custom objects for attention layers
        custom_objects_ = {
            "WeightedGlobalAverage1D": WeightedGlobalAverage1D,
            "WeightedPtResponse": WeightedPtResponse
        }

        # Load the model
        self.jet_model = load_qmodel(f"{out_dir}/model/saved_model.keras", custom_objects=custom_objects_)

