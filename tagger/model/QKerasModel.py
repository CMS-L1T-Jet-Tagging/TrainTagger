"""QKeras model parent class

Written 29/09/2025 cebrown@cern.ch
"""

import json
import os

import hls4ml
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from qkeras import QConv1D
from qkeras.qlayers import QActivation, QDense

# Qkeras
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.utils import load_qmodel
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, BatchNormalization

from tagger.model.common import AAtt, AttentionPooling, choose_aggregator
from tagger.model.JetTagModel import JetModelFactory, JetTagModel

class QKerasModel(JetTagModel):
    """QKerasModel class

    Args:
        JetTagModel (_type_): Base class of a JetTagModel
    """

    def set_dictionary(self):

                # Define some common arguments, taken from the yaml config
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

        # Set some tensorflow constants
        NUM_THREADS = self.run_config['num_threads']
        os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(NUM_THREADS)
        os.environ["TF_NUM_INTEROP_THREADS"] = str(NUM_THREADS)

        tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)
        tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)

        tf.keras.utils.set_random_seed(420)  # not a special number


    def _prune_model(self, num_samples: int):
        """Pruning setup for the model, internal model function called by compile

        Args:
            num_samples (int): number of samples in the training set used for scheduling
        """

        print("Begin pruning the model...")

        # Calculate the ending step for pruning
        end_step = (
            np.ceil(num_samples / self.training_config['batch_size']).astype(np.int32) * self.training_config['epochs']
        )

        # Define the pruned model
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=self.training_config['initial_sparsity'],
                final_sparsity=self.training_config['final_sparsity'],
                begin_step=0,
                end_step=end_step,
            )
        }
        self.jet_model = tfmot.sparsity.keras.prune_low_magnitude(self.jet_model, **pruning_params)

        # Add preface to loss name
        self.loss_name = 'prune_low_magnitude_'

        # Add pruning callback
        self.callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())

    def compile_model(self, num_samples: int):
        """compile the model generating callbacks and loss function
        Args:
            num_samples (int): Number of samples in the training set used for scheduling
        """

        # Define the callbacks using hyperparameters in the config
        self.callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.training_config['EarlyStopping_patience']),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.training_config['ReduceLROnPlateau_factor'],
                patience=self.training_config['ReduceLROnPlateau_patience'],
                min_lr=self.training_config['ReduceLROnPlateau_min_lr'],
            ),
        ]

        # Define the pruning
        self._prune_model(num_samples)

        # compile the tensorflow model setting the loss and metrics
        self.jet_model.compile(
            optimizer='adam',
            loss={
                self.loss_name + self.output_id_name: 'categorical_crossentropy',
                self.loss_name + self.output_pt_name: tf.keras.losses.Huber(),
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

        # Train the model using hyperparameters in yaml config
        self.history = self.jet_model.fit(
            {'model_input': X_train},
            {self.loss_name + self.output_id_name: y_train, self.loss_name + self.output_pt_name: pt_target_train},
            sample_weight=sample_weight,
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            verbose=self.run_config['verbose'],
            validation_split=self.training_config['validation_split'],
            callbacks=self.callbacks,
            shuffle=True,
        )

    # Decorated with save decorator for added functionality
    @JetTagModel.save_decorator
    def save(self, out_dir: str = "None"):
        """Save the model file

        Args:
            out_dir (str, optional): Where to save it if not in the output_directory. Defaults to "None".
        """
        # Export the model
        model_export = tfmot.sparsity.keras.strip_pruning(self.jet_model)

        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        # Use keras save format !NOT .h5! due to depreciation
        export_path = os.path.join(out_dir, "model/saved_model.keras")
        model_export.save(export_path)
        print(f"Model saved to {export_path}")

    @JetTagModel.load_decorator
    def load(self, out_dir: str = "None"):
        """Load the model file

        Args:
            out_dir (str, optional): Where to load it if not in the output_directory. Defaults to "None".
        """

        # Additional custom objects for attention layers
        custom_objects_ = {
            "AAtt": AAtt,
            "AttentionPooling": AttentionPooling,
        }
        # Load the model
        self.jet_model = load_qmodel(f"{out_dir}/model/saved_model.keras", custom_objects=custom_objects_)
