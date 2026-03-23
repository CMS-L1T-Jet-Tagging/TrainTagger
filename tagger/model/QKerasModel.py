"""QKeras model parent class

Written 29/09/2025 cebrown@cern.ch
"""

import json
import os
import re

import hls4ml
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from schema import Schema, And, Use, Optional

# Qkeras
from qkeras.quantizers import quantized_bits
from qkeras.utils import load_qmodel
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tagger.model.common import AAtt, AttentionPooling, choose_aggregator
from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tagger.data.tools import constituents_mask

class QKerasModel(JetTagModel):
    """QKerasModel class

    Args:
        JetTagModel (_type_): Base class of a JetTagModel
    """

    quantization_schema = {'quantizer_bits' : And(int, lambda s: 64 >= s >= 0),
                           'quantizer_bits_int' : And(int, lambda s: 32 >= s >= 0),
                           'quantizer_alpha_val' : And(float, lambda s: 1.0 >= s >= 0.0),
                           'pt_output_quantization' : list,
                           # 'pt_layers_bits': list,
                           }

    training_config_schema =    {"weight_method" : And(str, lambda s: s in  ["none", "ptref", "onlyclass"]),
                                 "validation_split" : And(float, lambda s: s > 0.0),
                                 "offsets" : bool,
                                 "epochs" : And(int, lambda s: s >= 1),
                                 "batch_size" : And(int, lambda s: s >= 1),
                                 "learning_rate" : And(float, lambda s: s > 0.0),
                                 "loss_weights" : And(list, lambda s: len(s) == 2),
                                 "initial_sparsity" : And(float, lambda s: 1.0 >= s >= 0.0),
                                 "final_sparsity" : And(float, lambda s: 1.0 >= s >= 0.0),
                                 "EarlyStopping_patience" : And(int, lambda s: s > 0),
                                 "ReduceLROnPlateau_factor" : And(float, lambda s: 1.0 >= s >= 0.0),
                                 "ReduceLROnPlateau_patience" : int,
                                 "ReduceLROnPlateau_min_lr" : And(float, lambda s: s >= 0.0)}

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

    def compile_model(self, num_samples: int, loss_weights: list = [1.0, 1.0]):
        """compile the model generating callbacks and loss function
        Args:
            num_samples (int): Number of samples in the training set used for scheduling
        """

        # Define the callbacks using hyperparameters in the config
        self.callbacks = [
            EarlyStopping(monitor='val_prune_low_magnitude_pT_output_loss',
                          patience=self.training_config['EarlyStopping_patience'],
                          restore_best_weights=True,
                          verbose=2),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.training_config['ReduceLROnPlateau_factor'],
                patience=self.training_config['ReduceLROnPlateau_patience'],
                min_lr=self.training_config['ReduceLROnPlateau_min_lr'],
                verbose=2,
            ),
            ModelCheckpoint(
                filepath=os.path.join(f"{self.output_directory}","weights_epoch_{epoch:02d}.h5"),
                save_weights_only=False,
                save_freq="epoch"
            )
        ]

        # Define the pruning
        if 'initial_sparsity' in self.training_config:
            self._prune_model(num_samples)

        def asymmetric_huber_loss(delta=.1, pu=2., alpha=1.5):
            """
            Huber loss with asymmetric penalization.

            Args:
                delta: Huber threshold.
                alpha: Weight for underestimation (y_true > y_pred).
            """
            def loss(y_true, y_pred):
                # Minbias: punish overestimation
                pu_punish = tf.where(
                    (y_true == -1) & (y_pred > 1),
                    1.0 + pu * (y_pred - 1.0), # scaling proportional to excess
                    0.)
                y_true = tf.where(y_true == -1, 0.0, y_true)

                # punish underestimation more for all other samples
                residual = y_true - y_pred
                overest = tf.where(residual > 0, (abs(residual) * alpha) + 1., 0.)  # Penalize overestimation more
                weights = overest + pu_punish + 1.0  # Add 1 as base value

                abs_res = tf.abs(residual)
                quadratic = tf.minimum(abs_res, delta)
                linear = abs_res - quadratic

                return tf.reduce_mean(weights * (0.5 * quadratic**2 + delta * linear))

            return loss

        # compile the tensorflow model setting the loss and metrics
        self.jet_model.compile(
            optimizer='adam',
            loss={
                self.loss_name + self.output_id_name: 'categorical_crossentropy',
                # self.loss_name + self.output_pt_name: tf.keras.losses.Huber(),
                self.loss_name + self.output_pt_name: asymmetric_huber_loss(),
            },
            loss_weights=loss_weights,
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
        train_dict: dict,
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
            train_dict,
            {self.loss_name + self.output_id_name: y_train, self.loss_name + self.output_pt_name: pt_target_train},
            sample_weight=sample_weight,
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            verbose=self.run_config['verbose'],
            validation_split=self.training_config['validation_split'],
            callbacks=self.callbacks,
            shuffle=True,
        )

    def prepare_inputs(self, raw_inputs: dict) -> dict:
        """Prepare the input dictionary for the model from a list of arrays

        Args:
            raw_inputs: Dictionary of all possible input arrays (currently requires basic_input, jet_pt and jet_eta)

        Returns:
            dict: Dictionary of required input arrays
        """
        input_dict = {
            'basic_input': raw_inputs['basic_input'],
            'basic_mask': constituents_mask(raw_inputs['basic_input'], 10),
            'pt_mask': constituents_mask(raw_inputs['basic_input'], 10)[:, :, 0],
            'constituent_pt': raw_inputs['basic_input'][:, :, 0],
            'constituent_fraction': raw_inputs['basic_input'][:, :, 0] / np.sum(raw_inputs['basic_input'][:, :, 0], axis=-1, keepdims=True),
            'inverse_jet_pt': 1 / raw_inputs['jet_pt'].reshape(-1, 1),
            'jet_pt': raw_inputs['jet_pt'],
        }

        for key in list(input_dict.keys()):
            if key not in self.inputs['basic_features']:
                del input_dict[key]

        if len(self.inputs['custom_features']) > 0:
            jet_features = np.empty((raw_inputs['basic_input'].shape[0], len(self.inputs['custom_features'])))
            for i, k in enumerate(self.inputs['custom_features']):
                jet_features[:, i] = raw_inputs[k]
            input_dict['jet_features'] = jet_features

        input_shapes = {k: v.shape[1:] for k, v in input_dict.items()}

        return input_dict, input_shapes

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
