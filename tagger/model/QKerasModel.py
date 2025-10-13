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
from schema import Schema, And, Use, Optional

# Qkeras
from qkeras.quantizers import quantized_bits
from qkeras.utils import load_qmodel
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from qkeras.qlayers import QDense
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D

from tagger.model.JetTagModel import JetModelFactory, JetTagModel


class AAtt(tf.keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer):
    """Attention Layer class

    Args:
        tf.keras.layers.Layer (_type_): tensorflow layer wrapper
        tfmot.sparsity.keras.PrunableLayer (_type_): prunable layer wrapper
    """

    def __init__(self, d_model=16, nhead=2, bits=9, bits_int=2, alpha_val=1, **kwargs):
        super(AAtt, self).__init__(**kwargs)

        self.d_model = d_model
        self.n_head = nhead
        self.bits = bits
        self.bits_int = bits_int
        self.alpha_val = alpha_val

        self.qD = QDense(self.d_model, **kwargs)
        self.kD = QDense(self.d_model, **kwargs)
        self.vD = QDense(self.d_model, **kwargs)
        self.outD = QDense(self.d_model, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "d_model": (self.d_model),
            "nhead": (self.n_head),
            "bits": (self.bits),
            "bits_int": (self.bits_int),
            "alpha_val": (self.alpha_val),
        }
        return {**base_config, **config}

    def call(self, input):
        """Call the layer

        Args:
            input (_type_): input to layer

        Returns:
            _type_: Output of layer
        """
        input_shape = input.shape
        shape_ = (-1, input_shape[1], self.n_head, self.d_model // self.n_head)
        perm_ = (0, 2, 1, 3)

        q = self.qD(input)
        q = tf.reshape(q, shape=shape_)
        q = tf.transpose(q, perm=perm_)

        k = self.kD(input)
        k = tf.reshape(k, shape=shape_)
        k = tf.transpose(k, perm=perm_)

        v = self.vD(input)
        v = tf.reshape(v, shape=shape_)
        v = tf.transpose(v, perm=perm_)

        a = tf.matmul(q, k, transpose_b=True)
        a = tf.nn.softmax(a / q.shape[3] ** 0.5, axis=3)

        out = tf.matmul(a, v)
        out = tf.transpose(out, perm=perm_)
        out = tf.reshape(out, shape=(-1, input_shape[1], self.d_model))
        out = self.outD(out)

        return out

    # define all prunable weights
    def get_prunable_weights(self):
        return (
            self.qD._trainable_weights
            + self.kD._trainable_weights
            + self.vD._trainable_weights
            + self.outD._trainable_weights
        )


class AttentionPooling(tf.keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer):
    """Attention Pooling layer class

    Args:
        tf.keras.layers.Layer (_type_): tensorflow layer class
        tfmot.sparsity.keras.PrunableLayer (_type_): prunable layer class
    """

    def __init__(self, bits, bits_int, alpha_val, **kwargs):
        super().__init__(**kwargs)

        self.score_dense = QDense(1, use_bias=False, **kwargs)

    def call(self, x):  # (B, N, d) -> (B,d) pooling via simple softmax
        """Call the layer

        Args:
            x (_type_): input to layer

        Returns:
            _type_: output to layer
        """
        a = tf.squeeze(self.score_dense(x), axis=-1)
        a = tf.nn.softmax(a, axis=1)

        out = tf.matmul(a[:, tf.newaxis, :], x)
        return tf.squeeze(out, axis=1)

    # define all prunable weights
    def get_prunable_weights(self):
        return self.score_dense._trainable_weights
    

def choose_aggregator(choice: str, name: str, bits=9, bits_int=2, alpha_val=1, **common_args) -> tf.keras.layers.Layer:
    """Choose the aggregator keras object based on an input string."""
    if choice not in ["mean", "max", "attention"]:
        raise ValueError(
            "Given aggregation string is not implemented in choose_aggregator(). "
            "See models.py and add string and corresponding object there."
        )
    if choice == "mean":
        return GlobalAveragePooling1D(name=name)
    elif choice == "max":
        return GlobalMaxPooling1D(name=name)
    elif choice == "attention":
        return AttentionPooling(name=name, bits=bits, bits_int=bits_int, alpha_val=alpha_val, **common_args)



class QKerasModel(JetTagModel):
    """QKerasModel class

    Args:
        JetTagModel (_type_): Base class of a JetTagModel
    """

    quantization_schema = {'quantizer_bits' : And(int, lambda s: 32 >= s >= 0),
                           'quantizer_bits_int' : And(int, lambda s: 32 >= s >= 0),
                           'quantizer_alpha_val' : And(float, lambda s: 1.0 >= s >= 0.0),
                           'pt_output_quantization' : list}

    training_config_schema =    {"weight_method" : And(str, lambda s: s in  ["none", "ptref", "onlyclass"]),
                                 "validation_split" : And(float, lambda s: s > 0.0),
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
        if 'initial_sparsity' in self.training_config:
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
        history = self.jet_model.fit(
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
        
        self.history = history.history

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
