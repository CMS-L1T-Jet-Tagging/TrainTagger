"""Common utilities for usage across all model child classes
Includes attention layers
Include from Yaml and Folder loading functionality

Written 28/05/2025 cebrown@cern.ch
"""

import os
import shutil

import tensorflow as tf
import tensorflow_model_optimization as tfmot
import yaml
from qkeras.qlayers import QDense
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Multiply, Reshape, RepeatVector, Permute

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

class WeightedGlobalAverage1D(tf.keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer):
    def call(self, inputs):
        inp, weights = inputs
        weights = tf.keras.layers.Reshape((16, 1), name='reshape_pt_weights')(weights)
        weighted_inputs = inp * weights
        weighted_pool = GlobalAveragePooling1D()(weighted_inputs)
        return weighted_pool

    def get_prunable_weights(self):
        return [] # Required for pruning support


class WeightedPtResponse(tf.keras.layers.Layer):
    def call(self, inputs):
        pt_weights, pt_correction, pt, ratio_correction_w, ratio_correction_delta = inputs
        weighted_pt = tf.reduce_sum(pt_weights * pt + pt_correction, axis=1)
        weighted_pt = tf.expand_dims(weighted_pt, axis=-1)
        response = weighted_pt / tf.expand_dims(tf.reduce_sum(pt, axis=1), axis=-1)
        response = ratio_correction_w * response + ratio_correction_delta
        return response

    def get_prunable_weights(self):
        return [] # Required for pruning support


def initialise_tensorflow(num_threads):
    # Set some tensorflow constants
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(num_threads)

    tf.keras.utils.set_random_seed(46)  # not a special number


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


def fromYaml(yaml_path: str, folder: str, recreate: bool = True) -> JetTagModel:
    """Create a model directly from a yaml input file

    Args:
        yaml_path (str): Path to yaml file
        folder (str): Output saving folder for model
        recreate (bool, optional): Rewrite the output directory?. Defaults to True.

    Returns:
        JetTagModel: The model
    """

    with open(yaml_path, 'r') as stream:
        yaml_dict = yaml.safe_load(stream)

    # Create a model based on what is specified in the yaml 'model' field
    # Model must be registered for this to function
    model = JetModelFactory.create_JetTagModel(yaml_dict['model'], folder)
    # Validate yaml dict before loading
    model.schema.validate(yaml_dict)
    model.load_yaml(yaml_path)
    if recreate:
        # Remove output dir if exists
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Re-created existing directory: {folder}.")
            # Create dir to save results
        os.makedirs(folder)
        os.system('cp ' + yaml_path + ' ' + folder)
    return model


def fromFolder(save_path: str, newoutput_dir: str = "None") -> JetTagModel:
    """Load a model from its save folder using the yaml file in the save folder

    Args:
        save_path (str): Where to load the model from
        newoutput_dir (str, optional): New folder to save the model to if needed. Defaults to "None".

    Returns:
        JetTagModel: The model
    """
    if newoutput_dir != "None":
        folder = newoutput_dir
        recreate = True
    else:
        folder = save_path
        recreate = False

    for file in os.listdir(folder):
        if file.endswith(".yaml"):
            yaml_path = os.path.join(folder, file)

    model = fromYaml(yaml_path, folder, recreate=recreate)
    model.load(folder)
    return model
