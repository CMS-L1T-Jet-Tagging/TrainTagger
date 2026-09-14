import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Multiply, Reshape, RepeatVector, Permute
import tensorflow_model_optimization as tfmot
import os


def huber_loss(delta=.1, pu=0., alpha=0.):
    """
    Huber loss with asymmetric penalization. Parameters spcified in model config.

    Args:
        delta: Huber threshold.
        alpha: Weight for underestimation (y_true > y_pred).
        pu: Weight for overestimation of pileup (y_true == -1 and y_pred > 1).
    """
    def loss(y_true, y_pred):
        # Minbias: punish overestimation
        pu_punish = tf.where(
            (y_true == -1) & (y_pred > 1),
            pu * (y_pred - 1.0), # scaling proportional to excess
            0.)
        pu_mask = y_true == -1
        y_true = tf.where(pu_mask, 0.95, y_true)

        # punish underestimation more for all other samples
        residual = y_true - y_pred
        overest = tf.where((residual > 0) & (~pu_mask), (abs(residual) * alpha), 0.)  # Penalize overestimation more
        weights = overest + pu_punish + 1.0  # Add 1 as base value

        abs_res = tf.abs(residual)
        quadratic = tf.minimum(abs_res, delta)
        linear = abs_res - quadratic

        return weights * (0.5 * quadratic**2 + delta * linear)

    return loss

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
