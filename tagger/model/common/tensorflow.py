import keras 
import tensorflow as tf
import os
import numpy as np
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D

def initialise_tensorflow(num_threads):
    os.environ["KERAS_BACKEND"] = "tf" 
    
    print("Using ")
    print(tf.config.list_physical_devices('GPU'))
    print("for training")

    # Set some tensorflow constants
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(num_threads)
    
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    tf.keras.utils.set_random_seed(46)  # not a special number
    
def set_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

# Normalize the features to unit length, which is important for contrastive learning
class L2NormalizeLayer(keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

class BatchNormalizationLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bn = tf.keras.layers.BatchNormalization()
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = tf.cast(mask, inputs.dtype)          # (batch, num_candidates)
            mask = tf.expand_dims(mask, axis=-1)       # (batch, num_candidates, 1)
            inputs = inputs * mask                      # zero out padded candidates
            mean = tf.reduce_sum(inputs, axis=1) / tf.maximum(tf.reduce_sum(mask, axis=1), 1e-6) # avoid division by zero
            variance = tf.reduce_sum(mask * tf.square(inputs - mean[:, None, :]), axis=1) / tf.maximum(tf.reduce_sum(mask, axis=1), 1e-6)
            return (inputs - mean[:, None, :]) / tf.sqrt(variance[:, None, :] + 1e-6)
        else:
            return self.bn(inputs)

class AveragePoolingLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = tf.cast(mask, inputs.dtype)          # (batch, num_candidates)
            mask = tf.expand_dims(mask, axis=-1)       # (batch, num_candidates, 1)
            inputs = inputs * mask                      # zero out padded candidates
            sum_inputs = tf.reduce_sum(inputs, axis=1) # (batch, feature_dim)
            count_inputs = tf.reduce_sum(mask, axis=1) # (batch, 1)
            return sum_inputs / tf.maximum(count_inputs, 1e-6) # avoid division by zero
        else:
            return tf.reduce_mean(inputs, axis=1)

class MaxPoolingLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = tf.cast(mask, inputs.dtype)          # (batch, num_candidates)
            mask = tf.expand_dims(mask, axis=-1)       # (batch, num_candidates, 1)
            inputs = tf.where(mask > 0, inputs, tf.float32.min) # set padded candidates to -inf
        return tf.reduce_max(inputs, axis=1)

@keras.saving.register_keras_serializable(package="Tagger")
class AttentionPoolingLayer(keras.layers.Layer):
    def __init__(self, hidden_dim, name='pool', **kwargs):
        super().__init__(**kwargs)
        self.attention = keras.Sequential([
            keras.layers.Dense(hidden_dim, activation='relu'),
            keras.layers.Dense(1)
        ], name=name)

    def call(self, inputs, mask=None):
        # inputs: (batch, num_candidates, feature_dim)
        attention_scores = self.attention(inputs)  # (batch, num_candidates, 1)

        if mask is not None:
            mask = tf.cast(mask, attention_scores.dtype)          # (batch, num_candidates)
            mask = tf.expand_dims(mask, axis=-1)                   # (batch, num_candidates, 1)
            attention_scores += (1.0 - mask) * -1e9                # push padded logits to -inf

        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        weighted_sum = tf.reduce_sum(attention_weights * inputs, axis=1)
        return weighted_sum

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.attention.layers[0].units,
        })
        return config

class PoolingFactory(keras.layers.Layer):
    def __init__(self, pooling_type, name='pool', hidden_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.pooling_type = pooling_type
        self.hidden_dim = hidden_dim

        if pooling_type == "average":
            self.pooling_layer = GlobalAveragePooling1D(name=name)
        elif pooling_type == "max":
            self.pooling_layer = GlobalMaxPooling1D(name=name)
        elif pooling_type == "attention":
            assert hidden_dim is not None, "hidden_dim must be specified for attention pooling"
            self.pooling_layer = AttentionPoolingLayer(hidden_dim, name=name)
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")

    def call(self, inputs, mask=None):
        return self.pooling_layer(inputs, mask)

class DelayedReduceLROnPlateau(keras.callbacks.ReduceLROnPlateau):
    def __init__(self, start_from_epoch=0, **kwargs):
        super().__init__(**kwargs)
        self.start_from_epoch = start_from_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_from_epoch:
            return

        super().on_epoch_end(epoch, logs)

    def get_config(self):
        config = super().get_config()
        config["start_from_epoch"] = self.start_from_epoch
        return config
    
# ---------- Pre Training Monitoring ----------

def effective_rank(z, eps=1e-12, norm=False):
    """Participation-ratio rank of covariance; higher values mean variance is distributed across more latent directions."""
    if norm:
        z = tf.math.l2_normalize(z, axis=1)
    z = z - tf.reduce_mean(z, axis=0, keepdims=True)
    n = tf.cast(tf.shape(z)[0], z.dtype)
    cov = tf.matmul(z, z, transpose_a=True) / (n - 1.0)
    eigvals = tf.linalg.eigvalsh(cov)
    eigvals = tf.clip_by_value(eigvals, eps, tf.reduce_max(eigvals) + eps)
    return tf.square(tf.reduce_sum(eigvals)) / tf.reduce_sum(tf.square(eigvals))

def alignment(z1, z2):
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)
    return tf.reduce_mean(tf.reduce_sum(z1 * z2, axis=1))

def embedding_dim_std(z, norm=False):
    """Per-dimension std of embeddings; the minimum flags any collapsed (unused) dimension."""
    if norm:
        z = tf.math.l2_normalize(z, axis=1)
    per_dim_std = tf.math.reduce_std(z, axis=0)
    return tf.reduce_min(per_dim_std), per_dim_std

    
def make_finetuning_dataset(x, y, pt, w, batch_size, output_id_name, output_pt_name, train=True):

    ds = tf.data.Dataset.from_tensor_slices((
            x,
            {output_id_name: y, output_pt_name: pt,},
            {output_id_name: w, output_pt_name: w,},
        ))

    if train:
        ds = (
            ds
            .shuffle(batch_size*10, reshuffle_each_iteration=True)
            .batch(batch_size, drop_remainder=False,)
            .prefetch(tf.data.AUTOTUNE)
        )

    else:
        ds = (
            ds
            .batch(batch_size, drop_remainder=False,)
            .prefetch(tf.data.AUTOTUNE)
        )


    return ds