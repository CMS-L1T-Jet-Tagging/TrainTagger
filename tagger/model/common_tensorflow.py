import tensorflow as tf
import os
import numpy as np

def initialise_tensorflow(num_threads):
    
    print("Using ")
    print(tf.config.list_physical_devices('GPU'))
    print("for training")
    
    gpu = tf.config.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(gpu[0], True)
    # Set some tensorflow constants
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(num_threads)
    
    seed = np.random.randint(2**32 - 1)
    tf.keras.utils.set_random_seed(seed)
    print("my seed is:", seed)

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

