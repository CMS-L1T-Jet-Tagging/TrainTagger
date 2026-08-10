import keras 
import tensorflow as tf
import os
import numpy as np

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

def _unmask_highest_pt_if_empty(mask, pt):
    """
    If all candidates are masked, unmask the highest-pT candidate.

    Args:
        mask: (batch, num_candidates) boolean or integer mask:
              1 = unmasked (keep)
              0 = masked (remove)
        pt: (batch, num_candidates) pT values

    Returns:
        mask: (batch, num_candidates) mask with the highest-pT candidate
              unmasked if all candidates were masked.
    """
    # Count the number of unmasked candidates per row
    mask_sum = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)

    # Find rows where all candidates are masked i.e. mask_sum == 0
    fully_masked_rows = tf.cast(tf.where(mask_sum == 0), tf.int32)  # (num_bad, 1)
    pt_max_idx = tf.argmax(pt, axis=1, output_type=tf.int32)  # (batch,)
    fix_cols = tf.gather(pt_max_idx, fully_masked_rows[:, 0])
    scatter_indices = tf.concat([fully_masked_rows, fix_cols[:, None]], axis=1,)

    # Set the highest-pT candidate to unmasked (1 / True)
    mask = tf.tensor_scatter_nd_update(mask, scatter_indices, tf.ones(tf.shape(scatter_indices)[0], dtype=mask.dtype,),)

    return mask

class FlatMaskingLayer(keras.layers.Layer):
    def __init__(self, isFilled_idx=15, pt_index=0, masking_probability=0.2):
        super().__init__()
        self.isFilled_idx = isFilled_idx
        self.pt_index = pt_index
        self.masking_probability = masking_probability

        print(f"Using FlatMaskingLayer for data augmentation with pt_index={self.pt_index} masking_probability={self.masking_probability}")

    def call(self, inputs):
        """Mask = 1 means keep, Mask = 0 means mask"""

        batch_size = tf.shape(inputs)[0]
        n_cand = tf.shape(inputs)[1]
        isFilled = inputs[:, :, self.isFilled_idx]  # (batch, cand)

        mask = tf.random.uniform((batch_size, n_cand))
        mask = tf.logical_and(mask > self.masking_probability, isFilled > 0)
        mask = _unmask_highest_pt_if_empty(mask, inputs[:, :, self.pt_index])

        # expand over feature dimension
        mask = mask[..., None] # (batch, cand, 1)
        outputs = tf.where(mask, inputs, 0.0)
        return tf.cast(outputs, inputs.dtype)
    
class PtDependentMaskingLayer(keras.layers.Layer):
    def __init__(self, isFilled_idx=15, pt_index=0, pt_scale=40.0):
        """Note pt_scale has nothing to do with the actual pt, it is just a scale factor to control the masking probability."""
        super().__init__()
        self.isFilled_idx = isFilled_idx
        self.pt_index = pt_index
        self.pt_scale = pt_scale
        self.scale = 1 / self.pt_scale

        print(f"Using PtDependentMaskingLayer for data augmentation with pt_scale={pt_scale}")

    def call(self, inputs):
        """Mask = 1 means keep, Mask = 0 means mask"""

        batch_size = tf.shape(inputs)[0]
        n_cand = tf.shape(inputs)[1]

        isFilled = inputs[:, :, self.isFilled_idx]  # (batch, cand)
        pt = inputs[:, :, self.pt_index]  # (batch, cand)
        pt_sum = tf.maximum(
            tf.reduce_sum(pt, axis=1, keepdims=True),
            tf.keras.backend.epsilon()
        )
        pt_rel = pt / pt_sum
        mask_prob = self.scale / (pt_rel + (2 * self.scale))

        mask = tf.random.uniform((batch_size, n_cand))
        mask = tf.logical_and(mask > mask_prob, isFilled > 0)
        mask = _unmask_highest_pt_if_empty(mask, pt)

        # expand over feature dimension
        mask = mask[..., None] # (batch, cand, 1)

        outputs = tf.where(mask, inputs, 0.0)
        return tf.cast(outputs, inputs.dtype)

class PtSmearingLayer(keras.layers.Layer):
    def __init__(self, pt_index, pt_log_index, smearing_std=0.02):
        super().__init__()
        self.pt_index = pt_index
        self.pt_log_index = pt_log_index
        self.smearing_std = smearing_std

        print(f"Using PtSmearingLayer for data augmentation with smearing_std={smearing_std}")

    def call(self, inputs):

        pt = inputs[:, :, self.pt_index]
        pt = pt * tf.random.normal(tf.shape(pt), mean=1.0, stddev=self.smearing_std, dtype=inputs.dtype,)
        pt = tf.round(4 * pt) / 4

        pt_log = tf.where(pt > 0, tf.math.log(pt) / tf.math.log(2.0), tf.zeros_like(pt))
        pt_log = tf.round(pt_log * 4096.0) / 4096.0 # assuming fractional part is 12 bits, so we round to 1/4096

        outputs = tf.concat([
            inputs[:, :, :self.pt_index],
            pt[:, :, None],
            inputs[:, :, self.pt_index + 1:self.pt_log_index],
            pt_log[:, :, None],
            inputs[:, :, self.pt_log_index + 1:]
        ], axis=-1)

        return tf.cast(outputs, inputs.dtype)
    
class EtaPhiSmearingLayer(keras.layers.Layer):
    def __init__(self, isFilled_idx=15, eta_index=2, phi_index=3, smearing_std=1):
        """
        Smearing is provided in LSBs. 1 LSB = pi/720.
        """
        super().__init__()
        self.isFilled_idx = isFilled_idx
        self.eta_index = eta_index
        self.phi_index = phi_index
        self.smearing_std = smearing_std

        print(f"Using EtaPhiSmearingLayer for data augmentation with smearing_std={smearing_std}")

    def call(self, inputs):

        eta = inputs[:, :, self.eta_index]
        eta = eta + tf.random.normal(tf.shape(eta), mean=0.0, stddev=self.smearing_std, dtype=inputs.dtype,)
        eta = tf.round(eta)

        phi = inputs[:, :, self.phi_index]
        phi = phi + tf.random.normal(tf.shape(phi), mean=0.0, stddev=self.smearing_std, dtype=inputs.dtype,)
        phi = tf.round(phi)

        is_filled = inputs[:, :, self.isFilled_idx]
        eta = tf.where(is_filled > 0, eta, 0.0)
        phi = tf.where(is_filled > 0, phi, 0.0)

        outputs = tf.concat([
            inputs[:, :, :self.eta_index],
            eta[:, :, None],
            inputs[:, :, self.eta_index + 1:self.phi_index],
            phi[:, :, None],
            inputs[:, :, self.phi_index + 1:]
        ], axis=-1)

        return tf.cast(outputs, inputs.dtype)
    
class EtaPhiRotationLayer(keras.layers.Layer):

    def __init__(self, isFilled_idx, eta_index, phi_index):
        super().__init__()
        self.isFilled_idx = isFilled_idx
        self.eta_index = eta_index
        self.phi_index = phi_index

        print("Using EtaPhiRotationLayer for data augmentation")

    def call(self, inputs):

        eta = inputs[:, :, self.eta_index]
        phi = inputs[:, :, self.phi_index]
        isFilled = inputs[:, :, self.isFilled_idx]

        eta_phys = eta * (np.pi / 720.0)
        phi_phys = phi * (np.pi / 720.0)

        theta = tf.random.uniform((tf.shape(eta)[0], 1), minval=0.0, maxval=2 * np.pi, dtype=inputs.dtype,)
        cos_theta = tf.cos(theta)
        sin_theta = tf.sin(theta)

        eta_phys_rot = cos_theta * eta_phys - sin_theta * phi_phys
        phi_phys_rot = sin_theta * eta_phys + cos_theta * phi_phys

        eta_rot = tf.round(eta_phys_rot * (720.0 / np.pi))
        phi_rot = tf.round(phi_phys_rot * (720.0 / np.pi))

        eta_rot = tf.where(isFilled > 0, eta_rot, 0.0)
        phi_rot = tf.where(isFilled > 0, phi_rot, 0.0)

        outputs = tf.concat([
            inputs[:, :, :self.eta_index],
            eta_rot[:, :, None],
            inputs[:, :, self.eta_index + 1:self.phi_index],
            phi_rot[:, :, None],
            inputs[:, :, self.phi_index + 1:]
        ], axis=-1)

        return tf.cast(outputs, inputs.dtype)
    
 
class AugmentationLayer(keras.layers.Layer):
    def __init__(
            self,
            masking_probability=0.2,
            smearing_std=0.02,
            pt_scale=40.0,
            isFilled_idx=15,
            pt_index=0,
            pt_log_index=1,
            eta_index=2,
            phi_index=3,
        ):
        super().__init__()
        self.augment = tf.keras.Sequential([
            PtSmearingLayer(pt_index=pt_index, pt_log_index=pt_log_index, smearing_std=smearing_std),
            EtaPhiSmearingLayer(isFilled_idx=isFilled_idx, eta_index=eta_index, phi_index=phi_index, smearing_std=smearing_std),
            EtaPhiRotationLayer(isFilled_idx=isFilled_idx, eta_index=eta_index, phi_index=phi_index),
            PtDependentMaskingLayer(isFilled_idx=isFilled_idx, pt_index=pt_index, pt_scale=pt_scale),
            FlatMaskingLayer(isFilled_idx=isFilled_idx, pt_index=pt_index, masking_probability=masking_probability),
        ])

    def call(self, x, y , w):
        # return self.augment(x), self.augment(x), y, w
        return x, self.augment(x), y, w

