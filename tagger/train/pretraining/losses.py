import keras 
import tensorflow as tf
import numpy as np

# ---------- SimCLR Losses ----------    

def compute_ntxent_loss(features, positive_mask, denominator_mask, anchor_weights, temperature=0.1):

    logits = tf.linalg.matmul(features, features, transpose_b=True,)
    logits /= tf.cast(temperature, logits.dtype,)
    masked_logits = tf.where(denominator_mask, logits, tf.cast(-1e9, logits.dtype),)

    log_denominator = tf.reduce_logsumexp(masked_logits, axis=1, keepdims=True,)

    log_prob = logits - log_denominator

    positive_mask = tf.cast(positive_mask, logits.dtype,)
    num_positives = tf.reduce_sum(positive_mask, axis=1,)

    mean_positive_log_prob = tf.math.divide_no_nan(
        tf.reduce_sum(log_prob * positive_mask,axis=1,),
        num_positives,
    )

    anchor_loss = -mean_positive_log_prob

    return tf.math.divide_no_nan(
        tf.reduce_sum(anchor_loss * anchor_weights),
        tf.reduce_sum(anchor_weights),
    )

class SupConLoss(tf.keras.layers.Layer):
    """
    Supervised Contrastive Loss.

    Reference:
        https://arxiv.org/abs/2004.11362

    Shapes:
        features: (views, batch_size, feature_dim)
        weights:  (batch_size,)
        labels:   (batch_size, num_classes), one-hot encoded
    """

    def __init__(self, temperature=0.1, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

        print("Initialized SupConLoss with temperature:", self.temperature)

    @tf.function
    def call(self, features: tf.Tensor, weights: tf.Tensor, labels: tf.Tensor,) -> tf.Tensor:

        # L2 normalize embeddings
        features = tf.math.l2_normalize(features, axis=-1)

        num_views = tf.shape(features)[0]
        batch_size = tf.shape(features)[1]
        num_embeddings = batch_size * num_views

        # (views, batch, dim) -> (batch * views, dim)
        features = tf.transpose(features, [1, 0, 2])
        features = tf.reshape(features, [num_embeddings, -1])

        # Repeat labels/weights for every view
        integer_labels = tf.cast(tf.argmax(labels, axis=1), tf.int32)
        labels_flat = tf.repeat(integer_labels, repeats=num_views,)
        anchor_weights = tf.repeat(tf.cast(weights, features.dtype), repeats=num_views,)

        # Positive pairs = same class
        positive_mask = tf.equal(labels_flat[:, None], labels_flat[None, :],)

        # Exclude self comparisons
        self_mask = tf.eye(num_embeddings, dtype=tf.bool,)
        positive_mask &= ~self_mask
        denominator_mask = ~self_mask

        return compute_ntxent_loss(
            features,
            positive_mask,
            denominator_mask,
            anchor_weights,
            temperature=self.temperature,
        )


class SimCLRLoss(tf.keras.layers.Layer):
    """
    SimCLR self-supervised contrastive loss.

    Positive pairs are different augmented views of the same sample.

    Shapes:
        features: (views, batch_size, feature_dim)
        weights:  (batch_size,)
    """

    def __init__(self, temperature=0.1, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

        print("Initialized SimCLRLoss with temperature:", self.temperature)

    @tf.function
    def call(self, features: tf.Tensor, weights: tf.Tensor, labels: tf.Tensor,) -> tf.Tensor:

        features = tf.math.l2_normalize(features, axis=-1,)

        num_views = tf.shape(features)[0]
        batch_size = tf.shape(features)[1]
        num_embeddings = batch_size * num_views

        # (views, batch, dim) -> (batch * views, dim)
        features = tf.transpose(features, [1, 0, 2],)
        features = tf.reshape(features, [num_embeddings, -1],)

        # Keep track of which embedding belongs to which sample
        sample_indices = tf.repeat(tf.range(batch_size), repeats=num_views,)
        same_sample = tf.equal(sample_indices[:, None], sample_indices[None, :],)
        self_mask = tf.eye(num_embeddings, dtype=tf.bool,)

        # Other view(s) of same sample are positives
        positive_mask = (same_sample & ~self_mask)
        denominator_mask = ~self_mask

        anchor_weights = tf.repeat(tf.cast(weights, features.dtype), repeats=num_views,)

        return compute_ntxent_loss(
            features,
            positive_mask,
            denominator_mask,
            anchor_weights,
            temperature=self.temperature,
        )

class HybridSupLoss(tf.keras.layers.Layer):
    """
    Hybrid supervised/self-supervised contrastive loss.

    Matched jets:
        Supervised Contrastive Loss.

    Unmatched jets:
        SimCLR Loss.

    Total:
        L = L_sup + alpha * L_simclr
    """

    def __init__(self, temperature=0.1, unmatched_class_index=8, alpha=1.0, **kwargs,):
        super().__init__(**kwargs)

        self.temperature = temperature
        self.unmatched_class_index = unmatched_class_index
        self.alpha = alpha

        self.supcon_loss = SupConLoss(temperature=temperature,)
        self.simclr_loss = SimCLRLoss(temperature=temperature,)

        print("Initialized HybridSupLoss with temperature:", self.temperature, "unmatched_class_index:", self.unmatched_class_index, "alpha:", self.alpha)

    @tf.function
    def call(self, features: tf.Tensor, weights: tf.Tensor, labels: tf.Tensor,) -> tf.Tensor:

        integer_labels = tf.cast(tf.argmax(labels, axis=1), tf.int32)
        unmatched = tf.equal(integer_labels, self.unmatched_class_index,)
        matched = ~unmatched

        # --------------------------------
        # Supervised loss on matched jets
        # --------------------------------
        supervised_loss = tf.cond(
            tf.reduce_any(matched),
            lambda: self.supcon_loss(
                tf.boolean_mask(features, matched, axis=1),
                tf.boolean_mask(weights, matched),
                tf.boolean_mask(labels, matched),
            ),
            lambda: tf.constant(
                0.0,
                dtype=features.dtype,
            ),
        )

        # --------------------------------
        # SimCLR loss on unmatched jets
        # --------------------------------
        self_supervised_loss = tf.cond(
            tf.reduce_any(unmatched),
            lambda: self.simclr_loss(
                tf.boolean_mask(features, unmatched, axis=1),
                tf.boolean_mask(weights, unmatched),
            ),
            lambda: tf.constant(
                0.0,
                dtype=features.dtype,
            ),
        )

        # --------------------------------
        # Weighted combination
        # --------------------------------
        alpha = tf.cast(self.alpha, supervised_loss.dtype,)
        return (supervised_loss + alpha * self_supervised_loss)
    

# ---------- VICReg Loss Layer ----------

class VICRegLoss(tf.keras.layers.Layer):
    """
    VICReg with optional supervised invariance loss.
    Args:
        sim_coeff: self-supervised invariance coefficient
        std_coeff: variance coefficient
        cov_coeff: covariance coefficient
    """

    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, gamma=1.0, eps=1e-4,):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma
        self.eps = eps

    @tf.function
    def call(self, features: tf.Tensor, weights: tf.Tensor, labels: tf.Tensor = None,):
        """
        Args:
            features: (2, B, D)
            weights:  (B,)
            labels:   (B,) integer class labels (optional)
        """
        z1 = features[0]
        z2 = features[1]
        batch_size = tf.cast(tf.shape(z1)[0], tf.float32)
        feature_dim = tf.cast(tf.shape(z1)[1], tf.float32)

        ############################################################
        # 1. Standard VICReg invariance loss
        ############################################################

        sim_loss_per_sample = tf.reduce_mean(tf.square(z1 - z2), axis=1)
        sim_loss = tf.reduce_mean(sim_loss_per_sample * weights)


        ############################################################
        # 2. Variance loss
        ############################################################

        z1_centered = z1 - tf.reduce_mean(z1, axis=0, keepdims=True)
        z2_centered = z2 - tf.reduce_mean(z2, axis=0, keepdims=True)
        std_z1 = tf.sqrt(tf.math.reduce_variance(z1_centered, axis=0) + self.eps)
        std_z2 = tf.sqrt(tf.math.reduce_variance(z2_centered, axis=0) + self.eps)
        var_loss = (
            tf.reduce_mean(tf.nn.relu(self.gamma - std_z1))
            + tf.reduce_mean(tf.nn.relu(self.gamma - std_z2))
        )

        ############################################################
        # 3. Covariance loss
        ############################################################

        cov_z1 = tf.matmul(z1_centered, z1_centered, transpose_a=True) / (batch_size - 1.0)
        cov_z2 = tf.matmul(z2_centered, z2_centered, transpose_a=True) / (batch_size - 1.0)

        diag = tf.eye(tf.shape(cov_z1)[0])
        cov_loss_z1 = (tf.reduce_sum(tf.square(cov_z1 * (1.0 - diag))) / feature_dim)
        cov_loss_z2 = (tf.reduce_sum(tf.square(cov_z2 * (1.0 - diag))) / feature_dim)
        cov_loss = cov_loss_z1 + cov_loss_z2

        ############################################################
        # Total
        ############################################################

        loss = (
            self.sim_coeff * sim_loss
            + self.std_coeff * var_loss
            + self.cov_coeff * cov_loss
        )

        return loss
    
# ---------- JEPA Loss Layer ----------

class JepaLoss(tf.keras.layers.Layer):
    @tf.function
    def call(self, prediction, target, weights=None):
        prediction = tf.math.l2_normalize(prediction, axis=-1)
        target = tf.math.l2_normalize(target, axis=-1)

        # cosine similarity loss
        loss = 1.0 - tf.reduce_sum(prediction * target, axis=-1)
        if weights is not None:
            loss = loss * weights

        return tf.reduce_mean(loss)



def get_loss_function(loss_name: str, **kwargs) -> tf.keras.layers.Layer:
    if loss_name == "SupCon":
        return SupConLoss(**kwargs)
    elif loss_name == "SimCLR":
        return SimCLRLoss(**kwargs)
    elif loss_name == "HybridSup":
        return HybridSupLoss(**kwargs)
    elif loss_name == "VICReg":
        return VICRegLoss(**kwargs)
    elif loss_name == "Jepa":
        return JepaLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")