import keras 
import tensorflow as tf
import os
import sys
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

    
def contrastive_loss(z1, z2, temperature=0.5):
        # First: Concatenate both batches of embeddings (positive pairs)
        z = tf.concat([z1, z2], axis=0)  # shape: (2N, D), where N is batch size

        # Cosine similarity matrix between all embeddings (assumes z is L2-normalized)
        sim = tf.matmul(z, z, transpose_b=True)  # shape: (2N, 2N), sim[i][j] = similarity between sample i and j
        sim /= temperature  # scale similarities by temperature (sharpening)

        # Create some positive/negative pair labels — position i matches with i + N ( same image, different view)
        batch_size = tf.shape(z1)[0]
        labels = tf.range(batch_size)
        labels = tf.concat([labels, labels], axis=0)  # shape: (2N,)

        # Remove self-similarities (the diagonal) from similarity matrix, dont need to do similarity with itselt
        mask = tf.eye(2 * batch_size)  # identity matrix
        sim = sim - 1e9 * mask  # set diagonal to a large negative number so it's ignored in softmax, HACKY! COuld do masking but expensive

        # Get positive similarities from the similarity matrix
        # Positive pairs are offset by +N and -N in the 2N batch
        positives = tf.concat([
            tf.linalg.diag_part(sim, k=batch_size),   # sim[i][i+N]
            tf.linalg.diag_part(sim, k=-batch_size)   # sim[i+N][i]
        ], axis=0)  # shape: (2N,)

        # Step 6: Compute the famous NT-Xent loss
        numerator = tf.exp(positives)  # exp(similarity of positive pairs)
        denominator = tf.reduce_sum(tf.exp(sim), axis=1)  # sum over all other similarities for each sample
        loss = -tf.math.log(numerator / denominator)  # -log(positive / all)

        # Step 7: Return average loss over the batch
        return tf.reduce_mean(loss)
    
    
class SimCLRLoss(tf.keras.layers.Layer):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
    
    # @tf.function
    # def call(self, features, labels=None, mask=None):
    #     """Compute loss for model. If both `labels` and `mask` are None,
    #     it degenerates to SimCLR unsupervised loss:
    #     https://arxiv.org/pdf/2002.05709.pdf
        
    #     Args:
    #         features: hidden vector of shape [n_views, bsz, ...].
    #         labels: one-hot encoded ground truth of shape [bsz, num_classes, 1].
    #         mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
    #             has the same class as sample i. Can be asymmetric.
    #     Returns:
    #         A loss scalar.
    #     """
    #     if len(features.shape) < 3:
    #         raise ValueError('`features` needs to be [n_views, bsz, ...], '
    #                          'at least 3 dimensions are required')
        
    #     if len(features.shape) > 3:
    #         n_views = tf.shape(features)[0]
    #         batch_size = tf.shape(features)[1]
    #         features = tf.reshape(features, [n_views, batch_size, -1])
    #     else:
    #         batch_size = tf.shape(features)[1]
        
    #     if labels is not None and mask is not None:
    #         raise ValueError('Cannot define both `labels` and `mask`')
    #     elif labels is None and mask is None:
    #         mask = tf.eye(batch_size, dtype=tf.float32)
    #     elif labels is not None:
    #         # Handle one-hot encoded labels [batch_size, num_classes, 1]
    #         # Remove the last dimension if it's 1
    #         labels = tf.squeeze(labels, axis=-1)  # [batch_size, num_classes]
            
    #         # Create mask by comparing one-hot vectors
    #         # Two samples have the same class if their one-hot vectors match
    #         mask = tf.matmul(labels, tf.transpose(labels))  # [batch_size, batch_size]
    #         mask = tf.cast(mask > 0, tf.float32)
    #     else:
    #         mask = tf.cast(mask, tf.float32)
        
    #     contrast_count = tf.shape(features)[0]
    #     # Unbind along n_views dimension and concatenate along batch dimension
    #     contrast_feature = tf.concat(tf.unstack(features, axis=0), axis=0)
        
    #     if self.contrast_mode == 'one':
    #         anchor_feature = features[0]
    #         anchor_count = 1
    #     elif self.contrast_mode == 'all':
    #         anchor_feature = contrast_feature
    #         anchor_count = contrast_count
    #     else:
    #         raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        
    #     # compute logits
    #     anchor_dot_contrast = tf.divide(
    #         tf.matmul(anchor_feature, tf.transpose(contrast_feature)),
    #         self.temperature)
        
    #     # for numerical stability
    #     logits_max = tf.reduce_max(anchor_dot_contrast, axis=1, keepdims=True)
    #     logits = anchor_dot_contrast - tf.stop_gradient(logits_max)
        
    #     # tile mask
    #     mask = tf.tile(mask, [anchor_count, contrast_count])
        
    #     # mask-out self-contrast cases - corrected version
    #     diagonal_indices = tf.range(batch_size * anchor_count)
    #     row_indices = diagonal_indices
    #     col_indices = diagonal_indices
        
    #     # Create a mask of ones, then set diagonal to zero
    #     logits_mask = tf.ones([batch_size * anchor_count, batch_size * contrast_count], dtype=tf.float32)
    #     indices = tf.stack([row_indices, col_indices], axis=1)
    #     updates = tf.zeros(batch_size * anchor_count, dtype=tf.float32)
    #     logits_mask = tf.tensor_scatter_nd_update(logits_mask, indices, updates)
        
    #     mask = mask * logits_mask
        
    #     # compute log_prob
    #     exp_logits = tf.exp(logits) * logits_mask
    #     log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True))
        
    #     # compute mean of log-likelihood over positive
    #     mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / tf.reduce_sum(mask, axis=1)
        
    #     # loss
    #     loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
    #     loss = tf.reduce_mean(tf.reshape(loss, [anchor_count, batch_size]))
        
    #     return loss
    
    @tf.function
    def call(self,features, labels, weights):
        # def SimCLRLoss(features, labels, temperature = 0.07):
        '''
        Computes SimCLRLoss as defined in https://arxiv.org/pdf/2004.11362.pdf
        '''
        batch_size = features.shape[1]
        if (features.shape[1] != labels.shape[0]):
            raise ValueError('Error in SIMCLRLOSS: Number of labels does not match number of features')

        # Generates mask indicating what samples are considered pos/neg
        #labels = tf.squeeze(labels, axis=-1) 
        labels = tf.argmax(labels, axis=1)
        positive_mask = tf.equal(labels, tf.transpose(labels))
        negative_mask = tf.logical_not(positive_mask)
        positive_mask = tf.cast(positive_mask, dtype=tf.float32)
        negative_mask = tf.cast(negative_mask, dtype=tf.float32)

        # Computes dp between pairs
        logits = tf.linalg.matmul(features, features, transpose_b=True)
        temperature = tf.cast(self.temperature, tf.float32)
        logits = logits / temperature

        # Subtract largest |logits| elt for numerical stability
        # Simply for numerical precision -> stop gradient
        max_logit = tf.reduce_max(tf.stop_gradient(logits), axis=1, keepdims=True)
        logits = logits - max_logit

        exp_logits = tf.exp(logits)
        num_positives_per_row = tf.reduce_sum(positive_mask, axis=1)

        denominator = tf.reduce_sum(exp_logits * negative_mask, axis = 1, keepdims=True)
        denominator += tf.reduce_sum(exp_logits * positive_mask, axis = 1, keepdims=True)

        # Compute L OUTSIDE -> defined in eq 2 of paper
        log_probs = (logits - tf.math.log(denominator)) * positive_mask
        log_probs = tf.reduce_sum(log_probs, axis=1)
        log_probs = tf.math.divide_no_nan(log_probs, num_positives_per_row)
        loss = -log_probs * temperature * weights
        loss = tf.reduce_mean(loss)
        return loss

# ----- Encoder and Projection Head -----
class L2NormalizeLayer(keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)


class FlatMaskingLayer(keras.layers.Layer):
    def __init__(self, probability=0.2):
        super().__init__()
        self.probability = probability

    def call(self, inputs):
        mask = np.random.rand((inputs.shape[0]))
        idx = mask < self.probability
        idx = idx.reshape(inputs.shape[0], 1, 1)
        # Convert boolean to TensorFlow and use where
        result = tf.where(idx, 0.0, inputs)  # Sets True positions to 0
        return result
    
    
    
    
# ----- Data Augmentation -----
class SimCLRPreprocessing(keras.layers.Layer):
    def __init__(self,masking_probability):
        super().__init__()
        self.augment = tf.keras.Sequential([
            FlatMaskingLayer(masking_probability)
            #tf.keras.layers.Rescaling(1./255),
            #tf.keras.layers.RandomCrop(16, 20),
            #layers.RandomCrop(28, 28),
            #tf.keras.layers.RandomFlip("horizontal"),
            #tf.keras.layers.RandomRotation(0.1),
            #tf.keras.layers.RandomZoom(0.2),
        ])

    def call(self, x,y , w):
        return x, self.augment(x),y, w
