import keras 
import tensorflow as tf
import os


def initialise_tensorflow(num_threads):
    os.environ["KERAS_BACKEND"] = "tf" 
    
    print("Using ")
    print(tf.config.list_physical_devices('GPU'))
    print("for training")

    # Set some tensorflow constants
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(num_threads)

    tf.keras.utils.set_random_seed(46)  # not a special number
    
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

# ----- Encoder and Projection Head -----
class L2NormalizeLayer(keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)
    
# ----- Data Augmentation -----
class SimCLRPreprocessing(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.augment = keras.Sequential([
            keras.layers.Rescaling(1./255),
            #tf.keras.layers.RandomCrop(16, 20),
            #layers.RandomCrop(28, 28),
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            #tf.keras.layers.RandomZoom(0.2),
        ])

    def call(self, x):
        return self.augment(x), self.augment(x)
