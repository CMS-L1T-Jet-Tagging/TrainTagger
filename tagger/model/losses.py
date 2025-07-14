# borrowed from https://github.com/thaarres/ML_nuggets/blob/main/exercises/simclr.ipynb

import tensorflow as tf

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
