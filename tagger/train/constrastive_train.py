import os
from argparse import ArgumentParser

# Third parties
import numpy as np

# Import from other modules
from tagger.data.tools import load_data, to_ML
from tagger.model.common import fromFolder, fromYaml, SimCLRPreprocessing,contrastive_loss
from tagger.plot.basic import basic
from tagger.train.train import save_test_data 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import tensorflow as tf

def train(model, out_dir, percent, fine=True):

    # Load the data, class_labels and input variables name, not really using input variable names to be honest
    data_train, data_test, class_labels, input_vars, extra_vars = load_data("training_data", percentage=percent)
    model.set_labels(
        input_vars,
        extra_vars,
        class_labels,
    )

    # Make into ML-like data for training
    X_train, y_train, pt_target_train, truth_pt_train, reco_pt_train = to_ML(data_train, class_labels)

    # Save X_test, y_test, and truth_pt_test for plotting later
    X_test, y_test, _, truth_pt_test, reco_pt_test = to_ML(data_test, class_labels)
    save_test_data(out_dir, X_test, y_test, truth_pt_test, reco_pt_test)

    # Get input shape
    input_shape = X_train.shape[1:]  # First dimension is batch size
    output_shape = y_train.shape[1:]
    
    if fine != True:
      model.build_model(input_shape, output_shape)
    
    model.create_encoder(input_shape,8)
    x_train = X_train[..., tf.newaxis].astype("float32")#→ (N, 16, 20, 1), a different way of adding a channel
    x_test = X_test[..., tf.newaxis].astype("float32")#→ (N, 16, 20, 1)
    
    if fine != True:
      batch_size = 512
      augment = SimCLRPreprocessing()
      train_ds = (
          tf.data.Dataset.from_tensor_slices(x_train)
          .shuffle(1024)
          .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE)
      )
      
      optimizer = tf.keras.optimizers.Adam()
      
      for epoch in range(10):
        losses = []
        for x1, x2 in train_ds:
            with tf.GradientTape() as tape:
                z1 = model.embedding_model(x1, training=True)
                z2 = model.embedding_model(x2, training=True)
                loss = contrastive_loss(z1, z2)
            grads = tape.gradient(loss, model.embedding_model.trainable_weights) #gradients of the loss wrt all trainable weights in the model
            optimizer.apply_gradients(zip(grads, model.embedding_model.trainable_weights)) #applies the gradients to update the weights using the Adam optimizer we set (zip matches gradients to weights)
            losses.append(loss.numpy()) #store loss so we can track it
        print(f"Epoch {epoch+1}: Loss = {np.mean(losses):.4f}")
    
    # ----- Embedding Visualization -----
    # Extract features and apply t-SNE
    features = model.embedding_model(x_test / 255.).numpy()
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init='random', random_state=42,verbose=1)
    embeddings_2d = tsne.fit_transform(features)
    labels = ["🐝","💖", "💡","🧴", "☦","✝️","🐮","⚡"]
    labels = ['b','c','light','gluon','tau +','tau -','muon','electron']
    colors = [np.where(y_test[i]==1) for i in range(len(y_test))]
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, cmap='tab10', alpha=0.6)
    cbar = plt.colorbar(scatter, ticks=range(8))
    cbar.ax.set_yticklabels(labels)
    plt.title("t-SNE of SimCLR embeddings (Jet Tagger)")
    plt.grid(True)
    os.makedirs(os.path.join(out_dir, 'plots'), exist_ok=True)
    plt.savefig(out_dir+'/plots/'+'Embedding_2D.png')
    
    frozen_encoder = tf.keras.Sequential([model.embedding_model, tf.keras.layers.Dense(64, activation='relu', trainable=False)])
    classifier = tf.keras.Sequential([
        frozen_encoder,
        tf.keras.layers.Dense(8, activation='softmax')
    ])

    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    classifier.fit(x_train / 255., y_train, epochs=1, batch_size=1024, validation_split=0.1)

    # And for the classifier
    features = classifier(x_test / 255.).numpy()
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init='random', random_state=42,verbose=1)
    embeddings_2d = tsne.fit_transform(features)

    plt.figure(figsize=(18, 16))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, cmap='tab10', alpha=0.6)
    cbar = plt.colorbar(scatter, ticks=range(8))
    cbar.ax.set_yticklabels(labels)
    plt.grid(True)

    plt.savefig(out_dir+'/plots/'+'2D_finetune.png')
        
    model.embedding_model.save(out_dir)

    return


if __name__ == "__main__":

    parser = ArgumentParser()
    # Training argument
    parser.add_argument(
        '-o', '--output', default='output/baseline', help='Output model directory path, also save evaluation plots'
    )
    parser.add_argument('-p', '--percent', default=100, type=int, help='Percentage of how much processed data to train on')
    parser.add_argument(
        '-y', '--yaml_config', default='tagger/model/configs/baseline_larger.yaml', help='YAML config for model'
    )

    # Basic ploting
    parser.add_argument('-fine','--fine_tune', default='true', help='Fine tune a pretrained model')

    args = parser.parse_args()

    if args.fine_tune:
      model = fromYaml(args.yaml_config, args.output)
      train(model, args.output, args.percent, fine=args.fine_tune)
      
    else:
      model = fromFolder('output/baseline')
      train(model, args.output, args.percent, fine=args.fine_tune)

