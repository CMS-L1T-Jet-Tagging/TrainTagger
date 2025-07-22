"""Transformer model child class

Written 15/07/2025 by Arianna Cox amc424@ic.ac.uk
"""

import os

import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import GlobalAveragePooling1D, BatchNormalization, LayerNormalization, MultiHeadAttention, Dense
from tagger.model.common import AAtt, AttentionPooling, choose_aggregator, L2NormalizeLayer

from tagger.model.JetTagModel import JetModelFactory, JetTagModel

# Set some tensorflow constants
NUM_THREADS = 24
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(NUM_THREADS)
os.environ["TF_NUM_INTEROP_THREADS"] = str(NUM_THREADS)

tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)
tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)

tf.keras.utils.set_random_seed(420)  # not a special number


# Register the model in the factory with the string name corresponding to what is in the yaml config
@JetModelFactory.register('TransformerModel')
class TransformerModel(JetTagModel):
    """TransformerModel class

    Args:
        JetTagModel (_type_): Base class of a JetTagModel
    """

    def build_model(self, inputs_shape: tuple, outputs_shape: tuple):
        """build model override, makes the model layer by layer

        Args:
            inputs_shape (tuple): Shape of the input
            outputs_shape (tuple): Shape of the output
        """

        # Define some common arguments, taken from the yaml config
        common_args = {
            'kernel_initializer': self.model_config['kernel_initializer'],
        }

        # Initialize inputs
        inputs = tf.keras.layers.Input(shape=inputs_shape, name='model_input')

        # Main branch
        main = BatchNormalization(name='norm_input')(inputs)

        # Embedding
        for i, nodes in enumerate(self.model_config['emb_layers']):
            main = Dense(nodes, activation=tf.keras.activations.relu, name='emb_'+str(i+1), **common_args)(main)

        # Transformer blocks
        for i, (num_heads, mha_hidden_dim, num_dense_layers, dim_dense_layers) in enumerate(self.model_config['transformer_layers']):
            mha = MultiHeadAttention(num_heads, mha_hidden_dim, name='mha_'+str(i+1))(main,main)
            # Layer norm and residual connection
            main = LayerNormalization()(mha+main)
            for j in range(num_dense_layers):
                if j == 0:
                    feedforward = Dense(dim_dense_layers, activation=tf.keras.activations.relu, name='dense_'+str(i+1)+'_'+str(j+1), **common_args)(main)
                else:
                    feedforward = Dense(dim_dense_layers, activation=tf.keras.activations.relu, name='dense_'+str(i+1)+'_'+str(j+1), **common_args)(feedforward)
            # Layer norm and residual connection
            main = LayerNormalization()(main+feedforward)

        # Global average pooling
        main = GlobalAveragePooling1D(data_format='channels_last')(main)

        # Now split into jet ID and pt regression

        # Make fully connected dense layers for classification task
        for iclass, depthclass in enumerate(self.model_config['classification_layers']):
            if iclass == 0:
                jet_id = Dense(depthclass, activation=tf.keras.activations.relu, name='Dense_' + str(iclass + 1) + '_jetID', **common_args)(main)
            else:
                jet_id = Dense(depthclass, activation=tf.keras.activations.relu, name='Dense_' + str(iclass + 1) + '_jetID', **common_args)(jet_id)

        # Make output layer for classification task
        jet_id = Dense(outputs_shape[0], activation=tf.keras.activations.softmax, name='jet_id_output', **common_args)(jet_id)

        # Make fully connected dense layers for pt regression task
        for ireg, depthreg in enumerate(self.model_config['regression_layers']):
            if ireg == 0:
                pt_regress = Dense(depthreg, activation=tf.keras.activations.relu, name='Dense_' + str(ireg + 1) + '_pT', **common_args)(main)
            else:
                pt_regress = Dense(depthreg, activation=tf.keras.activations.relu, name='Dense_' + str(ireg + 1) + '_pT', **common_args)(pt_regress)

        pt_regress = Dense(1, name='pT_output',kernel_initializer='lecun_uniform')(pt_regress)

        # Define the model using both branches
        self.jet_model = tf.keras.Model(inputs=inputs, outputs=[jet_id, pt_regress])

        print(self.jet_model.summary())

    def compile_model(self, num_samples: int):
        """compile the model generating callbacks and loss function
        Args:
            num_samples (int): Number of samples in the training set used for scheduling
        """

        # Define the callbacks using hyperparameters in the config
        self.callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.training_config['EarlyStopping_patience']),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.training_config['ReduceLROnPlateau_factor'],
                patience=self.training_config['ReduceLROnPlateau_patience'],
                min_lr=self.training_config['ReduceLROnPlateau_min_lr'],
            ),
        ]

        # compile the tensorflow model setting the loss and metrics
        self.jet_model.compile(
            optimizer='adam',
            loss={
                self.loss_name + self.output_id_name: 'categorical_crossentropy',
                self.loss_name + self.output_pt_name: tf.keras.losses.Huber(),
            },
            loss_weights=self.training_config['loss_weights'],
            metrics={
                self.loss_name + self.output_id_name: 'categorical_accuracy',
                self.loss_name + self.output_pt_name: ['mae', 'mean_squared_error'],
            },
            weighted_metrics={
                self.loss_name + self.output_id_name: 'categorical_accuracy',
                self.loss_name + self.output_pt_name: ['mae', 'mean_squared_error'],
            },
        )

    def fit(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        pt_target_train: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64],
    ):
        """Fit the model to the training dataset

        Args:
            X_train (npt.NDArray[np.float64]): X train dataset
            y_train (npt.NDArray[np.float64]): y train classification targets
            pt_target_train (npt.NDArray[np.float64]): y train pt regression targets
            sample_weight (npt.NDArray[np.float64]): sample weighting
        """

        # Train the model using hyperparameters in yaml config
        self.history = self.jet_model.fit(
            {'model_input': X_train},
            {self.loss_name + self.output_id_name: y_train, self.loss_name + self.output_pt_name: pt_target_train},
            sample_weight=sample_weight,
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            verbose=self.run_config['verbose'],
            validation_split=self.training_config['validation_split'],
            callbacks=self.callbacks,
            shuffle=True,
        )

    # Decorated with save decorator for added functionality
    @JetTagModel.save_decorator
    def save(self, out_dir: str = "None"):
        # Export the model
        model_export = tfmot.sparsity.keras.strip_pruning(self.jet_model)

        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        export_path = os.path.join(out_dir, "model/saved_model.keras")
        model_export.save(export_path)
        print(f"Model saved to {export_path}")

    @JetTagModel.load_decorator
    def load(self, out_dir: str = "None"):

        self.jet_model = tf.keras.models.load_model(f"{out_dir}/model/saved_model.keras")

    def hls4ml_convert(self, firmware_dir: str, build: bool = False):
        print('AMEC: hls4ml_convert was called but will do nothing!')

    def create_encoder(self,inputs_shape, projection_dim):
        embedding_model = tf.keras.models.Model(inputs=self.jet_model.input, outputs=self.jet_model.get_layer('global_average_pooling1d').output)

        inputs = tf.keras.Input(inputs_shape)
        #inputs = tf.keras.Input(shape=(28, 28, 1))
        features = embedding_model(inputs)

        # Projection head, the z's remember?
        outputs = tf.keras.Sequential([
            Dense(64, activation='relu'),
            Dense(projection_dim)
        ])(features)
        # Normalize to unit vectors so dot product equals cosine similarity (required for contrastive loss)
        outputs = L2NormalizeLayer()(outputs)
        self.embedding_model = tf.keras.Model(inputs, outputs)
     