"""DeepSet model child class

Written 28/05/2025 cebrown@cern.ch
"""

import json
import os

import hls4ml
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from qkeras import QConv1D
from qkeras.qlayers import QActivation, QDense

# Qkeras
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.utils import load_qmodel
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, BatchNormalization, Dense

from tagger.model.common import AAtt, AttentionPooling, choose_aggregator, L2NormalizeLayer, contrastive_loss, SimCLRPreprocessing
from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tqdm import tqdm

# Set some tensorflow constants
NUM_THREADS = 24
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(NUM_THREADS)
os.environ["TF_NUM_INTEROP_THREADS"] = str(NUM_THREADS)

tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)
tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)

tf.keras.utils.set_random_seed(420)  # not a special number


# Register the model in the factory with the string name corresponding to what is in the yaml config
@JetModelFactory.register('DeepSetModel')
class DeepSetModel(JetTagModel):
    """DeepSetModel class

    Args:
        JetTagModel (_type_): Base class of a JetTagModel
    """

    def build_model(self, inputs_shape: tuple, outputs_shape: tuple):
        """build model override, makes the model layer by layer

        Args:
            inputs_shape (tuple): Shape of the input
            outputs_shape (tuple): Shape of the output

        Additional hyperparameters in the config
            conv1d_layers: List of number of nodes for each layer of the conv1d layers.
            classifier_layers: List of number of nodes for each layer of the classifier MLP.
            regression_layers: List of number of nodes for each layer of the regression MLP
            aggregator: String that specifies the type of aggregator to use after the conv1D net.
        """

        # Define some common arguments, taken from the yaml config
        common_args = {
            'kernel_quantizer': quantized_bits(
                self.quantization_config['quantizer_bits'],
                self.quantization_config['quantizer_bits_int'],
                alpha=self.quantization_config['quantizer_alpha_val'],
            ),
            'bias_quantizer': quantized_bits(
                self.quantization_config['quantizer_bits'],
                self.quantization_config['quantizer_bits_int'],
                alpha=self.quantization_config['quantizer_alpha_val'],
            ),
            'kernel_initializer': self.model_config['kernel_initializer'],
        }

        # Initialize inputs
        inputs = tf.keras.layers.Input(shape=inputs_shape, name='model_input')

        # Main branch
        main = BatchNormalization(name='norm_input')(inputs)

        # Make Conv1D layers
        for iconv1d, depthconv1d in enumerate(self.model_config['conv1d_layers']):
            main = QConv1D(filters=depthconv1d, kernel_size=1, name='Conv1D_' + str(iconv1d + 1), **common_args)(main)
            main = QActivation(
                activation=quantized_relu(self.quantization_config['quantizer_bits'], 0), name='relu_' + str(iconv1d + 1)
            )(main)
            # ToDo: fix the bits_int part later, ie use the default not 0

        # Linear activation to change HLS bitwidth to fix overflow in AveragePooling
        main = QActivation(activation='quantized_bits(18,8)', name='act_pool')(main)
        agg = choose_aggregator(choice=self.model_config['aggregator'], name="pool")
        main = agg(main)

        # Now split into jet ID and pt regression

        # Make fully connected dense layers for classification task
        for iclass, depthclass in enumerate(self.model_config['classification_layers']):
            if iclass == 0:
                jet_id = QDense(depthclass, name='Dense_' + str(iclass + 1) + '_jetID', **common_args)(main)
            else:
                jet_id = QDense(depthclass, name='Dense_' + str(iclass + 1) + '_jetID', **common_args)(jet_id)
            jet_id = QActivation(
                activation=quantized_relu(self.quantization_config['quantizer_bits'], 0),
                name='relu_' + str(iclass + 1) + '_jetID',
            )(jet_id)
            # ToDo: fix the bits_int part later, ie use the default not 0

        # Make output layer for classification task
        jet_id = QDense(outputs_shape[0], name='Dense_' + str(iclass + 2) + '_jetID', **common_args)(jet_id)
        jet_id = Activation('softmax', name='jet_id_output')(jet_id)

        # Make fully connected dense layers for pt regression task
        for ireg, depthreg in enumerate(self.model_config['regression_layers']):
            if ireg == 0:
                pt_regress = QDense(depthreg, name='Dense_' + str(ireg + 1) + '_pT', **common_args)(main)
            else:
                pt_regress = QDense(depthreg, name='Dense_' + str(ireg + 1) + '_pT', **common_args)(pt_regress)
            pt_regress = QActivation(
                activation=quantized_relu(self.quantization_config['quantizer_bits'], 0),
                name='relu_' + str(ireg + 1) + '_pT',
            )(pt_regress)

        pt_regress = QDense(
            1,
            name='pT_output',
            kernel_quantizer=quantized_bits(
                self.quantization_config['pt_output_quantization'][0],
                self.quantization_config['pt_output_quantization'][1],
                alpha=self.quantization_config['quantizer_alpha_val'],
            ),
            bias_quantizer=quantized_bits(
                self.quantization_config['pt_output_quantization'][0],
                self.quantization_config['pt_output_quantization'][1],
                alpha=self.quantization_config['quantizer_alpha_val'],
            ),
            kernel_initializer='lecun_uniform',
        )(pt_regress)

        # Define the model using both branches
        self.jet_model = tf.keras.Model(inputs=inputs, outputs=[jet_id, pt_regress])

        print(self.jet_model.summary())

    def _prune_model(self, num_samples: int):
        """Pruning setup for the model, internal model function called by compile

        Args:
            num_samples (int): number of samples in the training set used for scheduling
        """

        print("Begin pruning the model...")

        # Calculate the ending step for pruning
        end_step = (
            np.ceil(num_samples / self.training_config['batch_size']).astype(np.int32) * self.training_config['epochs']
        )

        # Define the pruned model
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=self.training_config['initial_sparsity'],
                final_sparsity=self.training_config['final_sparsity'],
                begin_step=0,
                end_step=end_step,
            )
        }
        self.jet_model = tfmot.sparsity.keras.prune_low_magnitude(self.jet_model, **pruning_params)

        # Add preface to loss name
        self.loss_name = 'prune_low_magnitude_'

        # Add pruning callback
        self.callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())

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

        # Define the pruning
        self._prune_model(num_samples)

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

        # --- OPTIMIZED FAST TRAINING using tf.data pipeline ---
        # Convert to float32 for TensorFlow (in-place, saves memory)
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        pt_target_train = np.asarray(pt_target_train, dtype=np.float32)
        sample_weight = np.asarray(sample_weight, dtype=np.float32)

        # Manual validation split for efficiency (use slicing, avoid copies)
        val_split = self.training_config.get('validation_split', 0.1)
        n_val = int(len(X_train) * val_split)
        X_val = None  # Ensure X_val is always defined
        if n_val > 0:
            X_val = X_train[:n_val]
            y_val = y_train[:n_val]
            pt_val = pt_target_train[:n_val]
            sw_val = sample_weight[:n_val]
            X_train = X_train[n_val:]
            y_train = y_train[n_val:]
            pt_target_train = pt_target_train[n_val:]
            sample_weight = sample_weight[n_val:]
            val_data = (
                {'model_input': X_val},
                {self.loss_name + self.output_id_name: y_val, self.loss_name + self.output_pt_name: pt_val},
                sw_val,
            )
            # Store validation length before deleting arrays
            val_len = len(X_val)
            del X_val, y_val, pt_val, sw_val
        # Remove redundant else: val_data = None (already handled above)
            val_len = 0

        # Use tf.data.Dataset.from_tensor_slices for efficient pipeline
        train_ds = tf.data.Dataset.from_tensor_slices((
            {'model_input': X_train},
            {self.loss_name + self.output_id_name: y_train, self.loss_name + self.output_pt_name: pt_target_train},
            sample_weight
        ))
        # Use large shuffle buffer for better randomness, autotune for best performance
        train_ds = train_ds.shuffle(buffer_size=min(10000, len(X_train)), reshuffle_each_iteration=True)
        train_ds = train_ds.batch(self.training_config['batch_size'], drop_remainder=True)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        # Validation dataset
        if val_data:
            val_ds = tf.data.Dataset.from_tensor_slices(val_data)
            val_ds = val_ds.batch(self.training_config['batch_size'], drop_remainder=False)
            val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
        else:
            val_ds = None

        # Use steps_per_epoch for large datasets to avoid partial batches
        steps_per_epoch = len(X_train) // self.training_config['batch_size']
        if n_val > 0 and val_len > 0:
            validation_steps = val_len // self.training_config['batch_size']
        else:
            validation_steps = None

        # Train the model using tf.data pipeline
        self.history = self.jet_model.fit(
            train_ds,
            epochs=self.training_config['epochs'],
            verbose=self.run_config['verbose'],
            validation_data=val_ds,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=self.callbacks,
        )

    # Decorated with save decorator for added functionality
    @JetTagModel.save_decorator
    def save(self, out_dir: str = "None"):
        """Save the model file

        Args:
            out_dir (str, optional): Where to save it if not in the output_directory. Defaults to "None".
        """
        # Export the model
        model_export = tfmot.sparsity.keras.strip_pruning(self.jet_model)

        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        # Use keras save format !NOT .h5! due to depreciation
        export_path = os.path.join(out_dir, "model/saved_model.keras")
        model_export.save(export_path)
        print(f"Model saved to {export_path}")

    @JetTagModel.load_decorator
    def load(self, out_dir: str = "None"):
        """Load the model file

        Args:
            out_dir (str, optional): Where to load it if not in the output_directory. Defaults to "None".
        """

        # Additional custom objects for attention layers
        custom_objects_ = {
            "AAtt": AAtt,
            "AttentionPooling": AttentionPooling,
        }
        # Load the model
        self.jet_model = load_qmodel(f"{out_dir}/model/saved_model.keras", custom_objects=custom_objects_)

    def hls4ml_convert(self, firmware_dir: str, build: bool = False):
        """Run the hls4ml model conversion

        Args:
            firmware_dir (str): Where to save the firmware
            build (bool, optional): Run the full hls4ml build? Or just create the project. Defaults to False.
        """

        # Remove the old directory if it exists
        hls4ml_outdir = firmware_dir + '/' + self.hls4ml_config['project_name']
        os.system(f'rm -rf {hls4ml_outdir}')

        # Create default config
        config = hls4ml.utils.config_from_keras_model(self.jet_model, granularity='name')
        config['IOType'] = 'io_parallel'
        config['LayerName']['model_input']['Precision']['result'] = self.hls4ml_config['input_precision']

        # Configuration for conv1d layers
        # hls4ml automatically figures out the paralellization factor
        # config['LayerName']['Conv1D_1']['ParallelizationFactor'] = 8
        # config['LayerName']['Conv1D_2']['ParallelizationFactor'] = 8

        # Additional config
        for layer in self.jet_model.layers:
            layer_name = layer.__class__.__name__

            if layer_name in ["BatchNormalization", "InputLayer"]:
                config["LayerName"][layer.name]["Precision"] = self.hls4ml_config['input_precision']
                config["LayerName"][layer.name]["result"] = self.hls4ml_config['input_precision']
                config["LayerName"][layer.name]["Trace"] = not build

            elif layer_name in ["Permute", "Concatenate", "Flatten", "Reshape", "UpSampling1D", "Add"]:
                print("Skipping trace for:", layer.name)
            else:
                config["LayerName"][layer.name]["Trace"] = not build

        config["LayerName"]["jet_id_output"]["Precision"]["result"] = self.hls4ml_config['class_precision']
        config["LayerName"]["jet_id_output"]["Implementation"] = "latency"
        config["LayerName"]["pT_output"]["Precision"]["result"] = self.hls4ml_config['reg_precision']
        config["LayerName"]["pT_output"]["Implementation"] = "latency"

        # Write HLS
        self.hls_jet_model = hls4ml.converters.convert_from_keras_model(
            self.jet_model,
            backend='Vitis',
            project_name=self.hls4ml_config['project_name'],
            clock_period=2.5,  # 1/360MHz = 2.8ns
            hls_config=config,
            output_dir=f'{hls4ml_outdir}',
            part='xcvu13p-flga2577-2-e',
        )

        # Compile the project
        self.hls_jet_model.compile()

        # Save config  as json file
        print("Saving default config as config.json ...")
        with open(hls4ml_outdir + '/config.json', 'w') as fp:
            json.dump(config, fp)

        if build:
            # build the project
            self.hls_jet_model.build(csim=False, reset=True)

# Register the model in the factory with the string name corresponding to what is in the yaml config
@JetModelFactory.register('DeepSetEmbeddingModel')
class DeepSetEmbeddingModel(DeepSetModel):
    """DeepSetEmbeddingModel class

    Args:
        DeepSetModel (_type_): Base class of a DeepSetModel
    """
    
    def __init__(self,output_dir):
        super().__init__(output_dir)
        self.backbone_model = None
        self.embedding_model = None
        

    def build_model(self, inputs_shape: tuple, outputs_shape: tuple):
        """build model override, makes the model layer by layer

        Args:
            inputs_shape (tuple): Shape of the input
            outputs_shape (tuple): Shape of the output

        Additional hyperparameters in the config
            conv1d_layers: List of number of nodes for each layer of the conv1d layers.
            classifier_layers: List of number of nodes for each layer of the classifier MLP.
            regression_layers: List of number of nodes for each layer of the regression MLP
            aggregator: String that specifies the type of aggregator to use after the conv1D net.
        """

        # Define some common arguments, taken from the yaml config
        common_args = {
            'kernel_quantizer': quantized_bits(
                self.quantization_config['quantizer_bits'],
                self.quantization_config['quantizer_bits_int'],
                alpha=self.quantization_config['quantizer_alpha_val'],
            ),
            'bias_quantizer': quantized_bits(
                self.quantization_config['quantizer_bits'],
                self.quantization_config['quantizer_bits_int'],
                alpha=self.quantization_config['quantizer_alpha_val'],
            ),
            'kernel_initializer': self.model_config['kernel_initializer'],
        }

        # Initialize inputs
        inputs = tf.keras.layers.Input(shape=inputs_shape, name='model_input')

        # Main branch
        main = BatchNormalization(name='norm_input')(inputs)

        # Make Conv1D layers
        for iconv1d, depthconv1d in enumerate(self.model_config['conv1d_layers']):
            main = QConv1D(filters=depthconv1d, kernel_size=1, name='Conv1D_' + str(iconv1d + 1), **common_args)(main)
            main = QActivation(
                activation=quantized_relu(self.quantization_config['quantizer_bits'], 0), name='relu_' + str(iconv1d + 1)
            )(main)
            # ToDo: fix the bits_int part later, ie use the default not 0

        # Linear activation to change HLS bitwidth to fix overflow in AveragePooling
        main = QActivation(activation='quantized_bits(18,8)', name='act_pool')(main)
        agg = choose_aggregator(choice=self.model_config['aggregator'], name="pool")
        main = agg(main)
        
        self.backbone_model = tf.keras.Model(inputs=inputs, outputs=main)

        # Now split into jet ID and pt regression
        # Make fully connected dense layers for classification task
        for iclass, depthclass in enumerate(self.model_config['classification_layers']):
            if iclass == 0:
                jet_id = QDense(depthclass, name='Dense_' + str(iclass + 1) + '_jetID', **common_args)(main)
            else:
                jet_id = QDense(depthclass, name='Dense_' + str(iclass + 1) + '_jetID', **common_args)(jet_id)
            jet_id = QActivation(
                activation=quantized_relu(self.quantization_config['quantizer_bits'], 0),
                name='relu_' + str(iclass + 1) + '_jetID',
            )(jet_id)
            # ToDo: fix the bits_int part later, ie use the default not 0

        # Make output layer for classification task
        jet_id = QDense(outputs_shape[0], name='Dense_' + str(iclass + 2) + '_jetID', **common_args)(jet_id)
        jet_id = Activation('softmax', name='jet_id_output')(jet_id)

        # Make fully connected dense layers for pt regression task
        for ireg, depthreg in enumerate(self.model_config['regression_layers']):
            if ireg == 0:
                pt_regress = QDense(depthreg, name='Dense_' + str(ireg + 1) + '_pT', **common_args)(main)
            else:
                pt_regress = QDense(depthreg, name='Dense_' + str(ireg + 1) + '_pT', **common_args)(pt_regress)
            pt_regress = QActivation(
                activation=quantized_relu(self.quantization_config['quantizer_bits'], 0),
                name='relu_' + str(ireg + 1) + '_pT',
            )(pt_regress)

        pt_regress = QDense(
            1,
            name='pT_output',
            kernel_quantizer=quantized_bits(
                self.quantization_config['pt_output_quantization'][0],
                self.quantization_config['pt_output_quantization'][1],
                alpha=self.quantization_config['quantizer_alpha_val'],
            ),
            bias_quantizer=quantized_bits(
                self.quantization_config['pt_output_quantization'][0],
                self.quantization_config['pt_output_quantization'][1],
                alpha=self.quantization_config['quantizer_alpha_val'],
            ),
            kernel_initializer='lecun_uniform',
        )(pt_regress)


        self.create_encoder(inputs_shape,self.model_config['projection_dims'])
                
        self.jet_model = tf.keras.Model(inputs, [jet_id,pt_regress])
                
        print(self.embedding_model.summary())
        
        print(self.jet_model.summary())
        
    def create_encoder(self,inputs_shape, projection_dim):

        inputs = tf.keras.Input(inputs_shape)
        #inputs = tf.keras.Input(shape=(28, 28, 1))
        features = self.backbone_model(inputs)

        # Projection head, the z's remember?
        outputs = tf.keras.Sequential([
            Dense(10, activation='relu'),
            Dense(projection_dim)
        ])(features)
        # Normalize to unit vectors so dot product equals cosine similarity (required for contrastive loss)
        outputs = L2NormalizeLayer()(outputs)
        self.embedding_model = tf.keras.Model(inputs, outputs)

    def _prune_model(self, num_samples: int):
        """Pruning setup for the model, internal model function called by compile

        Args:
            num_samples (int): number of samples in the training set used for scheduling
        """

        print("Begin pruning the model...")

        # Calculate the ending step for pruning
        embedding_end_step = (
            np.ceil(num_samples / self.training_config['batch_size']).astype(np.int32) * self.training_config['embedding_epochs']
        )
        
        fine_tuning_end_step = (
            np.ceil(num_samples / self.training_config['batch_size']).astype(np.int32) * self.training_config['finetuning_epochs']
        )

        # Define the pruned model
        embedding_prune_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=self.training_config['initial_sparsity'],
                final_sparsity=self.training_config['final_sparsity'],
                begin_step=0,
                end_step=embedding_end_step,
            )
        }
        
        fine_tune_prune_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=self.training_config['initial_sparsity'],
                final_sparsity=self.training_config['final_sparsity'],
                begin_step=0,
                end_step=fine_tuning_end_step,
            )
        }
        
        self.embedding_model = tfmot.sparsity.keras.prune_low_magnitude(self.embedding_model, **embedding_prune_params)
        self.jet_model = tfmot.sparsity.keras.prune_low_magnitude(self.jet_model, **fine_tune_prune_params)

        # Add preface to loss name
        self.loss_name = 'prune_low_magnitude_'

        # Add pruning callback
        self.callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())

    def compile_model(self, num_samples: int):
        """compile the model generating callbacks and loss function
        Args:
            num_samples (int): Number of samples in the training set used for scheduling
        """
        # Define the callbacks using hyperparameters in the config
        self.fine_tune_callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.training_config['EarlyStopping_patience']),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.training_config['ReduceLROnPlateau_factor'],
                patience=self.training_config['ReduceLROnPlateau_patience'],
                min_lr=self.training_config['ReduceLROnPlateau_min_lr'],
            ),
        ]

        # Define the pruning
        self._prune_model(num_samples)
        
        self.constrastive_optimizer = tf.keras.optimizers.Adam()
        self.embedding_model.optimizer = self.constrastive_optimizer  
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
        """Fit the model to the training dataset (embedding + finetune, optimized for speed)

        Args:
            X_train (npt.NDArray[np.float64]): X train dataset
            y_train (npt.NDArray[np.float64]): y train classification targets
            pt_target_train (npt.NDArray[np.float64]): y train pt regression targets
            sample_weight (npt.NDArray[np.float64]): sample weighting
        """
        # Ensure backbone and embedding model are built
        input_shape = X_train.shape[1:]
        output_shape = (y_train.shape[-1],) if len(y_train.shape) > 1 else (1,)
        if self.backbone_model is None or self.jet_model is None:
            self.build_model(input_shape, output_shape)
        # Always ensure encoder is created after build_model, as build_model may not set embedding_model
        projection_dim = self.model_config['projection_dims']
        if self.embedding_model is None:
            self.create_encoder(input_shape, projection_dim)
        # Force assignment in case create_encoder does not set self.embedding_model
        if self.embedding_model is None:
            # Try to get the encoder as return value (if create_encoder returns it)
            encoder = self.create_encoder(input_shape, projection_dim)
            if encoder is not None:
                self.embedding_model = encoder
            else:
                raise RuntimeError("self.embedding_model is still None after create_encoder. Please check create_encoder implementation.")

        # --- Embedding (SimCLR) training (FAST) ---
        x_train = X_train[..., tf.newaxis].astype("float32")
        augment = SimCLRPreprocessing()
        train_ds = (
            tf.data.Dataset.from_tensor_slices(x_train)
            .shuffle(self.training_config['batch_size'])
            .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.training_config['batch_size'])
            .prefetch(tf.data.AUTOTUNE)
        )

        optimizer = self.constrastive_optimizer
        embedding_model = self.embedding_model
        callbacks = tf.keras.callbacks.CallbackList(self.callbacks, add_history=True, model=embedding_model)
        logs = {}
        callbacks.on_train_begin(logs=logs)

        @tf.function
        def train_step(x1, x2):
            # Record operations for automatic differentiation
            with tf.GradientTape() as tape:
                # Forward pass: compute embeddings for both augmented views
                z1 = embedding_model(x1, training=True)
                z2 = embedding_model(x2, training=True)
                # Compute SimCLR contrastive loss between the two views
                loss = contrastive_loss(z1, z2)
            # Compute gradients of loss w.r.t. model trainable weights
            grads = tape.gradient(loss, embedding_model.trainable_weights)
            # Apply gradients to update model weights using optimizer
            optimizer.apply_gradients(zip(grads, embedding_model.trainable_weights))
            # Return the computed loss for logging
            return loss

        for epoch in range(self.training_config['embedding_epochs']):
            callbacks.on_epoch_begin(epoch, logs=logs)
            losses = []
            ibatch = 0
            n_batches = tf.data.experimental.cardinality(train_ds).numpy()
            with tqdm(total=n_batches, desc=f"Embedding Epoch {epoch+1}", unit="batch") as pbar:
                for x1, x2 in train_ds:
                    ibatch += 1
                    callbacks.on_train_batch_begin(ibatch, logs=logs)
                    loss = train_step(x1, x2)
                    losses.append(loss.numpy())
                    callbacks.on_train_batch_end(ibatch, logs=logs)
                    pbar.set_postfix({"loss": f"{np.mean(losses):.4f}"})
                    pbar.update(1)
            logs['loss'] = np.mean(losses)
            print(f"Epoch {epoch+1}: Loss = {logs['loss']:.4f}")
            callbacks.on_epoch_end(epoch, logs=logs)
        callbacks.on_train_end(logs=logs)

        # --- Finetuning (jet model) training ---
        # Convert to float32 for TensorFlow
        X_train = X_train.astype("float32")
        y_train = y_train.astype("float32")
        pt_target_train = pt_target_train.astype("float32")
        sample_weight = sample_weight.astype("float32")

        # Manual validation split for efficiency
        val_split = self.training_config.get('validation_split', 0.1)
        n_val = int(len(X_train) * val_split)
        if n_val > 0:
            X_val, y_val, pt_val, sw_val = X_train[:n_val], y_train[:n_val], pt_target_train[:n_val], sample_weight[:n_val]
            X_train, y_train, pt_target_train, sample_weight = (
                X_train[n_val:], y_train[n_val:], pt_target_train[n_val:], sample_weight[n_val:]
            )
            val_data = (
                {'model_input': X_val},
                {self.loss_name + self.output_id_name: y_val, self.loss_name + self.output_pt_name: pt_val},
                sw_val,
            )
        else:
            val_data = None

        # Build tf.data.Dataset for training
        train_ds = tf.data.Dataset.from_tensor_slices((
            {'model_input': X_train},
            {self.loss_name + self.output_id_name: y_train, self.loss_name + self.output_pt_name: pt_target_train},
            sample_weight
        ))
        train_ds = train_ds.shuffle(buffer_size=4096)
        train_ds = train_ds.batch(self.training_config['batch_size'])
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        # Validation dataset
        if val_data:
            val_ds = tf.data.Dataset.from_tensor_slices(val_data)
            val_ds = val_ds.batch(self.training_config['batch_size'])
            val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
        else:
            val_ds = None

        # Freeze layers 
        for layer in self.jet_model.layers[:7]:
            layer.trainable = False
        for i, layer in enumerate(self.jet_model.layers):
            if i in [12, 13, 14, 15, 16]:
                layer.trainable = False

        # Train the model using tf.data pipeline
        self.history = self.jet_model.fit(
            train_ds,
            epochs=self.training_config['finetuning_epochs'],
            verbose=self.run_config['verbose'],
            validation_data=val_ds,
            callbacks=self.callbacks + self.fine_tune_callbacks,
        )
