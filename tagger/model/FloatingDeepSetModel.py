"""DeepSet model child class

Written 07/10/2025 cebrown@cern.ch
"""

import json
import os

#import hls4ml
import numpy as np
import numpy.typing as npt
from schema import Schema, And, Use, Optional

from tagger.model.common_tensorflow import *
from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tagger.model.common import cosine_decay_restarts

import keras
from keras.models import load_model
from keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D,AveragePooling1D, Dense, Conv1D, Flatten, ReLU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


import tensorflow as tf
# Register the model in the factory with the string name corresponding to what is in the yaml config
@JetModelFactory.register('FloatingDeepSetModel')
class FloatingDeepSetModel(JetTagModel):

    """FloatingDeepSetModel class

    Args:
        JetTagModel (_type_): Base class of a JetTagModel
    """

    schema = Schema(
            {
                "model": str,
                ## generic run config coniguration
                "run_config" : JetTagModel.run_schema,
                "model_config" : {"name" : str,
                                  "conv1d_layers" : list,
                                  "classification_layers" : list,
                                  "regression_layers" : list,
                                  "kernel_initializer" : str,
                                  "aggregator" : And(str, lambda s: s in  ["mean", "max", "attention"])},
                "quantization_config" : {'pt_output_quantization' : list},
                "training_config" : {"weight_method" : And(str, lambda s: s in  ["none", "ptref", "onlyclass"]),
                                     "validation_split" : And(float, lambda s: s > 0.0),
                                     "epochs" : And(int, lambda s: s >= 1),
                                     "batch_size" : And(int, lambda s: s >= 1),
                                     "learning_rate" : And(float, lambda s: s > 0.0),
                                     "loss_weights" : And(list, lambda s: len(s) == 2),
                                     "EarlyStopping_patience" : And(int, lambda s: s > 0),
                                     "ReduceLROnPlateau_factor" : And(float, lambda s: 1.0 >= s >= 0.0),
                                     "ReduceLROnPlateau_patience" : int,
                                     "ReduceLROnPlateau_min_lr" : And(float, lambda s: s >= 0.0)},
                ## generic hls4ml configuration
                "firmware_config" : {"input_precision" : str,
                                     "class_precision" : str,
                                     "reg_precision": str,
                                     "clock_period" : And(float, lambda s: 0.0 < s <= 10),
                                     "fpga_part" : str,
                                     "project_name" : str}
            }
    )

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
        initialise_tensorflow(self.run_config['num_threads'])
        
        self.common_args = {
            'kernel_initializer': self.model_config['kernel_initializer'],
        }

        L, C = inputs_shape
        #Initialize inputs
        inputs = keras.layers.Input(shape=inputs_shape, name='model_input')
        
        #Main branch
        main = BatchNormalization(name='norm_input')(inputs)
                        
        # Make Conv1D layers
        for iconv1d, depthconv1d in enumerate(self.model_config['conv1d_layers']):
            main = Conv1D(filters=depthconv1d, kernel_size=1, name='Conv1D_' + str(iconv1d + 1), activation='relu',**self.common_args)(main)

        # Linear activation to change HLS bitwidth to fix overflow in AveragePooling
        #main = AveragePooling1D(L,name='pool')(main)
        main = GlobalAveragePooling1D(data_format='channels_last',name="pool")(main)
        #main = Flatten()(main)

        #Now split into jet ID and pt regression
        
        # Make fully connected dense layers for classification task
        for iclass, depthclass in enumerate(self.model_config['classification_layers']):
            if iclass == 0:
                jet_id = Dense(depthclass, name='Dense_' + str(iclass + 1) + '_jetID', **self.common_args)(main)
            else:
                jet_id = Dense(depthclass, name='Dense_' + str(iclass + 1) + '_jetID', activation='relu', **self.common_args)(jet_id)

        jet_id = Dense(outputs_shape[0], name='Dense_3_jetID',activation='linear',kernel_initializer='lecun_uniform')(jet_id)
        jet_id = Activation('softmax', name='jet_id_output')(jet_id)

        #pT regression branch
        
        # Make fully connected dense layers for pt regression task
        for ireg, depthreg in enumerate(self.model_config['regression_layers']):
            if ireg == 0:
                pt_regress = Dense(depthreg, name='Dense_' + str(ireg + 1) + '_pT', **self.common_args)(main)
            else:
                pt_regress = QDense(depthreg, name='Dense_' + str(ireg + 1) + '_pT', activation='relu',**self.common_args)(pt_regress)

        pt_regress = Dense(1, name='pT_output',
                            kernel_initializer='lecun_uniform')(pt_regress)

        #Define the model using both branches
        self.jet_model = keras.Model(inputs = [inputs], outputs = [jet_id, pt_regress])

        print(self.jet_model.summary())

    def firmware_convert(self, firmware_dir: str, build: bool = False):
        """Run the hls4ml model conversion

        Args:
            firmware_dir (str): Where to save the firmware
            build (bool, optional): Run the full hls4ml build? Or just create the project. Defaults to False.
        """

        # Remove the old directory if it exists
        hls4ml_outdir = firmware_dir + '/' + self.firmware_config['project_name']
        os.system(f'rm -rf {hls4ml_outdir}')

        # Create default config
        config = hls4ml.utils.config_from_keras_model(self.jet_model, granularity='name')
        config['IOType'] = 'io_parallel'
        config['LayerName']['model_input']['Precision']['result'] = self.firmware_config['input_precision']

        # Configuration for conv1d layers
        # hls4ml automatically figures out the paralellization factor
        # config['LayerName']['Conv1D_1']['ParallelizationFactor'] = 8
        # config['LayerName']['Conv1D_2']['ParallelizationFactor'] = 8

        # Additional config
        for layer in self.jet_model.layers:
            layer_name = layer.__class__.__name__

            if layer_name in ["BatchNormalization", "InputLayer"]:
                config["LayerName"][layer.name]["Precision"] = self.firmware_config['input_precision']
                config["LayerName"][layer.name]["result"] = self.firmware_config['input_precision']
                config["LayerName"][layer.name]["Trace"] = not build

            elif layer_name in ["Permute", "Concatenate", "Flatten", "Reshape", "UpSampling1D", "Add"]:
                print("Skipping trace for:", layer.name)
            else:
                config["LayerName"][layer.name]["Trace"] = not build

        config["LayerName"]["jet_id_output"]["Precision"]["result"] = self.firmware_config['class_precision']
        config["LayerName"]["jet_id_output"]["Implementation"] = "latency"
        config["LayerName"]["pT_output"]["Precision"]["result"] = self.firmware_config['reg_precision']
        config["LayerName"]["pT_output"]["Implementation"] = "latency"

        # Write HLS
        self.hls_jet_model = hls4ml.converters.convert_from_keras_model(
            self.jet_model,
            backend='Vitis',
            project_name=self.firmware_config['project_name'],
            clock_period=self.firmware_config['clock_period'],
            hls_config=config,
            output_dir=f'{hls4ml_outdir}',
            part= self.firmware_config['fpga_part'],
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
        if 'initial_sparsity' in self.training_config:
            self._prune_model(num_samples)

        # compile the tensorflow model setting the loss and metrics
        self.jet_model.compile(
            optimizer='adam',
            loss={
                self.loss_name + self.output_id_name: 'categorical_crossentropy',
                self.loss_name + self.output_pt_name: keras.losses.Huber(),
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
        keras.config.disable_traceback_filtering()
        sample_weight_dict = {
                            "jet_id_output": sample_weight,
                            "pT_output": sample_weight,
        }
        # Train the model using hyperparameters in yaml config
        history = self.jet_model.fit(
            {'model_input': X_train},
            [y_train,pt_target_train],
            sample_weight = [sample_weight, sample_weight],
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            verbose=self.run_config['verbose'],
            validation_split=self.training_config['validation_split'],
            callbacks=self.callbacks,
            shuffle=True,
        )
        
        self.history = history.history
        
    def embedding_predict(self, X_test: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:        
        embedding_model = keras.Model(self.jet_model.input, self.jet_model.get_layer('pool').output)
        return embedding_model.predict(X_test,verbose=0)


    # Decorated with save decorator for added functionality
    @JetTagModel.save_decorator
    def save(self, out_dir: str = "None"):
        """Save the model file

        Args:
            out_dir (str, optional): Where to save it if not in the output_directory. Defaults to "None".
        """
        # Export the model
        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        # Use keras save format !NOT .h5! due to depreciation
        export_path = os.path.join(out_dir, "model/saved_model.keras")
        self.jet_model.save(export_path)
        print(f"Model saved to {export_path}")

    @JetTagModel.load_decorator
    def load(self, out_dir: str = "None"):
        """Load the model file

        Args:
            out_dir (str, optional): Where to load it if not in the output_directory. Defaults to "None".
        """
        # Load the model
        self.jet_model = load_model(f"{out_dir}/model/saved_model.keras")
        
        
@JetModelFactory.register('FloatingDeepSetEmbeddingModel')
class FloatingDeepSetEmbeddingModel(JetTagModel):

    """FloatingDeepSetEmbeddingModel class

    Args:
        JetTagModel (_type_): Base class of a JetTagModel
    """

    schema = Schema(
            {
                "model": str,
                ## generic run config coniguration
                "run_config" : JetTagModel.run_schema,
                "model_config" : {"name" : str,
                                  "conv1d_layers" : list,
                                  "classification_layers" : list,
                                  "regression_layers" : list,
                                  "kernel_initializer" : str,
                                  "aggregator" : And(str, lambda s: s in  ["mean", "max", "attention"]),
                                  "projection_dims" : And(int, lambda s: s >= 1),},
                "quantization_config" : {'pt_output_quantization' : list},
                "training_config" : {"weight_method" : And(str, lambda s: s in  ["none", "ptref", "onlyclass"]),
                                     "validation_split" : And(float, lambda s: s > 0.0),
                                     "embedding_epochs" : And(int, lambda s: s >= 1),
                                     "finetuning_epochs" : And(int, lambda s: s >= 1),
                                     "batch_size" : And(int, lambda s: s >= 1),
                                     "learning_rate" : And(float, lambda s: s > 0.0),
                                     "embedding_lr" : And(float, lambda s: s > 0.0),
                                     "loss_weights" : And(list, lambda s: len(s) == 2),
                                     "EarlyStopping_patience" : And(int, lambda s: s > 0),
                                     "ReduceLROnPlateau_factor" : And(float, lambda s: 1.0 >= s >= 0.0),
                                     "ReduceLROnPlateau_patience" : int,
                                     "ReduceLROnPlateau_min_lr" : And(float, lambda s: s >= 0.0)},
                ## generic hls4ml configuration
                "firmware_config" : {"input_precision" : str,
                                     "class_precision" : str,
                                     "reg_precision": str,
                                     "clock_period" : And(float, lambda s: 0.0 < s <= 10),
                                     "fpga_part" : str,
                                     "project_name" : str}
            }
    )

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
        initialise_tensorflow(self.run_config['num_threads'])
        
        self.common_args = {
            'kernel_initializer': self.model_config['kernel_initializer'],
        }

        L, C = inputs_shape

        # Initialize inputs
        inputs = keras.layers.Input(shape=inputs_shape, name='model_input')
        
        main = BatchNormalization(name='norm_input')(inputs)
                        
        # Encoder
        for iconv1d, depthconv1d in enumerate(self.model_config['conv1d_layers']):
            main = Conv1D(filters=depthconv1d, kernel_size=1, name='Conv1D_' + str(iconv1d + 1), activation='relu',**self.common_args)(main)
        
        main = GlobalAveragePooling1D(data_format='channels_last', name="pool")(main)

        # Projection head
        proj = BatchNormalization(name='norm_embedding')(main)
        for iproj, depthproj in enumerate(self.model_config['projection_layers'][:-1]):
            proj = Dense(depthproj, name='Dense_' + str(iproj + 1) + '_proj', activation='relu', **self.common_args)(proj)
        proj = Dense(self.model_config['projection_layers'][-1], name='Dense_' + str(len(self.model_config['projection_layers'])) + '_proj', activation='linear', **self.common_args)(proj)

        # Jet ID and pT regression heads
        bn_main = BatchNormalization(name='norm_embedding')(main)

        for iclass, depthclass in enumerate(self.model_config['classification_layers']):
            jet_id = Dense(depthclass, name='Dense_' + str(iclass + 1) + '_jetID', activation='relu', **self.common_args)(bn_main if iclass == 0 else jet_id)
        jet_id = Dense(outputs_shape[0], name='Dense_3_jetID', activation='linear', **self.common_args)(jet_id)
        jet_id = Activation('softmax', name='jet_id_output')(jet_id)

        for ireg, depthreg in enumerate(self.model_config['regression_layers']):
            pt_regress = Dense(depthreg, name='Dense_' + str(ireg + 1) + '_pT', activation='relu', **self.common_args)(bn_main if ireg == 0 else pt_regress)
        pt_regress = Dense(1, name='pT_output', **self.common_args)(pt_regress)

        self.backbone_model = keras.Model(inputs=inputs, outputs=main)
        self.embedding_model = keras.Model(inputs=inputs, outputs=proj)
        self.jet_model = keras.Model(inputs=inputs, outputs=[jet_id, pt_regress])

        print("========== Embedding Model Summary ==========")
        print(self.embedding_model.summary())

        print("========== Jet Model Summary ==========")
        print(self.jet_model.summary())
        

    def compile_model(self, num_samples: int):
        """compile the model generating callbacks and loss function
        Args:
            num_samples (int): Number of samples in the training set used for scheduling
        """

        # Pretraining
        steps = self.training_config['embedding_epochs'] * (num_samples // self.training_config['batch_size'])
        scheduler = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.training_config['embedding_lr'], decay_steps=steps
        )

        self.embedding_callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="loss", patience=5, restore_best_weights=True
            )
        ]
        self.embedding_optimizer = tf.keras.optimizers.Adam(scheduler)
        
        self.fine_tune_callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.training_config['EarlyStopping_patience']),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.training_config['ReduceLROnPlateau_factor'],
                patience=self.training_config['ReduceLROnPlateau_patience'],
                min_lr=self.training_config['ReduceLROnPlateau_min_lr'],
            ),
        ]
        
        self.fine_tuning_optimizer = tf.keras.optimizers.Adam(learning_rate=self.training_config['learning_rate'])

        # compile the tensorflow model setting the loss and metrics
        self.jet_model.compile(
            optimizer=self.fine_tuning_optimizer,
            loss={
                self.loss_name + self.output_id_name: 'categorical_crossentropy',
                self.loss_name + self.output_pt_name: keras.losses.Huber(),
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
        Y_train: npt.NDArray[np.float64],
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
        output_shape = (Y_train.shape[-1],) if len(Y_train.shape) > 1 else (1,)
        if self.backbone_model is None or self.jet_model is None:
            self.build_model(input_shape, output_shape)
        # --- Embedding (SimCLR) training (FAST) ---
        x_train = X_train[..., tf.newaxis].astype("float32")
        y_train = Y_train[..., tf.newaxis].astype("float32")
        augment = SimCLRPreprocessing(0.2)
        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train,y_train,sample_weight))
            .shuffle(20_000, reshuffle_each_iteration=True)
            .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.training_config['batch_size'])
            .prefetch(tf.data.AUTOTUNE)
        )

        self.embedding_model.loss = SimCLRLoss()
        callbacks = keras.callbacks.CallbackList(self.embedding_callbacks, add_history=True, model=self.embedding_model)
        logs = {}
        callbacks.on_train_begin(logs=logs)

        @tf.function
        def train_step(x1, x2, y, w):
            # Record operations for automatic differentiation
            with tf.GradientTape() as tape:
                # Forward pass: compute embeddings for both augmented views
                z1 = self.embedding_model(x1, training=True)
                z2 = self.embedding_model(x2, training=True)
                zs = tf.stack([z1,z2])
                # Compute SimCLR contrastive loss between the two views
                loss = self.embedding_model.loss(features=zs,labels=y,weights=w)
            # Compute gradients of loss w.r.t. model trainable weights
            grads = tape.gradient(loss, self.embedding_model.trainable_weights)
            # Apply gradients to update model weights using optimizer
            self.embedding_optimizer.apply_gradients(zip(grads, self.embedding_model.trainable_weights))
            # Return the computed loss for logging
            return loss

        for epoch in range(self.training_config['embedding_epochs']):
            callbacks.on_epoch_begin(epoch, logs=logs)
            losses = []
            ibatch = 0
            for x1, x2, y, w in train_ds:
                ibatch += 1
                callbacks.on_train_batch_begin(ibatch, logs=logs)
                loss = train_step(x1, x2, y, w)
                losses.append(loss.numpy())
                callbacks.on_train_batch_end(ibatch, logs=logs)
            logs['loss'] = np.mean(losses)
            print(f"Epoch {epoch+1}: Loss = {logs['loss']:.4f}")
            callbacks.on_epoch_end(epoch, logs=logs)
        callbacks.on_train_end(logs=logs)

        # --- Finetuning (jet model) training ---
        
        print(self.jet_model.get_layer('Dense_1_jetID').get_weights())
        
        #Freeze layers 
        fine_tune_layers = ['norm_embedding','Dense_1_jetID', 'Dense_2_jetID', 'Dense_3_jetID','Dense_1_pT', 'jet_id_output','pT_output']
        for i, layer in enumerate(self.jet_model.layers):
            if layer.name not in fine_tune_layers:
                print(layer.name)
                self.jet_model.get_layer(layer.name).trainable = False
        
        
        keras.config.disable_traceback_filtering()
        sample_weight_dict = {
                            "jet_id_output": sample_weight,
                            "pT_output": sample_weight,
        }
        history = self.jet_model.fit(
            {'model_input': X_train},
            [Y_train,pt_target_train],
            sample_weight = [sample_weight, sample_weight],
            epochs=self.training_config['finetuning_epochs'],
            batch_size=self.training_config['batch_size'],
            verbose=self.run_config['verbose'],
            validation_split=self.training_config['validation_split'],
            callbacks=self.fine_tune_callbacks,
            shuffle=True,
        )
        
        print(self.jet_model.get_layer('Dense_1_jetID').get_weights())
        
        self.history = history.history
        
    def embedding_predict(self, X_test: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:        
        embedding_model = keras.Model(self.jet_model.input, self.jet_model.get_layer('pool').output)
        return embedding_model.predict(X_test,verbose=0)

    # Decorated with save decorator for added functionality
    @JetTagModel.save_decorator
    def save(self, out_dir: str = "None"):
        """Save the model file

        Args:
            out_dir (str, optional): Where to save it if not in the output_directory. Defaults to "None".
        """
        # Export the model
        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        # Use keras save format !NOT .h5! due to depreciation
        export_path = os.path.join(out_dir, "model/saved_model.keras")
        self.jet_model.save(export_path)
        print(f"Model saved to {export_path}")

    @JetTagModel.load_decorator
    def load(self, out_dir: str = "None"):
        """Load the model file

        Args:
            out_dir (str, optional): Where to load it if not in the output_directory. Defaults to "None".
        """
        # Load the model
        self.jet_model = load_model(f"{out_dir}/model/saved_model.keras")