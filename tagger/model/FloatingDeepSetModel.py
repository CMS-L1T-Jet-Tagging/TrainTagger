"""DeepSet model child class

Written 07/10/2025 cebrown@cern.ch
"""

from collections import defaultdict
import gc
import gc
import json
import os

#import hls4ml
import tensorflow as tf
import numpy as np
import numpy.typing as npt
from schema import Schema, And, Use, Optional
from tqdm import tqdm

from tagger.model.common.tensorflow import *
from tagger.train.pretraining.losses import SimCLRLoss, SupConLoss, HybridSupLoss, VICRegLoss, JepaLoss
from tagger.train.pretraining.augmentations import AugmentationLayer
from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tagger.train.pretraining.trainer import ContrastiveTrainer, EmbeddingDiagnostics

import keras
from keras.models import load_model
from keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D, GlobalMaxPooling1D, Dense, Conv1D, ReLU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers.schedules import CosineDecay


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
                                     "loss_weights" : And(dict, lambda s: len(s) == 2),
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

        inputs = keras.layers.Input(shape=inputs_shape, name='model_input')
        
        # ----- Encoder -----
        # Everthing here is frozen during fine-tuning
        main = BatchNormalization(name='norm_input')(inputs)
                        
        for iconv1d, depthconv1d in enumerate(self.model_config['conv1d_layers']):
            is_last = (iconv1d == len(self.model_config['conv1d_layers']) - 1)
            main = Conv1D(
                filters=depthconv1d, 
                kernel_size=1, 
                name='Conv1D_' + str(iconv1d + 1), 
                activation='relu', 
                **self.common_args
            )(main)

        if self.model_config['aggregator'] == 'mean':
            main = GlobalAveragePooling1D(data_format='channels_last',name="pool")(main)
        elif self.model_config['aggregator'] == 'max':
            main = GlobalMaxPooling1D(data_format='channels_last',name="pool")(main)
        elif self.model_config['aggregator'] == 'attention':
            main = AttentionPoolingLayer(name="pool")(main)

        # ---- Output heads -----
        bn_main = BatchNormalization(name='norm_embedding')(main)

        # ----- Classification branch -----
        for iclass, depthclass in enumerate(self.model_config['classification_layers']):
            jet_id = Dense(
                depthclass, 
                name='Dense_' + str(iclass + 1) + '_jet_id', 
                activation='relu', 
                **self.common_args
            )(bn_main if iclass == 0 else jet_id)

        jet_id = Dense(outputs_shape[0], name='Dense_output_jet_id', **self.common_args)(jet_id)
        jet_id = Activation('softmax', name='jet_id')(jet_id)

        # ----- pT regression branch -----
        for ireg, depthreg in enumerate(self.model_config['regression_layers']):
            pt_regress = Dense(depthreg, name='Dense_' + str(ireg + 1) + '_pT', activation='relu', **self.common_args)(bn_main if ireg == 0 else pt_regress)

        pt_regress = Dense(1, name='pT', **self.common_args)(pt_regress)

        # Define the model using all branches
        self.jet_model = keras.Model(inputs=inputs, outputs={self.output_id_name: jet_id, self.output_pt_name: pt_regress})

        print("========== Jet Model ==========")
        print(self.jet_model.summary())

    def compile_model(self, num_samples: int):
        """
        Configure the training process, including optimizers, learning rate schedulers, loss functions, and metrics.
        Args:
            num_samples (int): Number of samples in the training set used for scheduling
        """
        
        # ----- Finetuning Model -----
        self.fine_tune_callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.training_config['EarlyStopping_patience'], restore_best_weights=True),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.training_config['ReduceLROnPlateau_factor'],
                patience=self.training_config['ReduceLROnPlateau_patience'],
                min_lr=self.training_config['ReduceLROnPlateau_min_lr'],
            ),
        ]
        self.fine_tuning_optimizer = tf.keras.optimizers.Adam(learning_rate=self.training_config['learning_rate'])
        
        self.jet_loss = {
            self.output_id_name: keras.losses.CategoricalCrossentropy(name='cce_loss'),
            self.output_pt_name: keras.losses.Huber(name='pT_loss'),
        }
        self.jet_loss_weights = {
            self.output_id_name: self.training_config['loss_weights']['jet_id'],
            self.output_pt_name: self.training_config['loss_weights']['pT'],
        }
        self.jet_metrics = {
            self.output_id_name: keras.metrics.CategoricalAccuracy(name='acc'),
            self.output_pt_name: [keras.metrics.MeanAbsoluteError(name='mae'), keras.metrics.MeanSquaredError(name='mse')],
        }
        self.jet_weighted_metrics = {
            self.output_id_name: keras.metrics.CategoricalAccuracy(name='w_acc'),
            self.output_pt_name: [keras.metrics.MeanAbsoluteError(name='w_mae'), keras.metrics.MeanSquaredError(name='w_mse')]
        }

    def fit(
        self,
        x_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        pt_train: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64],
    ):
        """Fit the model to the training dataset (embedding + finetune, optimized for speed)

        Args:
            x_train (npt.NDArray[np.float64]): X train dataset
            y_train (npt.NDArray[np.float64]): y train classification targets
            pt_train (npt.NDArray[np.float64]): y train pt regression targets
            sample_weight (npt.NDArray[np.float64]): sample weighting
        """
        keras.config.disable_traceback_filtering()
        x_train = x_train.astype("float32")
        y_train = y_train.astype("float32")
        pt_train = pt_train.astype("float32")
        sample_weight = sample_weight.astype("float32")
        
        # Ensure backbone and embedding model are built
        input_shape = x_train.shape[1:]
        output_shape = (y_train.shape[-1],) if len(y_train.shape) > 1 else (1,)
        if self.backbone_model is None or self.jet_model is None:
            self.build_model(input_shape, output_shape)

        # Create validation split for embedding training
        rng = np.random.default_rng(42)
        val_split = self.training_config['validation_split']
        val_indices = rng.choice(len(x_train), size=int(x_train.shape[0] * val_split), replace=False,)
        is_val = np.zeros(len(x_train), dtype=bool)
        is_val[val_indices] = True
        is_train = ~is_val

        x_train, x_val = x_train[is_train], x_train[is_val]
        y_train, y_val = y_train[is_train], y_train[is_val]
        pt_train, pt_val = pt_train[is_train], pt_train[is_val]
        w_train, w_val = sample_weight[is_train], sample_weight[is_val]

        # --- Finetuning (jet model) training ---
        keras.config.disable_traceback_filtering()

        # # Construct datasets for finetuning to avoid excessive memory usage
        # train_finetune_ds = tf.data.Dataset.from_tensor_slices((
        #         x_train,
        #         {self.output_id_name: y_train, self.output_pt_name: pt_train,},
        #         {self.output_id_name: w_train, self.output_pt_name: w_train,},
        #     ))
        # train_finetune_ds = (
        #     train_finetune_ds
        #     .shuffle(self.training_config['batch_size']*10, reshuffle_each_iteration=True)
        #     .batch(self.training_config['batch_size'], drop_remainder=False,)
        #     .prefetch(tf.data.AUTOTUNE)
        # )

        # val_finetune_ds = tf.data.Dataset.from_tensor_slices((
        #     x_val,
        #     {self.output_id_name: y_val, self.output_pt_name: pt_val,},
        #     {self.output_id_name: w_val, self.output_pt_name: w_val,},
        # ))
        # val_finetune_ds = (
        #     val_finetune_ds
        #     .batch(20_000, drop_remainder=False,)
        #     .prefetch(tf.data.AUTOTUNE)
        # )

        self.jet_model.compile(
            optimizer=self.fine_tuning_optimizer,
            loss=self.jet_loss,
            loss_weights=self.jet_loss_weights,
            metrics=self.jet_metrics,
            weighted_metrics=self.jet_weighted_metrics,
        )
        history = self.jet_model.fit(
            x=x_train,
            y={self.output_id_name: y_train, self.output_pt_name: pt_train,},
            sample_weight={self.output_id_name: w_train, self.output_pt_name: w_train,},
            validation_data=(
                x_val, 
                {self.output_id_name: y_val, self.output_pt_name: pt_val,},
                {self.output_id_name: w_val, self.output_pt_name: w_val,}
            ),
            epochs=self.training_config["epochs"],
            batch_size=self.training_config["batch_size"],
            verbose=self.run_config["verbose"],
            callbacks=self.fine_tune_callbacks,
        )
        self.history = history.history

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
                "model_config" : {
                    "name" : str,
                    "conv1d_layers" : list,
                    "projection_layers": And(list, lambda s: len(s) >= 1),
                    "classification_layers" : list,
                    "regression_layers" : list,
                    "kernel_initializer" : str,
                    "aggregator" : And(str, lambda s: s in  ["mean", "max", "attention"]),
                },
                "quantization_config" : {'pt_output_quantization' : list},
                "training_config" : {
                    "loss": dict,
                    "weight_method" : And(str, lambda s: s in  ["none", "ptref", "onlyclass"]),
                    "validation_split" : And(float, lambda s: s > 0.0),
                    "embedding_epochs" : And(int, lambda s: s >= 1),
                    "finetuning_epochs" : And(int, lambda s: s >= 1),
                    "freeze_backbone" : bool,
                    "batch_size" : And(int, lambda s: s >= 1),
                    "learning_rate" : And(float, lambda s: s > 0.0),
                    "embedding_lr" : And(float, lambda s: s > 0.0),
                    "loss_weights" : And(dict, lambda s: len(s) == 2),
                    "EarlyStopping_patience" : And(int, lambda s: s > 0),
                    "ReduceLROnPlateau_factor" : And(float, lambda s: 1.0 >= s >= 0.0),
                    "ReduceLROnPlateau_patience" : int,
                    "ReduceLROnPlateau_min_lr" : And(float, lambda s: s >= 0.0)
                },
                ## generic hls4ml configuration
                "firmware_config" : {
                    "input_precision" : str,
                    "class_precision" : str,
                    "reg_precision": str,
                    "clock_period" : And(float, lambda s: 0.0 < s <= 10),
                    "fpga_part" : str,
                    "project_name" : str
                }
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

        inputs = keras.layers.Input(shape=inputs_shape, name='model_input')
        
        # ----- Encoder -----
        # Everthing here is frozen during fine-tuning
        main = BatchNormalization(name='norm_input')(inputs)
                        
        for iconv1d, depthconv1d in enumerate(self.model_config['conv1d_layers']):
            is_last = (iconv1d == len(self.model_config['conv1d_layers']) - 1)
            main = Conv1D(
                filters=depthconv1d, 
                kernel_size=1, 
                name='Conv1D_' + str(iconv1d + 1), 
                activation='relu', 
                **self.common_args
            )(main)

        if self.model_config['aggregator'] == 'mean':
            main = GlobalAveragePooling1D(data_format='channels_last',name="pool")(main)
        elif self.model_config['aggregator'] == 'max':
            main = GlobalMaxPooling1D(data_format='channels_last',name="pool")(main)
        elif self.model_config['aggregator'] == 'attention':
            main = AttentionPoolingLayer(name="pool")(main)

        # ---- Output heads -----
        bn_main = BatchNormalization(name='norm_embedding')(main)

        # ----- Embedding Projection branch -----
        # Thrown away after pre-training
        z = BatchNormalization(name='norm_projection')(main)
        for iproj, depthproj in enumerate(self.model_config['projection_layers']):
            is_last = (iproj == len(self.model_config['projection_layers']) - 1)
            z = Dense(depthproj, use_bias=False, name=f'Dense_projection_{iproj+1}', **self.common_args)(z)
            if not is_last:
                z = BatchNormalization(name=f'norm_projection_{iproj+1}')(z)
                z = ReLU(name=f'relu_projection_{iproj+1}')(z)

        # ----- Classification branch -----
        for iclass, depthclass in enumerate(self.model_config['classification_layers']):
            jet_id = Dense(
                depthclass, 
                name='Dense_' + str(iclass + 1) + '_jet_id', 
                activation='relu', 
                **self.common_args
            )(bn_main if iclass == 0 else jet_id)

        jet_id = Dense(outputs_shape[0], name='Dense_output_jet_id', **self.common_args)(jet_id)
        jet_id = Activation('softmax', name='jet_id')(jet_id)

        # ----- pT regression branch -----
        for ireg, depthreg in enumerate(self.model_config['regression_layers']):
            pt_regress = Dense(depthreg, name='Dense_' + str(ireg + 1) + '_pT', activation='relu', **self.common_args)(bn_main if ireg == 0 else pt_regress)

        pt_regress = Dense(1, name='pT', **self.common_args)(pt_regress)

        # Define the model using all branches
        self.backbone_model = keras.Model(inputs=inputs, outputs=main)
        self.embedding_model = keras.Model(inputs=inputs, outputs=z)
        self.jet_model = keras.Model(inputs=inputs, outputs={'jet_id': jet_id, 'pT': pt_regress})
                
        print("========== Backbone Model ==========")
        print(self.backbone_model.summary())

        print("========== Embedding Model ==========")
        print(self.embedding_model.summary())

        print("========== Jet Model ==========")
        print(self.jet_model.summary())
        
    def compile_model(self, num_samples: int):
        """
        Configure the training process, including optimizers, learning rate schedulers, loss functions, and metrics.
        Args:
            num_samples (int): Number of samples in the training set used for scheduling
        """

        # ----- Embedding Model -----
        batch_size = self.training_config["batch_size"]
        val_split = self.training_config["validation_split"]
        num_train_samples = int(num_samples * (1.0 - val_split))
        steps_per_epoch = num_train_samples // batch_size
        decay_steps = (self.training_config["embedding_epochs"] * steps_per_epoch)
        scheduler = CosineDecay(initial_learning_rate=self.training_config['embedding_lr'], decay_steps=decay_steps)
        self.embedding_optimizer = tf.keras.optimizers.Adam(scheduler)
        
        # ----- Finetuning Model -----
        self.fine_tune_callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.training_config['EarlyStopping_patience'], restore_best_weights=True),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.training_config['ReduceLROnPlateau_factor'],
                patience=self.training_config['ReduceLROnPlateau_patience'],
                min_lr=self.training_config['ReduceLROnPlateau_min_lr'],
            ),
        ]
        self.fine_tuning_optimizer = tf.keras.optimizers.Adam(learning_rate=self.training_config['learning_rate'])
        
        self.jet_loss = {
            self.output_id_name: keras.losses.CategoricalCrossentropy(name='cce_loss'),
            self.output_pt_name: keras.losses.Huber(name='pT_loss'),
        }
        self.jet_loss_weights = {
            self.output_id_name: self.training_config['loss_weights']['jet_id'],
            self.output_pt_name: self.training_config['loss_weights']['pT'],
        }
        self.jet_metrics = {
            self.output_id_name: keras.metrics.CategoricalAccuracy(name='acc'),
            self.output_pt_name: [keras.metrics.MeanAbsoluteError(name='mae'), keras.metrics.MeanSquaredError(name='mse')],
        }
        self.jet_weighted_metrics = {
            self.output_id_name: keras.metrics.CategoricalAccuracy(name='w_acc'),
            self.output_pt_name: [keras.metrics.MeanAbsoluteError(name='w_mae'), keras.metrics.MeanSquaredError(name='w_mse')]
        }

    def fit(
        self,
        x_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        pt_train: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64],
    ):
        """Fit the model to the training dataset (embedding + finetune, optimized for speed)

        Args:
            x_train (npt.NDArray[np.float64]): X train dataset
            y_train (npt.NDArray[np.float64]): y train classification targets
            pt_train (npt.NDArray[np.float64]): y train pt regression targets
            sample_weight (npt.NDArray[np.float64]): sample weighting
        """
        keras.config.disable_traceback_filtering()
        x_train = x_train.astype("float32")
        y_train = y_train.astype("float32")
        pt_train = pt_train.astype("float32")
        sample_weight = sample_weight.astype("float32")
        
        # Ensure backbone and embedding model are built
        input_shape = x_train.shape[1:]
        output_shape = (y_train.shape[-1],) if len(y_train.shape) > 1 else (1,)
        if self.backbone_model is None or self.jet_model is None:
            self.build_model(input_shape, output_shape)

        # Create validation split for embedding training
        rng = np.random.default_rng(42)
        val_split = self.training_config['validation_split']
        val_indices = rng.choice(len(x_train), size=int(x_train.shape[0] * val_split), replace=False,)
        is_val = np.zeros(len(x_train), dtype=bool)
        is_val[val_indices] = True
        is_train = ~is_val

        x_train, x_val = x_train[is_train], x_train[is_val]
        y_train, y_val = y_train[is_train], y_train[is_val]
        pt_train, pt_val = pt_train[is_train], pt_train[is_val]
        w_train, w_val = sample_weight[is_train], sample_weight[is_val]

        # --- Embedding (SimCLR) training (FAST) ---

        # Create TensorFlow datasets for training and validation
        augment = AugmentationLayer()
        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train, w_train))
            .shuffle(self.training_config['batch_size']*20, reshuffle_each_iteration=True)
            .batch(self.training_config['batch_size'], drop_remainder=True)
            .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = (tf.data.Dataset
            .from_tensor_slices((x_val, y_val, w_val))
            .batch(self.training_config["batch_size"], drop_remainder=True,)
            .map(augment, num_parallel_calls=tf.data.AUTOTUNE,)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

        loss_name = self.training_config['loss'].get('name', 'SupCon')
        if loss_name == 'SimCLR':
            embedding_loss = SimCLRLoss(**self.training_config['loss'].get('params', {'temperature': 0.1}))
        elif loss_name == 'SupCon':
            embedding_loss = SupConLoss(**self.training_config['loss'].get('params', {'temperature': 0.1}))
        elif loss_name == 'HybridSupLoss':
            embedding_loss = HybridSupLoss(**self.training_config['loss'].get('params', {'temperature': 0.1, 'alpha': 0.1}))
        elif loss_name == 'VICReg':
            embedding_loss = VICRegLoss(**self.training_config['loss'].get('params', {'mu': 25.0, 'lambda': 25.0, 'nu': 1.0}))
        else:
            raise ValueError(f"Unknown loss name: {loss_name}")

        callbacks = [
            EmbeddingDiagnostics(self.backbone_model, x_train, x_val),
            EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        ]
        trainer = ContrastiveTrainer(self.embedding_model, embedding_loss)
        trainer.compile(optimizer=self.embedding_optimizer, jit_compile=False,)
        pretrain_history = trainer.fit(
            train_ds, 
            validation_data=val_ds, 
            epochs=self.training_config["embedding_epochs"], 
            callbacks=callbacks, 
            verbose=self.run_config['verbose'],
        )
        del train_ds, val_ds, trainer, embedding_loss, callbacks

        # --- Finetuning (jet model) training ---
        keras.config.disable_traceback_filtering()

        # # Construct datasets for finetuning to avoid excessive memory usage
        # train_finetune_ds = tf.data.Dataset.from_tensor_slices((
        #         x_train,
        #         {self.output_id_name: y_train, self.output_pt_name: pt_train,},
        #         {self.output_id_name: w_train, self.output_pt_name: w_train,},
        #     ))
        # train_finetune_ds = (
        #     train_finetune_ds
        #     .shuffle(self.training_config['batch_size']*10, reshuffle_each_iteration=True)
        #     .batch(self.training_config['batch_size'], drop_remainder=False,)
        #     .prefetch(tf.data.AUTOTUNE)
        # )

        # val_finetune_ds = tf.data.Dataset.from_tensor_slices((
        #     x_val,
        #     {self.output_id_name: y_val, self.output_pt_name: pt_val,},
        #     {self.output_id_name: w_val, self.output_pt_name: w_val,},
        # ))
        # val_finetune_ds = (
        #     val_finetune_ds
        #     .batch(20_000, drop_remainder=False,)
        #     .prefetch(tf.data.AUTOTUNE)
        # )

        # Freeze backbone layers if specified in the config
        if self.training_config['freeze_backbone']:
            print("Freezing backbone layers:")
            for layer in self.backbone_model.layers:
                print(f"  {layer.name}")
                layer.trainable = False
        else:
            print("Keeping backbone layers as trainable")
            for layer in self.backbone_model.layers:
                layer.trainable = True

        self.jet_model.compile(
            optimizer=self.fine_tuning_optimizer,
            loss=self.jet_loss,
            loss_weights=self.jet_loss_weights,
            metrics=self.jet_metrics,
            weighted_metrics=self.jet_weighted_metrics,
        )
        history = self.jet_model.fit(
            x=x_train,
            y={self.output_id_name: y_train, self.output_pt_name: pt_train,},
            sample_weight={self.output_id_name: w_train, self.output_pt_name: w_train,},
            validation_data=(
                x_val, 
                {self.output_id_name: y_val, self.output_pt_name: pt_val,},
                {self.output_id_name: w_val, self.output_pt_name: w_val,}
            ),
            epochs=self.training_config["finetuning_epochs"],
            batch_size=self.training_config["batch_size"],
            verbose=self.run_config["verbose"],
            callbacks=self.fine_tune_callbacks,
        )
        self.pretrain_history = pretrain_history.history      
        self.history = history.history
        

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

        def json_default(obj):
            if isinstance(obj, np.generic):
                return obj.item()

            if isinstance(obj, np.ndarray):
                return obj.tolist()

            raise TypeError(
                f"Object of type {type(obj).__name__} is not JSON serializable"
            )

        with open(os.path.join(out_dir, 'history_pretrain.json'), 'w') as f:
            json.dump(self.pretrain_history, f, indent=4, default=json_default)

        print(f"Model saved to {export_path}")

    @JetTagModel.load_decorator
    def load(self, out_dir: str = "None"):
        """Load the model file

        Args:
            out_dir (str, optional): Where to load it if not in the output_directory. Defaults to "None".
        """
        # Load the model
        self.jet_model = load_model(f"{out_dir}/model/saved_model.keras")



@JetModelFactory.register('FloatingDeepSetJepaModel')
class FloatingDeepSetJepaModel(JetTagModel):

    """FloatingDeepSetJepaModel class

    Args:
        JetTagModel (_type_): Base class of a JetTagModel
    """

    schema = Schema(
            {
                "model": str,
                ## generic run config coniguration
                "run_config" : JetTagModel.run_schema,
                "model_config" : {
                    "name" : str,
                    "conv1d_layers" : list,
                    "projection_layers": And(list, lambda s: len(s) >= 1),
                    "classification_layers" : list,
                    "regression_layers" : list,
                    "kernel_initializer" : str,
                    "aggregator" : And(str, lambda s: s in  ["mean", "max", "attention"]),
                },
                "quantization_config" : {'pt_output_quantization' : list},
                "training_config" : {
                    # "loss": dict,
                    "weight_method" : And(str, lambda s: s in  ["none", "ptref", "onlyclass"]),
                    "validation_split" : And(float, lambda s: s > 0.0),
                    "embedding_epochs" : And(int, lambda s: s >= 1),
                    "finetuning_epochs" : And(int, lambda s: s >= 1),
                    "freeze_backbone" : bool,
                    "batch_size" : And(int, lambda s: s >= 1),
                    "learning_rate" : And(float, lambda s: s > 0.0),
                    "embedding_lr" : And(float, lambda s: s > 0.0),
                    "loss_weights" : And(list, lambda s: len(s) == 2),
                    "EarlyStopping_patience" : And(int, lambda s: s > 0),
                    "ReduceLROnPlateau_factor" : And(float, lambda s: 1.0 >= s >= 0.0),
                    "ReduceLROnPlateau_patience" : int,
                    "ReduceLROnPlateau_min_lr" : And(float, lambda s: s >= 0.0)
                },
                ## generic hls4ml configuration
                "firmware_config" : {
                    "input_precision" : str,
                    "class_precision" : str,
                    "reg_precision": str,
                    "clock_period" : And(float, lambda s: 0.0 < s <= 10),
                    "fpga_part" : str,
                    "project_name" : str
                }
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
        inputs = keras.layers.Input(shape=inputs_shape, name='model_input')
        
        # ----- Main branch -----
        # Everthing here is frozen during fine-tuning
        main = BatchNormalization(name='norm_input')(inputs)
                        
        for iconv1d, depthconv1d in enumerate(self.model_config['conv1d_layers']):
            main = Conv1D(filters=depthconv1d, kernel_size=1, name='Conv1D_' + str(iconv1d + 1), activation='relu', **self.common_args)(main)

        if self.model_config['aggregator'] == 'mean':
            main = GlobalAveragePooling1D(data_format='channels_last',name="pool")(main)
        elif self.model_config['aggregator'] == 'max':
            main = GlobalMaxPooling1D(data_format='channels_last',name="pool")(main)
        elif self.model_config['aggregator'] == 'attention':
            main = AttentionPoolingLayer(name="pool")(main)

        # Batchnorm for the output heads - not used in embedding projection branch
        # Unfrozen during fine-tuning
        bn_main = BatchNormalization(name='norm_embedding')(main)

        # ----- Embedding Projection branch -----
        # Thrown away after pre-training
        z = BatchNormalization(name='norm_projection')(main)
        for iproj, depthproj in enumerate(self.model_config['projection_layers']):
            proj = Dense(depthproj, use_bias=False, name=f'Dense_projection_{iproj+1}', **self.common_args)(z if iproj == 0 else proj)
            proj = BatchNormalization(name=f'norm_projection_{iproj+1}')(proj)
            proj = ReLU(name=f'relu_projection_{iproj+1}')(proj)

        proj = Dense(depthconv1d, use_bias=False, activation=None, name=f'Dense_projection_output', **self.common_args)(proj)

        # ----- Classification branch -----
        for iclass, depthclass in enumerate(self.model_config['classification_layers']):
            jet_id = Dense(depthclass, name='Dense_' + str(iclass + 1) + '_jet_id', activation='relu', **self.common_args)(bn_main if iclass == 0 else jet_id)

        jet_id = Dense(outputs_shape[0], name='Dense_output_jet_id', **self.common_args)(jet_id)
        jet_id = Activation('softmax', name='jet_id_output')(jet_id)

        # ----- pT regression branch -----
        for ireg, depthreg in enumerate(self.model_config['regression_layers']):
            pt_regress = Dense(depthreg, name='Dense_' + str(ireg + 1) + '_pT', activation='relu', **self.common_args)(bn_main if ireg == 0 else pt_regress)

        pt_regress = Dense(1, name='pT_output', **self.common_args)(pt_regress)

        # Define the model using all branches
        self.backbone_model = keras.Model(inputs=inputs, outputs=main)
        self.embedding_model = keras.Model(inputs=inputs, outputs=proj)
        self.jet_model = keras.Model(inputs, [jet_id, pt_regress])

        # Target model produces the embedding from weights which are the EMA of the backbone model 
        # weights during training, not updated by gradients. Used for the JEPA loss.
        self.target_model = keras.models.clone_model(self.backbone_model)
        self.target_model.set_weights(self.backbone_model.get_weights())
                
        print("========== Backbone Model ==========")
        print(self.backbone_model.summary())

        print("========== Embedding Model ==========")
        print(self.embedding_model.summary())

        print("========== Jet Model ==========")
        print(self.jet_model.summary())


    def compile_model(self, num_samples: int):
        """compile the model generating callbacks and loss function
        Args:
            num_samples (int): Number of samples in the training set used for scheduling
        """

        # ----- Embedding Model -----
        self.embedding_callbacks = [
            EarlyStopping(monitor="loss", patience=5, restore_best_weights=True),
        ]
        self.pretrain_history = defaultdict(list)

        steps = self.training_config['embedding_epochs'] * (num_samples // self.training_config['batch_size'])
        scheduler = CosineDecay(initial_learning_rate=self.training_config['embedding_lr'], decay_steps=steps)
        self.embedding_optimizer = tf.keras.optimizers.Adam(scheduler)
        
        # ----- Finetuning Model -----
        self.fine_tune_callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.training_config['EarlyStopping_patience'], restore_best_weights=True),
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
        keras.config.disable_traceback_filtering()
        # Ensure backbone and embedding model are built
        input_shape = X_train.shape[1:]
        output_shape = (Y_train.shape[-1],) if len(Y_train.shape) > 1 else (1,)
        if self.backbone_model is None or self.jet_model is None:
            self.build_model(input_shape, output_shape)

        # --- Embedding (SimCLR) training (FAST) ---
        x_train = X_train.astype("float32")
        y_train = Y_train.astype("float32")

        print("Shapes:")
        print("  X_train:", x_train.shape)
        print("  Y_train:", y_train.shape)
        print("  sample_weight:", sample_weight.shape)

        augment = AugmentationLayer()
        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train,y_train,sample_weight))
            .shuffle(max(self.training_config['batch_size'], 50_000), reshuffle_each_iteration=True)
            .batch(self.training_config['batch_size'], drop_remainder=True)
            .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

        embedding_loss = JepaLoss()
        normed_space = False  # JEPA does not require normalized space

        callbacks = keras.callbacks.CallbackList(self.embedding_callbacks, add_history=True, model=self.embedding_model)
        logs = {}
        callbacks.on_train_begin(logs=logs)
        ema_decay = self.training_config.get('ema_decay', 0.996)

        @tf.function
        def train_step(x1, x2, y, w):
            with tf.GradientTape() as tape:
                # target sees original, embedding model sees augmented
                target = tf.stop_gradient(self.target_model(x1, training=False))
                pred   = self.embedding_model(x2, training=True)                
                loss   = embedding_loss(pred, target, weights=w)

            grads = tape.gradient(loss, self.embedding_model.trainable_weights)
            self.embedding_optimizer.apply_gradients(zip(grads, self.embedding_model.trainable_weights))

            # EMA update — target encoder is updated by exponential moving average of the online encoder weights.
            for t_weight, e_weight in zip(self.target_model.weights, self.backbone_model.weights):
                t_weight.assign(ema_decay * t_weight + (1.0 - ema_decay) * e_weight)

            return loss

        x1_eval, x2_eval, _, _ = augment(x_train, y_train, sample_weight)
        self.embedding_model.stop_training = False
        for epoch in range(self.training_config['embedding_epochs']):
            callbacks.on_epoch_begin(epoch, logs=logs)
            logs['epoch'] = epoch + 1
            losses = []
            ibatch = 0
            progress_bar = tqdm(train_ds, desc="Training", unit="batch", leave=False)
            for x1, x2, y, w in progress_bar:
                ibatch += 1
                callbacks.on_train_batch_begin(ibatch, logs=logs)
                loss = train_step(x1, x2, y, w)
                loss_value = loss.numpy()
                losses.append(loss_value)
                callbacks.on_train_batch_end(ibatch, logs=logs)
                progress_bar.set_postfix(loss=f"{loss_value:.4f}")

            logs['loss'] = np.mean(losses)
            logs["lr"] = float(keras.ops.convert_to_numpy(self.embedding_optimizer.learning_rate))

            # Compute additional metrics on full dataset for logging
            z = self.backbone_model.predict(x_train, batch_size=20_000, verbose=0)
            z1 = self.target_model.predict(x1_eval, batch_size=20_000, verbose=0)
            z2 = self.embedding_model.predict(x2_eval, batch_size=20_000, verbose=0)
            logs['norm'] = np.mean(np.linalg.norm(z, axis=1))
            logs['effective_rank'] = effective_rank(z, norm=normed_space).numpy()
            logs['alignment'] = alignment(z1, z2).numpy()
            min_embedding_std, _ = embedding_dim_std(z, norm=normed_space)
            logs['min_std'] = min_embedding_std.numpy()

            callbacks.on_epoch_end(epoch, logs=logs)

            print(
                f"Epoch {epoch+1:>3}: | "
                f"Loss={logs['loss']:.3f}, "
                f"Norm={logs['norm']:.3f}, "
                f"EffRank={logs['effective_rank']:.3f}, "
                f"Alignment={logs['alignment']:.3f}, "
                f"MinStd={logs['min_std']:.3f} | "
                f"LR={logs['lr']:.3e}, "
            )

            for k, v in logs.items():
                self.pretrain_history[k].append(float(v))

            if self.embedding_model.stop_training:  
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        callbacks.on_train_end(logs=logs)

        # --- Finetuning (jet model) training ---
        # Convert to float32 for TensorFlow
        pt_target_train = pt_target_train.astype("float32")
        sample_weight = sample_weight.astype("float32")

        if self.training_config['freeze_backbone']:
            print("Freezing layers:")
            fine_tune_layers = ['norm_embedding', 'jet_id', 'pT']
            for i, layer in enumerate(self.jet_model.layers):
                if not any(ft_layer in layer.name for ft_layer in fine_tune_layers):
                    print(f"  {layer.name}")
                    self.jet_model.get_layer(layer.name).trainable = False
        else:
            print("Setting encoder layers to trainable")

        keras.config.disable_traceback_filtering()
        history = self.jet_model.fit(
            x                = {'model_input': x_train},
            y                = [y_train, pt_target_train],
            sample_weight    = [sample_weight, sample_weight],
            epochs           = self.training_config['finetuning_epochs'],
            batch_size       = self.training_config['batch_size'],
            verbose          = self.run_config['verbose'],
            validation_split = self.training_config['validation_split'],
            callbacks        = self.fine_tune_callbacks,
            shuffle          = True,
        )
                
        self.history = history.history
        
    def embedding_predict(self, X_test: npt.NDArray[np.float64], batch_size: int = 2048, verbose: int = 0) -> npt.NDArray[np.float64]:        
        embedding_model = keras.Model(self.jet_model.input, self.jet_model.get_layer('pool').output)
        return embedding_model.predict(X_test, batch_size=batch_size, verbose=verbose)

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

        with open(os.path.join(out_dir, 'history_pretrain.json'), 'w') as f:
            json.dump(self.pretrain_history, f, indent=4)

        print(f"Model saved to {export_path}")

    @JetTagModel.load_decorator
    def load(self, out_dir: str = "None"):
        """Load the model file

        Args:
            out_dir (str, optional): Where to load it if not in the output_directory. Defaults to "None".
        """
        # Load the model
        self.jet_model = load_model(f"{out_dir}/model/saved_model.keras")