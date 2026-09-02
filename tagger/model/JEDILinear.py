import json
import os
from schema import Schema, And, Use, Optional
from math import log2

import numpy as np
import numpy.typing as npt

import tensorflow as tf
import keras
from keras.models import load_model
from keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D, GlobalMaxPooling1D, Dense, ReLU, Add, UpSampling1D, EinsumDense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers.schedules import CosineDecay

from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tagger.train.pretraining.losses import SimCLRLoss, SupConLoss, HybridSupLoss, VICRegLoss, JepaLoss
from tagger.train.pretraining.augmentations import AugmentationLayer
from tagger.model.common.tensorflow import *
from tagger.train.pretraining.trainer import EmbeddingDiagnostics, ContrastiveTrainer

@JetModelFactory.register('JEDILinearModel')
class JEDILinearModel(JetTagModel):

    """JEDILinearEmbeddingModel class

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
                    "embedding_layers" : list,
                    "local_layers" : list,
                    "global_layers" : list,
                    "interaction_layers" : list,
                    "classification_layers" : list,
                    "regression_layers" : list,
                    "aggregator" : And(str, lambda s: s in ["mean", "max"]),
                    "kernel_initializer" : str,},
                "quantization_config" : {'pt_output_quantization' : list},
                "training_config" : {
                    "weight_method" : And(str, lambda s: s in  ["none", "ptref", "onlyclass"]),
                    "validation_split" : And(float, lambda s: s > 0.0),
                    "epochs" : And(int, lambda s: s >= 1),
                    "batch_size" : And(int, lambda s: s >= 1),
                    "learning_rate" : And(float, lambda s: s > 0.0),
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

    def build_model(self, inputs_shape, outputs_shape):
        # initialise_tensorflow(self.run_config['num_threads'])
        self.common_args = {
            'kernel_initializer': self.model_config['kernel_initializer'],
        }

        inputs = Input(shape=inputs_shape, name='input')
        main = BatchNormalization(name='norm_input')(inputs)

        # ----- Main Branch -----
        # Feature embedding
        for iembed, depth_embed in enumerate(self.model_config['embedding_layers']):
            main = Dense(depth_embed, activation='relu', name='dense_embed_' + str(iembed + 1), **self.common_args)(main)

        # Local feature mixing
        local_cxt = main
        for ilocal, depth_local in enumerate(self.model_config['local_layers']):
            local_cxt = Dense(depth_local, activation='relu', name='dense_local_' + str(ilocal + 1), **self.common_args)(local_cxt) # (batch, n_constituents, depth_local)

        # Global feature mixing
        global_cxt = GlobalAveragePooling1D(name='pool_global')(main)
        for iglobal, depth_global in enumerate(self.model_config['global_layers']):
            global_cxt = Dense(depth_global, activation='relu', name='dense_global')(global_cxt) # (batch, depth_global)

        # Combine local and global context
        interaction = Add(name='add_interaction')([local_cxt, global_cxt])
        for iinter, depth_inter in enumerate(self.model_config['interaction_layers']):
            interaction = Dense(depth_inter, activation='relu', name='dense_interaction_' + str(iinter + 1), **self.common_args)(interaction)

        main = GlobalAveragePooling1D(name='pool')(interaction)
        
        # Batchnorm for the output heads - not used in embedding projection branch
        # Unfrozen during fine-tuning
        bn_main = BatchNormalization(name='norm_embedding')(main) 

        # ----- Classification branch -----
        for iclass, depthclass in enumerate(self.model_config['classification_layers']):
            jet_id = Dense(depthclass, name='Dense_' + str(iclass + 1) + '_jet_id', activation='relu', **self.common_args)(bn_main if iclass == 0 else jet_id)

        jet_id = Dense(outputs_shape[0], name='Dense_output_jet_id', **self.common_args)(jet_id)
        jet_id = Activation('softmax', name='jet_id')(jet_id)

        # ----- pT regression branch -----
        for ireg, depthreg in enumerate(self.model_config['regression_layers']):
            pt_regress = Dense(depthreg, name='Dense_' + str(ireg + 1) + '_pT', activation='relu', **self.common_args)(bn_main if ireg == 0 else pt_regress)      

        pt_regress = Dense(1, name='pT', **self.common_args)(pt_regress)

        self.jet_model = keras.Model(inputs = inputs, outputs = {self.output_id_name: jet_id, self.output_pt_name: pt_regress})

        print("========== Jet Model ==========")
        print(self.jet_model.summary())

    def compile_model(self, num_samples: int):
        """
        Configure the training process, including optimizers, learning rate schedulers, loss functions, and metrics.
        Args:
            num_samples (int): Number of samples in the training set used for scheduling
        """

        # ----- Embedding Model -----
        self.embedding_optimizer = tf.keras.optimizers.Adam(learning_rate=self.training_config['learning_rate'])
        
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



@JetModelFactory.register('JEDILinearEmbeddingModel')
class JEDILinearEmbeddingModel(JetTagModel):

    """JEDILinearEmbeddingModel class

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
                    "embedding_layers" : list,
                    "local_layers" : list,
                    "global_layers" : list,
                    "interaction_layers" : list,
                    "projection_layers": And(list, lambda s: len(s) >= 1),
                    "classification_layers" : list,
                    "regression_layers" : list,
                    "aggregator" : And(str, lambda s: s in ["mean", "max"]),
                    "kernel_initializer" : str,},
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

    def build_model(self, inputs_shape, outputs_shape):
        # initialise_tensorflow(self.run_config['num_threads'])
        self.common_args = {
            'kernel_initializer': self.model_config['kernel_initializer'],
        }

        n_constituents = inputs_shape[0]
        inputs = Input(shape=inputs_shape, name='input')
        main = BatchNormalization(name='norm_input')(inputs)

        # ----- Main Branch -----
        # Feature embedding
        for iembed, depth_embed in enumerate(self.model_config['embedding_layers']):
            main = Dense(depth_embed, activation='relu', name='dense_embed_' + str(iembed + 1), **self.common_args)(main)

        # Local feature mixing
        local_cxt = main
        for ilocal, depth_local in enumerate(self.model_config['local_layers']):
            local_cxt = Dense(depth_local, activation='relu', name='dense_local_' + str(ilocal + 1), **self.common_args)(local_cxt) # (batch, n_constituents, depth_local)

        # Global feature mixing
        global_cxt = GlobalAveragePooling1D(name='pool_global')(main)
        for iglobal, depth_global in enumerate(self.model_config['global_layers']):
            global_cxt = Dense(depth_global, activation='relu', name='dense_global_' + str(iglobal + 1))(global_cxt) # (batch, depth_global)

        # Combine local and global context
        interaction = Add(name='add_interaction')([local_cxt, global_cxt])
        for iinter, depth_inter in enumerate(self.model_config['interaction_layers']):
            interaction = Dense(depth_inter, activation='relu', name='dense_interaction_' + str(iinter + 1), **self.common_args)(interaction)

        main = GlobalAveragePooling1D(name='pool')(interaction)

        # Batchnorm for the output heads - not used in embedding projection branch
        # Unfrozen during fine-tuning
        bn_main = BatchNormalization(name='norm_embedding')(main) 

        # ----- Embedding Projection branch -----
        # Thrown away after pre-training
        z = BatchNormalization(name='norm_projection')(main)
        for iproj, depthproj in enumerate(self.model_config['projection_layers']):
            is_last = (iproj == len(self.model_config['projection_layers']) - 1)
            proj = Dense(depthproj, use_bias=False, name=f'Dense_projection_{iproj+1}', **self.common_args)(z if iproj == 0 else proj)
            if not is_last:
                proj = BatchNormalization(name=f'norm_projection_{iproj+1}')(proj)
                proj = ReLU(name=f'relu_projection_{iproj+1}')(proj)  

        # ----- Classification branch -----
        for iclass, depthclass in enumerate(self.model_config['classification_layers']):
            jet_id = Dense(depthclass, name='Dense_' + str(iclass + 1) + '_jet_id', activation='relu', **self.common_args)(bn_main if iclass == 0 else jet_id)

        jet_id = Dense(outputs_shape[0], name='Dense_output_jet_id', **self.common_args)(jet_id)
        jet_id = Activation('softmax', name='jet_id')(jet_id)

        # ----- pT regression branch -----
        for ireg, depthreg in enumerate(self.model_config['regression_layers']):
            pt_regress = Dense(depthreg, name='Dense_' + str(ireg + 1) + '_pT', activation='relu', **self.common_args)(bn_main if ireg == 0 else pt_regress)      

        pt_regress = Dense(1, name='pT', **self.common_args)(pt_regress)

        self.backbone_model = keras.Model(inputs=inputs, outputs=main)
        self.embedding_model = keras.Model(inputs=inputs, outputs=proj)
        self.jet_model = keras.Model(inputs = inputs, outputs = {self.output_id_name: jet_id, self.output_pt_name: pt_regress})

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

