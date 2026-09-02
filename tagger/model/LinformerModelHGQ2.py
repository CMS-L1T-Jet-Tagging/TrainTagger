import json
import os
import gc
from schema import Schema, And, Use, Optional

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras
from keras.models import load_model
from keras.layers import BatchNormalization, Input, Activation, Dense, ReLU
from hgq.layers import QDense, QBatchNormDense, QBatchNormalization, QLinformerAttention, QGlobalAveragePooling1D, QAdd
from hgq.utils.sugar import FreeEBOPs, BetaPID
import hls4ml

from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tagger.model.common.tensorflow import *
from tagger.model.common.hgq import quantization_decorator, get_ebops, ReduceLROnPlateauWithEbopsThres, EarlyStoppingWithEbopsThres
from tagger.train.pretraining.augmentations import AugmentationLayer
from tagger.train.pretraining.losses import get_loss_function
from tagger.train.pretraining.trainer import ContrastiveTrainer, EmbeddingDiagnostics, InnerModelCallback

@JetModelFactory.register('LinformerModelHGQ2')
class LinformerModelHGQ2(JetTagModel):

    schema = Schema(
            {
                "model": str,
                "run_config" : JetTagModel.run_schema,
                "model_config" : {
                    "name" : str,
                    "embedding_layers" : int,
                    "transformer_blocks" : And(int, lambda s: s >= 1),
                    "transformer_dim" : And(int, lambda s: s >= 1),
                    "num_heads" : And(int, lambda s: s >= 1),
                    "kv_proj_dim": int,
                    "classification_layers" : list,
                    "regression_layers" : list,
                    "kernel_initializer" : str,
                    "aggregator" : And(str, lambda s: s in ["mean", "max"]),
                    "target_ebops": And(int, lambda s: s > 0),
                },
                "quantization_config" : {'pt_output_quantization' : list},
                "training_config" :     {
                    "weight_method" : And(str, lambda s: s in  ["none", "ptref", "onlyclass"]),
                    "validation_split" : And(float, lambda s: s > 0.0),
                    "epochs" : And(int, lambda s: s >= 1),
                    "batch_size" : And(int, lambda s: s >= 1),
                    "learning_rate": And(float, lambda s: s > 0.0),
                    "loss_weights" : And(dict, lambda s: len(s) == 2),
                    "EarlyStopping_patience" : And(int, lambda s: s > 0),
                    "ReduceLROnPlateau_factor" : And(float, lambda s: 1.0 >= s >= 0.0),
                    "ReduceLROnPlateau_patience" : int,
                    "ReduceLROnPlateau_min_lr" : And(float, lambda s: s >= 0.0),
                    "beta": And(float, lambda s: 1.0 >= s >= 0.0),
                },
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

    @quantization_decorator
    def build_model(self, inputs_shape, outputs_shape):
        
        # initialise_tensorflow(self.run_config['num_threads'])

        self.common_args = {
            'kernel_initializer': self.model_config['kernel_initializer'],
        }

        N_constituents = inputs_shape[0]
        n_features = inputs_shape[1]

        tdim = self.model_config['transformer_dim']
        nheads = self.model_config['num_heads']

        inputs = keras.layers.Input((N_constituents, n_features), name='model_input')
        main = QBatchNormalization(name='norm_input')(inputs)

        # ----- Main branch -----                                
        # Embedding
        for i in range(self.model_config['embedding_layers']):
            main = QDense(tdim, activation='relu', parallelization_factor=tdim, name='emb_'+str(i+1), **self.common_args)(main)

        # Transformer blocks
        for i in range(self.model_config['transformer_blocks']):
            norm1 = QBatchNormalization(name='norm_mha_'+str(i+1))(main)
            mha = QLinformerAttention(nheads, self.model_config["kv_proj_dim"], tdim//nheads, parallelization_factor=tdim, name='mha_'+str(i+1))(norm1, norm1)
            main = QAdd(name='res_mha_'+str(i+1))([main, mha])
            ff = QBatchNormDense(tdim, activation='relu', parallelization_factor=tdim, name='dense_'+str(i+1)+'_1', **self.common_args)(main)
            ff = QDense(tdim, activation=None, parallelization_factor=tdim, name='dense_'+str(i+1)+'_2', **self.common_args)(ff)
            main = QAdd(name='res_ff_'+str(i+1))([main, ff])

        # Global pooling
        main = QGlobalAveragePooling1D(name='pool', oq_conf=None)(main)
                
        # ---- Output heads -----
        bn_main = QBatchNormalization(name='norm_embedding')(main)

        # ----- Classification branch -----
        for iclass, depthclass in enumerate(self.model_config['classification_layers']):
            jet_id = QDense(
                units=depthclass, 
                parallelization_factor=depthclass, 
                name='Dense_' + str(iclass + 1) + '_jet_id', 
                activation='relu', 
                **self.common_args,
            )(bn_main if iclass == 0 else jet_id)

        jet_id = QDense(units=outputs_shape[0], parallelization_factor=outputs_shape[0], name='Dense_output_jet_id', **self.common_args)(jet_id)
        jet_id = Activation('softmax', name='jet_id')(jet_id)

        # ----- pT regression branch -----
        for ireg, depthreg in enumerate(self.model_config['regression_layers']):
            pt_regress = QDense(
                units=depthreg, 
                parallelization_factor=depthreg, 
                name='Dense_' + str(ireg + 1) + '_pT', 
                activation='relu', 
                **self.common_args,
            )(bn_main if ireg == 0 else pt_regress)      

        pt_regress = QDense(1, name='pT', oq_conf=None, **self.common_args)(pt_regress)

        self.jet_model = keras.Model(inputs = inputs, outputs = {self.output_id_name: jet_id, self.output_pt_name: pt_regress})

        print("========== Jet Model ==========")
        print(self.jet_model.summary())

    def compile_model(self, num_samples: int):
        
        """compile the model generating callbacks and loss function
        Args:
            num_samples (int): Number of samples in the training set used for scheduling
        """

        # ----- Fine-tuning Model -----
        self.callbacks = [
            BetaPID(
                p=1, i=0.1, d=0,
                target_ebops=self.model_config['target_ebops'],
                init_beta=1e-10, warmup=10,
                max_beta=1e-6, damp_beta_on_target=0.5
            ),
            EarlyStoppingWithEbopsThres(
                monitor="val_loss",
                patience=self.training_config['EarlyStopping_patience'],
                verbose=1,
                mode="min",
                restore_best_weights=True,
                ebops_threshold=self.model_config['target_ebops'],
            ),
            ReduceLROnPlateauWithEbopsThres(
                monitor='val_loss',
                factor=self.training_config['ReduceLROnPlateau_factor'],
                patience=self.training_config['ReduceLROnPlateau_patience'],
                min_lr=self.training_config['ReduceLROnPlateau_min_lr'],
                ebops_threshold=self.model_config['target_ebops'],
            ),
            FreeEBOPs(),
        ]
        self.optimizer = keras.optimizers.Adam(learning_rate=self.training_config['learning_rate'])
        
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
        """Fit the model to the training dataset

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
        x_train, x_val, y_train, y_val, pt_train, pt_val, w_train, w_val = train_test_split(
            x_train, y_train, pt_train, sample_weight,
            test_size=self.training_config['validation_split'],
            random_state=42,
            shuffle=True,
        )

        # --- Finetuning (jet model) training ---
        keras.config.disable_traceback_filtering()

        # # Construct datasets for finetuning to avoid excessive memory usage when using over 50% of dataset
        # train_finetune_ds = make_finetuning_dataset(x_train, y_train, pt_train, w_train, self.training_config['batch_size'], self.output_id_name, self.output_pt_name, train=True)
        # val_finetune_ds = make_finetuning_dataset(x_val, y_val, pt_val, w_val, self.training_config['batch_size'], self.output_id_name, self.output_pt_name, train=False)

        self.jet_model.compile(
            optimizer=self.optimizer,
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
            callbacks=self.callbacks,
        )
        self.history = history.history

        print("========== EBOPs ==========")
        total_ebops = get_ebops(self.jet_model, print_layers=True)
        print(f"Total EBOPs: {total_ebops}")

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


@JetModelFactory.register('LinformerHGQ2EmbeddingModel')
class LinformerHGQ2EmbeddingModel(JetTagModel):

    schema = Schema(
            {
                "model": str,
                "run_config" : JetTagModel.run_schema,
                "model_config" : {
                    "name" : str,
                    "embedding_layers" : int,
                    "transformer_blocks" : And(int, lambda s: s >= 1),
                    "transformer_dim" : And(int, lambda s: s >= 1),
                    "num_heads" : And(int, lambda s: s >= 1),
                    "kv_proj_dim": int,
                    "projection_layers" : And(list, lambda s: len(s) >= 1),
                    "classification_layers" : list,
                    "regression_layers" : list,
                    "kernel_initializer": str,
                    "aggregator" : And(str, lambda s: s in ["mean", "max"]),
                    "target_ebops_encoder": And(int, lambda s: s > 0),
                    "target_ebops_output": And(int, lambda s: s > 0),
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
                    "ReduceLROnPlateau_min_lr" : And(float, lambda s: s >= 0.0),
                    "beta": And(float, lambda s: 1.0 >= s >= 0.0),
                },
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

    @quantization_decorator
    def build_model(self, inputs_shape, outputs_shape):
        
        self.common_args = {
            'kernel_initializer': self.model_config['kernel_initializer'],
        }

        N_constituents = inputs_shape[0]
        n_features = inputs_shape[1]

        tdim = self.model_config['transformer_dim']
        nheads = self.model_config['num_heads']

        inputs = keras.layers.Input((N_constituents, n_features), name='model_input')
        main = QBatchNormalization(name='norm_input')(inputs)

        # ----- Main branch -----                                
        # Embedding
        for i in range(self.model_config['embedding_layers']):
            main = QDense(tdim, activation='relu', parallelization_factor=tdim, name='emb_'+str(i+1), **self.common_args)(main)

        # Linformer blocks
        for i in range(self.model_config['transformer_blocks']):
            norm1 = QBatchNormalization(name='norm_mha_'+str(i+1))(main)
            mha = QLinformerAttention(
                num_heads=nheads, 
                lin_kv_proj_dim=self.model_config["kv_proj_dim"], 
                key_dim=tdim//nheads,
                parallelization_factor=tdim, 
                name='mha_'+str(i+1)
            )(norm1, norm1)
            main = QAdd(name='res_mha_'+str(i+1))([main, mha])
            ff = QBatchNormDense(tdim, activation='relu', parallelization_factor=tdim, name='dense_'+str(i+1)+'_1', **self.common_args)(main)
            ff = QDense(tdim, activation=None, parallelization_factor=tdim, name='dense_'+str(i+1)+'_2', **self.common_args)(ff)
            main = QAdd(name='res_ff_'+str(i+1))([main, ff])

        # Global pooling
        main = QGlobalAveragePooling1D(name='pool', oq_conf=None)(main)
                
        # ----- Embedding Projection branch -----
        # Thrown away after pre-training
        z = BatchNormalization(name='norm_projection')(main)
        for iproj, depthproj in enumerate(self.model_config['projection_layers']):
            is_last = (iproj == len(self.model_config['projection_layers']) - 1)
            z = Dense(depthproj, use_bias=False, name=f'Dense_projection_{iproj+1}', **self.common_args)(z)
            if not is_last:
                z = BatchNormalization(name=f'norm_projection_{iproj+1}')(z)
                z = ReLU(name=f'relu_projection_{iproj+1}')(z)  
        
        # ---- Output heads -----
        bn_main = QBatchNormalization(name='norm_embedding')(main)

        # ----- Classification branch -----
        for iclass, depthclass in enumerate(self.model_config['classification_layers']):
            jet_id = QDense(
                units=depthclass, 
                parallelization_factor=depthclass, 
                name='Dense_' + str(iclass + 1) + '_jet_id', 
                activation='relu', 
                **self.common_args,
            )(bn_main if iclass == 0 else jet_id)

        jet_id = QDense(units=outputs_shape[0], parallelization_factor=outputs_shape[0], name='Dense_output_jet_id', **self.common_args)(jet_id)
        jet_id = Activation('softmax', name='jet_id')(jet_id)

        # ----- pT regression branch -----
        for ireg, depthreg in enumerate(self.model_config['regression_layers']):
            pt_regress = QDense(
                units=depthreg, 
                parallelization_factor=depthreg, 
                name='Dense_' + str(ireg + 1) + '_pT', 
                activation='relu', 
                **self.common_args,
            )(bn_main if ireg == 0 else pt_regress)      

        pt_regress = QDense(1, name='pT', oq_conf=None, **self.common_args)(pt_regress)

        # Define the model using all branches
        self.backbone_model = keras.Model(inputs=inputs, outputs=main)
        self.embedding_model = keras.Model(inputs=inputs, outputs=z)
        self.jet_model = keras.Model(inputs = inputs, outputs = {self.output_id_name: jet_id, self.output_pt_name: pt_regress})
                
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
        self.embedding_optimizer = tf.keras.optimizers.Adam(learning_rate=self.training_config['embedding_lr'])

        # ----- Fine-tuning Model -----
        target_ebops = self.model_config['target_ebops_encoder'] + self.model_config['target_ebops_output']
        self.fine_tune_callbacks = [
            BetaPID(
                p=1, i=0.1, d=0,
                target_ebops=target_ebops,
                init_beta=1e-10, warmup=10,
                max_beta=1e-6, damp_beta_on_target=0.5
            ),
            EarlyStoppingWithEbopsThres(
                monitor="val_loss",
                patience=self.training_config['EarlyStopping_patience'],
                verbose=1,
                mode="min",
                restore_best_weights=True,
                ebops_threshold=target_ebops,
            ),
            ReduceLROnPlateauWithEbopsThres(
                monitor='val_loss',
                factor=self.training_config['ReduceLROnPlateau_factor'],
                patience=self.training_config['ReduceLROnPlateau_patience'],
                min_lr=self.training_config['ReduceLROnPlateau_min_lr'],
                ebops_threshold=target_ebops,
            ),
            FreeEBOPs(),
        ]
        self.fine_tuning_optimizer = keras.optimizers.Adam(learning_rate=self.training_config['learning_rate'])
        
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
        x_train, x_val, y_train, y_val, pt_train, pt_val, w_train, w_val = train_test_split(
            x_train, y_train, pt_train, sample_weight,
            test_size=self.training_config['validation_split'],
            random_state=42,
            shuffle=True,
        )

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
        embedding_loss = get_loss_function(loss_name, **self.training_config['loss'].get('params', {}))

        target_ebops = self.model_config['target_ebops_encoder']
        callbacks = [
            InnerModelCallback(FreeEBOPs(), model_attr="embedding_model"),
            InnerModelCallback(BetaPID(
                p=1, i=0.1, d=0,
                target_ebops=target_ebops,
                init_beta=1e-9, warmup=5,
                max_beta=1e-6, damp_beta_on_target=0.5
            ), model_attr="embedding_model"),
            ReduceLROnPlateauWithEbopsThres(
                monitor='val_emb_loss',
                mode='min',
                factor=self.training_config['ReduceLROnPlateau_factor'],
                patience=5,
                min_lr=self.training_config['ReduceLROnPlateau_min_lr'],
                ebops_threshold=target_ebops,
                model_attr="embedding_model",
            ),
            EarlyStoppingWithEbopsThres(
                monitor="val_emb_loss",
                mode="min",
                patience=20,
                verbose=1,
                restore_best_weights=True,
                ebops_threshold=target_ebops,
                model_attr="embedding_model",
            ),
            EmbeddingDiagnostics(self.backbone_model, x_train, x_val),
        ]
        
        trainer = ContrastiveTrainer(self.embedding_model, embedding_loss, use_hgq=True)
        trainer.compile(optimizer=self.embedding_optimizer, jit_compile=False,)
        pretrain_history = trainer.fit(
            train_ds, 
            validation_data=val_ds, 
            epochs=self.training_config["embedding_epochs"], 
            callbacks=callbacks, 
            verbose=self.run_config['verbose'],
        )
        del train_ds, val_ds, trainer, embedding_loss, callbacks; gc.collect()  # Free up memory

        print("========== EBOPs ==========")
        total_ebops = get_ebops(self.jet_model, print_layers=True)
        print(f"Total EBOPs: {total_ebops}")

        # --- Finetuning (jet model) training ---
        keras.config.disable_traceback_filtering()

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

        # # Construct datasets for finetuning to avoid excessive memory usage when using over 50% of dataset
        # train_finetune_ds = make_finetuning_dataset(x_train, y_train, pt_train, w_train, self.training_config['batch_size'], self.output_id_name, self.output_pt_name, train=True)
        # val_finetune_ds = make_finetuning_dataset(x_val, y_val, pt_val, w_val, self.training_config['batch_size'], self.output_id_name, self.output_pt_name, train=False)

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

        print("========== EBOPs ==========")
        total_ebops = get_ebops(self.jet_model, print_layers=True)
        print(f"Total EBOPs: {total_ebops}")

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
