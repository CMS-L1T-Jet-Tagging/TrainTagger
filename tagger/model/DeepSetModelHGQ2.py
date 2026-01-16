import json
import os
from schema import Schema, And, Use, Optional
import scipy

import keras
import numpy as np
import numpy.typing as npt
from keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D, AveragePooling1D, Flatten
from hgq.layers import QConv1D, QDense, QMeanPow2,QBatchNormalization, QSoftmax,QLayerBaseSingleInput,QLayerBaseMultiInputs,  QEinsumDenseBatchnorm, QGlobalAveragePooling1D
from hgq.config import LayerConfigScope, QuantizerConfigScope, QuantizerConfig
from hgq.regularizers import MonoL1
from hgq.constraints import MinMax
from hgq.utils.sugar import FreeEBOPs, BetaScheduler, PieceWiseSchedule
# Qkeras

from keras.models import load_model
import hls4ml
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tagger.data.tools import load_data, to_ML
from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tagger.model.common import log_beta_schedule,cosine_decay_restarts
from tagger.model.common_tensorflow import *

@JetModelFactory.register('DeepSetModelHGQ2')
class DeepSetModelHGQ2(JetTagModel):

    schema = Schema(
            {
                "model": str,
                ## generic run config coniguration
                "run_config" : JetTagModel.run_schema,
                "model_config" : {"name" : str,
                                  "conv1d_layers" : list,
                                  "conv1d_parallelisation_factor" : list,
                                  "classification_layers" : list,
                                  "classification_parallelisation_factor" : list,
                                  "regression_layers" : list,
                                  "regression_parallelisation_factor" : list,
                                  "beta": And(float, lambda s: 1.0 >= s >= 0.0),
                                  },

                "quantization_config" : {'pt_output_quantization' : list},

                "training_config" :     {"weight_method" : And(str, lambda s: s in  ["none", "ptref", "onlyclass"]),
                                         "validation_split" : And(float, lambda s: s > 0.0),
                                         "epochs" : And(int, lambda s: s >= 1),
                                         "batch_size" : And(int, lambda s: s >= 1),
                                         "learning_rate": And(float, lambda s: s > 0.0),
                                         "loss_weights" : And(list, lambda s: len(s) == 2),
                                        },
                
                "firmware_config" : {"input_precision" : str,
                                    "class_precision" : str,
                                    "reg_precision": str,
                                    "clock_period" : And(float, lambda s: 0.0 < s <= 10),
                                    "fpga_part" : str,
                                    "project_name" : str}
            }
    )

    def build_model(self, inputs_shape, outputs_shape):

        initialise_tensorflow(self.run_config['num_threads'])
        
        scope0 = QuantizerConfigScope(default_q_type='kbi',
                                      b0=7,
                                      overflow_mode='wrap',
                                      i0=0,
                                      fr=MonoL1(1.e-8),
                                      ir=MonoL1(1.e-8),
                                    )

        scope1 = QuantizerConfigScope(default_q_type='kif',
                                      place='datalane',
                                      overflow_mode='wrap',
                                      f0=7,
                                      fr=MonoL1(1.e-8),
                                      ic=MinMax(0, 12),
                                    )
        heterogeneous_axis = None

        scope2 = LayerConfigScope(enable_ebops=True, heterogeneous_axis=heterogeneous_axis,beta0=1e-8)
        
        with scope0, scope1, scope2:
            
                iq_conf = QuantizerConfig(k0=1, i0=11, f0=12, trainable=False,round_mode='RND',overflow_mode='SAT')
                oq_conf_jetid = QuantizerConfig(k0=0, i0=12, f0=12, trainable=False,round_mode='RND',overflow_mode='SAT')
                oq_conf_pt = QuantizerConfig(k0=1, i0=9, f0=6, trainable=False,round_mode='RND',overflow_mode='SAT')

                N_constituents = inputs_shape[0]
                n_features = inputs_shape[1]
            
                inputs = Input(shape=(N_constituents,n_features), name='model_input')
                main = QBatchNormalization(name='norm_input',iq_conf=iq_conf)(inputs)
                
                for iconv1d, depthconv1d in enumerate(self.model_config['conv1d_layers']):
                    main = QConv1D(filters=depthconv1d, parallelization_factor=self.model_config['conv1d_parallelisation_factor'][iconv1d], kernel_size=1, name='Conv1D_' + str(iconv1d + 1),activation='relu')(main)                
                main = GlobalAveragePooling1D(name='avgpool')(main)
             
                #jetID branch, 3 layer MLP
                
                for iclass, depthclass in enumerate(self.model_config['classification_layers']):
                    if iclass == 0:
                        jet_id = QDense(depthclass, parallelization_factor=self.model_config['classification_parallelisation_factor'][iclass], name='Dense_' + str(iclass + 1) + '_jetID',activation='relu')(main)
                    else:
                        jet_id = QDense(depthclass, parallelization_factor=self.model_config['classification_parallelisation_factor'][iclass], name='Dense_' + str(iclass + 1) + '_jetID',activation='relu')(jet_id)                
                jet_id = QDense(outputs_shape[0], parallelization_factor=outputs_shape[0], activation='relu')(jet_id)
                jet_id = Activation('softmax', name='jet_id_output')(jet_id)
                #pT regression branch
                for ireg, depthreg in enumerate(self.model_config['regression_layers']):
                    if ireg == 0:
                        pt_regress = QDense(depthreg, parallelization_factor=self.model_config['regression_parallelisation_factor'][ireg], name='Dense_' + str(ireg + 1) + '_pT',activation='relu')(main)
                    else:
                        pt_regress = QDense(depthreg, parallelization_factor=self.model_config['regression_parallelisation_factor'][ireg], name='Dense_' + str(ireg + 1) + '_pT',activation='relu')(pt_regress)      
                pt_regress = QDense(1,name='pT_output', enable_oq=True,oq_conf=oq_conf_pt,iq_conf=oq_conf_pt)(pt_regress)#1.1e-7
    
                #Define the model using both branches
                self.jet_model = keras.Model(inputs = inputs, outputs = [jet_id, pt_regress])

                # Define the model using both branches

                print(self.jet_model.summary())

    # Redefine save and load for HGQ due to needing h5 format
    @JetTagModel.save_decorator
    def save(self, out_dir):
        # Export the model
        #model_export = tfmot.sparsity.keras.strip_pruning(self.jet_model)
        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        export_path = os.path.join(out_dir, "model/saved_model.h5")
        self.jet_model.save(export_path)
        print(f"Model saved to {export_path}")

    @JetTagModel.load_decorator
    def load(self, out_dir=None):
        # Load model
        self.jet_model = load_model(f"{out_dir}/model/saved_model.h5")
    
    def predict(self, X_test: npt.NDArray[np.float64]) -> tuple:
        model_outputs = self.jet_model.predict(X_test)
        class_predictions = model_outputs[0]
        pt_ratio_predictions = model_outputs[1].flatten()
        return (class_predictions, pt_ratio_predictions)

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
            config["Model"]["Strategy"]="distributed_arithmetic"
            config["Model"]["ReuseFactor"]=1
            config['IOType'] = 'io_parallel'
           
            # Configuration for conv1d layers
            # hls4ml automatically figures out the paralellization factor
            config['LayerName']['Conv1D_1']['ParallelizationFactor'] = 16
            config['LayerName']['Conv1D_2']['ParallelizationFactor'] = 16
            
            #config['LayerName']['model_input']['Precision']['result'] = self.firmware_config['input_precision']            
            #config["LayerName"]["jet_id_output"]["Precision"]["result"] = self.firmware_config['class_precision']
            #config["LayerName"]["pT_output"]["Precision"]["result"] = self.firmware_config['reg_precision']

            # Additional config

            # Write HLS
            self.hls_jet_model = hls4ml.converters.convert_from_keras_model(
                self.jet_model,
                backend='Vitis',
                project_name=self.firmware_config['project_name'],
                clock_period=self.firmware_config['clock_period'],
                hls_config=config,
                output_dir=f'{hls4ml_outdir}',
                part= self.firmware_config['fpga_part'],
                # namespace='hls4ml_'+self.firmware_config['project_name'],
                # write_weights_txt=False,
                # write_emulation_constants=True,
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

        scheduler = keras.callbacks.LearningRateScheduler(schedule = lambda epoch : cosine_decay_restarts(epoch, 
                                                                                                          initial_learning_rate=self.training_config['learning_rate'],
                                                                                                          max_epochs=self.training_config['epochs']))

        beta_scheduler = BetaScheduler(PieceWiseSchedule([(0, 0.2e-7, 'linear'), (20, 3e-7, 'log'), (100, 3e-6, 'constant')] ))
        # Define the callbacks using hyperparameters in the config
        self.callbacks = [
            FreeEBOPs(),
            scheduler,
            beta_scheduler

        ]
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
        
# Register the model in the factory with the string name corresponding to what is in the yaml config
@JetModelFactory.register('DeepSetHGQ2EmbeddingModel')
class DeepSetHGQ2EmbeddingModel(DeepSetModelHGQ2):
    """DeepSetEmbeddingModel class

    Args:
        DeepSetModel (_type_): Base class of a DeepSetModel
    """
    
    schema = Schema(
            {
                "model": str,
                ## generic run config coniguration
                "run_config" : JetTagModel.run_schema,
                "model_config" : {"name" : str,
                                  "conv1d_layers" : list,
                                  "conv1d_parallelisation_factor" : list,
                                  "classification_layers" : list,
                                  "classification_parallelisation_factor" : list,
                                  "regression_layers" : list,
                                  "regression_parallelisation_factor" : list,
                                  "projection_dims": And(int, lambda s: s >= 1)},
                "quantization_config" : {'pt_output_quantization' : list},
                "training_config" : {"weight_method" : And(str, lambda s: s in  ["none", "ptref", "onlyclass"]),
                                     "validation_split" : And(float, lambda s: s > 0.0),
                                     "embedding_epochs" : And(int, lambda s: s >= 1),
                                     "finetuning_epochs" : And(int, lambda s: s >= 1),
                                     "batch_size" : And(int, lambda s: s >= 1),
                                     "learning_rate" : And(float, lambda s: s > 0.0),
                                     "embeddding_lr" : And(float, lambda s: s > 0.0),
                                     "loss_weights" : And(list, lambda s: len(s) == 2)},
                ## generic hls4ml configuration
                "firmware_config" : {"input_precision" : str,
                                    "class_precision" : str,
                                    "reg_precision": str,
                                    "clock_period" : And(float, lambda s: 0.0 < s <= 10),
                                    "fpga_part" : str,
                                    "project_name" : str}
            }
    )
    
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
        
        initialise_tensorflow(self.run_config['num_threads'])
        
        scope0 = QuantizerConfigScope(default_q_type='kbi',
                                      b0=7,
                                      overflow_mode='wrap',
                                      i0=0,
                                      fr=MonoL1(1.e-8),
                                      ir=MonoL1(1.e-8),
                                    )

        scope1 = QuantizerConfigScope(default_q_type='kif',
                                      place='datalane',
                                      overflow_mode='wrap',
                                      f0=7,
                                      fr=MonoL1(1.e-8),
                                      ic=MinMax(0, 12),
                                    )
        heterogeneous_axis = None

        scope2 = LayerConfigScope(enable_ebops=True, heterogeneous_axis=heterogeneous_axis,beta0=1e-8)
        
        with scope0, scope1, scope2:
        
                N_constituents = inputs_shape[0]
                n_features = inputs_shape[1]
            
                inputs = Input(shape=(N_constituents,n_features), name='model_input')
                main = QBatchNormalization(name='norm_input')(inputs)
                
                for iconv1d, depthconv1d in enumerate(self.model_config['conv1d_layers']):
                    main = QConv1D(filters=depthconv1d, parallelization_factor=self.model_config['conv1d_parallelisation_factor'][iconv1d], kernel_size=1, name='Conv1D_' + str(iconv1d + 1),activation='relu')(main)                
                main = GlobalAveragePooling1D(name='avgpool')(main)
                self.backbone_model = keras.Model(inputs=inputs, outputs=main)
                #jetID branch, 3 layer MLP
                
                for iclass, depthclass in enumerate(self.model_config['classification_layers']):
                    if iclass == 0:
                        jet_id = QDense(depthclass, parallelization_factor=self.model_config['classification_parallelisation_factor'][iclass], name='Dense_' + str(iclass + 1) + '_jetID',activation='relu')(main)
                    else:
                        jet_id = QDense(depthclass, parallelization_factor=self.model_config['classification_parallelisation_factor'][iclass], name='Dense_' + str(iclass + 1) + '_jetID',activation='relu')(jet_id)                
                jet_id = QDense(outputs_shape[0], parallelization_factor=outputs_shape[0], activation='relu')(jet_id)
                jet_id = Activation('softmax', name='jet_id_output')(jet_id)
                #pT regression branch
                for ireg, depthreg in enumerate(self.model_config['regression_layers']):
                    if ireg == 0:
                        pt_regress = QDense(depthreg, parallelization_factor=self.model_config['regression_parallelisation_factor'][ireg], name='Dense_' + str(ireg + 1) + '_pT',activation='relu')(main)
                    else:
                        pt_regress = QDense(depthreg, parallelization_factor=self.model_config['regression_parallelisation_factor'][ireg], name='Dense_' + str(ireg + 1) + '_pT',activation='relu')(pt_regress)      
                pt_regress = QDense(1,name='pT_output')(pt_regress)#1.1e-7
    
                self.create_encoder(inputs_shape,self.model_config['projection_dims'])
                #Define the model using both branches
                self.jet_model = keras.Model(inputs = inputs, outputs = [jet_id, pt_regress])

                print(self.embedding_model.summary())
        
                print(self.jet_model.summary())
        
    def create_encoder(self,inputs_shape, projection_dim):

        inputs = keras.Input(inputs_shape)
        features = self.backbone_model(inputs)

        # Projection head, the z's remember?
        outputs = keras.Sequential([
            keras.layers.Dense(10, activation='linear'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dense(10, activation='linear'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dense(10, activation='linear'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dense(projection_dim)
        ])(features)
        # Normalize to unit vectors so dot product equals cosine similarity (required for contrastive loss)
        outputs = L2NormalizeLayer()(outputs)
        self.embedding_model = keras.Model(inputs, outputs)
        
    def compile_model(self, num_samples: int):
        """compile the model generating callbacks and loss function
        Args:
            num_samples (int): Number of samples in the training set used for scheduling
        """
        scheduler = keras.callbacks.LearningRateScheduler(schedule = lambda epoch : cosine_decay_restarts(epoch, 
                                                                                                          initial_learning_rate=self.training_config['embeddding_lr'],
                                                                                                          max_epochs=self.training_config['embedding_epochs']))

        beta_scheduler = BetaScheduler(PieceWiseSchedule([(0, 0.2e-7, 'linear'), (self.training_config['embedding_epochs']/3, 3e-7, 'log'), (self.training_config['embedding_epochs'], 3e-6, 'constant')] ))
        # Define the callbacks using hyperparameters in the config
        self.callbacks = [
            FreeEBOPs(),
            scheduler,
            beta_scheduler
        ]
        
        ft_scheduler = keras.callbacks.LearningRateScheduler(schedule = lambda epoch : cosine_decay_restarts(epoch, 
                                                                                                          initial_learning_rate=self.training_config['learning_rate'],
                                                                                                          max_epochs=self.training_config['finetuning_epochs']))
        ft_beta_scheduler = BetaScheduler(PieceWiseSchedule([(0, 0.2e-7, 'linear'), (self.training_config['finetuning_epochs']/3, 3e-7, 'log'), (self.training_config['finetuning_epochs'], 3e-6, 'constant')] ))

        self.fine_tune_callbacks = [
            FreeEBOPs(),
            ft_scheduler,
            ft_beta_scheduler
        ]
        
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
        keras.config.disable_traceback_filtering()
        # Ensure backbone and embedding model are built
        input_shape = X_train.shape[1:]
        output_shape = (y_train.shape[-1],) if len(y_train.shape) > 1 else (1,)
        if self.backbone_model is None or self.jet_model is None:
            self.build_model(input_shape, output_shape)
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

        callbacks = tf.keras.callbacks.CallbackList(self.callbacks, add_history=True, model=self.embedding_model)
        logs = {'ebops':0}
        callbacks.on_train_begin(logs=logs)

        @tf.function
        def train_step(x1, x2):
            # Record operations for automatic differentiation
            with tf.GradientTape() as tape:
                # Forward pass: compute embeddings for both augmented views
                z1 = self.embedding_model(x1, training=True)
                z2 = self.embedding_model(x2, training=True)
                # Compute SimCLR contrastive loss between the two views
                loss = contrastive_loss(z1, z2)
            # Compute gradients of loss w.r.t. model trainable weights
            grads = tape.gradient(loss, self.embedding_model.trainable_weights)
            # Apply gradients to update model weights using optimizer
            self.constrastive_optimizer.apply_gradients(zip(grads, self.embedding_model.trainable_weights))
            # Return the computed loss for logging
            return loss

        for epoch in range(self.training_config['embedding_epochs']):
            callbacks.on_epoch_begin(epoch, logs=logs)
            losses = []
            ibatch = 0
            for x1, x2 in train_ds:
                ibatch += 1
                callbacks.on_train_batch_begin(ibatch, logs=logs)
                loss = train_step(x1, x2)
                losses.append(loss.numpy())
                callbacks.on_train_batch_end(ibatch, logs=logs)
            logs['loss'] = np.mean(losses)
            callbacks.on_epoch_end(epoch, logs=logs)
            print(self.callbacks)
            print(f"Epoch {epoch+1}: Loss = {logs['loss']:.4f} : ebops = {logs['ebops']:.4f}")
        callbacks.on_train_end(logs=logs)

        # --- Finetuning (jet model) training ---
        # Convert to float32 for TensorFlow
        X_train = X_train.astype("float32")
        y_train = y_train.astype("float32")
        pt_target_train = pt_target_train.astype("float32")
        sample_weight = sample_weight.astype("float32")

        # Freeze layers 
        for i, layer in enumerate(self.jet_model.layers):
            if i in [0,1,2,3,4]:
                print(layer.name)
                self.jet_model.get_layer(layer.name).trainable = False
        
        history = self.jet_model.fit(
            {'model_input': X_train},
            [y_train,pt_target_train],
            sample_weight=[sample_weight, sample_weight],
            epochs=self.training_config['finetuning_epochs'],
            batch_size=self.training_config['batch_size'],
            verbose=self.run_config['verbose'],
            validation_split=self.training_config['validation_split'],
            callbacks= self.fine_tune_callbacks,
            shuffle=True,
        )

        self.history = history.history