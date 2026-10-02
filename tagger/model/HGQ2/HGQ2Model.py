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
from hgq.utils.sugar import FreeEBOPs, BetaScheduler, PieceWiseSchedule,EarlyStoppingWithEbopsThres,BetaPID
# Qkeras

from keras.models import load_model
#import hls4ml
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tagger.data.tools import load_data, to_ML
from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tagger.model.common import log_beta_schedule,cosine_decay_restarts
from tagger.model.common_tensorflow import initialise_tensorflow, huber_loss

class HGQ2Model(JetTagModel):

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
                
                "input_config" : {"basic_input_config": list,
                                  "basic_features": list,
                                  "jet_features": list},
                
                "training_config" :     {"weight_method": And(
                                        list,
                                        lambda lst: len(lst) == 2,
                                        lambda lst: all(x in ["none", "ptref", "onlyclass"] for x in lst)
                                            ),
                                         "validation_split" : And(float, lambda s: s > 0.0),
                                         "epochs" : And(int, lambda s: s >= 1),
                                         "batch_size" : And(int, lambda s: s >= 1),
                                         "learning_rate": And(float, lambda s: s > 0.0),
                                         "loss_weights" : And(list, lambda s: len(s) == 2),
                                         "target_ebops" : int,
                                         "pileup": And(bool),
                                         "huber_weights": And(list, lambda s: len(s) == 2),
                                         "huber_delta": And(float, lambda s: s > 0.0),
                                        },

                "firmware_config" : {"input_precision" : dict,
                                    "class_precision" : str,
                                    "reg_precision": str,
                                    "clock_period" : And(float, lambda s: 0.0 < s <= 10),
                                    "fpga_part" : str,
                                    "project_name" : str}
            }
    )

    # Redefine save and load for HGQ due to needing h5 format
    @JetTagModel.save_decorator
    def save(self, out_dir):
        # Export the model
        #model_export = tfmot.sparsity.keras.strip_pruning(self.jet_model)
        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        export_path = os.path.join(out_dir, "model/saved_model.keras")
        self.jet_model.save(export_path)
        print(f"Model saved to {export_path}")

    @JetTagModel.load_decorator
    def load(self, out_dir=None):
        # Load model
        self.jet_model = load_model(f"{out_dir}/model/saved_model.keras")

    def predict(self, X_test: npt.NDArray[np.float64]) -> tuple:
        model_outputs = self.jet_model.predict(X_test['basic_input'])
        class_predictions = scipy.special.softmax(model_outputs[0],axis=1)
        pt_ratio_predictions = model_outputs[1].flatten()
        return (class_predictions, pt_ratio_predictions)

    def compile_model(self, num_samples: int):

        """compile the model generating callbacks and loss function
        Args:
            num_samples (int): Number of samples in the training set used for scheduling
        """

        scheduler = keras.callbacks.LearningRateScheduler(schedule = lambda epoch : cosine_decay_restarts(epoch, 
                                                                                                          initial_learning_rate=self.training_config['learning_rate'],
                                                                                                          max_epochs=self.training_config['epochs']))

        es = EarlyStoppingWithEbopsThres(monitor="val_loss",
                                         patience=150,
                                         verbose=1,
                                         mode="min",
                                         restore_best_weights=True,
                                         start_from_epoch=75,
                                         ebops_threshold=self.training_config['target_ebops'] + 100000
                                        )
        terminate_on_nan = keras.callbacks.TerminateOnNaN()

        ebops_tracker = FreeEBOPs()
        ebops_scheduler = BetaPID(
            p=1, i=0.1, d=0,
            target_ebops=self.training_config['target_ebops'],
            init_beta=1e-10, warmup=10,
            max_beta=5e-6, damp_beta_on_target=0.5
        )
        # Define the callbacks using hyperparameters in the config
        self.callbacks = [
            scheduler,
            ebops_tracker,
            terminate_on_nan,
            ebops_scheduler,
            es

        ]
        # compile the tensorflow model setting the loss and metrics
        self.jet_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.training_config['learning_rate']),
            loss={
                self.loss_name + self.output_id_name: keras.losses.CategoricalCrossentropy(from_logits=True),
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
            }
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

        # Train the model using hyperparameters in yaml config
        history = self.jet_model.fit(
            {'basic_input': X_train},
            [y_train,pt_target_train],
            sample_weight = [sample_weight[0], sample_weight[1]],
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            verbose=self.run_config['verbose'],
            validation_split=self.training_config['validation_split'],
            callbacks=self.callbacks,
            shuffle=True,
        )

        self.history = history.history
        
        self.results_dict['best_loss'] = np.min(history.history['loss'])
        self.results_dict['best_val_loss'] = np.min(history.history['val_loss'])
        self.results_dict['best_jet_id_loss'] = np.min(history.history['jet_id_output_loss'])
        self.results_dict['best_pT_output_loss'] = np.min(history.history['pT_output_loss'])
        
        
    def get_keras_trace_model(self):
        # Create a sub-model that outputs all intermediate layers and get keras trace
        layer_outputs = [layer.output for layer in self.jet_model.layers]
        keras_trace_model = Model(inputs=self.jet_model.input, outputs=layer_outputs)
        return keras_trace_model

    def get_branch_model(self, output_name):
        output_tensor = self.jet_model.get_layer(output_name).output
        input_layers = self._get_branch_inputs(output_tensor)
        return keras.Model(input_layers, outputs=output_tensor)
    
    @staticmethod
    def _get_branch_inputs(output_tensor):
        # Keras 3: walk the graph via operation/node API
        visited = set()
        inputs = {}
        stack = [output_tensor]

        while stack:
            t = stack.pop()
            if id(t) in visited:
                continue
            visited.add(id(t))

            kh = t._keras_history          # still exists in Keras 3
            op = kh.operation              # NOTE: 'operation', not 'layer'
            node_index = kh.node_index

            if isinstance(op, keras.layers.InputLayer):
                inputs[op.name] = op.output
                continue

            node = op._inbound_nodes[node_index]
            stack.extend(node.arguments.keras_tensors)   # replaces node.input_tensors

        return list(inputs.values())
