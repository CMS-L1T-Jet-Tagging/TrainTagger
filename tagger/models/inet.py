# Implementation of the permutation invariant Deep Sets network from the
# https://arxiv.org/abs/1703.06114 paper.

import numpy as np
import itertools

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as KL

#Third parties
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

import keras

import os

import json

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.pyplot import cm
import mplhep as hep
plt.style.use(hep.style.ROOT)

from sklearn.metrics import roc_curve, auc

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'medium',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)


class NodeEdgeProjection(KL.Layer):
    """Layer that build the adjacency matrix for the interaction network graph.

    Attributes:
        receiving: Whether we are building the receiver (True) or sender (False)
            adjency matrix.
        node_to_edge: Whether the projection happens from nodes to edges (True) or
            the edge matrix gets projected into the nodes (False).
    """

    def __init__(self, receiving: bool = True, node_to_edge: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._receiving = receiving
        self._node_to_edge = node_to_edge

    def build(self, input_shape: tuple):
        if self._node_to_edge:
            self._n_nodes = input_shape[-2]
            self._n_edges = self._n_nodes * (self._n_nodes - 1)
        else:
            self._n_edges = input_shape[-2]
            self._n_nodes = int((np.sqrt(4 * self._n_edges + 1) + 1) / 2)

        self._adjacency_matrix = self._assign_adjacency_matrix()

    def _assign_adjacency_matrix(self):
        receiver_sender_list = itertools.permutations(range(self._n_nodes), r=2)
        if self._node_to_edge:
            shape, adjacency_matrix = self._assign_node_to_edge(receiver_sender_list)
        else:
            shape, adjacency_matrix = self._assign_edge_to_node(receiver_sender_list)

        return tf.Variable(
            initial_value=adjacency_matrix,
            name="adjacency_matrix",
            dtype="float32",
            shape=shape,
            trainable=False,
        )

    def _assign_node_to_edge(self, receiver_sender_list: list):
        shape = (1, self._n_edges, self._n_nodes)
        adjacency_matrix = np.zeros(shape, dtype=float)
        for i, (r, s) in enumerate(receiver_sender_list):
            if self._receiving:
                adjacency_matrix[0, i, r] = 1
            else:
                adjacency_matrix[0, i, s] = 1

        return shape, adjacency_matrix

    def _assign_edge_to_node(self, receiver_sender_list: list):
        shape = (1, self._n_nodes, self._n_edges)
        adjacency_matrix = np.zeros(shape, dtype=float)
        for i, (r, s) in enumerate(receiver_sender_list):
            if self._receiving:
                adjacency_matrix[0, r, i] = 1
            else:
                adjacency_matrix[0, s, i] = 1

        return shape, adjacency_matrix

    def call(self, inputs):
        return tf.matmul(self._adjacency_matrix, inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "receiving": self._receiving,
                "node_to_edge": self._node_to_edge,
            }
        )
        return config

class IntNet:        
    def __init__(
        self,
        inputs_shape: tuple,
        output_shape: tuple,
        nparts = 16,
        effects_layers: list = [32, 32, 32],
        objects_layers: list = [16],
        classifier_layers: list = [32, 32],
        regression_layers: list = [32, 32],
        activ: str = "relu",
        aggreg: str = "mean",
    ):
        self.inputs_shape = inputs_shape
        self.output_shape = output_shape
        self.number_edges = inputs_shape[1] * (inputs_shape[1] - 1)
        self.effects_layers = effects_layers
        self.objects_layers = objects_layers
        self.classifier_layers = classifier_layers
        self.regression_layers = regression_layers

        self.aggreg = aggreg
        self.activ = activ

        self.classification_loss =  'binary_crossentropy'
        self.regression_loss =  'mean_absolute_error'
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        # GLOBAL PARAMETERS TO BE DEFINED WHEN TRAINING
        self.VALIDATION_SPLIT = 0.1
        self.BATCH_SIZE = 1024
        self.EPOCHS = 30

        self.GAMMA = 0.1

        self.I_SPARSITY = 0.0 #Initial sparsity
        self.F_SPARSITY = 0.6 #Final sparsity

    def build_model(self):
        self.classificationOutputLayerName = 'jet_id_output'
        self.regressionOutputLayerName = 'pT_output'

        self.receiver_matrix_proj = NodeEdgeProjection(
            name="receiver_matrix", receiving=True, node_to_edge=True
        )
        self.sender_matrix_proj = NodeEdgeProjection(
            name="sender_matrix", receiving=False, node_to_edge=True
        )
        self.effects_proj = NodeEdgeProjection(
            name="prj_effects", receiving=True, node_to_edge=False
        )

        self._build_effects_net()
        self._build_objects_net()
        self._build_agg()
        self._build_classifier()
        self._build_regressor()
        self.output_class_layer = KL.Dense(self.output_shape[0], name='jet_id_dense')
        self.output_regress_layer = KL.Dense(1, name= self.regressionOutputLayerName)

        inputs = tf.keras.layers.Input(shape=self.inputs_shape, name='model_input')
        batchnorm = tf.keras.layers.BatchNormalization(name='norm_input')(inputs)
        # Shape the input to a graph.
        receiver_matrix = self.receiver_matrix_proj(batchnorm)
        sender_matrix = self.sender_matrix_proj(batchnorm)
        input_effects = KL.Concatenate(axis=-1, name="concat_eff")(
            [receiver_matrix, sender_matrix]
        )

        # Compute the effects between the nodes.
        effects_output = self.effects(input_effects)
        effects_output = self.effects_proj(effects_output)

        # Compute the effects on the nodes.
        input_objects = KL.Concatenate(axis=-1, name="concat_obj")(
            [inputs, effects_output]
        )
        objects_output = self.objects(input_objects)

        # Aggregate the output and classify the graph.
        aggreg_output = self.agg(objects_output, axis=1)
        classifier_output = self.classifier(aggreg_output)
        logits = self.output_class_layer(classifier_output)
        probabilities = KL.Activation('softmax', name='jet_id_output')(logits)

        regression_output = self.regressor(aggreg_output)
        pt_regress = self.output_regress_layer(regression_output)

        #Define the model using both branches
        self.model = tf.keras.Model(inputs = inputs, outputs = [probabilities, pt_regress])

        return self.model

    def _build_effects_net(self):
        input_shape = [self.inputs_shape[1], self.inputs_shape[-1] * 2]
        self.effects = keras.Sequential(name="EffectsNetwork")

        for layer in self.effects_layers:
            self.effects.add(KL.Conv1D(layer, kernel_size=1))
            input_shape[-1] = layer

            self.effects.add(KL.Activation(self.activ))

    def _build_objects_net(self):
        input_shape = [
            self.inputs_shape[1],
            self.inputs_shape[-1] + self.effects_layers[-1],
        ]
        self.objects = keras.Sequential(name="ObjectsNetwork")

        for layer in self.objects_layers:
            self.objects.add(KL.Conv1D(layer, kernel_size=1))
            input_shape[-1] = layer

            self.objects.add(KL.Activation(self.activ))

    def _build_agg(self):
        switcher = {
            "mean": lambda: self._get_mean_aggregator(),
            "max": lambda: self._get_max_aggregator(),
        }
        self.agg = switcher.get(self.aggreg, lambda: None)()
        if self.agg is None:
            raise ValueError(
                "Given aggregation string is not implemented. "
                "See deepsets.py and add string and corresponding object there."
            )

    def _get_mean_aggregator(self):
        """Get mean aggregator object and calculate number of flops."""
        # Sum number of inputs into the aggregator + 1 division times number of feats.
        return tf.reduce_mean

    def _get_max_aggregator(self):
        """Get max aggregator object and calculate number of flops."""
        # FLOPs calculation WIP.

        return tf.reduce_max

    def _build_classifier(self):
        input_shape = self.objects_layers[-1]
        self.classifier = keras.Sequential(name="ClassifierNetwork")
        for layer in self.classifier_layers:
            self.classifier.add(KL.Dense(layer))
            input_shape = layer

            self.classifier.add(KL.Activation(self.activ))

    def _build_regressor(self):
        input_shape = self.objects_layers[-1]
        self.regressor = keras.Sequential(name="RegressionNetwork")
        for layer in self.regression_layers:
            self.regressor.add(KL.Dense(layer))
            input_shape = layer

            self.regressor.add(KL.Activation(self.activ))

    def compile_model(self,num_samples,prune=False):

        self.build_model()

        self.callbacks = []
        
        if prune:
            self.model = self.prune_model(num_samples) 
            self.callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())
            self.classificationOutputLayerName = 'prune_low_magnitude_jet_id_output'
            self.regressionOutputLayerName = 'prune_low_magnitude_pT_output'

        self.model.compile(optimizer=self.optimizer,
                           loss={self.classificationOutputLayerName: self.classification_loss, self.regressionOutputLayerName: self.regression_loss},
                           loss_weights={self.classificationOutputLayerName: self.GAMMA, self.regressionOutputLayerName: 1 - self.GAMMA}, metrics=['accuracy'])

        self.callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=2, patience=5))


    def fit(self,X_train,y_train,pt_target_train):
        history = self.model.fit({'model_input': X_train},
                            {self.classificationOutputLayerName: y_train, self.regressionOutputLayerName: pt_target_train},
                            epochs=self.EPOCHS, batch_size=self.BATCH_SIZE, verbose=2, validation_split=self.VALIDATION_SPLIT, callbacks = [self.callbacks])
        return history


    def prune_model(self, num_samples):
        """
        Pruning settings for the model. Return the pruned model
        """

        print("Begin pruning the model...")

        #Calculate the ending step for pruning
        end_step = np.ceil(num_samples / self.BATCH_SIZE).astype(np.int32) * self.EPOCHS

        #Define the pruned model
        pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=self.I_SPARSITY, final_sparsity=self.F_SPARSITY, begin_step=0, end_step=end_step)}
        self.pruned_model = tfmot.sparsity.keras.prune_low_magnitude(self.model, **pruning_params)

        return self.pruned_model

    def save_model(self,out_dir):
        export_path = os.path.join(out_dir, "model/")
        os.makedirs(export_path, exist_ok=True)
        self.model.save(export_path+'/saved_model.keras')
        print(f"Model saved to {export_path}")

    def load_model(self,in_dir):
        model = keras.models.load_model(f"{in_dir}/model/saved_model.keras", compile=False)

    def basic_ROC(self,model_dir):
        """
        Plot the basic ROCs for different classes. Does not reflect L1 rate
        """

        plot_dir = os.path.join(model_dir, "plots/training")

        with open(f"{model_dir}/class_label.json", 'r') as file: class_labels = json.load(file)

        #Load the testing data & model
        X_test = np.load(f"{model_dir}/testing_data/X_test.npy")
        y_test = np.load(f"{model_dir}/testing_data/y_test.npy")
            
        #Load the metada for class_label
        model_outputs = self.model.predict(X_test)
        #Get classification outputs
        y_pred = model_outputs[0]
        print(y_pred)

        # Create a colormap for unique colors
        colormap = cm.get_cmap('Set1', len(class_labels))  # Use 'tab10' with enough colors

        # Create a plot for ROC curves
        plt.figure(figsize=(16, 16))
        for i, class_label in enumerate(class_labels):

            # Get true labels and predicted probabilities for the current class
            y_true = y_test[:, i]  # Extract the one-hot column for the current class
            y_score = y_pred[:, i] # Predicted probabilities for the current class

            # Compute FPR, TPR, and AUC
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            # Plot the ROC curve for the current class
            plt.plot(tpr, fpr, label=f'{class_label} (AUC = {roc_auc:.2f})',
                    color=colormap(i), linewidth=5)

        # Plot formatting
        plt.grid(True)
        plt.ylabel('False Positive Rate')
        plt.xlabel('True Positive Rate')
        hep.cms.text("Phase 2 Simulation")
        hep.cms.lumitext("PU 200 (14 TeV)")
        plt.legend(loc='lower right')

        plt.yscale('log')
        plt.ylim([1e-3, 1.1])

        # Save the plot
        save_path = os.path.join(plot_dir, "basic_ROC")
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.close()

        return

    def basic_residual(self,model_dir):
        """
        Plot the basic residual for different classes. Does not reflect L1 rate
        """

        plot_dir = os.path.join(model_dir, "plots/training")

        with open(f"{model_dir}/class_label.json", 'r') as file: class_labels = json.load(file)

        #Load the testing data & model
        X_test = np.load(f"{model_dir}/testing_data/X_test.npy")
        y_test = np.load(f"{model_dir}/testing_data/pt_target_test.npy")
            
        #Load the metada for class_label
        model_outputs = self.model.predict(X_test)
        #Get classification outputs
        y_pred = model_outputs[1]
        print(y_pred)

        # Create a colormap for unique colors

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(18,15))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)

        ax.hist(y_test-y_pred[:,0],bins=50,range=(-1,1),histtype="step",
                        linewidth=5,
                        color = 'r',
                        label='IntNet Residual',
                        density=False) 
 
        ax.grid(True)
        ax.set_xlabel('pT regression residual (truth - predicted)',ha="right",x=1)
        ax.set_ylabel('# Jets',ha="right",y=1)
        ax.legend(loc='best')

        plt.tight_layout()

        # Save the plot
        save_path = os.path.join(plot_dir, "basic_residual")
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.close()

        return

    def basic_histo(self,model_dir):
        """
        Plot the basic residual for different classes. Does not reflect L1 rate
        """

        plot_dir = os.path.join(model_dir, "plots/training")

        with open(f"{model_dir}/class_label.json", 'r') as file: class_labels = json.load(file)

        #Load the testing data & model
        X_test = np.load(f"{model_dir}/testing_data/X_test.npy")
        y_test = np.load(f"{model_dir}/testing_data/pt_target_test.npy")
            
        #Load the metada for class_label
        model_outputs = self.model.predict(X_test)
        #Get classification outputs
        y_pred = model_outputs[1]

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(18,15))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)

        ax.hist(y_test,bins=50,range=(0,3),histtype="step",
                        linewidth=5,
                        color = 'k',
                        label='Truth Values',
                        density=True) 
        ax.hist(y_pred,bins=50,range=(0,3),histtype="step",
                        linewidth=5,
                        color = 'r',
                        label='IntNet Prediction',
                        density=True) 
  
        ax.grid(True)
        ax.set_xlabel('pT regression',ha="right",x=1)
        ax.set_ylabel('# Jets',ha="right",y=1)
        ax.legend(loc='best')

        plt.tight_layout()

        # Save the plot
        save_path = os.path.join(plot_dir, "basic_histo")
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.close()


    def __str__(self):
        self.model.summary(expand_nested=True)
        return("=======")