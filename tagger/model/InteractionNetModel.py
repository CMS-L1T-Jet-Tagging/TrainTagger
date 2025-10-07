"""Interaction net model child class

Written 28/05/2025 cebrown@cern.ch
"""

import itertools
import json
import os

import hls4ml
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from schema import Schema, And, Use, Optional

from qkeras.quantizers import quantized_bits
from tagger.model.common import initialise_tensorflow
from tagger.model.JetTagModel import JetModelFactory, JetTagModel
from tagger.model.QKerasModel import QKerasModel, AAtt, AttentionPooling, choose_aggregator

from qkeras import QConv1D
from qkeras.utils import load_qmodel
from qkeras.qlayers import QActivation, QDense
from qkeras.quantizers import quantized_bits, quantized_relu
from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Layer

# Custom NodeEdgeProjection needed for the InteractionNet
class NodeEdgeProjection(Layer, tfmot.sparsity.keras.PrunableLayer):
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

    def get_prunable_weights(self):
        return []  # Empty as this layer learns nothing?

# Register the model in the factory with the string name corresponding to what is in the yaml config
@JetModelFactory.register('InteractionNetModel')
class InteractionNetModel(QKerasModel):
    """InteractionNetModel class

    Args:
        JetTagModel (_type_): Base class of a JetTagModel
    """
    schema = Schema(
            {
                "model": str,
                ## generic run config coniguration
                "run_config" : JetTagModel.run_schema,
                "model_config" : {"name" : str,
                                 "effects_layers" : list,
                                 "objects_layers" : list,
                                 "classification_layers" : list,
                                 "regression_layers" : list,
                                 "kernel_initializer" : str,
                                 "aggregator" : And(str, lambda s: s in  ["mean", "max", "attention"])},
                "quantization_config" : QKerasModel.quantization_schema,
                "training_config"     : QKerasModel.training_config_schema,
            }
    )

    def build_model(self, inputs_shape: tuple, outputs_shape: tuple):
        """Interaction network model from https://arxiv.org/abs/1612.00222.

        Args:
            inputs_shape (tuple): Shape of the input
            outputs_shape (tuple): Shape of the output

        Additional hyperparameters in the config
            effects_layers: List of number of nodes for each layer of the effects MLP.
            objects_layers: List of number of nodes for each layer of the objects MLP.
            classifier_layers: List of number of nodes for each layer of the classifier MLP.
            regression_layers: List of number of nodes for each layer of the regression MLP
            aggregator: String that specifies the type of aggregator to use after the obj net.
        """

        initialise_tensorflow(self.run_config['num_threads'])

        self.common_args = {
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

        receiver_matrix = NodeEdgeProjection(name="receiver_matrix", receiving=True, node_to_edge=True)(main)
        sender_matrix = NodeEdgeProjection(name="sender_matrix", receiving=False, node_to_edge=True)(main)

        input_effects = Concatenate(axis=-1, name="concat_eff")([receiver_matrix, sender_matrix])

        x = QConv1D(
            self.model_config['effects_layers'][0],
            kernel_size=1,
            name=f"effects_{1}",
            **self.common_args,
        )(input_effects)

        x = QActivation(
            activation=quantized_relu(
                self.quantization_config['quantizer_bits'], self.quantization_config['quantizer_bits_int']
            )
        )(x)

        for i, layer in enumerate(self.model_config['effects_layers'][1:]):
            x = QConv1D(
                layer,
                kernel_size=1,
                **self.common_args,
                name=f"effects_{i+2}",
            )(x)
            x = QActivation(
                activation=quantized_relu(
                    self.quantization_config['quantizer_bits'], self.quantization_config['quantizer_bits_int']
                )
            )(x)

        x = NodeEdgeProjection(name="prj_effects", receiving=True, node_to_edge=False)(x)

        input_objects = Concatenate(axis=-1, name="concat_obj")([inputs, x])

        # Objects network.
        x = QConv1D(
            self.model_config['objects_layers'][0],
            kernel_size=1,
            **self.common_args,
            name=f"objects_{1}",
        )(input_objects)
        x = QActivation(
            activation=quantized_relu(
                self.quantization_config['quantizer_bits'], self.quantization_config['quantizer_bits_int']
            )
        )(x)
        for i, layer in enumerate(self.model_config['objects_layers'][1:]):
            x = QConv1D(
                layer,
                kernel_size=1,
                **self.common_args,
                name=f"objects_{i+2}",
            )(x)
            x = QActivation(
                activation=quantized_relu(
                    self.quantization_config['quantizer_bits'], self.quantization_config['quantizer_bits_int']
                )
            )(x)

        # Linear activation to change HLS bitwidth to fix overflow in AveragePooling
        x = QActivation(activation='quantized_bits(18,8)', name='act_pool')(x)

        # Aggregator
        agg = choose_aggregator(choice=self.model_config['aggregator'], name="pool")
        main = agg(x)

        # Now split into jet ID and pt regression

        # Make fully connected dense layers for classification task
        for iclass, depthclass in enumerate(self.model_config['classification_layers']):
            if iclass == 0:
                jet_id = QDense(depthclass, name='Dense_' + str(iclass + 1) + '_jetID', **self.common_args)(main)
            else:
                jet_id = QDense(depthclass, name='Dense_' + str(iclass + 1) + '_jetID', **self.common_args)(jet_id)
            jet_id = QActivation(
                activation=quantized_relu(self.quantization_config['quantizer_bits'], 0),
                name='relu_' + str(iclass + 1) + '_jetID',
            )(jet_id)
            # ToDo: fix the bits_int part later, ie use the default not 0

        # Make output layer for classification task
        jet_id = QDense(outputs_shape[0], name='Dense_' + str(iclass + 2) + '_jetID', **self.common_args)(jet_id)
        jet_id = Activation('softmax', name='jet_id_output')(jet_id)

        # Make fully connected dense layers for pt regression task
        for ireg, depthreg in enumerate(self.model_config['regression_layers']):
            if ireg == 0:
                pt_regress = QDense(depthreg, name='Dense_' + str(ireg + 1) + '_pT', **self.common_args)(main)
            else:
                pt_regress = QDense(depthreg, name='Dense_' + str(ireg + 1) + '_pT', **self.common_args)(pt_regress)
            pt_regress = QActivation(
                activation=quantized_relu(self.quantization_config['quantizer_bits'], 0),
                name='relu_' + str(ireg + 1) + '_pT',
            )(pt_regress)

        pt_regress = QDense(
            1,
            name='pT_output_dense',
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

        pt_regress = QActivation(name = 'pT_output',
                                 activation= quantized_bits( self.quantization_config['pt_output_quantization'][0],
                                                 self.quantization_config['pt_output_quantization'][1],
                                                 alpha=self.quantization_config['quantizer_alpha_val'],
                                                ))(pt_regress)

        # Define the model using both branches
        self.jet_model = tf.keras.Model(inputs=inputs, outputs=[jet_id, pt_regress])

        print(self.jet_model.summary())

        return self.jet_model

    # Override load to allow node edge projection to also be loaded
    @JetTagModel.load_decorator
    def load(self, out_dir: str = "None"):
        """Load the model file

        Args:
            out_dir (str, optional): Where to load it if not in the output_directory. Defaults to "None".
        """

        # Additional custom objects for attention layers
        custom_objects_ = {
            "NodeEdgeProjection": NodeEdgeProjection,
            "AAtt": AAtt,
            "AttentionPooling": AttentionPooling,
        }

        # Load the model
        self.jet_model = load_qmodel(f"{out_dir}/model/saved_model.keras", custom_objects=custom_objects_)
