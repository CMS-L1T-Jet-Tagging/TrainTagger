"""
Here all the models are defined to be called in train.py
"""
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate, Layer

# Qkeras
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.qlayers import QDense, QActivation
from qkeras import QConv1D
from qkeras.utils import load_qmodel
import itertools
import numpy as np


# Attention layers in case we want to try at some point
class AAtt(tf.keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer):
    def __init__(self, d_model = 16, nhead = 2, bits=9, bits_int=2, alpha_val=1, **kwargs):
        super(AAtt, self).__init__(**kwargs)

        self.d_model = d_model
        self.n_head = nhead
        self.bits = bits
        self.bits_int = bits_int
        self.alpha_val = alpha_val

        self.qD = QDense(self.d_model, **kwargs)
        self.kD = QDense(self.d_model, **kwargs)
        self.vD = QDense(self.d_model, **kwargs)
        self.outD = QDense(self.d_model, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "d_model": (self.d_model),
            "nhead": (self.n_head),
            "bits": (self.bits),
            "bits_int": (self.bits_int),
            "alpha_val": (self.alpha_val),
        }
        return {**base_config, **config}

    def call(self, input):
        input_shape = input.shape
        shape_ = (-1, input_shape[1], self.n_head, self.d_model//self.n_head)
        perm_ = (0, 2, 1, 3)
        
        q = self.qD(input)
        q = tf.reshape(q, shape = shape_)
        q = tf.transpose(q, perm = perm_)

        k = self.kD(input)
        k = tf.reshape(k, shape = shape_)
        k = tf.transpose(k, perm = perm_)

        v = self.vD(input)
        v = tf.reshape(v, shape = shape_)
        v = tf.transpose(v, perm = perm_)

        a = tf.matmul(q, k, transpose_b=True)
        a = tf.nn.softmax(a / q.shape[3]**0.5, axis = 3)

        out = tf.matmul(a, v)
        out = tf.transpose(out, perm = perm_)
        out = tf.reshape(out, shape = (-1, input_shape[1], self.d_model))
        out = self.outD(out)

        return out
    
    # define all prunable weights
    def get_prunable_weights(self):
        return self.qD._trainable_weights + self.kD._trainable_weights + \
        self.vD._trainable_weights + self.outD._trainable_weights

class AttentionPooling(Layer, tfmot.sparsity.keras.PrunableLayer):
    def __init__(self, bits, bits_int, alpha_val, **kwargs):
        super().__init__(**kwargs)
        
        self.score_dense = QDense(1, use_bias=False, **kwargs)

    def call(self, x):  # (B, N, d) -> (B,d) pooling via simple softmax

        a = tf.squeeze(self.score_dense(x), axis=-1)
        a = tf.nn.softmax(a, axis=1)

        out = tf.matmul(a[:, tf.newaxis, :], x) 
        return tf.squeeze(out, axis=1) 

    # define all prunable weights
    def get_prunable_weights(self):
        return self.score_dense._trainable_weights


# Some helper functions used in the InteractionNet base model
def choose_aggregator(choice, name, bits=9, bits_int=2, alpha_val=1, **common_args):
    """Choose the aggregator keras object based on an input string."""
    if choice not in ["mean", "max", "attention"]:
        raise ValueError(
            "Given aggregation string is not implemented in choose_aggregator(). "
            "See models.py and add string and corresponding object there."
        )
    if choice == "mean":
        return GlobalAveragePooling1D(name = name)
    elif choice == "max":
        return GlobalMaxPooling1D(name = name)
    elif choice == "attention":
        return AttentionPooling(name = name, bits=bits, bits_int=bits_int, alpha_val=alpha_val, **common_args)

def format_quantiser(nbits: int, bits_int : int, alpha : float):
    """Format the quantisation of the ml floats in a QKeras way."""
    if nbits == 1:
        return "binary(alpha=1)"
    elif nbits == 2:
        return "ternary(alpha=1)"
    else:
        return f"quantized_bits({nbits}, {bits_int}, alpha={alpha})"

def format_qactivation(activation, nbits: int, bits_int : int, alpha : float) -> str:
    """Format the activation function strings in a QKeras friendly way."""
    return f"quantized_{activation}({nbits}, {bits_int}, alpha={alpha})"

# baseline DeepSet model
def baseline(inputs_shape, output_shape, bits=9, bits_int=2, alpha_val=1, 
            aggregator = "mean", conv1d_layers = [10, 10], class_layers = [32, 16], reg_layers = [10]):

    # Define a dictionary for common arguments
    common_args = {
        'kernel_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'bias_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'kernel_initializer': 'lecun_uniform',
    }

    #Initialize inputs
    inputs = tf.keras.layers.Input(shape=inputs_shape, name='model_input')

    #Main branch
    main = BatchNormalization(name='norm_input')(inputs)
    
    # Make Conv1D layers
    for iconv1d, depthconv1d in enumerate(conv1d_layers):
        main = QConv1D(filters=depthconv1d, kernel_size=1, name='Conv1D_'+str(iconv1d+1), **common_args)(main)
        main = QActivation(activation=quantized_relu(bits, 0), name='relu_'+str(iconv1d+1))(main)
        #ToDo: fix the bits_int part later, ie use the default not 0

    # Linear activation to change HLS bitwidth to fix overflow in AveragePooling
    main = QActivation(activation='quantized_bits(18,8)', name = 'act_pool')(main)
    agg = choose_aggregator(choice = aggregator, name = "pool")
    main = agg(main)

    #Now split into jet ID and pt regression

    # Make fully connected dense layers for classification task
    for iclass, depthclass in enumerate(class_layers):
        if iclass == 0:
            jet_id = QDense(depthclass, name='Dense_'+str(iclass+1)+'_jetID', **common_args)(main)
        else:
            jet_id = QDense(depthclass, name='Dense_'+str(iclass+1)+'_jetID', **common_args)(jet_id)
        jet_id = QActivation(activation=quantized_relu(bits, 0), name='relu_'+str(iclass+1)+'_jetID')(jet_id)
        #ToDo: fix the bits_int part later, ie use the default not 0

    # Make output layer for classification task
    jet_id = QDense(output_shape[0], name='Dense_'+str(len(class_layers)+1)+'_jetID', **common_args)(jet_id)
    jet_id = Activation('softmax', name='jet_id_output')(jet_id)

    ## Make fully connected dense layers for pt regression task
    for ireg, depthreg in enumerate(reg_layers):
        if ireg == 0:
            pt_regress = QDense(depthreg, name='Dense_'+str(ireg+1)+'_pT', **common_args)(main)
        else:
            pt_regress = QDense(depthreg, name='Dense_'+str(ireg+1)+'_pT', **common_args)(pt_regress)
        pt_regress = QActivation(activation=quantized_relu(bits, 0), name='relu_'+str(ireg+1)+'_pT')(pt_regress)

    pt_regress = QDense(1, name='pT_output',
                        kernel_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                        bias_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                        kernel_initializer='lecun_uniform')(pt_regress)

    #Define the model using both branches
    model = tf.keras.Model(inputs = inputs, outputs = [jet_id, pt_regress])

    print(model.summary())

    return model

def baseline_larger(inputs_shape, output_shape, bits=9, bits_int=2, alpha_val=1, aggregator = "mean"):
    return baseline(inputs_shape, output_shape, bits=9, bits_int=2, alpha_val=1, 
            aggregator = "mean",conv1d_layers = [30, 15, 10], class_layers = [32, 16, 8], reg_layers = [16, 8, 4])

# DeepSet model w/ attention pooling
def DeepSetAttPool(inputs_shape, output_shape, bits=9, bits_int=2, alpha_val=1, 
            aggregator = "mean",
            conv1d_layers = [10, 10],
            class_layers = [32, 16],
            reg_layers = [16, 8],
            ):

    # Define a dictionary for common arguments
    common_args = {
        'kernel_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'bias_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'kernel_initializer': 'lecun_uniform'
    }

    #Initialize inputs
    inputs = tf.keras.layers.Input(shape=inputs_shape, name='model_input')

    #Main branch
    main = BatchNormalization(name='norm_input')(inputs)
    
    # Make Conv1D layers
    for iconv1d, depthconv1d in enumerate(conv1d_layers):
        main = QConv1D(filters=depthconv1d, kernel_size=1, name='Conv1D_'+str(iconv1d+1), **common_args)(main)
        main = QActivation(activation=quantized_relu(bits, bits_int), name='relu_'+str(iconv1d+1))(main)

    # Linear activation to change HLS bitwidth to fix overflow in AveragePooling
    main = QActivation(activation='quantized_bits(18,8)', name = 'act_pool')(main)
    main = choose_aggregator(choice = "attention", name='pool', bits=bits, bits_int=bits_int, alpha_val=alpha_val, **common_args)(main)
    # main = AttentionPooling(name='avgpool', bits=bits, bits_int=bits_int, alpha_val=alpha_val, **common_args)(main)

    #Now split into jet ID and pt regression

    # Make fully connected dense layers for classification task
    for iclass, depthclass in enumerate(class_layers):
        if iclass == 0:
            jet_id = QDense(depthclass, name='Dense_'+str(iclass+1)+'_jetID', **common_args)(main)
        else:
            jet_id = QDense(depthclass, name='Dense_'+str(iclass+1)+'_jetID', **common_args)(jet_id)
        jet_id = QActivation(activation=quantized_relu(bits, bits_int), name='relu_'+str(iclass+1)+'_jetID')(jet_id)

    # Make output layer for classification task
    jet_id = QDense(output_shape[0], name='Dense_'+str(len(class_layers)+1)+'_jetID', **common_args)(jet_id)
    jet_id = Activation('softmax', name='jet_id_output')(jet_id)

    ## Make fully connected dense layers for pt regression task
    for ireg, depthreg in enumerate(reg_layers):
        if ireg == 0:
            pt_regress = QDense(depthreg, name='Dense_'+str(ireg+1)+'_pT', **common_args)(main)
        else:
            pt_regress = QDense(depthreg, name='Dense_'+str(ireg+1)+'_pT', **common_args)(pt_regress)
        pt_regress = QActivation(activation=quantized_relu(bits, bits_int), name='relu_'+str(ireg+1)+'_pT')(pt_regress)

    pt_regress = QDense(1, name='pT_output',
                        kernel_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                        bias_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                        kernel_initializer='lecun_uniform')(pt_regress)

    #Define the model using both branches
    model = tf.keras.Model(inputs = inputs, outputs = [jet_id, pt_regress])

    print(model.summary())

    return model

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
        return [] # Empty as this layer learns nothing?

def interaction_net_base(
        inputs_shape,
        output_shape,
        effects_layers: list = [16, 16, 16],
        objects_layers: list = [16],
        aggreg: str = "mean",
        bits: int = 9,
        bits_int : int = 2,
        alpha_val : float = 1,
    ):

    """Interaction network model from https://arxiv.org/abs/1612.00222.

    Attributes:
        input_size: Tuple with the shape of the input data.
        effects_layers: List of number of nodes for each layer of the effects MLP.
        objects_layers: List of number of nodes for each layer of the objects MLP.
        classifier_layers: List of number of nodes for each layer of the classifier MLP.
        activ: String that specifies Activation function to use between the dense layers.
        aggreg: String that specifies the type of aggregator to use after the obj net.
        output_dim: The output dimension of the network. For a supervised task, this is
            equal to the number of classes, which in our case is 5.
        nbits: Number of bits that the model is quantised to.
    """

    # Define a dictionary for common arguments
    common_args = {
        'kernel_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'bias_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'kernel_initializer': 'lecun_uniform'
    }

    #Initialize inputs
    inputs = tf.keras.layers.Input(shape=inputs_shape, name='model_input')
        
    #Main branch
    main = BatchNormalization(name='norm_input')(inputs)

    receiver_matrix = NodeEdgeProjection(
        name="receiver_matrix", receiving=True, node_to_edge=True
    )(main)
    sender_matrix = NodeEdgeProjection(
        name="sender_matrix", receiving=False, node_to_edge=True
    )(main)

    input_effects = Concatenate(axis=-1, name="concat_eff")(
        [receiver_matrix, sender_matrix]
    )

    x = QConv1D(
        effects_layers[0],
        kernel_size=1,
        name=f"effects_{1}",
        **common_args,
    )(input_effects)

    x = QActivation(activation=quantized_relu(bits, bits_int))(x)
    
    for i, layer in enumerate(effects_layers[1:]):
        x = QConv1D(
            layer,
            kernel_size=1,
            **common_args,
            name=f"effects_{i+2}",
        )(x)
        x = QActivation(activation=quantized_relu(bits, bits_int))(x)

    x = NodeEdgeProjection(
            name="prj_effects", receiving=True, node_to_edge=False
        )(x)
    
    input_objects = Concatenate(axis=-1, name="concat_obj")([inputs, x])

    # Objects network.
    x = QConv1D(
        objects_layers[0],
        kernel_size=1,
        **common_args,
        name=f"objects_{1}",
    )(input_objects)
    x = QActivation(activation=quantized_relu(bits, bits_int))(x)
    for i, layer in enumerate(objects_layers[1:]):
        x = QConv1D(
            layer,
            kernel_size=1,
            **common_args,
            name=f"objects_{i+2}",
        )(x)
        x = QActivation(activation=quantized_relu(bits, bits_int))(x)

    # Linear activation to change HLS bitwidth to fix overflow in AveragePooling
    x = QActivation(activation='quantized_bits(18,8)', name = 'act_pool')(x)

    # Aggregator
    agg = choose_aggregator(choice = aggreg, name = "pool")
    main = agg(x)
    
    #Now split into jet ID and pt regression

    #jetID branch, 3 layer MLP
    jet_id = QDense(32, name='Dense_1_jetID', **common_args)(main)
    jet_id = QActivation(activation=quantized_relu(bits, bits_int), name='relu_1_jetID')(jet_id)

    jet_id = QDense(16, name='Dense_2_jetID', **common_args)(jet_id)
    jet_id = QActivation(activation=quantized_relu(bits, bits_int), name='relu_2_jetID')(jet_id)

    jet_id = QDense(output_shape[0], name='Dense_3_jetID', **common_args)(jet_id)
    jet_id = Activation('softmax', name='jet_id_output')(jet_id)

    #pT regression branch, 3 layer MLP
    pt_regress = QDense(16, name='Dense_1_pT', **common_args)(main)
    pt_regress = QActivation(activation=quantized_relu(bits, bits_int), name='relu_1_pT')(pt_regress)
    pt_regress = QDense(8, name='Dense_2_pT', **common_args)(pt_regress)
    pt_regress = QActivation(activation=quantized_relu(bits, bits_int), name='relu_2_pT')(pt_regress)

    pt_regress = QDense(1, name='pT_output',
                        kernel_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                        bias_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                        kernel_initializer='lecun_uniform')(pt_regress)

    #Define the model using both branches
    model = tf.keras.Model(inputs = inputs, outputs = [jet_id, pt_regress])

    print(model.summary())

    return model

# define a function callable like the baseline model
def interaction_net(inputs_shape, output_shape, bits=9, bits_int=2, alpha_val=1):

    return interaction_net_base(
        inputs_shape = inputs_shape,
        output_shape = output_shape,
        effects_layers = [10, 10, 10], objects_layers = [10],
        bits = bits, bits_int = bits_int, alpha_val = alpha_val
    )