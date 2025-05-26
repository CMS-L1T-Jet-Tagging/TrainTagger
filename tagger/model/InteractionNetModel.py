import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate, Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow_model_optimization as tfmot

# Qkeras
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.qlayers import QDense, QActivation
from qkeras import QConv1D
from qkeras.utils import load_qmodel

from tagger.model.JetTagModel import JetTagModel, JetModelFactory
from tagger.model.common import choose_aggregator

import hls4ml

import itertools

import numpy as np

import os,json

num_threads = 24
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = str(num_threads)

tf.config.threading.set_inter_op_parallelism_threads(
    num_threads
)
tf.config.threading.set_intra_op_parallelism_threads(
    num_threads
)

# GLOBAL PARAMETERS TO BE DEFINED WHEN TRAINING
tf.keras.utils.set_random_seed(420) # not a special number


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


@JetModelFactory.register('InteractionNetModel')
class InteractionNetModel(JetTagModel):
    def __init__(self,output_dir):
        super().__init__(output_dir)

    def build_model(self,inputs_shape,outputs_shape):

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
                'kernel_quantizer': quantized_bits(self.quantization_config['quantizer_bits'], 
                                                self.quantization_config['quantizer_bits_int'], 
                                            alpha=self.quantization_config['quantizer_alpha_val']),
                'bias_quantizer':   quantized_bits(self.quantization_config['quantizer_bits'], 
                                                self.quantization_config['quantizer_bits_int'], 
                                            alpha=self.quantization_config['quantizer_alpha_val']),
                'kernel_initializer': self.model_config['kernel_initializer']
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
            self.model_config['effects_layers'][0],
            kernel_size=1,
            name=f"effects_{1}",
            **common_args,
        )(input_effects)

        x = QActivation(activation=quantized_relu(self.quantization_config['quantizer_bits'],
                            self.quantization_config['quantizer_bits_int']))(x)
        
        for i, layer in enumerate(self.model_config['effects_layers'][1:]):
            x = QConv1D(
                layer,
                kernel_size=1,
                **common_args,
                name=f"effects_{i+2}",
            )(x)
            x = QActivation(activation=quantized_relu(self.quantization_config['quantizer_bits'],
                            self.quantization_config['quantizer_bits_int']))(x)

        x = NodeEdgeProjection(
                name="prj_effects", receiving=True, node_to_edge=False
            )(x)
        
        input_objects = Concatenate(axis=-1, name="concat_obj")([inputs, x])

        # Objects network.
        x = QConv1D(
            self.model_config['objects_layers'][0],
            kernel_size=1,
            **common_args,
            name=f"objects_{1}",
        )(input_objects)
        x = QActivation(activation=quantized_relu(self.quantization_config['quantizer_bits'],
                            self.quantization_config['quantizer_bits_int']))(x)
        for i, layer in enumerate(self.model_config['objects_layers'][1:]):
            x = QConv1D(
                layer,
                kernel_size=1,
                **common_args,
                name=f"objects_{i+2}",
            )(x)
            x = QActivation(activation=quantized_relu(self.quantization_config['quantizer_bits'],
                            self.quantization_config['quantizer_bits_int']))(x)

        # Linear activation to change HLS bitwidth to fix overflow in AveragePooling
        x = QActivation(activation='quantized_bits(18,8)', name = 'act_pool')(x)

        # Aggregator
        agg = choose_aggregator(choice = self.model_config['aggregator'], name = "pool")
        main = agg(x)
        
    #Now split into jet ID and pt regression

        # Make fully connected dense layers for classification task
        for iclass, depthclass in enumerate(self.model_config['classification_layers']):
            if iclass == 0:
                jet_id = QDense(depthclass, name='Dense_'+str(iclass+1)+'_jetID', **common_args)(main)
            else:
                jet_id = QDense(depthclass, name='Dense_'+str(iclass+1)+'_jetID', **common_args)(jet_id)
            jet_id = QActivation(activation=quantized_relu(self.quantization_config['quantizer_bits'], 0), name='relu_'+str(iclass+1)+'_jetID')(jet_id)
                #ToDo: fix the bits_int part later, ie use the default not 0

        # Make output layer for classification task
        jet_id = QDense(outputs_shape[0], name='Dense_'+str(iclass+2)+'_jetID', **common_args)(jet_id)
        jet_id = Activation('softmax', name='jet_id_output')(jet_id)

        ## Make fully connected dense layers for pt regression task
        for ireg, depthreg in enumerate(self.model_config['regression_layers']):
            if ireg == 0:
                pt_regress = QDense(depthreg, name='Dense_'+str(ireg+1)+'_pT', **common_args)(main)
            else:
                pt_regress = QDense(depthreg, name='Dense_'+str(ireg+1)+'_pT', **common_args)(pt_regress)
            pt_regress = QActivation(activation=quantized_relu(self.quantization_config['quantizer_bits'], 0), name='relu_'+str(ireg+1)+'_pT')(pt_regress)

        pt_regress = QDense(1, name='pT_output',
                            kernel_quantizer=quantized_bits(self.quantization_config['pt_output_quantization'][0],
                                                                self.quantization_config['pt_output_quantization'][1], 
                                                                alpha=self.quantization_config['quantizer_alpha_val']),
                            bias_quantizer=quantized_bits(self.quantization_config['pt_output_quantization'][0],
                                                            self.quantization_config['pt_output_quantization'][1], 
                                                            alpha=self.quantization_config['quantizer_alpha_val']),
                            kernel_initializer='lecun_uniform')(pt_regress)

            #Define the model using both branches
        self.model = tf.keras.Model(inputs = inputs, outputs = [jet_id, pt_regress])

        print(self.model.summary())

        return self.model

    def _prune_model(self, num_samples):
        """
        Pruning settings for the model. Return the pruned model
        """

        print("Begin pruning the model...")

        #Calculate the ending step for pruning
        end_step = np.ceil(num_samples / self.training_config['batch_size']).astype(np.int32) * self.training_config['epochs']

        #Define the pruned model
        pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=self.training_config['initial_sparsity'], final_sparsity=self.training_config['final_sparsity'], begin_step=0, end_step=end_step)}
        self.model = tfmot.sparsity.keras.prune_low_magnitude(self.model, **pruning_params)

        self.loss_name = 'prune_low_magnitude_'

        self.callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())

    def compile_model(self, num_samples):

        self.callbacks = [EarlyStopping(monitor='val_loss', 
                                        patience=self.training_config['EarlyStopping_patience']),
                          ReduceLROnPlateau(monitor='val_loss', 
                                            factor=self.training_config['ReduceLROnPlateau_factor'], 
                                            patience=self.training_config['ReduceLROnPlateau_patience'], 
                                            min_lr=self.training_config['ReduceLROnPlateau_min_lr'],)]

        self._prune_model(num_samples)

        self.model.compile(optimizer='adam',
                           loss_weights=self.training_config['loss_weights'],
                            loss={self.loss_name+self.output_id_name: 'categorical_crossentropy', self.loss_name+self.output_pt_name: tf.keras.losses.Huber()},
                            metrics = {self.loss_name+self.output_id_name: 'categorical_accuracy', self.loss_name+self.output_pt_name: ['mae', 'mean_squared_error']},
                            weighted_metrics = {self.loss_name+self.output_id_name: 'categorical_accuracy', self.loss_name+self.output_pt_name: ['mae', 'mean_squared_error']})

    def fit(self,X_train,y_train,pt_target_train,sample_weight):
        self.history = self.model.fit({'model_input': X_train},
                            {self.loss_name+self.output_id_name: y_train, self.loss_name+self.output_pt_name: pt_target_train},
                            sample_weight=sample_weight,
                            epochs=self.training_config['epochs'],
                            batch_size=self.training_config['batch_size'],
                            verbose=self.run_config['verbose'],
                            validation_split=self.training_config['validation_split'],
                            callbacks = self.callbacks,
                            shuffle=True)
    
    @JetTagModel.save_decorator
    def save(self,out_dir):  
        #Export the model
        model_export = tfmot.sparsity.keras.strip_pruning(self.model)
        export_path = os.path.join(out_dir, "model/saved_model.h5")
        model_export.save(export_path)
        print(f"Model saved to {export_path}")

    @JetTagModel.load_decorator
    def load(self,out_dir=None):
        #Load model
        custom_objects_ = {
                "NodeEdgeProjection": NodeEdgeProjection,
        }
        self.model = load_qmodel(f"{out_dir}/model/saved_model.h5", custom_objects=custom_objects_)
       
       
    def hls4ml_convert(self,firmware_dir,build=False):

        #Remove the old directory if they exist
        hls4ml_outdir = self.output_directory + '/' + firmware_dir + '/' + self.hls4ml_config['project_name']
        os.system(f'rm -rf {hls4ml_outdir}')

        #Create default config
        config = hls4ml.utils.config_from_keras_model(self.model, granularity='name')
        config['IOType'] = 'io_parallel'
        config['LayerName']['model_input']['Precision']['result'] = self.hls4ml_config['input_precision']

        #Configuration for conv1d layers
        #hls4ml automatically figures out the paralellization factor
        #config['LayerName']['Conv1D_1']['ParallelizationFactor'] = 8
        #config['LayerName']['Conv1D_2']['ParallelizationFactor'] = 8

        #Additional config
        for layer in self.model.layers:
            layer_name = layer.__class__.__name__

            if layer_name in ["BatchNormalization", "InputLayer"]:
                config["LayerName"][layer.name]["Precision"] = self.hls4ml_config['input_precision']
                config["LayerName"][layer.name]["result"] = self.hls4ml_config['input_precision']
                config["LayerName"][layer.name]["Trace"] = not build

            elif layer_name in ["Permute","Concatenate","Flatten","Reshape","UpSampling1D","Add"]:
                print("Skipping trace for:", layer.name)
            else:
                config["LayerName"][layer.name]["Trace"]  = not build

        
        config["LayerName"]["jet_id_output"]["Precision"]["result"] =  self.hls4ml_config['class_precision']
        config["LayerName"]["jet_id_output"]["Implementation"] = "latency"
        config["LayerName"]["pT_output"]["Precision"]["result"] =  self.hls4ml_config['class_precision']
        config["LayerName"]["pT_output"]["Implementation"] = "latency"

        #Write HLS
        self.hls_model = hls4ml.converters.convert_from_keras_model(self.model,
                                                        backend='Vitis',
                                                        project_name=self.hls4ml_config['project_name'],
                                                        clock_period=2.8, #1/360MHz = 2.8ns
                                                        hls_config=config,
                                                        output_dir=f'{hls4ml_outdir}',
                                                        part='xcvu9p-flga2104-2L-e')
        

        # Compile and build the project
        self.hls_model.compile()

        # Save config  as json file
        print("Saving default config as config.json ...")
        with open(hls4ml_outdir+'/config.json', 'w') as fp: json.dump(config, fp)

        if build == True:
            self.hls_model.build(csim=False, reset = True)