import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam, Nadam
from tensorflow.keras.layers import BatchNormalization, Input, Activation, Dense, Conv1D, Add, RepeatVector
from tensorflow.keras.layers import Flatten, Reshape, GlobalAveragePooling1D, Concatenate, UpSampling1D, AveragePooling1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras import utils, regularizers
from tensorflow.keras.layers import Layer
from qkeras import *
import tensorflow.keras.layers as KL
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
# import tensorflow_addons as tfa
from tensorflow import math, matmul, reshape, shape, transpose, cast, float32
from tensorflow.keras.layers import Dense, Layer


class AAtt(keras.layers.Layer):
    def __init__(self, d_model = 16, nhead = 1, nbits = 8, **kwargs):
        super(AAtt, self).__init__(**kwargs)
        
        self.nbits = nbits
        if self.nbits == 1:
            qbits = 'binary(alpha=1)'
        elif self.nbits == 2:
            qbits = 'ternary(alpha=1)'
        else:
            qbits = 'quantized_bits({},0,alpha=1)'.format(nbits)

        # qact = 'quantized_relu({},0)'.format(nbits)

        dense_kwargs = dict(
            # kernel_initializer = tf.keras.initializers.glorot_normal(),
            # kernel_regularizer = regularizers.l2(0.0001),
            # bias_regularizer = regularizers.l2(0.0001),
            # kernel_constraint = tf.keras.constraints.max_norm(5),
            kernel_quantizer = qbits,
            bias_quantizer = qbits,
        )

        self.d_model = d_model
        self.n_head = nhead

        self.qD = QDense(self.d_model, **dense_kwargs)
        self.kD = QDense(self.d_model, **dense_kwargs)
        self.vD = QDense(self.d_model, **dense_kwargs)
        self.outD = QDense(self.d_model, **dense_kwargs)


    def get_config(self):
        base_config = super().get_config()
        config = {
            "d_model": (self.d_model),
            "nhead": (self.n_head),
            "nbits": (self.nbits),
            # "qbits": (self.qbits),
        }
        return {**base_config, **config}

    def call(self, input):
        input_shape = input.shape
        shape_ = (-1, input_shape[1], self.n_head, self.d_model//self.n_head)
        perm_ = (0, 2, 1, 3)
        
        q = self.qD(input)
        q = tf.reshape(q, shape = shape_)
        q = tf.transpose(q, perm = perm_)

        # k = self.qD(input)
        k = self.kD(input)
        k = tf.reshape(k, shape = shape_)
        k = tf.transpose(k, perm = perm_)

        # v = self.qD(input)
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

class AAttMLP(keras.layers.Layer):
    def __init__(self, d_model = 16, nhead = 1, nbits = 8, **kwargs):
        super(AAttMLP, self).__init__(**kwargs)
        
        self.nbits = nbits
        if self.nbits == 1:
            qbits = 'binary(alpha=1)'
        elif self.nbits == 2:
            qbits = 'ternary(alpha=1)'
        else:
            qbits = 'quantized_bits({},0,alpha=1)'.format(nbits)

        # qact = 'quantized_relu({},0)'.format(nbits)

        dense_kwargs = dict(
            # kernel_initializer = tf.keras.initializers.glorot_normal(),
            # kernel_regularizer = regularizers.l2(0.0001),
            # bias_regularizer = regularizers.l2(0.0001),
            # kernel_constraint = tf.keras.constraints.max_norm(5),
            kernel_quantizer = qbits,
            bias_quantizer = qbits,
        )

        self.d_model = d_model
        self.n_head = nhead

        self.qD = QDense(self.d_model, **dense_kwargs)
        self.kD = QDense(self.d_model, **dense_kwargs)
        self.vD = QDense(self.d_model, **dense_kwargs)
        self.outD = QDense(self.d_model, **dense_kwargs)


    def get_config(self):
        base_config = super().get_config()
        config = {
            "d_model": (self.d_model),
            "nhead": (self.n_head),
            "nbits": (self.nbits),
            # "qbits": (self.qbits),
        }
        return {**base_config, **config}

    def call(self, input):
        input_shape = input.shape
        # shape_ = (-1, input_shape[1], self.n_head, self.d_model//self.n_head)
        shape_ = (-1, input_shape[1], self.n_head)
        # perm_ = (0, 2, 1, 3)
        perm_ = (0, 2, 1)
        
        q = self.qD(input)
        q = tf.reshape(q, shape = shape_)
        q = tf.transpose(q, perm = perm_)

        # k = self.qD(input)
        k = self.kD(input)
        k = tf.reshape(k, shape = shape_)
        k = tf.transpose(k, perm = perm_)

        # v = self.qD(input)
        v = self.vD(input)
        v = tf.reshape(v, shape = shape_)
        v = tf.transpose(v, perm = perm_)

        a = tf.matmul(q, k, transpose_b=True)
        # a = tf.nn.softmax(a / q.shape[3]**0.5, axis = 3)
        a = tf.nn.softmax(a / q.shape[2]**0.5, axis = 2)

        out = tf.matmul(a, v)
        out = tf.transpose(out, perm = perm_)
        out = tf.reshape(out, shape = (-1, input_shape[1], self.d_model))
        out = self.outD(out)

        return out

def getDeepSetWAttention(nclasses, input_shape, nnodes_phi = 16, nnodes_rho = 16, nbits = 8, integ = 0, n_head = 1, dim = 8, dim2 = 16, addRegression = False):
    if nbits == 1:
        qbits = 'binary(alpha=1)'
    elif nbits == 2:
        qbits = 'ternary(alpha=1)'
    else:
        qbits = 'quantized_bits({},0,alpha=1)'.format(nbits)

    qact = 'quantized_relu({},0)'.format(nbits)

    #############################################################################
    # nnodes_phi = 16
    # nnodes_rho = 16
    # REGL = regularizers.l2(0.0001)
    # kernel_initializer_=tf.keras.initializers.glorot_normal()
    # kernel_constraint = tf.keras.constraints.max_norm(5)

    dense_kwargs = dict(
        # kernel_initializer = tf.keras.initializers.glorot_normal(),
        # kernel_regularizer = regularizers.l2(0.0001),
        # bias_regularizer = regularizers.l2(0.0001),
        # kernel_constraint = tf.keras.constraints.max_norm(5),
        kernel_quantizer = qbits,
        bias_quantizer = qbits,
    )

    ###################
    n_head = n_head  # Number of self-attention heads
    # d_k = dim  # Dimensionality of the linearly projected queries and keys
    # d_v = dim  # Dimensionality of the linearly projected values
    d_model = dim2  # Dimensionality of the model sub-layers' outputs

    inp = Input(shape=input_shape, name="inputs")
    # Input point features BatchNormalization 
    h = QBatchNormalization(name='qBatchnorm', beta_quantizer=qbits, gamma_quantizer=qbits)(inp)

    h = AAtt(d_model = d_model, nhead = n_head, nbits= nbits)(h)
    # h = AAtt(d_model = d_model, nhead = n_head, nbits= nbits)(h)
    h = QDense(nnodes_phi, name='qDense_phi1', **dense_kwargs)(h)
    phi_out = QActivation(qact, name='qActivation_phi3')(h)

    # Aggregate features (taking mean) over set elements  
    mean = GlobalAveragePooling1D(name='avgpool')(phi_out)      # return mean of features over elements
    # mean = GlobalAveragePooling2D(name='avgpool')(phi_out)      # return mean of features over elements

    # Rho MLP
    h = QDense(nnodes_rho, name='qDense_rho1', **dense_kwargs)(mean)
    h = QActivation(qact, name='qActivation_rho1')(h)
    # h = QDense(nnodes_rho, name='qDense_rho2', **dense_kwargs)(h)
    # h = QActivation(qact,name='qActivation_rho2')(h)
    # h = QDense(nclasses, name='qDense_rho3', **dense_kwargs)(h)
    # out = Activation('softmax',name='outputs_softmax')(h)

    h_out = QDense(nnodes_rho, name='qDense_rho3', **dense_kwargs)(h)
    h_out = QActivation(qact,name='qActivation_rho3')(h_out)
    h_out = QDense(nclasses, name='qDense_rho4', **dense_kwargs)(h_out)
    out = Activation('softmax', name='output_class')(h_out)

    if addRegression:
        h_reg = QDense(nnodes_rho, name='qDense_rho3_reg', **dense_kwargs)(h)
        h_reg = QActivation(qact,name='qActivation_rho3_reg')(h_reg)
        h_reg = QDense(1, name='qDense_rho4_reg', **dense_kwargs)(h_reg)
        out_reg = Activation('linear', name='output_reg')(h_reg)

    # Build the model
    
    if addRegression:
        model = Model(inputs=inp, outputs=[out, out_reg])
    else:
        model = Model(inputs=inp, outputs=out)

    # model = Model(inputs=inp, outputs=out)

    # Set NN and output name
    arch = 'QDeepSetsWithAttention_PermutationInv'
    fname = arch+'_nconst_'+str(input_shape[0])+"_nfeatures_"+str(input_shape[1])+'_nbits_'+str(nbits)

    custom_objects = {
        "AAtt": AAtt,
        "QDense": QDense,
        "QActivation": QActivation,
        "quantized_bits": quantized_bits,
        "ternary": ternary,
        "binary": binary,
        "QBatchNormalization": QBatchNormalization
        }
    print ("Model has custom objects:")
    print (custom_objects)

    return model, fname, custom_objects


def getMLPWAttention(nclasses, input_shape, nnodes_phi = 16, nnodes_rho = 16, nbits = 8, integ = 0, n_head = 1, dim = 8, dim2 = 16, addRegression = False, nLayers = 3, nFeatures = None):
    if nbits == 1:
        qbits = 'binary(alpha=1)'
    elif nbits == 2:
        qbits = 'ternary(alpha=1)'
    else:
        qbits = 'quantized_bits({},0,alpha=1)'.format(nbits)

    qact = 'quantized_relu({},0)'.format(nbits)

    #############################################################################
    # nnodes_phi = 16
    # nnodes_rho = 16
    # REGL = regularizers.l2(0.0001)
    # kernel_initializer_=tf.keras.initializers.glorot_normal()
    # kernel_constraint = tf.keras.constraints.max_norm(5)

    dense_kwargs = dict(
        # kernel_initializer = tf.keras.initializers.glorot_normal(),
        # kernel_regularizer = regularizers.l2(0.0001),
        # bias_regularizer = regularizers.l2(0.0001),
        # kernel_constraint = tf.keras.constraints.max_norm(5),
        kernel_quantizer = qbits,
        bias_quantizer = qbits,
    )

    ###################
    n_head = n_head  # Number of self-attention heads
    # d_k = dim  # Dimensionality of the linearly projected queries and keys
    # d_v = dim  # Dimensionality of the linearly projected values
    d_model = dim2  # Dimensionality of the model sub-layers' outputs

    # Instantiate Tensorflow input tensors in Batch mode 
    inp = Input(shape = input_shape, name = "inputs")

    # Input point features BatchNormalization 
    h = BatchNormalization(name='batchnorm')(inp)
    # h = QDense(nnodes_phi, name='qDense_phi1', **dense_kwargs)(h)
    for iLayer in range(1, nLayers):
        # Phi MLP ( permutation equivariant layers )
        h = AAttMLP(d_model = d_model, nhead = n_head, nbits= nbits)(h)
    h = QDense(nnodes_phi, name='qDense_phi'+str(nLayers), **dense_kwargs)(h)
    phi_out = QActivation(qact,name='qActivation_phi'+str(nLayers))(h)

    # Rho MLP
    h = QDense(nnodes_rho, name='qDense_rho1', **dense_kwargs)(phi_out)
    h = QActivation(qact,name='qActivation_rho1')(h)
    for iLayer in range(2,nLayers):
        h = QDense(nnodes_rho*2, name='qDense_rho'+str(iLayer), **dense_kwargs)(h)
        h = QActivation(qact,name='qActivation_rho'+str(iLayer))(h)

    h_out = QDense(nnodes_rho, name='qDense_rho'+str(nLayers)+'_class', **dense_kwargs)(h)
    h_out = QActivation(qact,name='qActivation_rho'+str(nLayers)+'_class')(h_out)
    h_out = QDense(nclasses, name='qDense_rho'+str(nLayers+1)+'_class', **dense_kwargs)(h_out)
    out = Activation('softmax', name='output_class')(h_out)

    if addRegression:
        h_reg = QDense(nnodes_rho, name='qDense_rho'+str(nLayers)+'_reg', **dense_kwargs)(h)
        h_reg = QActivation(qact,name='qActivation_rho'+str(nLayers)+'_reg')(h_reg)
        h_reg = QDense(1, name='qDense_rho4_reg', **dense_kwargs)(h_reg)
        out_reg = Activation('linear', name='output_reg')(h_reg)

    # Build the model
    
    if addRegression:
        model = Model(inputs=inp, outputs=[out, out_reg])
    else:
        model = Model(inputs=inp, outputs=out)


    # Set NN and output name
    arch = 'QMLPWithAttention'
    # fname = arch+'_nconst_'+str(input_shape[0])+"_nfeatures_"+str(input_shape[1])+'_nbits_'+str(nbits)
    fname = arch+'_nconst_'+str(int(input_shape/nFeatures))+"_nfeatures_"+str(nFeatures)+'_nbits_'+str(nbits)

    custom_objects = {
        "AAtt": AAtt,
        "AAttMLP": AAttMLP,
        "QDense": QDense,
        "QActivation": QActivation,
        "quantized_bits": quantized_bits,
        "ternary": ternary,
        "binary": binary,
        "QBatchNormalization": QBatchNormalization
        }
    print ("Model has custom objects:")
    print (custom_objects)

    return model, fname, custom_objects


def getDeepSet(nclasses, input_shape, nnodes_phi = 16, nnodes_rho = 16, nbits = 8, integ = 0, addRegression = False, nLayers = 3):

    # Define DeepSet Permutation Invariant Model

    # baseline keras model

    #########################################################################################################
    '''
    # Silence the info from tensorflow in which it brags that it can run on cpu nicely.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    keras.utils.set_random_seed(123)
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)

    from util.terminal_colors import tcols
    from . import util as dsutil

    tf.keras.backend.set_floatx("float64")

    util.util.device_info()
    outdir = util.util.make_output_directory("trained_deepsets", args["outdir"])
    util.util.save_hyperparameters_file(args, outdir)

    data = Data.shuffled(**args["data_hyperparams"])
    '''
    #########################################################################################################

    # Quantized bits
    #qbits = quantized_bits(nbits,integ,alpha=1.0)
    #qact = 'quantized_relu('+str(nbits)+',0)'

    # Set QKeras quantizer and activation 
    if nbits == 1:
        qbits = 'binary(alpha=1)'
    elif nbits == 2:
        qbits = 'ternary(alpha=1)'
    else:
        qbits = 'quantized_bits({},0,alpha=1)'.format(nbits)

    qbits = 'quantized_bits(9,2,alpha=1)'
    # qact = 'quantized_relu({},0)'.format(nbits)
    qact = 'quantized_relu({},0)'.format(9)

    # Print
    # print("Training with max # of contituents = ", nconstit)
    # print("Number of node features = ", nfeat)
    print("Quantization with nbits =",nbits)
    print("Quantization of integer part =",integ)

    #############################################################################
    REGL = regularizers.l2(0.0001)

    dense_kwargs = dict(
        # kernel_initializer = 'glorot_normal',
        # kernel_initializer = 'glorot_uniform',
        # kernel_regularizer = REGL,
        # bias_regularizer = REGL,
        # kernel_constraint = tf.keras.constraints.max_norm(5),
        kernel_quantizer = qbits,
        bias_quantizer = qbits,
    )
    dense_kwargs_out = dict(
        kernel_quantizer = 'quantized_bits(16,6,alpha=1)',
        bias_quantizer = 'quantized_bits(16,6,alpha=1)',
    )

    # Instantiate Tensorflow input tensors in Batch mode 
    inp = Input(shape = input_shape, name = "inputs")

    # Input feature BatchNormalization 
    h = BatchNormalization(name='batchnorm_inputs')(inp)
    for iLayer in range(1, nLayers):
        h = QDense(nnodes_phi, name='qDense_phi_'+str(iLayer), **dense_kwargs)(h)
        h = QActivation(qact,name='qActivation_phi_'+str(iLayer))(h)
    h = QDense(nnodes_phi, name='qDense_phi_'+str(nLayers), **dense_kwargs)(h)
    phi_out = QActivation(qact,name='qActivation_phi_'+str(nLayers))(h)
    
    # Linear activation to change HLS bitwidth to fix overflow in AveragePooling
    # phi_out = QActivation(activation='quantized_bits(15,7)', name = 'qActivationForPool')(phi_out)
    phi_out = QActivation(activation='quantized_bits(18,8)', name = 'qActivationForPool')(phi_out)

    # Aggregate features (taking mean) over set elements  
    mean = GlobalAveragePooling1D(name='avgpool')(phi_out)      # return mean of features over elements

    # Rho MLP
    # h = BatchNormalization(name='batchnorm_rho')(mean)
    h = QDense(nnodes_rho, name='qDense_rho_1', **dense_kwargs)(mean)
    h = QActivation(qact,name='qActivation_rho_1')(h)
    for iLayer in range(2,nLayers):
        h = QDense(nnodes_rho, name='qDense_rho_'+str(iLayer), **dense_kwargs)(h)
        h = QActivation(qact,name='qActivation_rho_'+str(iLayer))(h)

    h_out = QDense(nnodes_rho, name='qDense_'+str(nLayers)+'_class', **dense_kwargs)(h)
    h_out = QActivation(qact,name='qActivation_'+str(nLayers)+'_class')(h_out)
    # h_out = QDense(nnodes_rho, name='qDense_'+str(nLayers)+'_class_2', **dense_kwargs)(h_out)
    # h_out = QActivation(qact,name='qActivation_'+str(nLayers)+'_class_2')(h_out)
    h_out = QDense(nclasses, name='qDense_out_class', **dense_kwargs_out)(h_out)
    out = Activation('softmax', name='output_class')(h_out)

    if addRegression:
        h_reg = QDense(nnodes_rho, name='qDense_'+str(nLayers)+'_reg', **dense_kwargs)(h)
        h_reg = QActivation(qact,name='qActivation_'+str(nLayers)+'_reg')(h_reg)
        # h_reg = QDense(nnodes_rho, name='qDense_'+str(nLayers)+'_reg_2', **dense_kwargs)(h_reg)
        # h_reg = QActivation(qact,name='qActivation_'+str(nLayers)+'_reg_2')(h_reg)
        h_reg = QDense(1, name='qDense_out_reg', **dense_kwargs_out)(h_reg)
        out_reg = Activation('linear', name='output_reg')(h_reg)

    # Build the model
    
    if addRegression:
        model = Model(inputs=inp, outputs=[out, out_reg])
    else:
        model = Model(inputs=inp, outputs=out)

    # Set NN and output name
    arch = 'QDeepSets_PermutationInv'
    fname = arch+'_nconst_'+str(input_shape[0])+"_nfeatures_"+str(input_shape[1])+'_nbits_'+str(nbits)

    custom_objects = {
        "AAtt": AAtt,
        "QDense": QDense,
        "QActivation": QActivation,
        "quantized_bits": quantized_bits,
        "ternary": ternary,
        "binary": binary,
        "QBatchNormalization": QBatchNormalization
        }


    return model, fname, custom_objects

def getMLP(nclasses, input_shape, nnodes_phi = 16, nnodes_rho = 16, nbits = 8, integ = 0, addRegression = False, nLayers = 3, nFeatures = None, addGlobal = False, inputShapeGlobal = None):
# baseline keras model

    #########################################################################################################
    '''
    # Silence the info from tensorflow in which it brags that it can run on cpu nicely.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    keras.utils.set_random_seed(123)
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)

    from util.terminal_colors import tcols
    from . import util as dsutil

    tf.keras.backend.set_floatx("float64")

    util.util.device_info()
    outdir = util.util.make_output_directory("trained_deepsets", args["outdir"])
    util.util.save_hyperparameters_file(args, outdir)

    data = Data.shuffled(**args["data_hyperparams"])
    '''
    #########################################################################################################

    # Quantized bits
    #qbits = quantized_bits(nbits,integ,alpha=1.0)
    #qact = 'quantized_relu('+str(nbits)+',0)'

    # Set QKeras quantizer and activation 
    if nbits == 1:
        qbits = 'binary(alpha=1)'
    elif nbits == 2:
        qbits = 'ternary(alpha=1)'
    else:
        qbits = 'quantized_bits({},0,alpha=1)'.format(nbits)
    qbits = 'quantized_bits(9,2,alpha=1)'

    # qact = 'quantized_relu({},0)'.format(nbits)
    qact = 'quantized_relu({},0)'.format(9)

    # Print
    # print("Training with max # of contituents = ", nconstit)
    # print("Number of node features = ", nfeat)
    print("Quantization with nbits =",nbits)
    print("Quantization of integer part =",integ)

    #############################################################################
    # REGL = regularizers.l2(0.0001)
    # kernel_initializer_=tf.keras.initializers.glorot_normal()
    # kernel_constraint = tf.keras.constraints.max_norm(5)

    dense_kwargs = dict(
        # kernel_initializer = tf.keras.initializers.glorot_normal(),
        # kernel_initializer = 'glorot_normal',
        # kernel_regularizer = REGL,
        # bias_regularizer = REGL,
        # kernel_constraint = tf.keras.constraints.max_norm(5),
        kernel_quantizer = qbits,
        bias_quantizer = qbits,
        # dropout=0.1,
    )
    dense_kwargs_out = dict(
        # kernel_initializer = tf.keras.initializers.glorot_normal(),
        # kernel_initializer = 'glorot_normal',
        # kernel_regularizer = REGL,
        # bias_regularizer = REGL,
        # kernel_constraint = tf.keras.constraints.max_norm(5),
        kernel_quantizer = 'quantized_bits(16,6,alpha=1)',
        bias_quantizer = 'quantized_bits(16,6,alpha=1)',
        # dropout=0.1,
    )

    # Instantiate Tensorflow input tensors in Batch mode 
    inp = Input(shape = input_shape, name = "inputs")
    # if addGlobal:
    #     inp_global = Input(shape = inputShapeGlobal, name = "inputs_global")
    #     h_global = BatchNormalization(name='batchnorm')(inp_global)
    #     h_global = QDense(nnodes_phi, name='qDense_global_phi1', **dense_kwargs)(h_global)

    # Input point features BatchNormalization 
    # h = QBatchNormalization(name='qBatchnorm', beta_quantizer=qbits, gamma_quantizer=qbits)(inp)
    h = BatchNormalization(name='batchnorm')(inp)
    # Phi MLP ( permutation equivariant layers )
    h = QDense(nnodes_phi, name='qDense_phi1', **dense_kwargs)(h)
    # h = QDense(nnodes_phi, name='qDense_phi1', **dense_kwargs)(inp)
    # h = QDense(12, name='qDense_phi'+str(1), **dense_kwargs)(h)
    h = QActivation(qact,name='qActivation_phi'+str(1))(h)
    for iLayer in range(2, nLayers):
        # Phi MLP ( permutation equivariant layers )
        h = QDense(nnodes_phi, name='qDense_phi'+str(iLayer), **dense_kwargs)(h)
        h = QActivation(qact,name='qActivation_phi'+str(iLayer))(h)
    h = QDense(nnodes_phi, name='qDense_phi'+str(nLayers), **dense_kwargs)(h)
    phi_out = QActivation(qact,name='qActivation_phi'+str(nLayers))(h)
    # phi_out = QActivation(activation='quantized_bits(18,8)', name = 'qActivationForPool')(phi_out)

    # Rho MLP
    h = QDense(nnodes_rho, name='qDense_rho1', **dense_kwargs)(phi_out)
    h = QActivation(qact,name='qActivation_rho1')(h)
    for iLayer in range(2,nLayers):
        h = QDense(nnodes_rho, name='qDense_rho'+str(iLayer), **dense_kwargs)(h)
        h = QActivation(qact,name='qActivation_rho'+str(iLayer))(h)

    h_out = QDense(nnodes_rho, name='qDense_rho'+str(nLayers)+'_class', **dense_kwargs)(h)
    h_out = QActivation(qact,name='qActivation_rho'+str(nLayers)+'_class')(h_out)
    # h_out = QDense(nnodes_rho, name='qDense_rho'+str(nLayers)+'_class2', **dense_kwargs)(h_out)
    # h_out = QActivation(qact,name='qActivation_rho'+str(nLayers)+'_class2')(h_out)
    # h_out = QDense(nclasses, name='qDense_rho'+str(nLayers+1)+'_class', **dense_kwargs)(h_out)
    # h_out = QDense(nclasses, name='qDense_rho'+str(nLayers+1)+'_class', **dense_kwargs_out)(h_out)
    h_out = QDense(nclasses, name='qDense_out_class', **dense_kwargs_out)(h_out)
    out = Activation('softmax', name='output_class')(h_out)

    if addRegression:
        h_reg = QDense(nnodes_rho, name='qDense_rho'+str(nLayers)+'_reg', **dense_kwargs)(h)
        h_reg = QActivation(qact,name='qActivation_rho'+str(nLayers)+'_reg')(h_reg)
        # h_reg = QDense(nnodes_rho, name='qDense_rho'+str(nLayers)+'_reg2', **dense_kwargs)(h_reg)
        # h_reg = QActivation(qact,name='qActivation_rho'+str(nLayers)+'_reg2')(h_reg)
        # h_reg = QDense(1, name='qDense_rho'+str(nLayers+1)+'_reg', **dense_kwargs_out)(h_reg)
        h_reg = QDense(1, name='qDense_out_reg', **dense_kwargs_out)(h_reg)
        out_reg = Activation('linear', name='output_reg')(h_reg)

    # Build the model
    
    if addRegression:
        model = Model(inputs=inp, outputs=[out, out_reg])
    else:
        model = Model(inputs=inp, outputs=out)

    # Set NN and output name
    arch = 'QMLP'
    fname = arch+'_nconst_'+str(int(input_shape/nFeatures))+"_nfeatures_"+str(nFeatures)+'_nbits_'+str(nbits)

    custom_objects = {
        "AAtt": AAtt,
        "QDense": QDense,
        "QActivation": QActivation,
        "quantized_bits": quantized_bits,
        "ternary": ternary,
        "binary": binary,
        "QBatchNormalization": QBatchNormalization
        }


    return model, fname, custom_objects