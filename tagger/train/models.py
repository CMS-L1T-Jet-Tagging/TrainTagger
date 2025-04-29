"""
Here all the models are defined to be called in train.py
"""
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D
#HGQ
<<<<<<< HEAD
from HGQ.layers import HDense, HConv1D, PAveragePooling1D,  HQuantize, HActivation, PFlatten, HConv1DBatchNorm, Signature
=======
from HGQ.layers import HDense, HConv1D, PAveragePooling1D,  HQuantize, HActivation, PFlatten, HConv1DBatchNorm
>>>>>>> 10fb28f (HGQ Deep Set model added)
# Qkeras
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.qlayers import QDense, QActivation
from qkeras import QConv1D


def baseline(inputs_shape, output_shape, bits=9, bits_int=2, alpha_val=1):

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
    
    #First Conv1D
    main = QConv1D(filters=10, kernel_size=1, name='Conv1D_1', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_1')(main)

    #Second Conv1D
    main = QConv1D(filters=10, kernel_size=1, name='Conv1D_2', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_2')(main)

    # Linear activation to change HLS bitwidth to fix overflow in AveragePooling
    main = QActivation(activation='quantized_bits(18,8)', name = 'act_pool')(main)
    main = GlobalAveragePooling1D(name='avgpool')(main)

    #Now split into jet ID and pt regression

    #jetID branch, 3 layer MLP
    jet_id = QDense(32, name='Dense_1_jetID', **common_args)(main)
    jet_id = QActivation(activation=quantized_relu(bits), name='relu_1_jetID')(jet_id)

    jet_id = QDense(16, name='Dense_2_jetID', **common_args)(jet_id)
    jet_id = QActivation(activation=quantized_relu(bits), name='relu_2_jetID')(jet_id)

    jet_id = QDense(output_shape[0], name='Dense_3_jetID', **common_args)(jet_id)
    jet_id = Activation('softmax', name='jet_id_output')(jet_id)

    #pT regression branch
    pt_regress = QDense(10, name='Dense_1_pT', **common_args)(main)
    pt_regress = QActivation(activation=quantized_relu(bits), name='relu_1_pt')(pt_regress)

    pt_regress = QDense(1, name='pT_output',
                        kernel_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                        bias_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                        kernel_initializer='lecun_uniform')(pt_regress)

    #Define the model using both branches
    model = tf.keras.Model(inputs = inputs, outputs = [jet_id, pt_regress])

    print(model.summary())

    return model


def deepset_HGQ(inputs_shape, output_shape):
<<<<<<< HEAD
    
=======

>>>>>>> 10fb28f (HGQ Deep Set model added)
    inputs = tf.keras.layers.Input(shape=inputs_shape, name='model_input')
  
    #Main branch
    main = HQuantize(name='norm_input',beta=3e-5)(inputs)
    main = HConv1DBatchNorm(filters=10, kernel_size=1, beta=1.1e-5,parallel_factor=1,activation='relu',name='Conv1D_1')(main)
    
    main = HConv1D(filters=10, kernel_size=1,beta=1.1e-5,parallel_factor=1,activation='relu',name='Conv1D_2')(main)

    sequence_length = main.shape[1]
    main = PAveragePooling1D(pool_size=sequence_length, name="avgpool")(main)
<<<<<<< HEAD
    main = PFlatten()(main)
    
    #jetID branch, 3 layer MLP
    jet_id = HDense(32, beta=1.1e-5,parallel_factor=1,name='Dense_1_jetID')(main)
    jet_id = HActivation(activation='relu', beta=1.1e-5,name='relu_1_jetID')(jet_id)

    jet_id = HDense(16, name='Dense_2_jetID',beta=1.1e-5,parallel_factor=1)(jet_id)
    jet_id = HActivation(activation='relu',beta=1.1e-5, name='relu_2_jetID')(jet_id)

    
    jet_id = HDense(output_shape[0],beta=1.1e-5, name='Dense_3_jetID')(jet_id)
    jet_id = Activation('softmax', name='act_jet')(jet_id)
    jet_id = Signature(bits=18, int_bits=8, keep_negative=0, name='jet_id_output')(jet_id)


    #pT regression branch
    pt_regress = HDense(10, name='Dense_1_pT', parallel_factor=1,beta=1.1e-5)(main)
    pt_regress = HActivation(activation='relu',beta=1.1e-5, name='relu_1_pt')(pt_regress)

    
    pt_regress = HDense(1, beta=1.1e-5,parallel_factor=1,name='pT_out')(pt_regress)
    pt_regress = Signature(bits=16, int_bits=6, keep_negative=0, name='pT_output')(pt_regress)
=======
    main=PFlatten()(main)
    
    #jetID branch, 3 layer MLP
    jet_id = HDense(32, beta=1.1e-11,parallel_factor=1,name='Dense_1_jetID')(main)
    jet_id = HActivation(activation='relu', beta=1.1e-11,name='relu_1_jetID')(jet_id)

    jet_id = HDense(16, name='Dense_2_jetID',beta=1.1e-11,parallel_factor=1)(jet_id)
    jet_id = HActivation(activation='relu',beta=1.1e-11, name='relu_2_jetID')(jet_id)

    jet_id = HDense(output_shape[0],beta=1.1e-11,parallel_factor=1, name='Dense_3_jetID')(jet_id)
    jet_id = Activation('softmax', name='jet_id_output')(jet_id)

    #pT regression branch
    pt_regress = HDense(10, name='Dense_1_pT', parallel_factor=1,beta=1.1e-11)(main)
    pt_regress = HActivation(activation='relu',beta=1.1e-11, name='relu_1_pt')(pt_regress)

    pt_regress = HDense(1, beta=1.1e-5,parallel_factor=1,name='pT_output')(pt_regress)

>>>>>>> 10fb28f (HGQ Deep Set model added)
    #Define the model using both branches
    model = tf.keras.Model(inputs = inputs, outputs = [jet_id, pt_regress])

    return model
