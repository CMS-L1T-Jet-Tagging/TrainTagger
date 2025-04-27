"""
Here all the models are defined to be called in train.py
"""
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D

#HGQ
from hgq.layers import QDense, QConv1D 
from hgq.layers.softmax import QSoftmax

# Qkeras
from qkeras.quantizers import quantized_bits


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
    main = QConv1D(filters=10, kernel_size=1, name='Conv1D_1',activation='relu')(main)
    #main = QActivation(activation=quantized_relu(bits), name='relu_1')(main)

    #Second Conv1D
    main = QConv1D(filters=10, kernel_size=1, name='Conv1D_2',activation='relu')(main)
    #main = QActivation(activation=quantized_relu(bits), name='relu_2')(main)

    # Linear activation to change HLS bitwidth to fix overflow in AveragePooling
    #main = QActivation(activation='quantized_bits(18,8)', name = 'act_pool')(main)
    main = GlobalAveragePooling1D(name='avgpool')(main)

    #Now split into jet ID and pt regression

    #jetID branch, 3 layer MLP
    jet_id = QDense(32, name='Dense_1_jetID',activation='relu')(main)
    #jet_id = QActivation(activation=quantized_relu(bits), name='relu_1_jetID')(jet_id)

    jet_id = QDense(16, name='Dense_2_jetID',activation='relu')(jet_id)
    #jet_id = QActivation(activation=quantized_relu(bits), name='relu_2_jetID')(jet_id)

    jet_id = QDense(output_shape[0], name='Dense_3_jetID',activation='relu')(jet_id)
    jet_id = Activation('softmax', name='jet_id_output')(jet_id)
    #jet_id = QSoftmax('softmax', name='jet_id_output')(jet_id)

    #pT regression branch
    pt_regress = QDense(10, name='Dense_1_pT',activation='relu')(main)

    pt_regress = QDense(1, name='pT_output',
                       kernel_initializer='lecun_uniform')(pt_regress)

    #Define the model using both branches
    model = tf.keras.Model(inputs = inputs, outputs = [jet_id, pt_regress])

    print(model.summary())

    return model