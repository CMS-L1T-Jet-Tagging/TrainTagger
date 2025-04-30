"""
Here all the models are defined to be called in train.py
"""
import keras
from keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D, Dense, Conv1D

#HGQ
from hgq.layers import QDense, QConv1D ,QBatchNormalization
from hgq.layers.softmax import QSoftmax
from hgq.config import LayerConfigScope, QuantizerConfig, QuantizerConfigScope
from hgq.regularizers import MonoL1



def baseline(inputs_shape, output_shape):
    # Skipping these should also work.
    # Usually, the default configs are good enough for most cases, but the initial number of bits, `[bif]0`
    # may need to be increased. If you see that the model is not converging, you can try increasing these values.
    scope0 = QuantizerConfigScope(place='all', k0=1, b0=3, i0=0, default_q_type='kbi', overflow_mode='sat_sym')
    scope1 = QuantizerConfigScope(place='datalane', k0=0, default_q_type='kif', overflow_mode='wrap', f0=3, i0=3)
    beta0=1e-5

    with scope0, scope1:
        iq_conf = QuantizerConfig(place='datalane', k0=1)
        oq_conf = QuantizerConfig(place='datalane', k0=1, fr=MonoL1(1e-3))
        #Initialize inputs
        inputs = keras.layers.Input(shape=inputs_shape, name='model_input')
        
        #Main branch
        main = BatchNormalization(name='norm_input')(inputs)
                    
        #First Conv1D
        main = Conv1D(filters=10, kernel_size=1, name='Conv1D_1',activation='relu',kernel_initializer='lecun_uniform')(main)
        #main = QActivation(activation=quantized_relu(bits), name='relu_1')(main)

        #Second Conv1D
        main = Conv1D(filters=10, kernel_size=1, name='Conv1D_2',activation='relu',kernel_initializer='lecun_uniform')(main)
        #main = QActivation(activation=quantized_relu(bits), name='relu_2')(main)

        # Linear activation to change HLS bitwidth to fix overflow in AveragePooling
        #main = Activation(activation='quantized_bits(18,8)', name = 'act_pool')(main)
        main = GlobalAveragePooling1D(name='avgpool')(main)

        #Now split into jet ID and pt regression

        #jetID branch, 3 layer MLP
        jet_id = Dense(32, name='Dense_1_jetID',activation='relu',kernel_initializer='lecun_uniform')(main)
        #jet_id = QActivation(activation=quantized_relu(bits), name='relu_1_jetID')(jet_id)

        jet_id = Dense(16, name='Dense_2_jetID',activation='relu',kernel_initializer='lecun_uniform')(jet_id)
        #jet_id = QActivation(activation=quantized_relu(bits), name='relu_2_jetID')(jet_id)

        jet_id = Dense(output_shape[0], name='Dense_3_jetID',activation='relu',kernel_initializer='lecun_uniform')(jet_id)
        jet_id = Activation('softmax', name='jet_id_output')(jet_id)
        #jet_id = QSoftmax('softmax', name='jet_id_output')(jet_id)

        #pT regression branch
        pt_regress = Dense(10, name='Dense_1_pT',activation='relu',kernel_initializer='lecun_uniform')(main)

        pt_regress = Dense(1, name='pT_output',
                            kernel_initializer='lecun_uniform')(pt_regress)

        #Define the model using both branches
        model = keras.Model(inputs = inputs, outputs = [jet_id, pt_regress])

        print(model.summary())

    return model