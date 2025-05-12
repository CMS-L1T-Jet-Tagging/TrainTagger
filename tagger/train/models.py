"""
Here all the models are defined to be called in train.py
"""
import keras
from keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D,AveragePooling1D, Dense, Conv1D, Flatten
#HGQ
from hgq.layers import QDense, QConv1D ,QBatchNormalization
from hgq.layers.softmax import QSoftmax
from hgq.config import LayerConfigScope, QuantizerConfig, QuantizerConfigScope
from hgq.regularizers import MonoL1



def baseline(inputs_shape, output_shape):
    L, C = inputs_shape
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
    #main = Activation(activation='linear', name = 'act_pool')(main)
    #main = AveragePooling1D(L,name='avgpool')(main)
    main = GlobalAveragePooling1D(name='avgpool')(main)
    #main = Flatten()(main)

    #Now split into jet ID and pt regression

    #jetID branch, 3 layer MLP
    jet_id = Dense(32, name='Dense_1_jetID',activation='relu',kernel_initializer='lecun_uniform')(main)
    #jet_id = QActivation(activation=quantized_relu(bits), name='relu_1_jetID')(jet_id)

    jet_id = Dense(16, name='Dense_2_jetID',activation='relu',kernel_initializer='lecun_uniform')(jet_id)
    #jet_id = QActivation(activation=quantized_relu(bits), name='relu_2_jetID')(jet_id)

    jet_id = Dense(output_shape[0], name='Dense_3_jetID',activation='relu',kernel_initializer='lecun_uniform')(jet_id)
    jet_id = Activation('softmax', name='jet_id_output')(jet_id)
    #jet_id = Flatten( name='jet_id_output')(jet_id)
    #jet_id = QSoftmax('softmax', name='jet_id_output')(jet_id)

    #pT regression branch
    pt_regress = Dense(10, name='Dense_1_pT',activation='relu',kernel_initializer='lecun_uniform')(main)

    pt_regress = Dense(1, name='pT_output',
                        kernel_initializer='lecun_uniform')(pt_regress)
    #pt_regress = Flatten(name='pT_output')(pt_regress)

    #Define the model using both branches
    model = keras.Model(inputs = [inputs], outputs = [jet_id, pt_regress])

    print(model.summary())

    return model


# from HGQ.utils import get_default_paq_conf, get_default_kq_conf
# from HGQ.utils import set_default_paq_conf, set_default_kq_conf


# def deepset_HGQ(inputs_shape, output_shape):
#     paq_conf = get_default_paq_conf()
#     paq_conf['skip_dims'] = 'except_last' # or just 'all'
#     set_default_paq_conf(paq_conf)
        
#     L, C = inputs_shape
#     inputs = keras.layers.Input(shape=inputs_shape, name='model_input')

#     #Main branch
#     main = BatchNormalization(name='norm')(inputs)
#     main = HQuantize(name='quant1' ,beta=3e-5)(main) 
#     main = HConv1D(filters=10, activation='relu',kernel_size=1, beta=1.1e-5, parallel_factor=1, name='Conv1D_1')(main)
#     main = HConv1D(filters=10, activation='relu', kernel_size=1, beta=1.1e-5, parallel_factor=1, name='Conv1D_2')(main)
   
#     main = PAveragePooling1D(L, name='avgpool')(main)

#     #jetID branch, 3 layer MLP
    
#     jet_id = HDense(32, beta=1.1e-5, activation='relu',parallel_factor=1,name='Dense_1_jetID')(main)
#     jet_id = HDense(16, name='Dense_2_jetID',beta=1.1e-5, activation='relu',parallel_factor=1)(jet_id)
     
#     jet_id = HDense(output_shape[0],beta=1.1e-5, name='Dense_3_jetID', activation='relu')(jet_id)

#     #pT regression branch
#     pt_regress = HDense(10, name='Dense_1_pT', parallel_factor=1,beta=1.1e-5, activation='relu')(main)

    
#     pt_regress = HDense(1, beta=1.1e-5,parallel_factor=1,name='pT_out')(pt_regress)
#     #Define the model using both branches
#     model = keras.Model(inputs = inputs, outputs = [jet_id, pt_regress])
#     return model

# loss_classification = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# loss_regression = keras.losses.MeanSquaredError()