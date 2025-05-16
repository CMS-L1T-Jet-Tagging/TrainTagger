import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Input, Activation, AveragePooling1D, Dense, Conv1D, Flatten
import tensorflow_model_optimization as tfmot

from tagger.model.models import JetTagModel

import numpy as np

import os

class baselineFPModel(JetTagModel):
    def __init__(self,inputs_shape, outputs_shape,output_directory):
        super().__init__(inputs_shape, outputs_shape,output_directory)

    def build_model(self):

        L, C = self.inputs_shape
        #Initialize inputs
        inputs = tf.keras.layers.Input(shape=self.inputs_shape,  name='model_input')

        #Main branch
        main = BatchNormalization(name='norm_input')(inputs)
        
        #First Conv1D
        main = Conv1D(filters=10, kernel_size=1, name='Conv1D_1',kernel_initializer = 'lecun_uniform', activation='relu')(main)
        #Second Conv1D
        main = Conv1D(filters=10, kernel_size=1, name='Conv1D_2',kernel_initializer = 'lecun_uniform', activation='relu')(main)

        main = AveragePooling1D(L,name='avgpool')(main)
        main = Flatten()(main)

        #Now split into jet ID and pt regression

        #jetID branch, 3 layer MLP
        jet_id = Dense(32, name='Dense_1_jetID',kernel_initializer = 'lecun_uniform', activation='relu')(main)
        jet_id = Dense(16, name='Dense_2_jetID',kernel_initializer = 'lecun_uniform', activation='relu')(jet_id)
        jet_id = Dense(self.outputs_shape[0], name='Dense_3_jetID',kernel_initializer = 'lecun_uniform', activation='linear')(jet_id)
        jet_id = Activation('softmax', name=self.output_id_name )(jet_id)

        #pT regression branch
        pt_regress = Dense(10, name='Dense_1_pT',kernel_initializer = 'lecun_uniform', activation='relu')(main)
        pt_regress = Dense(1, name=self.output_pt_name, activation='linear',
                            kernel_initializer='lecun_uniform')(pt_regress)

        #Define the model using both branches
        self.model = tf.keras.Model(inputs = inputs, outputs = [jet_id, pt_regress])
        print(self.model.summary())

    def compile_model(self, num_samples):

        self.model.compile(optimizer='adam',
                            loss={self.loss_name+self.output_id_name: 'categorical_crossentropy', self.loss_name+self.output_pt_name: tf.keras.losses.Huber()},
                            metrics = {self.loss_name+self.output_id_name: 'categorical_accuracy', self.loss_name+self.output_pt_name: ['mae', 'mean_squared_error']},
                            weighted_metrics = {self.loss_name+self.output_id_name: 'categorical_accuracy', self.loss_name+self.output_pt_name: ['mae', 'mean_squared_error']})

    def fit(self,X_train,y_train,pt_target_train,sample_weight):
        history = self.model.fit({'model_input': X_train},
                            {self.loss_name+self.output_id_name: y_train, self.loss_name+self.output_pt_name: pt_target_train},
                            sample_weight=sample_weight,
                            epochs=self.hyperparameters['epochs'],
                            batch_size=self.hyperparameters['batch_size'],
                            verbose=2,
                            validation_split=self.hyperparameters['validation_split'],
                            callbacks = self.callbacks,
                            shuffle=True)

        return history

    def save(self,out_dir=None):  
        if out_dir is None:
          out_dir = self.output_directory
        #Export the model
        model_export = self.model

        export_path = os.path.join(out_dir, "model/saved_model.h5")
        model_export.save(export_path)
        print(f"Model saved to {export_path}")