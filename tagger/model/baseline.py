"""
Here all the models are defined to be called in train.py
"""
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D

# Qkeras
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.qlayers import QDense, QActivation
from qkeras import QConv1D

class baseline(JetTagModel):
    def __init__(self):
        super().__init__()

    def build_model(self,bits=9, bits_int=2, alpha_val=1):
        # Define a dictionary for common arguments
        common_args = {
            'kernel_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
            'bias_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
            'kernel_initializer': 'lecun_uniform'
        }

        #Initialize inputs
        inputs = tf.keras.layers.Input(shape=self.inputs_shape, name='model_input')

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

        jet_id = QDense(self.output_shape[0], name='Dense_3_jetID', **common_args)(jet_id)
        jet_id = Activation('softmax', name=self.output_id_name )(jet_id)

        #pT regression branch
        pt_regress = QDense(10, name='Dense_1_pT', **common_args)(main)
        pt_regress = QActivation(activation=quantized_relu(bits), name='relu_1_pt')(pt_regress)

        pt_regress = QDense(1, name=self.output_pt_name,
                            kernel_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                            bias_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                            kernel_initializer='lecun_uniform')(pt_regress)

        #Define the model using both branches
        self.model = tf.keras.Model(inputs = inputs, outputs = [jet_id, pt_regress])

    def _prune_model(model, num_samples):
        """
        Pruning settings for the model. Return the pruned model
        """

        print("Begin pruning the model...")

        #Calculate the ending step for pruning
        end_step = np.ceil(num_samples / self.hyperparameters['batch_size']).astype(np.int32) * self.hyperparameters['epochs']

        #Define the pruned model
        pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=self.hyperparameters['initial_sparsity'], final_sparsity=self.hyperparameters['final_sparsity'], begin_step=0, end_step=end_step)}
        self.model = tfmot.sparsity.keras.prune_low_magnitude(self.model, **pruning_params)

        self.loss_name = 'prune_low_magnitude_'

        self.callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())

    def compile_model(self, num_samples):

        self._prune_model(num_samples)

        self.model.compile(optimizer='adam',
                            loss={self.loss_name+self.output_id_name: 'categorical_crossentropy', self.loss_name+self.output_pt_name: tf.keras.losses.Huber()},
                            metrics = {self.loss_name+self.output_id_name: 'categorical_accuracy', self.loss_name+self.output_pt_name: ['mae', 'mean_squared_error']},
                            weighted_metrics = {self.loss_name+self.output_id_name: 'categorical_accuracy', self.loss_name+self.output_pt_name: ['mae', 'mean_squared_error']})

    def fit(self,X_train,y_train,sample_weight):
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