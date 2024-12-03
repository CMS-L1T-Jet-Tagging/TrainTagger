"""
Baseline Model class
"""
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D

# Qkeras
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.qlayers import QDense, QActivation
from qkeras import QConv1D

#Third parties
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

import os

import json

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.pyplot import cm
import mplhep as hep
plt.style.use(hep.style.ROOT)

from sklearn.metrics import roc_curve, auc

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'medium',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)

class Baseline:
    def __init__(self,inputs_shape, output_shape,bits=9, bits_int=2, alpha_val=1):
        self.inputs_shape = inputs_shape
        self.output_shape = output_shape
        self.bits = bits
        self.bits_int = bits_int
        self.alpha = alpha_val

        self.classification_loss =  'binary_crossentropy'
        self.regression_loss = 'mean_absolute_error'
        self.optimizer = 'adam'

        # GLOBAL PARAMETERS TO BE DEFINED WHEN TRAINING
        self.VALIDATION_SPLIT = 0.1
        self.BATCH_SIZE = 1024
        self.EPOCHS = 100
        # Sparsity parameters
        self.I_SPARSITY = 0.0 #Initial sparsity
        self.F_SPARSITY = 0.6 #Final sparsity

        # Loss function parameters
        self.GAMMA = 0.1 #Loss weight for classification, 1-GAMMA for pt regression

        # Define a dictionary for common arguments
        self.common_args = common_args = {'kernel_quantizer': quantized_bits(self.bits, self.bits_int, alpha=self.alpha),
                                          'bias_quantizer': quantized_bits(self.bits, self.bits_int, alpha=self.alpha),
                                          'kernel_initializer': 'lecun_uniform' }

    def build_model(self):
        self.classificationOutputLayerName = 'jet_id_output'
        self.regressionOutputLayerName = 'pT_output'

        #Initialize inputs
        inputs = tf.keras.layers.Input(shape=self.inputs_shape, name='model_input')

        #Main branch
        main = BatchNormalization(name='norm_input')(inputs)
        
        #First Conv1D
        main = QConv1D(filters=10, kernel_size=1, name='Conv1D_1', **self.common_args)(main)
        main = QActivation(activation=quantized_relu(self.bits), name='relu_1')(main)

        #Second Conv1D
        main = QConv1D(filters=10, kernel_size=2, strides=2, name='Conv1D_2', **self.common_args)(main)
        main = QActivation(activation=quantized_relu(self.bits), name='relu_2')(main)

        # Linear activation to change HLS bitwidth to fix overflow in AveragePooling
        main = QActivation(activation='quantized_bits(18,8)', name = 'act_pool')(main)
        main = GlobalAveragePooling1D(name='avgpool')(main)

        #Now split into jet ID and pt regression

        #jetID branch, 3 layer MLP
        jet_id = QDense(32, name='Dense_1_jetID', **self.common_args)(main)
        jet_id = QActivation(activation=quantized_relu(self.bits), name='relu_1_jetID')(jet_id)

        jet_id = QDense(16, name='Dense_2_jetID', **self.common_args)(jet_id)
        jet_id = QActivation(activation=quantized_relu(self.bits), name='relu_2_jetID')(jet_id)

        jet_id = QDense(self.output_shape[0], name='Dense_3_jetID', **self.common_args)(jet_id)
        jet_id = QActivation(activation='quantized_bits(18,8)', name='jet_id_preactivation_output')(jet_id)
        jet_id = Activation('softmax', name='jet_id_output')(jet_id)

        #pT regression branch
        pt_regress = QDense(10, name='Dense_1_pT', **self.common_args)(main)
        pt_regress = QActivation(activation=quantized_relu(self.bits), name='relu_1_pt')(pt_regress)
        
        pt_regress = QDense(1, name='pT_output',
                            kernel_quantizer=quantized_bits(16, 6, alpha=self.alpha),
                            bias_quantizer=quantized_bits(16, 6, alpha=self.alpha),
                            kernel_initializer='lecun_uniform')(pt_regress)

        #Define the model using both branches
        self.model = tf.keras.Model(inputs = inputs, outputs = [jet_id, pt_regress])

        return self.model

    def compile_model(self,num_samples,prune=True):

        self.build_model()

        self.callbacks = []
        
        if prune:
            self.model = self.prune_model(num_samples) 
            self.callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())
            self.classificationOutputLayerName = 'prune_low_magnitude_jet_id_output'
            self.regressionOutputLayerName = 'prune_low_magnitude_pT_output'

        self.model.compile(optimizer=self.optimizer,
                           loss={self.classificationOutputLayerName: self.classification_loss, self.regressionOutputLayerName: self.regression_loss},
                           loss_weights={self.classificationOutputLayerName: self.GAMMA, self.regressionOutputLayerName: 1 - self.GAMMA}, metrics=['accuracy'])

        self.callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=2, patience=5))


    def fit(self,X_train,y_train,pt_target_train):
        history = self.model.fit({'model_input': X_train},
                            {self.classificationOutputLayerName: y_train, self.regressionOutputLayerName: pt_target_train},
                            epochs=self.EPOCHS, batch_size=self.BATCH_SIZE, verbose=2, validation_split=self.VALIDATION_SPLIT, callbacks = [self.callbacks])
        return history


    def prune_model(self, num_samples):
        """
        Pruning settings for the model. Return the pruned model
        """

        print("Begin pruning the model...")

        #Calculate the ending step for pruning
        end_step = np.ceil(num_samples / self.BATCH_SIZE).astype(np.int32) * self.EPOCHS

        #Define the pruned model
        pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=self.I_SPARSITY, final_sparsity=self.F_SPARSITY, begin_step=0, end_step=end_step)}
        self.model = tfmot.sparsity.keras.prune_low_magnitude(self.model, **pruning_params)

        return self.model

    def save_model(self,out_dir):
        model_export = tfmot.sparsity.keras.strip_pruning(self.model)
        export_path = os.path.join(out_dir, "model/")
        os.makedirs(export_path, exist_ok=True)
        model_export.save(export_path+"/saved_model.keras")
        print(f"Model saved to {export_path}")

    def basic_ROC(self,model_dir):
        """
        Plot the basic ROCs for different classes. Does not reflect L1 rate
        """

        plot_dir = os.path.join(model_dir, "plots/training")

        with open(f"{model_dir}/class_label.json", 'r') as file: class_labels = json.load(file)

        #Load the testing data & model
        X_test = np.load(f"{model_dir}/testing_data/X_test.npy")
        y_test = np.load(f"{model_dir}/testing_data/y_test.npy")
            
        #Load the metada for class_label
        model_outputs = self.model.predict(X_test)
        #Get classification outputs
        y_pred = model_outputs[0]

        # Create a colormap for unique colors
        colormap = cm.get_cmap('Set1', len(class_labels))  # Use 'tab10' with enough colors

        # Create a plot for ROC curves
        plt.figure(figsize=(16, 16))
        for i, class_label in enumerate(class_labels):

            # Get true labels and predicted probabilities for the current class
            y_true = y_test[:, i]  # Extract the one-hot column for the current class
            y_score = y_pred[:, i] # Predicted probabilities for the current class

            # Compute FPR, TPR, and AUC
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            # Plot the ROC curve for the current class
            plt.plot(tpr, fpr, label=f'{class_label} (AUC = {roc_auc:.2f})',
                    color=colormap(i), linewidth=5)

        # Plot formatting
        plt.grid(True)
        plt.ylabel('False Positive Rate')
        plt.xlabel('True Positive Rate')
        hep.cms.text("Phase 2 Simulation")
        hep.cms.lumitext("PU 200 (14 TeV)")
        plt.legend(loc='lower right')

        plt.yscale('log')
        plt.ylim([1e-3, 1.1])

        # Save the plot
        save_path = os.path.join(plot_dir, "basic_ROC")
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.close()

        return

    def basic_residual(self,model_dir):
        """
        Plot the basic residual for different classes. Does not reflect L1 rate
        """

        plot_dir = os.path.join(model_dir, "plots/training")

        with open(f"{model_dir}/class_label.json", 'r') as file: class_labels = json.load(file)

        #Load the testing data & model
        X_test = np.load(f"{model_dir}/testing_data/X_test.npy")
        y_test = np.load(f"{model_dir}/testing_data/pt_target_test.npy")
            
        #Load the metada for class_label
        model_outputs = self.model.predict(X_test)
        #Get classification outputs
        y_pred = model_outputs[1]

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(18,15))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)

        ax.hist(y_test-y_pred[:,0],bins=50,range=(-1,1),histtype="step",
                        linewidth=5,
                        color = 'r',
                        label='DeepSet Residual',
                        density=True) 
  
        ax.grid(True)
        ax.set_xlabel('pT regression residual (truth - predicted)',ha="right",x=1)
        ax.set_ylabel('# Jets',ha="right",y=1)
        ax.legend(loc='best')

        plt.tight_layout()

        # Save the plot
        save_path = os.path.join(plot_dir, "basic_residual")
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.close()

        return

    def basic_histo(self,model_dir):
        """
        Plot the basic residual for different classes. Does not reflect L1 rate
        """

        plot_dir = os.path.join(model_dir, "plots/training")

        with open(f"{model_dir}/class_label.json", 'r') as file: class_labels = json.load(file)

        #Load the testing data & model
        X_test = np.load(f"{model_dir}/testing_data/X_test.npy")
        y_test = np.load(f"{model_dir}/testing_data/pt_target_test.npy")
            
        #Load the metada for class_label
        model_outputs = self.model.predict(X_test)
        #Get classification outputs
        y_pred = model_outputs[1]

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(18,15))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)

        ax.hist(y_test,bins=50,range=(0,3),histtype="step",
                        linewidth=5,
                        color = 'k',
                        label='Truth Values',
                        density=True) 
        ax.hist(y_pred,bins=50,range=(0,3),histtype="step",
                        linewidth=5,
                        color = 'r',
                        label='DeepSet Prediction',
                        density=True) 
  
        ax.grid(True)
        ax.set_xlabel('pT regression',ha="right",x=1)
        ax.set_ylabel('# Jets',ha="right",y=1)
        ax.legend(loc='best')

        plt.tight_layout()

        # Save the plot
        save_path = os.path.join(plot_dir, "basic_histo")
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.close()

        return

    def __str__(self):
        self.model.summary()
        return("=======")

