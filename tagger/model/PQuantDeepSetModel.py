"""DeepSet model child class

Written 07/10/2025 cebrown@cern.ch
"""

import json
import os
import time

import hls4ml
import numpy as np
import numpy.typing as npt
from schema import Schema, And, Use, Optional

from tagger.model.JetTagModel import JetModelFactory, JetTagModel

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

from tagger.model.TorchDeepSetModel import JetTagDataset, TorchDeepSetNetwork,TorchDeepSetModel

from pquant import get_default_config
from quantizers.fixed_point.fixed_point_ops import get_fixed_quantizer
from pquant import get_layer_keep_ratio, get_model_losses,add_compression_layers
from pquant import iterative_train,remove_pruning_from_model

from sklearn import model_selection, metrics

# Register the model in the factory with the string name corresponding to what is in the yaml config
@JetModelFactory.register('PQuantDeepSetModel')
class PQuantDeepSetModel(TorchDeepSetModel):

    """PQuantDeepSetModel class

    Args:
        JetTagModel (_type_): Base class of a JetTagModel
    """
    

    schema = Schema(
            {
                "model": str,
                ## generic run config coniguration
                "run_config" : JetTagModel.run_schema,
                "model_config" : {"name" : str,
                                  "conv1d_layers" : list,
                                  "classification_layers" : list,
                                  "regression_layers" : list,
                                  "kernel_initializer" : str,
                                  "aggregator" : And(str, lambda s: s in  ["mean", "max", "attention"])},
                "quantization_config" : {'input_quantization' : list,
                                         'pt_output_quantization' : list},
                "training_config" : {"weight_method" : And(str, lambda s: s in  ["none", "ptref", "onlyclass"]),
                                     "validation_split" : And(float, lambda s: s > 0.0),
                                     "epochs" : And(int, lambda s: s >= 1),
                                     "batch_size" : And(int, lambda s: s >= 1),
                                     "learning_rate" : And(float, lambda s: s > 0.0),
                                     "loss_weights" : And(list, lambda s: len(s) == 2),
                                     "EarlyStopping_patience" : And(int, lambda s: s > 0),
                                     "ReduceLROnPlateau_factor" : And(float, lambda s: 1.0 >= s >= 0.0),
                                     "ReduceLROnPlateau_patience" : int,
                                     "ReduceLROnPlateau_min_lr" : And(float, lambda s: s >= 0.0)},
                ## generic hls4ml configuration
                "firmware_config" : {"input_precision" : str,
                                     "class_precision" : str,
                                     "reg_precision": str,
                                     "clock_period" : And(float, lambda s: 0.0 < s <= 10),
                                     "fpga_part" : str,
                                     "project_name" : str},
                "pquant_config" : {"pruning_parameters" : dict,
                                   "quantization_parameters" : dict,
                                   "fitcompress_parameters" : dict,
                                   "training_parameters" : dict,
                                   'batch_size': int, 
                                   'cosine_tmax': int, 
                                   'gamma': float, 
                                   'l2_decay': float, 
                                   'label_smoothing': float, 
                                   'lr': float, 
                                   'lr_schedule': str, 
                                   'milestones': list, 
                                   'momentum': float, 
                                   'optimizer': str, 
                                   'plot_frequency': int}
            }
    )

    def __init__(self, out_dir):
        super().__init__(out_dir)            
        self.quantizer = get_fixed_quantizer(overflow_mode="SAT")

    def build_model(self, inputs_shape: tuple, outputs_shape: tuple):
        """build model override, makes the model layer by layer

        Args:
            inputs_shape (tuple): Shape of the input
            outputs_shape (tuple): Shape of the output

        Additional hyperparameters in the config
            conv1d_layers: List of number of nodes for each layer of the conv1d layers.
            classifier_layers: List of number of nodes for each layer of the classifier MLP.
            regression_layers: List of number of nodes for each layer of the regression MLP
            aggregator: String that specifies the type of aggregator to use after the conv1D net.
        """
        
        self.input_shape = inputs_shape
        self.output_shape = outputs_shape
        self.pquant_config = self.yaml_dict['pquant_config']

        self.jet_model = TorchDeepSetNetwork( self.model_config, inputs_shape, outputs_shape)
        print(self.jet_model)
        self.jet_model.to(self.device)
        #Define the model using both branches
        self.jet_model = add_compression_layers(self.jet_model, self.pquant_config, (1,inputs_shape[0],inputs_shape[1]))
        print(self.jet_model)

    def loss_function_wrapper(self,y,y_true,y_pt, y_true_pt,sample_weight):
            return self.class_loss_fn(y,y_true), self.regression_loss_fn(torch.squeeze(y_pt),y_true_pt)
        
    def train_func(self, model, trainloader, device, loss_func, epoch, optimizer, scheduler, *args, **kwargs):
        accuracy_step = 0
        mae_step = 0
        mse_step = 0
        len_step = 0
        for data in trainloader:
            inputs, y, y_pt, sample_weight = data['X'].to(device), data['y'].to(device), data['y_pt'].to(device), data['sample_weight'].to(device)
            inputs = self.quantizer(inputs, k=torch.tensor(1.), i=torch.tensor(self.quantization_config['input_quantization'][0] - self.quantization_config['input_quantization'][1] -1), f=torch.tensor(self.quantization_config['input_quantization'][1])) 
            optimizer.zero_grad()
            output_class, outputs_pt = model(inputs)
            loss_class, loss_pt = loss_func(output_class,y,outputs_pt,y_pt,sample_weight)
            loss_class = sum(self.class_loss_fn(output_class,y)*sample_weight)/len(output_class)
            loss_pt =  sum(self.regression_loss_fn(torch.squeeze(outputs_pt),y_pt)*sample_weight)/len(outputs_pt)
            loss = loss_class + loss_pt
            losses = get_model_losses(model, torch.tensor(0.).to(device))
            loss += losses
            loss.backward()
            
            optimizer.step()
            
            _,predicted_class = torch.max(output_class,1)
            y = nn.functional.softmax(y,dim=1) 
            y = y.cpu().detach().numpy()
            y_categorical = [ np.argmax(y[i])  for i in range(len(y))]
            accuracy_step += (metrics.accuracy_score(predicted_class.cpu().detach().numpy(), y_categorical))
            mae_step += (metrics.mean_absolute_error(outputs_pt.cpu().detach().numpy(), y_pt.cpu().detach().numpy()))
            mse_step += (metrics.mean_squared_error(outputs_pt.cpu().detach().numpy(), y_pt.cpu().detach().numpy()))
            len_step += 1
            
        self.history[self.loss_name + self.output_id_name+ '_loss'].append(loss_class.mean().cpu().detach().numpy())
        self.history[self.loss_name + self.output_pt_name+ '_loss'].append(loss_pt.mean().cpu().detach().numpy())
            
        self.history['train_acc'].append(accuracy_step/len_step)
        self.history['train_mae'].append(mae_step/len_step)
        self.history['train_mse'].append(mse_step/len_step)
        
        epoch += 1
        
        print(f'Epoch {epoch.cpu().detach().numpy():d} \nloss: {loss.mean().cpu().detach().numpy():1.3f} '
              f'jet_id_output_loss: {loss_class.mean().cpu().detach().numpy():1.3f} '
              f'pT_output_loss: {loss_pt.mean().cpu().detach().numpy():1.3f} '
              f'jet_id_output_categorical_accuracy: {self.history['train_acc'][-1]:1.3f} '
              f'pT_output_mae:  {self.history['train_mae'][-1]:1.3f} '
              f'pT_output_mean_squared_error:  {self.history['train_mse'][-1]:1.3f} '
              )
        
    def validation_func(self,model, testloader, device, loss_func, epoch, scheduler, *args, **kwargs):
        self.jet_model.eval()
        accuracy_step = 0
        mae_step = 0
        mse_step = 0
        len_step = 0
        with torch.no_grad():
            for data in testloader:
                inputs, y, y_pt, sample_weight = data['X'].to(device), data['y'].to(device), data['y_pt'].to(device), data['sample_weight'].to(device)
                inputs = self.quantizer(inputs, k=torch.tensor(1.), i=torch.tensor(self.quantization_config['input_quantization'][0] - self.quantization_config['input_quantization'][1] -1), f=torch.tensor(self.quantization_config['input_quantization'][1]))                
                output_class, outputs_pt = self.jet_model(inputs)
                loss_class, loss_pt = loss_func(output_class,y,outputs_pt,y_pt,sample_weight)
                loss_class = sum(self.class_loss_fn(output_class,y)*sample_weight)/len(output_class)
                loss_pt = sum(self.regression_loss_fn(torch.squeeze(outputs_pt),y_pt)*sample_weight)/len(outputs_pt)
                loss = loss_class + loss_pt
                losses = get_model_losses(model, torch.tensor(0.).to(device))
                loss += losses
                _,predicted_class = torch.max(output_class,1)
                y = nn.functional.softmax(y,dim=1)
                y = y.cpu().detach().numpy()
                y_categorical = [ np.argmax(y[i])  for i in range(len(y))]
                
                accuracy_step += metrics.accuracy_score(predicted_class.cpu().detach().numpy(), y_categorical)
                mae_step += metrics.mean_absolute_error(outputs_pt.cpu().detach().numpy(), y_pt.cpu().detach().numpy())
                mse_step += metrics.mean_squared_error(outputs_pt.cpu().detach().numpy(), y_pt.cpu().detach().numpy())
                len_step += 1
            
            if scheduler is not None:
                scheduler.step(loss.mean())
                
        self.history['val_' + self.loss_name + self.output_id_name+ '_loss'].append(loss_class.mean().cpu().detach().numpy())
        self.history['val_' + self.loss_name + self.output_pt_name+ '_loss'].append(loss_pt.mean().cpu().detach().numpy())
            
        self.history['test_acc'].append(accuracy_step/len_step)
        self.history['test_mae'].append(mae_step/len_step)
        self.history['test_mse'].append(mse_step/len_step)
        
        print(f'val_loss: {loss.mean().cpu().detach().numpy():1.3f} '
              f'val_jet_id_output_loss: {loss_class.mean().cpu().detach().numpy():1.3f} '
              f'val_pT_output_loss: {loss_pt.mean().cpu().detach().numpy():1.3f} '
              f'val_jet_id_output_categorical_accuracy: {self.history['test_acc'][-1]:1.3f} '
              f'val_pT_output_mae: {self.history['test_mae'][-1]:1.3f} '
              f'val_pT_output_mean_squared_error:  {self.history['test_mse'][-1]:1.3f} '
              f'lr: {self.scheduler.get_last_lr()[0]}'
              )
      
    def fit(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        pt_target_train: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64],
    ):
        """Fit the model to the training dataset

        Args:
            X_train (npt.NDArray[np.float64]): X train dataset
            y_train (npt.NDArray[np.float64]): y train classification targets
            pt_target_train (npt.NDArray[np.float64]): y train pt regression targets
            sample_weight (npt.NDArray[np.float64]): sample weighting
        """
        x_train, x_test, y_train, y_test, pt_target_train, pt_target_test, sample_weight_train, sample_weight_test = model_selection.train_test_split(
            X_train, y_train,pt_target_train, sample_weight, test_size=self.training_config['validation_split'])
        
        train_dataset = JetTagDataset(x_train,y_train,pt_target_train,sample_weight_train)
        test_dataset = JetTagDataset(x_test,y_test,pt_target_test,sample_weight_test)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.training_config['batch_size'],
                                          shuffle=True, num_workers=self.n_workers, pin_memory=self.pin_memory)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.training_config['batch_size'],
                                            shuffle=False, num_workers=self.n_workers, pin_memory=self.pin_memory)
        
        
        
        self.jet_model = iterative_train(model = self.jet_model, 
                                         config = self.pquant_config, 
                                         train_func = self.train_func, 
                                         valid_func = self.validation_func, 
                                         trainloader = train_loader, 
                                         testloader = test_loader, 
                                         device = self.device,
                                         loss_func = self.loss_function_wrapper,
                                         optimizer = self.optimizer, 
                                         scheduler = self.scheduler
                                        )
        
        self.jet_model = remove_pruning_from_model(self.jet_model, self.pquant_config)
        
    
    def predict(self, X_test: npt.NDArray[np.float64]) -> tuple:
        """Predict method for model

        Args:
            X_test (npt.NDArray[np.float64]): Input X test

        Returns:
            tuple: (class_predictions , pt_ratio_predictions)
        """
        self.jet_model.to(self.device)
        self.jet_model.eval()
        batch_dim = X_test.shape[0]

        test_dataset = JetTagDataset(X_test,np.zeros([batch_dim,8]),np.zeros([batch_dim,1]),np.zeros([batch_dim,1]))
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.training_config['batch_size'],
                                            shuffle=False, num_workers=self.n_workers, pin_memory=self.pin_memory)
        class_predictions = []
        pt_ratio_predictions = []
        with torch.no_grad():
            for data in testloader:
                inputs, y, y_pt, sample_weight = data['X'].to(self.device), data['y'].to(self.device), data['y_pt'].to(self.device), data['sample_weight'].to(self.device)
                inputs = self.quantizer(inputs, k=torch.tensor(1.), i=torch.tensor(self.quantization_config['input_quantization'][0] - self.quantization_config['input_quantization'][1] -1), f=torch.tensor(self.quantization_config['input_quantization'][1]))                
                output_class, outputs_pt = self.jet_model(inputs)
                class_predictions.append(output_class.cpu().detach().numpy())
                pt_ratio_predictions.append(outputs_pt.cpu().detach().numpy().flatten())

        class_predictions = np.concatenate(class_predictions)
        pt_ratio_predictions = np.concatenate(pt_ratio_predictions)

        return (class_predictions, pt_ratio_predictions)
    
    @JetTagModel.load_decorator
    def load(self, out_dir: str = "None"):
        """Load the model file

        Args:
            out_dir (str, optional): Where to load it if not in the output_directory. Defaults to "None".
        """
        # Load the model
        with open(f"{out_dir}/model/meta_data.json", 'r') as fp:
            meta_dict = json.load(fp)
        self.input_shape = meta_dict['input_shape']
        self.output_shape = meta_dict['output_shape']
        
        self.jet_model = TorchDeepSetNetwork(self.model_config, self.input_shape, self.output_shape )
        self.jet_model.to(self.device)
        self.pquant_config = self.yaml_dict['pquant_config']
        self.jet_model = add_compression_layers(self.jet_model, self.pquant_config, (1,self.input_shape[0],self.input_shape[1]))
        self.jet_model.load_state_dict(torch.load(f"{out_dir}/model/saved_model.keras", weights_only=True))
