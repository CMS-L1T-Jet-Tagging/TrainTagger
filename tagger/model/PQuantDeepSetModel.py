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

from pquant import get_default_config
from quantizers.fixed_point.fixed_point_ops import get_fixed_quantizer
from pquant import get_layer_keep_ratio, get_model_losses,add_compression_layers
from pquant import iterative_train,remove_pruning_from_model

from sklearn import model_selection, metrics



class TorchDeepSetNetwork(nn.Module):
    def __init__(self, model_config, inputs_shape, outputs_shape ):
        super(TorchDeepSetNetwork, self).__init__()
        
        C_in = inputs_shape[1]  # channels
        L = inputs_shape[0]     # sequence length
        
        self.norm_input = nn.BatchNorm1d(C_in)
        
        # Conv1D layers (Keras Conv1D: (batch, L, C) → PyTorch Conv1d: (batch, C, L))
        conv_layers = []
        in_channels = C_in
        for i, depth in enumerate(model_config['conv1d_layers']):
            conv_layers.append(nn.Conv1d(in_channels, depth, kernel_size=1))
            conv_layers.append(nn.ReLU())
            in_channels = depth
        self.conv1d_layers = nn.Sequential(*conv_layers)
        
        # Average pooling (same as Keras AveragePooling1D)
        self.avgpool = nn.AvgPool1d(kernel_size=L)
        
        # Compute flattened size after pooling
        self.flatten_dim = in_channels  # because AvgPool1d reduces L → 1
        
        # ---- Jet ID (classification) branch ----
        class_layers = []
        in_features = self.flatten_dim
        for i, depth in enumerate(model_config['classification_layers']):
            class_layers.append(nn.Linear(in_features, depth))
            if i != 0:
                class_layers.append(nn.ReLU())
            in_features = depth
        class_layers.append(nn.Linear(in_features, outputs_shape[0]))
        self.jet_id_head = nn.Sequential(*class_layers)

        # ---- pT Regression branch ----
        reg_layers = []
        in_features = self.flatten_dim
        for i, depth in enumerate(model_config['regression_layers']):
            reg_layers.append(nn.Linear(in_features, depth))
            if i != 0:
                reg_layers.append(nn.ReLU())
            in_features = depth
        reg_layers.append(nn.Linear(in_features, 1))
        self.pt_head = nn.Sequential(*reg_layers)
        
    def forward(self, x):
        # Keras: (batch, L, C) → PyTorch expects (batch, C, L)
        x = x.type(torch.float32)
        x = x.permute(0, 2, 1)
        x = self.norm_input(x)
        x = self.conv1d_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)

        jet_id = self.jet_id_head(x)
        jet_id = F.softmax(jet_id, dim=1)

        pt_regress = self.pt_head(x)

        return jet_id, pt_regress


class JetTagDataset(Dataset):

    def __init__(self, X, y, y_pt, sample_weight):
        
        self.X = X
        self.y = y
        self.y_pt = y_pt
        self.sample_weight = sample_weight
        

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {'X': self.X[idx], 'y' : self.y[idx], 'y_pt':self.y_pt[idx], 'sample_weight':self.sample_weight[idx]}

        return sample
    

# Register the model in the factory with the string name corresponding to what is in the yaml config
@JetModelFactory.register('PQuantDeepSetModel')
class PQuantDeepSetModel(JetTagModel):

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
        self.device = "cpu"
        self.n_workers = 8
        self.pin_memory = False
        if torch.cuda.is_available():
            self.device = "cuda"
            self.n_workers = 24
            self.pin_memory= True
            
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
        self.jet_model.to(self.device)
        #Define the model using both branches
        self.jet_model = add_compression_layers(self.jet_model, self.pquant_config, (1,inputs_shape[0],inputs_shape[1]))
        
        
        
    def firmware_convert(self, firmware_dir: str, build: bool = False):
        """Run the hls4ml model conversion

        Args:
            firmware_dir (str): Where to save the firmware
            build (bool, optional): Run the full hls4ml build? Or just create the project. Defaults to False.
        """

        # Remove the old directory if it exists
        hls4ml_outdir = firmware_dir + '/' + self.firmware_config['project_name']
        os.system(f'rm -rf {hls4ml_outdir}')

        # Create default config
        config = hls4ml.utils.config_from_keras_model(self.jet_model, granularity='name')
        config['IOType'] = 'io_parallel'
        config['LayerName']['model_input']['Precision']['result'] = self.firmware_config['input_precision']

        # Configuration for conv1d layers
        # hls4ml automatically figures out the paralellization factor
        # config['LayerName']['Conv1D_1']['ParallelizationFactor'] = 8
        # config['LayerName']['Conv1D_2']['ParallelizationFactor'] = 8

        # Additional config
        for layer in self.jet_model.layers:
            layer_name = layer.__class__.__name__

            if layer_name in ["BatchNormalization", "InputLayer"]:
                config["LayerName"][layer.name]["Precision"] = self.firmware_config['input_precision']
                config["LayerName"][layer.name]["result"] = self.firmware_config['input_precision']
                config["LayerName"][layer.name]["Trace"] = not build

            elif layer_name in ["Permute", "Concatenate", "Flatten", "Reshape", "UpSampling1D", "Add"]:
                print("Skipping trace for:", layer.name)
            else:
                config["LayerName"][layer.name]["Trace"] = not build

        config["LayerName"]["jet_id_output"]["Precision"]["result"] = self.firmware_config['class_precision']
        config["LayerName"]["jet_id_output"]["Implementation"] = "latency"
        config["LayerName"]["pT_output"]["Precision"]["result"] = self.firmware_config['reg_precision']
        config["LayerName"]["pT_output"]["Implementation"] = "latency"

        # Write HLS
        self.hls_jet_model = hls4ml.converters.convert_from_keras_model(
            self.jet_model,
            backend='Vitis',
            project_name=self.firmware_config['project_name'],
            clock_period=self.firmware_config['clock_period'],
            hls_config=config,
            output_dir=f'{hls4ml_outdir}',
            part= self.firmware_config['fpga_part'],
        )

        # Compile the project
        self.hls_jet_model.compile()

        # Save config  as json file
        print("Saving default config as config.json ...")
        with open(hls4ml_outdir + '/config.json', 'w') as fp:
            json.dump(config, fp)

        if build:
            # build the project
            self.hls_jet_model.build(csim=False, reset=True)

    def compile_model(self, num_samples: int):
        """compile the model generating callbacks and loss function
        Args:
            num_samples (int): Number of samples in the training set used for scheduling
        """
            
        self.optimizer = torch.optim.Adam(self.jet_model.parameters(), lr=self.training_config['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer , factor=self.training_config['ReduceLROnPlateau_factor'], patience=self.training_config['ReduceLROnPlateau_patience'],min_lr=self.training_config['ReduceLROnPlateau_min_lr'])
        
        # Instantiate a loss function.
        self.class_loss_fn = nn.CrossEntropyLoss()
        self.regression_loss_fn = nn.HuberLoss()
        
       
        
        self.history = { self.loss_name + self.output_id_name + '_loss':[], self.loss_name + self.output_pt_name + '_loss':[], 
                         'val_' + self.loss_name + self.output_id_name + '_loss' :[], 'val_' + self.loss_name + self.output_pt_name + '_loss':[], 
                     "train_acc":[], 'train_mae':[],'train_mse':[],
                     "test_acc":[], 'test_mae':[],'test_mse':[],}
    
    def loss_function_wrapper(self,y,y_true,y_pt, y_true_pt,sample_weight):
            return self.class_loss_fn(y,y_true)*sample_weight, self.regression_loss_fn(torch.squeeze(y_pt),y_true_pt)*sample_weight
        
    def train_func(self, model, trainloader, device, loss_func, epoch, optimizer, scheduler, *args, **kwargs):
        accuracy_step = 0
        mae_step = 0
        mse_step = 0
        len_step = 0
        for data in trainloader:
            inputs, y, y_pt, sample_weight = data['X'].to(device), data['y'].to(device), data['y_pt'].to(device), data['sample_weight'].to(device)
            inputs = self.quantizer(inputs, k=torch.tensor(1.), i=torch.tensor(self.quantization_config['input_quantization'][1]), f=torch.tensor(self.quantization_config['input_quantization'][0]) - 1) 
            optimizer.zero_grad()
            output_class, outputs_pt = model(inputs)
            loss_class, loss_pt = loss_func(output_class,y,outputs_pt,y_pt,sample_weight)
            loss = self.training_config['loss_weights'][0]*loss_class + self.training_config['loss_weights'][0]*loss_pt
            losses = get_model_losses(model, torch.tensor(0.).to(device))
            loss += losses
            loss.mean().backward()
            
            optimizer.step()
            
            
            _,predicted_class = torch.max(output_class,1)
            y = y.cpu().detach().numpy()
            y_categorical = [ np.argmax(y[i])  for i in range(len(y))]
            accuracy_step += (metrics.accuracy_score(predicted_class.detach().numpy(), y_categorical))
            mae_step += (metrics.mean_absolute_error(outputs_pt.detach().numpy(), y_pt.detach().numpy()))
            mse_step += (metrics.mean_squared_error(outputs_pt.detach().numpy(), y_pt.detach().numpy()))
            len_step += 1
            
        self.history[self.loss_name + self.output_id_name+ '_loss'].append(loss_class.mean().detach().numpy())
        self.history[self.loss_name + self.output_pt_name+ '_loss'].append(loss_pt.mean().detach().numpy())
            
        self.history['train_acc'].append(accuracy_step/len_step)
        self.history['train_mae'].append(mae_step/len_step)
        self.history['train_mse'].append(mse_step/len_step)
        
        epoch += 1
        
        print(f'Epoch {epoch.detach().numpy():d} \nloss: {loss.mean().detach().numpy():1.3f} '
              f'jet_id_output_loss: {loss_class.mean().detach().numpy():1.3f} '
              f'pT_output_loss: {loss_pt.mean().detach().numpy():1.3f} '
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
                inputs = self.quantizer(inputs, k=torch.tensor(1.), i=torch.tensor(self.quantization_config['input_quantization'][1]), f=torch.tensor(self.quantization_config['input_quantization'][0]) - 1) 
                output_class, outputs_pt = self.jet_model(inputs)
                loss_class, loss_pt = loss_func(output_class,y,outputs_pt,y_pt,sample_weight)
                loss = self.training_config['loss_weights'][0]*loss_class + self.training_config['loss_weights'][0]*loss_pt
                losses = get_model_losses(model, torch.tensor(0.).to(device))
                loss += losses
                _,predicted_class = torch.max(output_class,1)
                y = y.cpu().detach().numpy()
                y_categorical = [ np.argmax(y[i])  for i in range(len(y))]
                
                accuracy_step += metrics.accuracy_score(predicted_class.detach().numpy(), y_categorical)
                mae_step += metrics.mean_absolute_error(outputs_pt.detach().numpy(), y_pt.detach().numpy())
                mse_step += metrics.mean_squared_error(outputs_pt.detach().numpy(), y_pt.detach().numpy())
                len_step += 1
            
            if scheduler is not None:
                scheduler.step(loss.mean())
                
        self.history['val_' + self.loss_name + self.output_id_name+ '_loss'].append(loss_class.mean().detach().numpy())
        self.history['val_' + self.loss_name + self.output_pt_name+ '_loss'].append(loss_pt.mean().detach().numpy())
            
        self.history['test_acc'].append(accuracy_step/len_step)
        self.history['test_mae'].append(mae_step/len_step)
        self.history['test_mse'].append(mse_step/len_step)
        
        print(f'val_loss: {loss.mean().detach().numpy():1.3f} '
              f'val_jet_id_output_loss: {loss_class.mean().detach().numpy():1.3f} '
              f'val_pT_output_loss: {loss_pt.mean().detach().numpy():1.3f} '
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
                inputs = self.quantizer(inputs, k=torch.tensor(1.), i=torch.tensor(self.quantization_config['input_quantization'][1]), f=torch.tensor(self.quantization_config['input_quantization'][0]) - 1) 
                output_class, outputs_pt = self.jet_model(inputs)
                class_predictions.append(output_class.cpu().detach().numpy())
                pt_ratio_predictions.append(outputs_pt.cpu().detach().numpy().flatten())

        class_predictions = np.concatenate(class_predictions)
        pt_ratio_predictions = np.concatenate(pt_ratio_predictions)

        return (class_predictions, pt_ratio_predictions)
    
    # Decorated with save decorator for added functionality
    @JetTagModel.save_decorator
    def save(self, out_dir: str = "None"):
        """Save the model file

        Args:
            out_dir (str, optional): Where to save it if not in the output_directory. Defaults to "None".
        """
        # Export the model
        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        # Use keras save format !NOT .h5! due to depreciation
        export_path = os.path.join(out_dir, "model/saved_model.keras")
        torch.save(self.jet_model.state_dict(), export_path)
        
        meta_dict = {"input_shape":self.input_shape, 
                     "output_shape":self.output_shape}
            
        with open(f'{out_dir}/model/meta_data.json', 'w') as fp:
            json.dump(meta_dict, fp)
        
        print(f"Model saved to {export_path}")

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
        self.pquant_config = self.yaml_dict['pquant_config']
        self.jet_model = add_compression_layers(self.jet_model, self.pquant_config, (1,self.input_shape[0],self.input_shape[1]))
        self.jet_model.load_state_dict(torch.load(f"{out_dir}/model/saved_model.keras", weights_only=True))