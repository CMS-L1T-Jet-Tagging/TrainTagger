"""DeepSet model child class

Written 28/05/2025 cebrown@cern.ch
"""

import json
import os

import hls4ml
import numpy as np
import numpy.typing as npt
import ydf

from tagger.model.JetTagModel import JetModelFactory, JetTagModel

# Register the model in the factory with the string name corresponding to what is in the yaml config
@JetModelFactory.register('VectorTreeModel')
class VectorTreeModel(JetTagModel):
    """VectroTreeModel class

    Args:
        JetTagModel (_type_): Base class of a JetTagModel
    """
    
    def build_model(self, inputs_shape: tuple, outputs_shape: tuple, num_workers: int = 4, tuner=None):
        """build model override, makes the model layer by layer

        Args:
            inputs_shape (tuple): Shape of the input
            outputs_shape (tuple): Shape of the output

        Additional hyperparameters in the config

        """
        if tuner != None:
            config = {'num_trees':100}
        else:
            config = self.model_config
            
        print(ydf.__version__)
        features = [('pt', ydf.Semantic.NUMERICAL_VECTOR_SEQUENCE),('pt_rel', ydf.Semantic.NUMERICAL_VECTOR_SEQUENCE),('pt_log', ydf.Semantic.NUMERICAL_VECTOR_SEQUENCE),('delta', ydf.Semantic.NUMERICAL_VECTOR_SEQUENCE),('pid', ydf.Semantic.NUMERICAL_VECTOR_SEQUENCE),('z0', ydf.Semantic.NUMERICAL_VECTOR_SEQUENCE),('dxy', ydf.Semantic.NUMERICAL_VECTOR_SEQUENCE),('puppiweight', ydf.Semantic.NUMERICAL_VECTOR_SEQUENCE),('emid', ydf.Semantic.NUMERICAL_VECTOR_SEQUENCE),('quality', ydf.Semantic.NUMERICAL_VECTOR_SEQUENCE)]
        self.learner = ydf.GradientBoostedTreesLearner(**config,
                                                       label="label",
                                                       weights="weights",
                                                       features=features,
                                                       num_threads= 24,
                                                       discretize_numerical_columns=True,
                                                       tuner=tuner,
                                                       working_dir=self.output_directory
                                                       )

    def compile_model(self, num_samples: int):
        pass
    
    def tune(self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        pt_target_train: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64]):
        self.fit(X_train,y_train,pt_target_train,sample_weight)
    
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
        
        X_train_dict = {'pt':[],'pt_rel':[],'pt_log':[],'delta':[],'pid':[],'z0':[],'dxy':[],'puppiweight':[],'emid':[],'quality':[]}
        y_train_array = []
        
        for ibatch,batch in enumerate(X_train):
            if ibatch % 250000 == 0:
                print(ibatch , " out of ", len(X_train) )
            '''
              0 pt
              1 pt_rel
              2 pt_log
              3 deta
              4 dphi
              5 mass
              6 isPhoton
              7 isElectronPlus
              8 isElectronMinus
              9 isMuonPlus
              10 isMuonMinus
              11 isNeutralHadron
              12 isChargedHadronPlus
              13 isChargedHadronMinus
              14 z0
              15 dxy
              16 isfilled
              17 puppiweight
              18 emid
              19 quality
            '''            
            X_train_dict['pt'].append(np.array([[batch[j,0]] for j in range(len(batch))]))
            X_train_dict['pt_rel'].append(np.array([[batch[j,1]] for j in range(len(batch))]))
            X_train_dict['pt_log'].append(np.array([[batch[j,2]] for j in range(len(batch))]))
            X_train_dict['delta'].append(np.array([[batch[j,3],batch[j,4]] for j in range(len(batch))]))
            X_train_dict['pid'].append(np.array([[batch[j,6],batch[j,7],batch[j,8],batch[j,9],batch[j,10],batch[j,11],batch[j,12],batch[j,13]] for j in range(len(batch))]))
            X_train_dict['z0'].append(np.array([[batch[j,14]] for j in range(len(batch))]))
            X_train_dict['dxy'].append(np.array([[batch[j,15]] for j in range(len(batch))]))
            X_train_dict['puppiweight'].append(np.array([[batch[j,17]] for j in range(len(batch))]))
            X_train_dict['emid'].append(np.array([[batch[j,18]] for j in range(len(batch))]))
            X_train_dict['quality'].append(np.array([[batch[j,19]] for j in range(len(batch))]))

            index = np.where(y_train[ibatch] == 1)
            y_train_array.append(index[0][0])
            
        X_train_dict["label"] = np.array(y_train_array,dtype=int)
        X_train_dict["weights"] = sample_weight
        self.jet_model = self.learner.train(X_train_dict,verbose=2)
        
        #print(self.jet_model.describe())
        
        #print(self.jet_model.describe()[4])

    # Decorated with save decorator for added functionality
    @JetTagModel.save_decorator
    def save(self, out_dir: str = "None"):
        """Save the model file

        Args:
            out_dir (str, optional): Where to save it if not in the output_directory. Defaults to "None".
        """
        # Export the model

        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        self.jet_model.save(f"{out_dir}/model/saved_model")
        print("Model saved to model/saved_model")

    @JetTagModel.load_decorator
    def load(self, out_dir: str = "None"):
        """Load the model file

        Args:
            out_dir (str, optional): Where to load it if not in the output_directory. Defaults to "None".
        """
        # Load the model
        self.jet_model = ydf.load_model(f"{out_dir}/model/saved_model")
        
    def hls4ml_convert(self):
        pass
    
    def plot_loss(self):
        """Plot the loss of the model to the output directory"""
        out_dir = self.output_directory
        # Produce some basic plots with the training for diagnostics
        plot_path = os.path.join(out_dir, "plots/training")
        os.makedirs(plot_path, exist_ok=True)

        # Plot history
        #loss_history(plot_path, [self.loss_name + self.output_id_name, self.loss_name + self.output_pt_name], self.history)

    def predict(self, X_test: npt.NDArray[np.float64] ) -> tuple:
        """Predict method for model

        Args:
            X_test (npt.NDArray[np.float64]): Input X test

        Returns:
            tuple: (class_predictions , pt_ratio_predictions)
        """
        
        X_test_dict = {'pt':[],'pt_rel':[],'pt_log':[],'delta':[],'pid':[],'z0':[],'dxy':[],'puppiweight':[],'emid':[],'quality':[]}
        y_test_array = []
        
        for ibatch,batch in enumerate(X_test):
            if ibatch % 250000 == 0:
                print(ibatch , " out of ", len(X_test) )
            X_test_dict['pt'].append(np.array([[batch[j,0]] for j in range(len(batch))]))
            X_test_dict['pt_rel'].append(np.array([[batch[j,1]] for j in range(len(batch))]))
            X_test_dict['pt_log'].append(np.array([[batch[j,2]] for j in range(len(batch))]))
            X_test_dict['delta'].append(np.array([[batch[j,3],batch[j,4]] for j in range(len(batch))]))
            X_test_dict['pid'].append(np.array([[batch[j,6],batch[j,7],batch[j,8],batch[j,9],batch[j,10],batch[j,11],batch[j,12],batch[j,13]] for j in range(len(batch))]))
            X_test_dict['z0'].append(np.array([[batch[j,14]] for j in range(len(batch))]))
            X_test_dict['dxy'].append(np.array([[batch[j,15]] for j in range(len(batch))]))
            X_test_dict['puppiweight'].append(np.array([[batch[j,17]] for j in range(len(batch))]))
            X_test_dict['emid'].append(np.array([[batch[j,18]] for j in range(len(batch))]))
            X_test_dict['quality'].append(np.array([[batch[j,19]] for j in range(len(batch))]))
            y_test_array.append(0)
            
        
        X_test_dict["label"] =  np.array(y_test_array,dtype=int)        
        model_outputs = self.jet_model.predict(X_test_dict)
        class_predictions = model_outputs
        pt_ratio_predictions = np.array([[1] for i in range(model_outputs.shape[0])])
        return (class_predictions, pt_ratio_predictions)
    
    def init_tuner(
        self
        ):
        
        self.tuner = ydf.RandomSearchTuner(num_trials=5, automatic_search_space=True)
