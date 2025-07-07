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
        self.learner = ydf.GradientBoostedTreesLearner(**config,
                                                       label="label",
                                                       weights="weights",
                                                       features=[('feature', ydf.Semantic.NUMERICAL_VECTOR_SEQUENCE)],
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
        
        X_train_array = []
        y_train_array = []
        
        for ibatch,batch in enumerate(X_train):
            vectors_list = []
            y_list = []
            if ibatch % 250000 == 0:
                print(ibatch , " out of ", len(X_train) )
            for icandidate,candidate in enumerate(X_train[ibatch]):
                if np.abs(np.sum(candidate)) > 0:
                    vectors_list.append([candidate])
                #print(np.sum(candidate),candidate,y_train[icandidate])
            vectors = np.array(np.concatenate(vectors_list, axis=0)) 
            X_train_array.append(vectors)
            index = np.where(y_train[ibatch] == 1)
            y_train_array.append(index[0][0])
            

        train_dataset  = {"label": np.array(y_train_array,dtype=int), "feature": X_train_array, "weights": sample_weight}
        self.jet_model = self.learner.train(train_dataset,verbose=2)
        
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
        
        X_test_array = []
        y_test_array = []
        
        for ibatch,batch in enumerate(X_test):
            vectors_list = []
            y_list = []
            if ibatch % 250000 == 0:
                print(ibatch , " out of ", len(X_test) )
            for icandidate,candidate in enumerate(X_test[ibatch]):
                if np.abs(np.sum(candidate)) > 0:
                    vectors_list.append([candidate])
            vectors = np.array(np.concatenate(vectors_list, axis=0)) 
            X_test_array.append(vectors)
            y_test_array.append(0)
            

        test_dataset  = {"label": np.array(y_test_array,dtype=int), "feature": X_test_array}
        
        model_outputs = self.jet_model.predict(test_dataset)
        class_predictions = model_outputs
        pt_ratio_predictions = np.array([[1] for i in range(model_outputs.shape[0])])
        return (class_predictions, pt_ratio_predictions)
    
    def init_tuner(
        self
        ):
        
        self.tuner = ydf.RandomSearchTuner(num_trials=5, automatic_search_space=True)
