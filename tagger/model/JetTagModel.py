"""Jet Tag Model base class and additional functionality for model registering

Written 28/05/2025, cebrown@cern.ch
"""

import functools
import json
import os
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import yaml

from tagger.plot.basic import loss_history


class JetTagModel(ABC):
    """Parent Class for Jet Tag Models

    Abstract Base Class not for use directly
    """

    def __init__(self, output_dir: str):
        """
        Args:
            output_dir (str): Saving directory for model artefacts
        """
        self.output_directory = output_dir

        self.jet_model = None
        self.hls_jet_model = None

        self.input_vars = []
        self.extra_vars = []
        self.class_labels = []

        self.run_config = {}
        self.model_config = {}
        self.quantization_config = {}
        self.training_config = {}
        self.hls4ml_config = {}

        self.output_id_name = 'jet_id_output'
        self.output_pt_name = 'pT_output'
        self.loss_name = ''

        self.callbacks = []

        self.history = None

    def load_yaml(self, yaml_path: str):
        """Load config dictionaries

        Args:
            yaml_path (str): Path to yaml file
        """

        with open(yaml_path, 'r') as stream:
            yaml_dict = yaml.safe_load(stream)
        self.run_config = yaml_dict['run_config']
        self.model_config = yaml_dict['model_config']
        self.quantization_config = yaml_dict['quantization_config']
        self.training_config = yaml_dict['training_config']
        self.hls4ml_config = yaml_dict['hls4ml_config']

    @abstractmethod
    def build_model(self, **kwargs):
        """
        Build the model layers, must be written for child class
        """

    @abstractmethod
    def compile_model(self, **kwargs):
        """
        Compile the model, adding loss function and callbacks
        Must be written for child class
        """

    @abstractmethod
    def fit(self, **kwargs):
        """
        Fit the model to the training data
        Must be written for child class
        """

    @abstractmethod
    def hls4ml_convert(self, **kwargs):
        """
        Convert the model in hls4ml
        Must be written for child class
        """

    def predict(self, X_test: npt.NDArray[np.float64]) -> tuple:
        """Predict method for model

        Args:
            X_test (npt.NDArray[np.float64]): Input X test

        Returns:
            tuple: (class_predictions , pt_ratio_predictions)
        """
        model_outputs = self.jet_model.predict(X_test)
        class_predictions = model_outputs[0]
        pt_ratio_predictions = model_outputs[1].flatten()
        return (class_predictions, pt_ratio_predictions)
    
    @staticmethod
    def save_decorator(save_func):
        """Decorator used to include additional
        saving functionality for child classes
        """

        @functools.wraps(save_func)
        def wrapper(self, out_dir: str = "None"):
            """Wrapper adding saving functionality

            Args:
                out_dir (str): Where to save the model. Defaults to
                None but overridden to output_directory.
            """
            if out_dir == "None":
                out_dir = self.output_directory
            # Save additional jsons associated with model
            # Dump input variables
            with open(os.path.join(out_dir, "input_vars.json"), "w") as f:
                json.dump(self.input_vars, f, indent=4)
            # Dump extra variables
            with open(os.path.join(out_dir, "extra_vars.json"), "w") as f:
                json.dump(self.extra_vars, f, indent=4)
            # Dump class variables
            with open(os.path.join(out_dir, "class_labels.json"), "w") as f:
                json.dump(self.class_labels, f, indent=4)
            # Do the rest of the saving, defined in child class
            save_func(self, out_dir)

        return wrapper

    @staticmethod
    def load_decorator(load_func):
        """Decorator used to include additional
        loading functionality for child classes
        """

        @functools.wraps(load_func)
        def wrapper(self, out_dir: str = "None"):
            """Wrapper adding loading functionality

            Args:
                out_dir (str): Where to load the model from. Defaults to
                None but overridden to output_directory.
            """
            if out_dir == "None":
                out_dir = self.output_directory
            # Save additional jsons associated with model
            # Dump input variables
            with open(os.path.join(out_dir, "input_vars.json"), "r") as f:
                self.input_vars = json.load(f)
            # Dump extra variables
            with open(os.path.join(out_dir, "class_labels.json"), "r") as f:
                self.class_labels = json.load(f)
            # Dump class variables
            with open(os.path.join(out_dir, "extra_vars.json"), "r") as f:
                self.extra_vars = json.load(f)
            # Do the rest of the loading, defined in child class
            load_func(self, out_dir)

        return wrapper

    def set_labels(self, input_vars: str, extra_vars: str, class_labels: str):
        """Set internal labels

        Args:
            input_vars (str): Input variable names
            extra_vars (str): Extra variable names
            class_labels (str): Class label names
        """
        self.input_vars = input_vars
        self.extra_vars = extra_vars
        self.class_labels = class_labels

    def plot_loss(self):
        """Plot the loss of the model to the output directory"""
        out_dir = self.output_directory
        # Produce some basic plots with the training for diagnostics
        plot_path = os.path.join(out_dir, "plots/training")
        os.makedirs(plot_path, exist_ok=True)

        # Plot history
        loss_history(plot_path, [self.loss_name + self.output_id_name, self.loss_name + self.output_pt_name], self.history)
        
class JetModelFactory:
    """The factory class for creating Jet Tag Models"""

    registry = {}
    """ Internal registry for available Jet Tag Models """

    @classmethod
    def register(cls, name: str):
        """Decorator for registering new jet tag models

        Args:
            name (str): Name of the model
        """

        def inner_wrapper(wrapped_class: JetTagModel):
            if name in cls.registry:
                print('Jet Tagger Model %s already exists. Will replace it', name)
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_JetTagModel(cls, name: str, folder: str, **kwargs) -> 'JetTagModel':
        """Factory command to create the Jet Tag Model"""

        jettag_class = cls.registry[name]
        model = jettag_class(folder, **kwargs)

        return model
