import os
import json
import functools
from abc import ABC, abstractmethod

import yaml

from tagger.plot.basic import loss_history


class JetTagModel(ABC):
    def __init__(self, output_dir):
        self.output_directory = output_dir

        self.model = None

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

    def load_yaml(self, yaml_path):
        with open(yaml_path, 'r') as stream:
            yaml_dict = yaml.safe_load(stream)
        self.run_config = yaml_dict['run_config']
        self.model_config = yaml_dict['model_config']
        self.quantization_config = yaml_dict['quantization_config']
        self.training_config = yaml_dict['training_config']
        self.hls4ml_config = yaml_dict['hls4ml_config']

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def compile_model(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    def predict(self, X_test):
        model_outputs = self.model.predict(X_test)
        y_pred = model_outputs[0]
        pt_ratio = model_outputs[1].flatten()
        return (y_pred, pt_ratio)

    def save_decorator(save_func):
        @functools.wraps(save_func)
        def wrapper(self, out_dir=None):
            if out_dir is None:
                out_dir = self.output_directory
            with open(os.path.join(out_dir, "input_vars.json"), "w") as f:
                json.dump(self.input_vars, f, indent=4)  # Dump input variables
            with open(os.path.join(out_dir, "extra_vars.json"), "w") as f:
                # Dump output variables
                json.dump(self.extra_vars, f, indent=4)
            with open(os.path.join(out_dir, "class_labels.json"), "w") as f:
                # Dump output variables
                json.dump(self.class_labels, f, indent=4)
            save_func(self, out_dir)
        return wrapper

    def load_decorator(load_func):
        @functools.wraps(load_func)
        def wrapper(self, out_dir):
            with open(os.path.join(out_dir, "input_vars.json"), "r") as f:
                self.input_vars = json.load(f)
            with open(os.path.join(out_dir, "class_labels.json"), "r") as f:
                self.class_labels = json.load(f)
            with open(os.path.join(out_dir, "extra_vars.json"), "r") as f:
                self.extra_vars = json.load(f)
            load_func(self, out_dir)
        return wrapper

    def set_labels(self, input_vars, extra_vars, class_labels):
        self.input_vars = input_vars
        self.extra_vars = extra_vars
        self.class_labels = class_labels

    @abstractmethod
    def hls4ml_convert(self):
        pass

    def plot_loss(self, out_dir=None):
        if out_dir is None:
            out_dir = self.output_directory

        # Produce some basic plots with the training for diagnostics
        plot_path = os.path.join(out_dir, "plots/training")
        os.makedirs(plot_path, exist_ok=True)

        # Plot history
        loss_history(plot_path, self.history)


class JetModelFactory:
    """ The factory class for creating Jet Tag Models"""

    registry = {}
    """ Internal registry for available Jet Tag Models """

    @classmethod
    def register(cls, name: str):

        def inner_wrapper(wrapped_class: JetTagModel):
            if name in cls.registry:
                logger.warning(
                    'Jet Tagger Model %s already exists. Will replace it', name)
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_JetTagModel(cls, name: str, folder: str, **kwargs) -> 'JetTagModel':
        """ Factory command to create the Jet Tag Model """

        JetTag_class = cls.registry[name]
        model = JetTag_class(folder, **kwargs)

        return model
