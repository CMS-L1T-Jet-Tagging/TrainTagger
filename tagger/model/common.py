"""Common utilities for usage across all model child classes
Includes attention layers
Include from Yaml and Folder loading functionality

Written 28/05/2025 cebrown@cern.ch
"""

import os
import shutil

import numpy as np
import yaml

from math import cos, pi

from tagger.model.JetTagModel import JetModelFactory, JetTagModel


def fromYaml(yaml_path: str, folder: str, recreate: bool = True) -> JetTagModel:
    """Create a model directly from a yaml input file

    Args:
        yaml_path (str): Path to yaml file
        folder (str): Output saving folder for model
        recreate (bool, optional): Rewrite the output directory?. Defaults to True.

    Returns:
        JetTagModel: The model
    """

    with open(yaml_path, 'r') as stream:
        yaml_dict = yaml.safe_load(stream)

    # Create a model based on what is specified in the yaml 'model' field
    # Model must be registered for this to function
    model = JetModelFactory.create_JetTagModel(yaml_dict['model'], folder)
    # Validate yaml dict before loading
    model.schema.validate(yaml_dict)
    model.load_yaml(yaml_path)
    if recreate:
        # Remove output dir if exists
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Re-created existing directory: {folder}.")
            # Create dir to save results
        os.makedirs(folder)
        os.system('cp ' + yaml_path + ' ' + folder)
    return model


def fromFolder(save_path: str, newoutput_dir: str = "None") -> JetTagModel:
    """Load a model from its save folder using the yaml file in the save folder

    Args:
        save_path (str): Where to load the model from
        newoutput_dir (str, optional): New folder to save the model to if needed. Defaults to "None".

    Returns:
        JetTagModel: The model
    """
    if newoutput_dir != "None":
        folder = newoutput_dir
        recreate = True
    else:
        folder = save_path
        recreate = False

    for file in os.listdir(folder):
        if file.endswith(".yaml"):
            yaml_path = os.path.join(folder, file)

    model = fromYaml(yaml_path, folder, recreate=recreate)
    model.load(folder)
    return model

def log_beta_schedule(epoch, max_epochs=100):
    log_beta_start = np.log10(1e-7)
    log_beta_end = np.log10(1e-4)
    log_beta = log_beta_start + (log_beta_end - log_beta_start) * (epoch / max_epochs)
    return 10 ** log_beta

def cosine_decay_restarts(global_step,initial_learning_rate, max_epochs):
    n_cycle = 1
    cycle_step = global_step
    cycle_len = max_epochs
    while cycle_step >= cycle_len:
        cycle_step -= cycle_len
        cycle_len *= 1
        n_cycle += 1

    cycle_t = min(cycle_step / (cycle_len - 10), 1)
    lr = 1.e-6 + 0.5 * (initial_learning_rate - 1.e-6) * (
                1 + cos(pi * cycle_t)
            ) * 1 ** max(n_cycle - 1, 0)
    return lr


def initialise_tensorflow(num_threads):
    import tensorflow as tf
    
    print("Using ")
    print(tf.config.list_physical_devices('GPU'))
    print("for training")
    
    gpu = tf.config.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(gpu[0], True)
    # Set some tensorflow constants
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(num_threads)
    
    seed = np.random.randint(2**32 - 1)
    tf.keras.utils.set_random_seed(seed)
    print("my seed is:", seed)

def huber_loss(delta=.1, pu=0., alpha=0.):
    import tensorflow as tf
    """
    Huber loss with asymmetric penalization. Parameters spcified in model config.

    Args:
        delta: Huber threshold.
        alpha: Weight for underestimation (y_true > y_pred).
        pu: Weight for overestimation of pileup (y_true == -1 and y_pred > 1).
    """
    def loss(y_true, y_pred):
        # Minbias: punish overestimation
        pu_punish = tf.where(
            (y_true == -1) & (y_pred > 1),
            pu * (y_pred - 1.0), # scaling proportional to excess
            0.)
        pu_mask = y_true == -1
        y_true = tf.where(pu_mask, 0.95, y_true)

        # punish underestimation more for all other samples
        residual = y_true - y_pred
        overest = tf.where((residual > 0) & (~pu_mask), (abs(residual) * alpha), 0.)  # Penalize overestimation more
        weights = overest + pu_punish + 1.0  # Add 1 as base value

        abs_res = tf.abs(residual)
        quadratic = tf.minimum(abs_res, delta)
        linear = abs_res - quadratic

        return weights * (0.5 * quadratic**2 + delta * linear)

    return loss
