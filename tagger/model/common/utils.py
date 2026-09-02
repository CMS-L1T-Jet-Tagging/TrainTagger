"""Common utilities for usage across all model child classes
Includes attention layers
Include from Yaml and Folder loading functionality

Written 28/05/2025 cebrown@cern.ch
"""

import os
import shutil
import yaml
from tagger.model.JetTagModel import JetModelFactory, JetTagModel


def fromDict(config_dict: dict, folder: str, config_name: str, recreate: bool = True) -> JetTagModel:
    """Create a model directly from a dictionary 

    Args:
        config dict (dict): Config dictionary
        folder (str): Output saving folder for model
        recreate (bool, optional): Rewrite the output directory?. Defaults to True.

    Returns:
        JetTagModel: The model
    """

    # Create a model based on what is specified in the yaml 'model' field
    # Model must be registered for this to function
    model = JetModelFactory.create_JetTagModel(config_dict['model'], folder, config_dict)

    if recreate:
        # Remove output dir if exists
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Re-created existing directory: {folder}.")
        # Create dir to save results
        os.makedirs(folder)

        # Save the config dictionary to a yaml file in the output folder
        yaml_path = os.path.join(folder, f"{config_name}.yaml")
        with open(yaml_path, 'w') as yaml_file:
            yaml.dump(config_dict, yaml_file)

    return model

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
    model = JetModelFactory.create_JetTagModel(yaml_dict['model'], folder, yaml_dict)

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
