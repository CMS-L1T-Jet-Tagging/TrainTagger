import tagger.model
from tagger.model.models import JetModelFactory
import os,yaml,shutil

def fromYaml(yaml_path,folder,recreate=True):
    with open(yaml_path, 'r') as stream:
        yaml_dict = yaml.safe_load(stream)
    model = JetModelFactory.create_JetTagModel(yaml_dict['model'],folder)
    model.load_yaml(yaml_path)
    if recreate:
        # Remove output dir if exists
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Re-created existing directory: {folder}.")
            # Create dir to save results
        os.makedirs(folder)
        os.system('cp '+yaml_path+' '+folder)
    return model

def fromFolder(save_path,newoutput_dir=None):
    if newoutput_dir != None:
        folder = newoutput_dir
        recreate = True
    else:
        folder = save_path
        recreate = False
    
    for file in os.listdir(folder):
        if file.endswith(".yaml"):
            yaml_path = os.path.join(folder, file)
    
    model = fromYaml(yaml_path,folder,recreate=recreate)
    model.load(folder)
    return model
