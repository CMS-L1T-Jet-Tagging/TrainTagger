from schema import Schema, And, Use, Optional, SchemaError
import sys
from pathlib import Path
import yaml

pathlist = Path("./").glob('*.yaml')
for path in pathlist:

    with open(str(path), 'r') as stream:
        yaml_dict = yaml.safe_load(stream)
    model = yaml_dict['model']

    training_config = {"weight_method" : And(str, lambda s: s in  ["none", "ptref", "onlyclass"]),
                       "validation_split" : And(float, lambda s: s > 0.0),
                       Optional("epochs" : And(int, lambda s: s >= 1)),
                       Optional("batch_size" : And(int, lambda s: s >= 1)),
                       Optional("learning_rate": And(float, lambda s: s > 0.0)),
                       Optional("loss_weights" : And(list, lambda s: len(s) == 2)),
                       Optional('EarlyStopping_patience' : And(int, lambda s: s > 0)),}


    if model == 'DeepSetModel':
        model_config = {"name" : str,
                        "conv1d_layers" : list,
                        "classification_layers" : list,
                        "regression_layers" : list,
                        "kernel_initializer" : str,
                        "aggregator" : And(str, lambda s: s in  ["mean", "max", "attention"])}
        
        quantization_config = {'quantizer_bits' : And(int, lambda s: 32 >= s >= 0),
                               'quantizer_bits_int' : And(int, lambda s: 32 >= s >= 0),
                               'quantizer_alpha_val' : And(float, lambda s: 1.0 >= s >= 0.0),
                               'pt_output_quantization' : list
                              }
        
        additional_training_config = { "initial_sparsity" : And(float, lambda s: 1.0 >= s >= 0.0),
                                       "final_sparsity" : And(float, lambda s: 1.0 >= s >= 0.0),
                                       "ReduceLROnPlateau_factor" : And(float, lambda s: 1.0 >= s >= 0.0),
                                       "ReduceLROnPlateau_patience" : int,
                                       "ReduceLROnPlateau_min_lr" : And(float, lambda s: s >= 0.0)}

    elif model == 'DeepSetModelHGQ':
        model_config = {"name" : str,
                        "conv1d_layers" : list,
                        "classification_layers" : list,
                        "regression_layers" : list,
                        "beta": And(float, lambda s: 1.0 >= s >= 0.0) }
        
        quantization_config = {'pt_output_quantization' : list}
        
        additional_training_config = { "ReduceLROnPlateau_factor" : And(float, lambda s: 1.0 >= s >= 0.0),
                                       "ReduceLROnPlateau_patience" : int,
                                       "ReduceLROnPlateau_min_lr" : And(float, lambda s: s >= 0.0)}
        
    elif model == 'InteractionNetModel':
        model_config = {"name" : str,
                        "effects_layers" : list,
                        "objects_layers" : list,
                        "classification_layers" : list,
                        "regression_layers" : list,
                        "kernel_initializer" : str,
                        "aggregator" : And(str, lambda s: s in  ["mean", "max", "attention"])}
        
        quantization_config = {"quantizer_bits" : And(int, lambda s: 32 >= s >= 0),
                               "quantizer_bits_int" : And(int, lambda s: 32 >= s >= 0),
                               "quantizer_alpha_val" : And(float, lambda s: 1.0 >= s >= 0.0),
                               "pt_output_quantization" : list
                            }
        
        additional_training_config = { "initial_sparsity" : And(float, lambda s: 1.0 >= s >= 0.0),
                                       "final_sparsity" : And(float, lambda s: 1.0 >= s >= 0.0),
                                       "ReduceLROnPlateau_factor" : And(float, lambda s: 1.0 >= s >= 0.0),
                                       "ReduceLROnPlateau_patience" : int,
                                       "ReduceLROnPlateau_min_lr" : And(float, lambda s: s >= 0.0)}
        
    else:
        model_config = {"name" : str}
        quantization_config = {}
        additional_training_config = {}

    schema = Schema(
            {
                "model": str,
                "run_config" : {"verbose" : And(int, lambda s: s in [1,2,3]),
                                "debug": bool,
                                "num_threads" : And(int, lambda s: 1 <= s <= 128)},
                "model_config" : model_config,
                "quantization_config" : quantization_config,
                "training_config" : training_config | additional_training_config,
                "hls4ml_config" : {"input_precision" : str,
                                "class_precision" : str,
                                "reg_precision": str,
                                "clock_period" : And(float, lambda s: 0.0 < s <= 10),
                                "fpga_part" : str,
                                "project_name" : str}
            }
    )
    
    schema.validate(yaml_dict)

    if not schema.is_valid(yaml_dict):
        print("Invalid yaml config:", path)
        exit()
    else:
        print("Valid yaml config:", path)