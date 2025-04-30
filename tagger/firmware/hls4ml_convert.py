import os, sys
import json
from argparse import ArgumentParser

#Third party
import hls4ml
from keras.models import load_model
import hgq
import mlflow
from pathlib import Path
import numpy as np
#----------------------------------------------

def convert(model, outpath,build=True):

    #Remove the old directory if they exist
    os.system(f'rm -rf {outpath}')

    #Auxilary variables
    input_precision = 'ap_fixed<24,12,AP_RND,AP_SAT>'
    class_precision = 'ap_ufixed<24,12,AP_RND,AP_SAT>'
    reg_precision = 'ap_fixed<16,6,AP_RND,AP_SAT>' 
    trace=True

    #Create default config
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['IOType'] = 'io_parallel'
    config['LayerName']['model_input']['Precision']['result'] = input_precision

    #Configuration for conv1d layers
    #hls4ml automatically figures out the paralellization factor
    config['LayerName']['Conv1D_1']['ParallelizationFactor'] = 8
    config['LayerName']['Conv1D_2']['ParallelizationFactor'] = 8

    #Additional config
    for layer in model.layers:
        layer_name = layer.__class__.__name__

        if layer_name in ["BatchNormalization", "InputLayer"]:
            config["LayerName"][layer.name]["Precision"] = input_precision
            config["LayerName"][layer.name]["result"] = input_precision
            config["LayerName"][layer.name]["Trace"] = trace

        elif layer_name in ["Permute","Concatenate","Flatten","Reshape","UpSampling1D","Add"]:
            print("Skipping trace for:", layer.name)
        else:
            config["LayerName"][layer.name]["Trace"] = trace

    
    config["LayerName"]["jet_id_output"]["Precision"]["result"] = class_precision
    config["LayerName"]["jet_id_output"]["Implementation"] = "latency"
    config["LayerName"]["pT_output"]["Precision"]["result"] = reg_precision
    config["LayerName"]["pT_output"]["Implementation"] = "latency"

    #Save config  as json file
    print("Saving default config as config.json ...")
    with open('tagger/firmware/config.json', 'w') as fp: json.dump(config, fp)

    #Write HLS
    hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                       backend='Vitis',
                                                       project_name='L1TSC4NGJetModel',
                                                       clock_period=2.8, #1/360MHz = 2.8ns
                                                       hls_config=config,
                                                       output_dir=f'{outpath}',
                                                       part='xcvu9p-flga2104-2L-e')


    #Compile and build the project
    hls_model.compile()
    if build == True:
        #hls_model.build(csim=False, reset = True)
        return [input_precision,class_precision,reg_precision]
    else:
        return hls_model

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='output/baseline/model/saved_model.keras' , help = 'Input model path for conversion')    
    parser.add_argument('-o','--outpath', default='tagger/firmware/L1TSC4NGJetModel' , help = 'Jet tagger synthesized output directory')    
    args = parser.parse_args()

    #Load the model
    model = load_model(args.model)
    precisions = convert(model,args.outpath)


    