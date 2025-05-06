import os, sys
import json
from argparse import ArgumentParser
from tagger.data.tools import make_data, load_data, to_ML
#Third party
import hls4ml
from qkeras.utils import load_qmodel
import mlflow
from pathlib import Path
import numpy as np

#HGQ
from tensorflow.keras.models import load_model
from HGQ import trace_minmax, to_proxy_model
#----------------------------------------------

def convert(model, outpath,build=True):

    #Remove the old directory if they exist
    os.system(f'rm -rf {outpath}')

    #Auxilary variables
    input_precision = 'ap_fixed<24,12,AP_RND,AP_SAT>'
    class_precision = 'ap_ufixed<24,12,AP_RND,AP_SAT>'
    reg_precision = 'ap_fixed<16,6>' 
    trace=True

    #Create default config
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['IOType'] = 'io_parallel'
    #config['LayerName']['input_1']['Precision']['result']  = input_precision

    #Configuration for conv1d layers
    #hls4ml automatically figures out the paralellization factor
    config['LayerName']['Conv1D_1']['ParallelizationFactor'] = 8
    config['LayerName']['Conv1D_2']['ParallelizationFactor'] = 8

    #Additional config
    for layer in model.layers:
        layer_name = layer.__class__.__name__

        if layer_name in ["BatchNormalization", "InputLayer"]:
            #config["LayerName"][layer.name]["Precision"] = input_precision
            #config["LayerName"][layer.name]["result"] = input_precision
            config["LayerName"][layer.name]["Trace"] = trace

        elif layer_name in ["Permute","Concatenate","Flatten","Reshape","UpSampling1D","Add"]:
            print("Skipping trace for:", layer.name)
        else:
            config["LayerName"][layer.name]["Trace"] = trace

    
    #config["LayerName"]["act_jet"]["Precision"]["result"] = class_precision
    #config["LayerName"]["act_jet"]["Implementation"] = "latency"
    #config["LayerName"]["pT_out"]["Precision"]["result"] = reg_precision
    #config["LayerName"]["pT_out"]["Implementation"] = "latency"

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
    #model_dir="output/baseline_pruned"
    #X_test = np.load(f"{model_dir}/testing_data/X_test.npy")
    #y_test = np.load(f"{model_dir}/testing_data/y_test.npy")
    #truth_pt_test = np.load(f"{model_dir}/testing_data/truth_pt_test.npy")
    #reco_pt_test = np.load(f"{model_dir}/testing_data/reco_pt_test.npy")
    #indices = np.random.permutation(16) 
    #print(indices) 
    #X_test = X_test[:, indices, :] 
    #model_outputs = model.predict(X_test)

    #Get classification outputs
    #y_pred = model_outputs[0]
    #pt_ratio = model_outputs[1].flatten()

   
    #print(y_pred[0],y_test[0])

    if build == True:
        #hls_model.build(csim=False, reset = True)
        return [input_precision,class_precision,reg_precision]
    else:
        return hls_model

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='deepset_HGQ' , help = 'model name')    
    parser.add_argument('-o','--outpath', default='tagger/firmware/L1TSC4NGJetModel' , help = 'Jet tagger synthesized output directory')  
    
    args = parser.parse_args()
    path=f'output/{args.model}/model/saved_model.h5'
    if args.model=="baseline_pruned":
        model = load_qmodel(path)
        precisions = convert(model,args.outpath)
    else:
        data_train, data_test, class_labels, input_vars, extra_vars = load_data("training_data/", percentage=0.1)
        X_train, y_train, pt_target_train, truth_pt_train, reco_pt_train = to_ML(data_train, class_labels)
        model = load_qmodel(path)
        print(model.summary())
        trace_minmax(model, X_train, cover_factor=1.0)
        proxy = to_proxy_model(model, aggressive=False)
        precisions = convert(proxy,args.outpath)



    