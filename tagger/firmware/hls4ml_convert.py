import os, sys
import json
from argparse import ArgumentParser

#Third party
import hls4ml
from qkeras.utils import load_qmodel
import mlflow

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
                                                       project_name='JetTaggerNN',
                                                       clock_period=2.8, #1/360MHz = 2.8ns
                                                       hls_config=config,
                                                       output_dir=f'{outpath}',
                                                       part='xcvu9p-flga2104-2L-e')


    #Compile and build the project
    hls_model.compile()
    if build == True:
        hls_model.build(csim=False, reset = True)
        report = hls_model.read_report()
        return report
    else:
        return hls_model

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='output/baseline/model/saved_model.h5' , help = 'Input model path for conversion')    
    parser.add_argument('-o','--outpath', default='tagger/firmware/JetTaggerNN' , help = 'Jet tagger synthesized output directory')    
    parser.add_argument('-n','--name', default='baseline', help = 'Model experiment name')

    args = parser.parse_args()

    #Load the model
    model=load_qmodel(args.model)
    print(model.summary())

    f = open("run_id.txt", "r")
    run_id = (f.read())
    mlflow.get_experiment_by_name(args.model)
    with mlflow.start_run(experiment_id=1,
                        run_name=str(args.name),
                        run_id=run_id # pass None to start a new run
                        ):
        report = convert(model, args.outpath)
        print(report)
        mlflow.log_metric('FF %',report)
        mlflow.log_metric('LUT %',report)
        mlflow.log_metric('BRAM %',report)
        mlflow.log_metric('DSP %',report)
        mlflow.log_metric('Latency ',report)
        mlflow.log_metric('Initiation Interval ',report)

    