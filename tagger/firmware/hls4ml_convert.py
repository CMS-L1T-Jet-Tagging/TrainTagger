import os, sys
import json
from argparse import ArgumentParser

#Third party
import hls4ml
from qkeras.utils import load_qmodel

#----------------------------------------------

def convert(model, outpath,build=True):

    #Remove the old directory if they exist
    os.system(f'rm -rf {outpath}')

    #Auxilary variables
    input_precision = 'ap_fixed<15,12,AP_RND,AP_SAT>'
    class_precision = 'ap_ufixed<8,0,AP_RND,AP_SAT>'
    reg_precision = 'ap_fixed<16,6,AP_RND,AP_SAT>' 
    trace=True

    #Create default config
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
  
    config['IOType'] = 'io_parallel'

    config['Model']['Precision'] = 'fixed<64,32>'

    #config['LayerName']['jet_id_output']['inv_table_t'] = 'ap_fixed<18,8>'
    #config['LayerName']['jet_id_output']['exp_table_t'] = 'ap_fixed<18,8>'
    #config['LayerName']['jet_id_output']['table_t'] = 'ap_fixed<18,8>'
    #config['LayerName']['jet_id_output']['table_size'] = 65536

    #weight_summary = hls4ml.model.profiling.weights_keras(model=weightmodel,fmt="summary")

    # config['LayerName']['model_input']['Precision']['weight'] = 'fixed<18,12>'
    # config['LayerName']['model_input']['Precision']['bias'] = 'fixed<18,12>'
    # config['LayerName']['model_input']['Precision']['result'] = 'fixed<18,12>'

    # config['LayerName']['norm_input']['Precision']['weight'] = 'fixed<11,4>'
    # config['LayerName']['norm_input']['Precision']['bias'] = 'fixed<7,1>'
    # config['LayerName']['norm_input']['Precision']['result'] = 'fixed<25,11>'

    # config['LayerName']['Conv1D_1']['Precision']['weight'] = 'fixed<9,3>'
    # config['LayerName']['Conv1D_1']['Precision']['bias'] = 'fixed<9,3>'
    # config['LayerName']['Conv1D_1']['Precision']['result'] = 'fixed<33,24>'

    # config['LayerName']['Conv1D_1_linear']['Precision']['result']  = 'fixed<32,24>'

    # config['LayerName']['relu_1']['Precision']['result'] = 'ufixed<32,0,AP_RND,AP_SAT>'

    # config['LayerName']['Conv1D_2']['Precision']['weight'] = 'fixed<9,3>'
    # config['LayerName']['Conv1D_2']['Precision']['bias'] = 'fixed<9,3>'
    # config['LayerName']['Conv1D_2']['Precision']['result'] = 'fixed<20,4>'

    # config['LayerName']['Conv1D_2_linear']['Precision']['result']  = 'fixed<20,3>'

    # config['LayerName']['relu_2']['Precision']['result'] = 'ufixed<32,0,AP_RND,AP_SAT>'

    # config['LayerName']['act_pool']['Precision']['result'] = 'ufixed<32,0,AP_RND,AP_SAT>'

    # config['LayerName']['avgpool']['Precision']['result'] = 'ufixed<32,0,AP_RND,AP_SAT>'

    # config['LayerName']['Dense_1_jetID']['Precision']['weight'] = 'fixed<9,3>'
    # config['LayerName']['Dense_1_jetID']['Precision']['bias'] = 'fixed<9,3>'
    # config['LayerName']['Dense_1_jetID']['Precision']['result'] = 'fixed<22,3>'

    # config['LayerName']['Dense_1_jetID_linear']['Precision']['result']  = 'fixed<22,3>'

    # config['LayerName']['relu_1_jetID']['Precision']['result'] = 'ufixed<32,0,AP_RND,AP_SAT>'

    # config['LayerName']['Dense_2_jetID']['Precision']['weight'] = 'fixed<9,3>'
    # config['LayerName']['Dense_2_jetID']['Precision']['bias'] = 'fixed<9,3>'
    # config['LayerName']['Dense_2_jetID']['Precision']['result'] = 'fixed<22,5>'

    # config['LayerName']['Dense_2_jetID_linear']['Precision']['result']  = 'fixed<25,5>'

    # config['LayerName']['relu_2_jetID']['Precision']['result'] = 'ufixed<32,0,AP_RND,AP_SAT>'

    # config['LayerName']['Dense_1_pT']['Precision']['weight'] = 'fixed<9,3>'
    # config['LayerName']['Dense_1_pT']['Precision']['bias'] = 'fixed<9,3>'
    # config['LayerName']['Dense_1_pT']['Precision']['result'] = 'fixed<20,3>'

    # config['LayerName']['Dense_1_pT_linear']['Precision']['result']  = 'fixed<20,3>'

    # config['LayerName']['Dense_3_jetID']['Precision']['weight'] = 'fixed<9,3>'
    # config['LayerName']['Dense_3_jetID']['Precision']['bias'] = 'fixed<9,3>'
    # config['LayerName']['Dense_3_jetID']['Precision']['result'] = 'fixed<21,5>'

    # config['LayerName']['Dense_3_jetID_linear']['Precision']['result'] = 'fixed<21,5>'

    # config['LayerName']['relu_1_pt']['Precision']['result'] = 'ufixed<32,0,AP_RND,AP_SAT>'
    
    #config['LayerName']['jet_id_output']['Precision']['result'] = 'ufixed<32,0,AP_RND,AP_SAT>'

    # config['LayerName']['pT_output']['Precision']['weight'] = 'fixed<6,1>'
    # config['LayerName']['pT_output']['Precision']['bias'] = 'fixed<6,1>'
    # config['LayerName']['pT_output']['Precision']['result'] = 'fixed<16,2>'

    # config['LayerName']['pT_output_linear']['Precision']['result'] = 'fixed<16,2>'

    # config['LayerName']['model_input']['Precision']['weight'] = 'fixed<64,32>'
    # config['LayerName']['model_input']['Precision']['bias'] = 'fixed<64,32>'
    # config['LayerName']['model_input']['Precision']['result'] = 'fixed<64,32>'

    # config['LayerName']['norm_input']['Precision']['weight'] = 'fixed<64,32>'
    # config['LayerName']['norm_input']['Precision']['bias'] = 'fixed<64,32>'
    # config['LayerName']['norm_input']['Precision']['result'] ='fixed<64,32>'

    # config['LayerName']['Conv1D_1']['Precision']['weight'] = 'fixed<64,32>'
    # config['LayerName']['Conv1D_1']['Precision']['bias'] = 'fixed<64,32>'
    # config['LayerName']['Conv1D_1']['Precision']['result'] = 'fixed<64,32>'

    # config['LayerName']['Conv1D_1_linear']['Precision']['result']  ='fixed<64,32>'

    # config['LayerName']['relu_1']['Precision']['result'] = 'ufixed<64,0,AP_RND_ZERO,AP_SAT_SYM>'

    # config['LayerName']['Conv1D_2']['Precision']['weight'] = 'fixed<64,32>'
    # config['LayerName']['Conv1D_2']['Precision']['bias'] = 'fixed<64,32>'
    # config['LayerName']['Conv1D_2']['Precision']['result'] = 'fixed<64,32>'

    # config['LayerName']['Conv1D_2_linear']['Precision']['result']  = 'fixed<64,32>'

    # config['LayerName']['relu_2']['Precision']['result'] = 'ufixed<1,64>'

    # config['LayerName']['act_pool']['Precision']['result'] = 'ufixed<1,64>'

    # config['LayerName']['avgpool']['Precision']['result'] = 'ufixed<1,64>'

    # config['LayerName']['Dense_1_jetID']['Precision']['weight'] = 'fixed<64,32>'
    # config['LayerName']['Dense_1_jetID']['Precision']['bias'] = 'fixed<64,32>'
    # config['LayerName']['Dense_1_jetID']['Precision']['result'] ='fixed<64,32>'

    # config['LayerName']['Dense_1_jetID_linear']['Precision']['result']  = 'fixed<64,32>'

    # config['LayerName']['relu_1_jetID']['Precision']['result'] = 'ufixed<1,64>'

    # config['LayerName']['Dense_2_jetID']['Precision']['weight'] = 'fixed<64,32>'
    # config['LayerName']['Dense_2_jetID']['Precision']['bias'] = 'fixed<64,32>'
    # config['LayerName']['Dense_2_jetID']['Precision']['result'] = 'fixed<64,32>'

    # config['LayerName']['Dense_2_jetID_linear']['Precision']['result']  = 'fixed<64,32>'

    # config['LayerName']['relu_2_jetID']['Precision']['result'] = 'ufixed<1,64>'

    # config['LayerName']['Dense_1_pT']['Precision']['weight'] = 'fixed<64,32>'
    # config['LayerName']['Dense_1_pT']['Precision']['bias'] = 'fixed<64,32>'
    # config['LayerName']['Dense_1_pT']['Precision']['result'] = 'fixed<64,32>'

    # config['LayerName']['Dense_1_pT_linear']['Precision']['result']  = 'fixed<64,32>'

    # config['LayerName']['Dense_3_jetID']['Precision']['weight'] = 'fixed<64,32>'
    # config['LayerName']['Dense_3_jetID']['Precision']['bias'] ='fixed<64,32>'
    # config['LayerName']['Dense_3_jetID']['Precision']['result'] ='fixed<64,32>'

    # config['LayerName']['Dense_3_jetID_linear']['Precision']['result'] = 'fixed<64,32>'

    # config['LayerName']['relu_1_pt']['Precision']['result'] = 'ufixed<1,64>'
    
    # config['LayerName']['jet_id_output']['Precision']['result'] = 'ufixed<64,0>'

    # config['LayerName']['pT_output']['Precision']['weight'] ='fixed<64,32>'
    # config['LayerName']['pT_output']['Precision']['bias'] = 'fixed<64,32>'
    # config['LayerName']['pT_output']['Precision']['result'] = 'fixed<64,32>'

    # config['LayerName']['pT_output_linear']['Precision']['result'] = 'fixed<64,32>'

    #Configuration for conv1d layers
    #hls4ml automatically figures out the paralellization factor
    config['LayerName']['Conv1D_1']['ParallelizationFactor'] = 10
    config['LayerName']['Conv1D_2']['ParallelizationFactor'] = 10

    #Additional config
    for layer in model.layers:
        layer_name = layer.__class__.__name__
        #config["LayerName"][layer.name]['Precision'] = 'fixed<64,32>'

        if layer_name in ["BatchNormalization", "InputLayer"]:
            #config["LayerName"][layer.name]["Precision"] = input_precision
            #config["LayerName"][layer.name]["result"] = input_precision
            config["LayerName"][layer.name]["Trace"] = trace

        # elif layer_name in ["Permute","Concatenate","Flatten","Reshape","UpSampling1D","Add"]:
        #     print("Skipping trace for:", layer.name)
        # else:
        config["LayerName"][layer.name]["Trace"] = trace
    
    #config["LayerName"]["jet_id_output"]["Precision"]["result"] = class_precision
    config["LayerName"]["jet_id_output"]["Implementation"] = "latency"
    #config["LayerName"]["pT_output"]["Precision"]["result"] = reg_precision
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
        return
    else:
        return hls_model

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='output/baseline/model/saved_model.h5' , help = 'Input model path for conversion')    
    parser.add_argument('-o','--outpath', default='tagger/firmware/JetTaggerNN' , help = 'Jet tagger synthesized output directory')    

    args = parser.parse_args()

    #Load the model
    model=load_qmodel(args.model)
    print(model.summary())

    convert(model, args.outpath)