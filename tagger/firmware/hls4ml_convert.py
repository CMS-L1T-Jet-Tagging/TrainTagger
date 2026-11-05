from argparse import ArgumentParser

import keras
from keras.models import load_model
import hls4ml
from hgq.utils import trace_minmax

from tagger.data.tools import load_data, to_ML

    
def convert(model, outpath, project_name, build=True):
        
        input_precision = 'ap_fixed<24,12,AP_RND,AP_SAT>'
        class_precision = 'ap_ufixed<24,12,AP_RND,AP_SAT>'
        reg_precision = 'ap_fixed<16,6,AP_RND,AP_SAT>' 

        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config["Model"]["Strategy"]="distributed_arithmetic"
        config["Model"]["ReuseFactor"] = 1
        config['IOType'] = 'io_parallel'
        trace = True
        config['LayerName']['Conv1D_1']['ParallelizationFactor'] = 16
        config['LayerName']['Conv1D_2']['ParallelizationFactor'] = 16

        #for layer in model.layers:
         #   layer_name = layer.__class__.__name__
            #if layer_name in ["InputLayer"]:
                #config["LayerName"][layer.name]["Precision"] = input_precision
                #config["LayerName"][layer.name]["Trace"] = trace
                #config["LayerName"][layer.name]["result"] = input_precision

            #elif layer_name in ["jet_id_output"]:

                #config["LayerName"]["jet_id_output"]["Precision"]["result"] = class_precision
                #config["LayerName"]["jet_id_output"]["Implementation"] = "latency"
            #if layer_name in ["pT_output"]:
                #config["LayerName"]["pT_output"]["Precision"]["result"] = reg_precision
                #config["LayerName"]["pT_output"]["Implementation"] = "latency"
            #if layer_name in ["Conv1D_1","Conv1D_2"]:
             #   config['LayerName']['Conv1D_1']['ParallelizationFactor'] = 16
             #   config['LayerName']['Conv1D_2']['ParallelizationFactor'] = 16
           # else: 
          #              continue
             
        hls_model = hls4ml.converters.convert_from_keras_model(
            model,
            backend='Vitis',
            project_name=project_name,
            clock_period=2.8, #1/360MHz = 2.8ns
            hls_config=config,
            output_dir=f'{outpath}',
            part='xcvu9p-flga2104-2L-e'
        )

        hls_model.compile()
        hls_model.build(csim=False, reset=True)


def main():
    parser = ArgumentParser()
    parser.add_argument('--model-path', '-m', help='Path to the Keras model folder', required=True)
    parser.add_argument('--outpath', '-o', help='Path to the output folder', default='firmware')
    parser.add_argument('--data-path', '-d', type=str, help='Path to the training data folder', default=None)
    args = parser.parse_args()

    model = load_model(f"{args.model_path}/model/saved_model.h5")
    embedding_model = keras.Model(model.get_layer('model_input').input, model.get_layer('embedding').output)
    jet_id_model = keras.Model(model.get_layer('Dense_1_jetID').input , model.get_layer('jet_id_output').output)
    pt_model = keras.Model(model.get_layer('Dense_1_pT').input,   model.get_layer('pT_output').output)
    ad_model = load_model(f"{args.model_path}/model/saved_autoencoder.h5")

    if args.data_path is not None:
        data_train, data_test, class_labels, input_vars, extra_vars = load_data(args.data_path, 10, model)
        x_train, y_train, pt_target_train, _, reco_pt_train = to_ML(data_train, class_labels)
        z_train = embedding_model.predict(x_train)

        trace_minmax(embedding_model, x_train)
        trace_minmax(jet_id_model, z_train)
        trace_minmax(pt_model, z_train)
        trace_minmax(ad_model, z_train)

    precisions_embedding = convert(embedding_model, args.outpath + '/embedding')
    precisions_jet_id = convert(jet_id_model, args.outpath + '/jetid')
    precisions_pt = convert(pt_model, args.outpath + '/ptregress')
    precisions_ad = convert(ad_model, args.outpath + '/ad_head')