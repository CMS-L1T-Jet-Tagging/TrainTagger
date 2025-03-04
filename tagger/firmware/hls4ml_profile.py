from argparse import ArgumentParser
import os, shutil, json

#Import from other modules
from tagger.data.tools import make_data, load_data, to_ML, get_input_mask
from tagger.firmware.hls4ml_convert import convert
import tagger.train.models
from tagger.plot.common import plot_2d

#Third parties
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import hls4ml
from qkeras.utils import load_qmodel
from sklearn.metrics import roc_curve, auc,precision_recall_curve

import matplotlib.pyplot as plt
import mplhep as hep
import tagger.plot.style as style

style.set_style()

def doPlots(model,outputdir,inputdir):
    os.makedirs(outputdir, exist_ok=True)

    modelsAndNames = {"model":model}
    N_FILTERS = model.get_layer('avgpool').output_shape[1]

    data, _, class_labels, input_vars, extra_vars = load_data(inputdir, percentage=100,test_ratio=0.0)
    X_test, Y_test, pt_target, truth_pt, _ = to_ML(data, class_labels)

    labels = list(class_labels.keys())

    hls_model = convert(model,"temp",build=False)
    input_mask = get_input_mask(X_test, N_FILTERS)
    model_inp = np.concatenate([X_test, input_mask], axis=1)
    model_inp = np.ascontiguousarray(model_inp)
    y_hls, y_ptreg_hls = hls_model.predict(model_inp)
    y_class, y_ptreg = model.predict(model_inp)

    for i, label in enumerate(labels):
        plt.clf()
        min_x = min(np.amin(y_hls[:,i]), np.amin(y_class[:,i]))
        max_x = max(np.amax(y_hls[:,i]), np.amax(y_class[:,i]))
        figure = plot_2d(np.array(y_class[:,i]), np.array(y_hls[:,i]) ,(min_x,max_x),(min_x,max_x),"Tensorflow","hls4ml",style.CLASS_LABEL_STYLE[label]+" score")
        plt.savefig("%s/%s_score_2D.png" % (outputdir,label),bbox_inches='tight')
        plt.savefig("%s/%s_score_2D.pdf" % (outputdir,label),bbox_inches='tight')

    plt.clf()
    figure = plot_2d(y_ptreg[:,0] ,y_ptreg_hls[:,0],
                     ( min(np.amin(y_ptreg_hls), np.amin(y_ptreg)),max(np.amax(y_ptreg_hls), np.amax(y_ptreg))),
                     ( min(np.amin(y_ptreg_hls), np.amin(y_ptreg)),max(np.amax(y_ptreg_hls), np.amax(y_ptreg))),
                     "Tensorflow","hls4ml","Regression score")
    plt.savefig("%s/%s_score_2D.png" % (outputdir,"Regression"),bbox_inches='tight')
    plt.savefig("%s/%s_score_2D.pdf" % (outputdir,"Regression"),bbox_inches='tight')
    plt.close()

    wp, wph, ap, aph = hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=X_test)
    ap.savefig(outputdir+"/model_activations_profile.png")
    wp.savefig(outputdir+"/model_weights_profile.png")
    aph.savefig(outputdir+"/model_activations_profile_opt.png")
    wph.savefig(outputdir+"/model_weights_profile_opt.png")

    y_hls, hls4ml_trace = hls_model.trace(np.ascontiguousarray(X_test))
    keras_trace = hls4ml.model.profiling.get_ymodel_keras(model, X_test)

    for layer in hls4ml_trace.keys():
        print ("Doing profiling 2d for layer", layer)
        min_x = min(np.amin(hls4ml_trace[layer]), np.amin(keras_trace[layer]))
        max_x = max(np.amax(hls4ml_trace[layer]), np.amax(keras_trace[layer]))
        plot_2d(hls4ml_trace[layer].flatten() ,keras_trace[layer].flatten(),
                     (min_x,max_x),
                     ( min_x,max_x),
                     "hls4ml {}".format(layer),"Tensorflow  {}".format(layer),layer +" agreement")
        plt.plot([min_x, max_x], [min_x, max_x], c="gray")
        plt.savefig(f"{outputdir}/profile_2d_{layer}.png",bbox_inches='tight')
        plt.savefig(f"{outputdir}/profile_2d_{layer}.pdf",bbox_inches='tight')
        plt.close()





    return

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='output/baseline/model/saved_model.h5' , help = 'Input model path for comparison')
    parser.add_argument('-o','--outpath', default='output/baseline/plots/profile' , help = 'Jet tagger plotting directory')
    parser.add_argument('-i','--input', default='data/jetTuple_extended_5.root' , help = 'Path to profiling data rootfile')
    parser.add_argument('-r','--remake', default=False , help = 'Remake profiling data? ')

    args = parser.parse_args()

    #Load the model
    model=load_qmodel(args.model)
    print(model.summary())

    if args.remake:
        make_data(infile=args.input,outdir="profiling_data/",extras='extra_emulation_fields',tree="outnano/Jets")

    doPlots(model,args.outpath,"profiling_data/")
