from argparse import ArgumentParser
import os, shutil, json

#Import from other modules
from tagger.data.tools import make_data, load_data, to_ML
from tagger.firmware.hls4ml_convert import convert
import tagger.train.models
from tagger.plot.makeEmulationPlot import plot_2d

#Third parties
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import hls4ml
from qkeras.utils import load_qmodel
import mplhep as hep
from sklearn.metrics import roc_curve, auc,precision_recall_curve
import matplotlib.pyplot as plt



# Setup plotting to CMS style
hep.cms.label()
hep.cms.text("Simulation")
plt.style.use(hep.style.CMS)

SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 35

LEGEND_WIDTH = 20
LINEWIDTH = 3
MARKERSIZE = 20

colormap = "jet"

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('axes', linewidth=LINEWIDTH+2)              # thickness of axes
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE-2)            # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


import matplotlib

matplotlib.rcParams['xtick.major.size'] = 20
matplotlib.rcParams['xtick.major.width'] = 5
matplotlib.rcParams['xtick.minor.size'] = 10
matplotlib.rcParams['xtick.minor.width'] = 4

matplotlib.rcParams['ytick.major.size'] = 20
matplotlib.rcParams['ytick.major.width'] = 5
matplotlib.rcParams['ytick.minor.size'] = 10
matplotlib.rcParams['ytick.minor.width'] = 4

#colours=["green","red","blue","black","orange","purple","goldenrod"]
colours = ["black","red","orange","green", "blue"]
linestyles = ["-","--","dotted",(0, (3, 5, 1, 5)),(0, (3, 5, 1,1,1,5,)),(0, (3, 10, 1, 10)),(0, (3, 10, 1, 10, 1, 10))]


def doPlots(model,outputdir,inputdir):
    os.makedirs(outputdir, exist_ok=True)

    modelsAndNames = {"model":model}
    
    data, _, class_labels, input_vars, extra_vars = load_data(inputdir, percentage=100,test_ratio=0.0)
    X_test, Y_test, pt_target, truth_pt, _ = to_ML(data, class_labels)

    labels = list(class_labels.keys())

    hls_model = convert(model,"temp",build=False) 
    y_hls, y_ptreg_hls = hls_model.predict(np.ascontiguousarray(X_test))
    y_class, y_ptreg = model.predict(np.ascontiguousarray(X_test))

    for i, label in enumerate(labels):
        plt.clf()
        min_x = min(np.amin(y_hls[:,i]), np.amin(y_class[:,i]))
        max_x = max(np.amax(y_hls[:,i]), np.amax(y_class[:,i]))
        figure = plot_2d(np.array(y_class[:,i]), np.array(y_hls[:,i]) ,(min_x,max_x),(min_x,max_x),"Tensorflow","hls4ml",label+" score")
        plt.savefig("%s/%s_score_2D.png" % (outputdir,label))

    plt.clf()
    figure = plot_2d(y_ptreg[:,0] ,y_ptreg_hls[:,0],
                     ( min(np.amin(y_ptreg_hls), np.amin(y_ptreg)),max(np.amax(y_ptreg_hls), np.amax(y_ptreg))),
                     ( min(np.amin(y_ptreg_hls), np.amin(y_ptreg)),max(np.amax(y_ptreg_hls), np.amax(y_ptreg))),
                     "Tensorflow","hls4ml","Regression score")
    plt.savefig("%s/%s_score_2D.png" % (outputdir,"Regression"))
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
        fig,ax = plt.subplots(1,1,figsize=(18,15))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        min_x = min(np.amin(hls4ml_trace[layer]), np.amin(keras_trace[layer]))
        max_x = max(np.amax(hls4ml_trace[layer]), np.amax(keras_trace[layer]))
        hist2d = ax.hist2d(hls4ml_trace[layer].flatten(), keras_trace[layer].flatten(), bins=50, range=((min_x,max_x),(min_x,max_x)), norm=matplotlib.colors.LogNorm(),cmap='jet')    
        plt.plot([min_x, max_x], [min_x, max_x], c="gray")
        ax.set_xlabel("hls4ml {}".format(layer), horizontalalignment='right', x=1.0)
        ax.set_ylabel("Tensorflow  {}".format(layer), horizontalalignment='right', y=1.0)
        plt.savefig(f"{outputdir}/profile_2d_{layer}.png")
        plt.close()





    return

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='output/baseline/model/saved_model.h5' , help = 'Input model path for comparison')    
    parser.add_argument('-o','--outpath', default='output/baseline/plots/profile' , help = 'Jet tagger plotting directory')    
    parser.add_argument('-i','--input', default='data/jetTuple.root' , help = 'Path to profiling data rootfile')
    parser.add_argument('-r','--remake', default=False , help = 'Remake profiling data? ')

    args = parser.parse_args()

    #Load the model
    model=load_qmodel(args.model)
    print(model.summary())

    if args.remake:
        make_data(infile=args.input,outdir="profiling_data/",extras='extra_emulation_fields')

    doPlots(model,args.outpath,"profiling_data/")