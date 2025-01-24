from argparse import ArgumentParser
import os, shutil, json

#Import from other modules
from tagger.data.tools import make_data, load_data, to_ML
from tagger.firmware.hls4ml_convert import convert
import tagger.train.models
import tagger.plot.common as common

#Third parties
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import hls4ml
from qkeras.utils import load_qmodel
from sklearn.metrics import roc_curve, auc,precision_recall_curve

#Plotting
import matplotlib.pyplot as plt
import mplhep as hep
import tagger.plot.style as style

style.set_style()


def rms(array):
   return np.sqrt(np.mean(array ** 2))


def doPlots(model,outputdir,inputdir):
    os.makedirs(outputdir, exist_ok=True)

    modelsAndNames = {"model":model}
    
    data, _, class_labels, input_vars, extra_vars = load_data(inputdir, percentage=100,test_ratio=0.0)
    X_test, Y_test, pt_target, truth_pt, _ = to_ML(data, class_labels) #Last thing was reconstructed pt

    labels = list(class_labels.keys())

    hls_model = convert(model,"temp",build=False)

    y_hls, y_ptreg_hls = hls_model.predict(np.ascontiguousarray(X_test))
    y_class, y_ptreg = model.predict(np.ascontiguousarray(X_test))
    
    jet_pt_phys= np.array(data['jet_pt_phys'])

    modelsAndNames["Y_predict"] = y_class
    modelsAndNames["Y_predict_reg"] = y_ptreg

    modelsAndNames["Y_hls_predict"] = y_hls
    modelsAndNames["Y_hls_predict_reg"] = y_ptreg_hls

    jet_pt_cor_reg = jet_pt_phys * modelsAndNames["Y_predict_reg"][:,0]
    jet_pt_cor_reg_hls = jet_pt_phys * modelsAndNames["Y_hls_predict_reg"][:,0]
    jet_pt_cor_reg_emu = jet_pt_phys * np.array(data['jet_multijetscore_regression'])

    figure = common.plot_2d(np.array(modelsAndNames["Y_predict_reg"][:,0]) ,np.array(data['jet_multijetscore_regression']) ,(0,2),(0,2),"Tensorflow","CMSSW Emulation","Jet Regression")
    plt.savefig("%s/jetRegression_2D.png" % outputdir,bbox_inches='tight')
    plt.savefig("%s/jetRegression_2D.pdf" % outputdir,bbox_inches='tight')

    plt.clf()
    figure = common.plot_histo([modelsAndNames["Y_predict_reg"][:,0],np.array(data['jet_multijetscore_regression']),np.array(modelsAndNames["Y_hls_predict_reg"][:,0])],["Tensorflow","CMSSW Emulation", "hls4ml"],"",'Regression Output','a.u.',range=(0,2))
    plt.savefig("%s/jetRegression_1D.png" % outputdir,bbox_inches='tight')
    plt.savefig("%s/jetRegression_1D.pdf" % outputdir,bbox_inches='tight')

    for i, label in enumerate(labels):
        plt.close()
        plt.clf()
        figure = common.plot_histo([np.array(modelsAndNames['Y_predict'][:,i]),np.array(data['jet_multijetscore_'+label]),np.array(modelsAndNames['Y_hls_predict'][:,i])],["Tensorflow","CMSSW Emulation", "hls4ml"],"",style.CLASS_LABEL_STYLE[label]+' score','a.u.',range=(0,1))
        plt.savefig("%s/%s_score_1D.png" % (outputdir,label),bbox_inches='tight')
        plt.savefig("%s/%s_score_1D.pdf" % (outputdir,label),bbox_inches='tight')

        plt.clf()
        figure = common.plot_2d(np.array(modelsAndNames['Y_predict'][:,i]),np.array(data['jet_multijetscore_'+label]),(0,1),(0,1),"Tensorflow","CMSSW Emulation",style.CLASS_LABEL_STYLE[label]+" score")
        plt.savefig("%s/%s_score_2D.png" % (outputdir,label),bbox_inches='tight')
        plt.savefig("%s/%s_score_2D.pdf" % (outputdir,label),bbox_inches='tight')

    fpr = {}
    tpr = {}
    auc1 = {}
    thresholds = {}
    # Loop over classes (labels) to get metrics per class
    for i, label in enumerate(labels):
        fpr[label], tpr[label], thresholds[label] = roc_curve(Y_test[:,i], modelsAndNames["Y_predict"][:,i])
        auc1[label] = auc(fpr[label], tpr[label])

    modelsAndNames["Tensorflow"] = {}
    modelsAndNames["Tensorflow"]["ROCs"] = {}
    modelsAndNames["Tensorflow"]["ROCs"]["tpr"] = tpr
    modelsAndNames["Tensorflow"]["ROCs"]["fpr"] = fpr
    modelsAndNames["Tensorflow"]["ROCs"]["auc"] = auc1

    fpr = {}
    tpr = {}
    auc1 = {}
    thresholds = {}
    for i, label in enumerate(labels):
        fpr[label], tpr[label], thresholds[label] = roc_curve(Y_test[:,i], modelsAndNames["Y_hls_predict"][:,i])
        auc1[label] = auc(fpr[label], tpr[label])

    modelsAndNames["hls4ml"] = {}
    modelsAndNames["hls4ml"]["ROCs"] = {}
    modelsAndNames["hls4ml"]["ROCs"]["tpr"] = tpr
    modelsAndNames["hls4ml"]["ROCs"]["fpr"] = fpr
    modelsAndNames["hls4ml"]["ROCs"]["auc"] = auc1

    fpr = {}
    tpr = {}
    auc1 = {}
    thresholds = {}
    # Get emulation ROCs
    for i, label in enumerate(labels):
        fpr[label], tpr[label], thresholds[label] = roc_curve(Y_test[:,i], data['jet_multijetscore_'+label])
        auc1[label] = auc(fpr[label], tpr[label])

    modelsAndNames["Emulation"] = {}
    modelsAndNames["Emulation"]["ROCs"] = {}
    modelsAndNames["Emulation"]["ROCs"]["tpr"] = tpr
    modelsAndNames["Emulation"]["ROCs"]["fpr"] = fpr
    modelsAndNames["Emulation"]["ROCs"]["auc"] = auc1

    #===========================#

    for i, label in enumerate(labels):
        plt.close()
        common.plot_roc(modelsAndNames,label,title=style.CLASS_LABEL_STYLE[label]+" ROC Comparison")
        plt.savefig(outputdir+"/ROC_Emulation_comparison_"+label+".png",bbox_inches='tight')
        plt.savefig(outputdir+"/ROC_Emulation_comparison_"+label+".pdf",bbox_inches='tight')

    response_reg = jet_pt_cor_reg / data['jet_genmatch_pt']
    response_emu = jet_pt_cor_reg_emu / data['jet_genmatch_pt']
    response_hls = jet_pt_cor_reg_hls / data['jet_genmatch_pt']

    figure = common.plot_histo([response_reg,response_emu,response_hls],
                        ["Tensorflow" + " median: "+str(np.round(np.median(response_reg),3))+" rms: "+str(np.round(rms(response_reg),3)),
                         "Emulation" + " median: "+str(np.round(np.median(response_emu),3))+" rms: "+str(np.round(rms(response_emu),3)),
                         "hls4ml" + " median: "+str(np.round(np.median(response_hls),3))+" rms: "+str(np.round(rms(response_hls),3)),],
                        "Jet Regression",'Jet Response (reco/gen)','a.u.',range=(0,2))
    plt.savefig(outputdir+"/response_emulation"+".png",bbox_inches='tight')
    plt.savefig(outputdir+"/response_emulation"+".pdf",bbox_inches='tight')
    plt.close()
    return

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='output/baseline/model/saved_model.h5' , help = 'Input model path for comparison')    
    parser.add_argument('-o','--outpath', default='output/baseline/plots/emulation' , help = 'Jet tagger plotting directory')    
    parser.add_argument('-i','--input', default='data/jetTuple_extended_5.root' , help = 'Path to emulation data rootfile')
    parser.add_argument('-r','--remake', default=False , help = 'Remake emulation data? ')

    args = parser.parse_args()

    #Load the model
    model=load_qmodel(args.model)
    print(model.summary())

    if args.remake:
        make_data(infile=args.input,outdir="emulation_data/",extras='extra_emulation_fields')

    doPlots(model,args.outpath,"emulation_data/")

    f = open("mlflow_run_id.txt", "r")
    run_id = (f.read())
    mlflow.log_artifacts("output/baseline/plots/emulation",artifact_path="plots/emulation",run_id=run_id)