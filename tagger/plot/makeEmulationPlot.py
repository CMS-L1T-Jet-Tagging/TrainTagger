from argparse import ArgumentParser
import os, shutil, json

#Import from other modules
from tagger.data.tools import make_data, load_data, to_ML
from tagger.firmware.hls4ml_convert import convert
import tagger.train.models

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

def plot_2d(variable_one,variable_two,range_one,range_two,name_one,name_two,title):
    fig,ax = plt.subplots(1,1,figsize=(18,15))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    hist2d = ax.hist2d(variable_one, variable_two, range=(range_one,range_two), bins=50, norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax.set_xlabel(name_one, horizontalalignment='right', x=1.0)
    ax.set_ylabel(name_two, horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Tracks')
    #ax.vlines(0,-20,20,linewidth=3,linestyle='dashed',color='k')
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def plot_histo(variable,name,title,xlabel,ylabel,range=(0,1)):
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(18,15))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    for i,histo in enumerate(variable):

        ax.hist(histo,bins=50,range=range,histtype="step",
                    linewidth=LINEWIDTH,
                    color = colours[i],
                    label=name[i],
                    density=True)    
    ax.grid(True)
    ax.set_xlabel(xlabel,ha="right",x=1)
    ax.set_ylabel(ylabel,ha="right",y=1)
    ax.legend(loc='best')

    plt.suptitle(title)
    plt.tight_layout()
    return fig

def plot_roc(modelsAndNames,truthclass,keys = ["Emulation","Tensorflow","hls4ml"],labels = ["CMSSW Emulation", "Tensorflow", "hls4ml"],title="None",colours=colours):
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(18,15))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)

    for i,key in enumerate(keys):
        tpr = modelsAndNames[key]["ROCs"]["tpr"]
        fpr = modelsAndNames[key]["ROCs"]["fpr"]
        auc1 = modelsAndNames[key]["ROCs"]["auc"]
        ax.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(labels[i], auc1[truthclass]*100.),linewidth=LINEWIDTH,color=colours[i])
    ax.semilogy()
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Mistag rate")
    ax.set_xlim(0.,1.)
    ax.set_ylim(0.001,1)
    ax.grid(True)
    ax.legend(loc='best')
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def rms(array):
   return np.sqrt(np.mean(array ** 2))


def doPlots(model,outputdir,inputdir):
    os.makedirs(outputdir, exist_ok=True)

    modelsAndNames = {"model":model}
    
    data, _, class_labels, input_vars, extra_vars = load_data(inputdir, percentage=100,test_ratio=0.0)
    X_test, Y_test, pt_target, truth_pt = to_ML(data, class_labels)

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

    figure = plot_2d(np.array(modelsAndNames["Y_predict_reg"][:,0]) ,np.array(data['jet_multijetscore_regression']) ,(0,2),(0,2),"Tensorflow","CMSSW Emulation","Jet Regression")
    plt.savefig("%s/jetRegression_2D.png" % outputdir)

    plt.clf()
    figure = plot_histo([modelsAndNames["Y_predict_reg"][:,0],np.array(data['jet_multijetscore_regression']),np.array(modelsAndNames["Y_hls_predict_reg"][:,0])],["Tensorflow","CMSSW Emulation", "hls4ml"],"Jet Regression",'Regression Score','# Jets',range=(0,2))
    plt.savefig("%s/jetRegression_1D.png" % outputdir)

    for i, label in enumerate(labels):
        plt.close()
        plt.clf()
        figure = plot_histo([np.array(modelsAndNames['Y_predict'][:,i]),np.array(data['jet_multijetscore_'+label]),np.array(modelsAndNames['Y_hls_predict'][:,i])],["Tensorflow","CMSSW Emulation", "hls4ml"],"Jet " + label + " Score",label+' Score','# Jets',range=(0,1))
        plt.savefig("%s/%s_score_1D.png" % (outputdir,label))

        plt.clf()
        figure = plot_2d(np.array(modelsAndNames['Y_predict'][:,i]),np.array(data['jet_multijetscore_'+label]),(0,1),(0,1),"Tensorflow","CMSSW Emulation",label+" score")
        plt.savefig("%s/%s_score_2D.png" % (outputdir,label))

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
        plt.figure()
        plot_roc(modelsAndNames,label,title=label+" ROC Comparison")
        plt.savefig(outputdir+"/ROC_Emulation_comparison_"+label+".png")

    response_reg = jet_pt_cor_reg / data['jet_genmatch_pt']
    response_emu = jet_pt_cor_reg_emu / data['jet_genmatch_pt']
    response_hls = jet_pt_cor_reg_hls / data['jet_genmatch_pt']

    figure = plot_histo([response_reg,response_emu,response_hls],
                        ["Tensorflow" + " median: "+str(np.round(np.median(response_reg),3))+" rms: "+str(np.round(rms(response_reg),3)),
                         "Emulation" + " median: "+str(np.round(np.median(response_emu),3))+" rms: "+str(np.round(rms(response_emu),3)),
                         "hls4ml" + " median: "+str(np.round(np.median(response_hls),3))+" rms: "+str(np.round(rms(response_hls),3)),],
                        "Jet Regression",'Jet Response (reco/gen)','# Jets',range=(0,2))
    plt.savefig(outputdir+"/response_emulation"+".png")
    plt.close()
    return

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='output/baseline/model/saved_model.h5' , help = 'Input model path for comparison')    
    parser.add_argument('-o','--outpath', default='output/baseline/plots/emulation' , help = 'Jet tagger plotting directory')    
    parser.add_argument('-i','--input', default='data/jetTuple.root' , help = 'Path to emulation data rootfile')
    parser.add_argument('-r','--remake', default=False , help = 'Remake emulation data? ')

    args = parser.parse_args()

    #Load the model
    model=load_qmodel(args.model)
    print(model.summary())

    if args.remake:
        make_data(infile=args.input,outdir="emulation_data/",extras='extra_emulation_fields')

    doPlots(model,args.outpath,"emulation_data/")