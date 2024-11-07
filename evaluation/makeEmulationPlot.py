from datatools.imports import *
from datatools.dataset import *
from datatools.createDataset import *
import argparse
from train.models import *
import tensorflow_model_optimization as tfmot

from array import array

from sklearn.metrics import roc_curve, auc,precision_recall_curve
import matplotlib.pyplot as plt
import json
import glob
import pdb

import shap
import pandas
import numpy
#from histbook import *


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

def rms(array):
   return np.sqrt(np.mean(array ** 2))


modelnamesDict = {
    "DeepSet": "QDeepSets_PermutationInv",
    "DeepSet-MHA": "QDeepSetsWithAttention_PermutationInv",
    "MLP": "QMLP",
    "MLP-MHA": "QMLPWithAttention",
}

nconstit = 16

def doPlots(
        test_data_dir,
        filetag,
        timestamp,
        flav,
        inputSetTag,
        modelname,
        outname,
        regression,
        pruning,
        inputQuant,
        test = False,
        save = True,
        workdir = "./",):

    modelsAndNames = {}

    # tempflav = "btgc"
    # PATH_load = workdir + '/datasets_050924/' + filetag + "/" + tempflav + "/"
    PATH_load = f"{test_data_dir}/"
    print("Loading data from: ", PATH_load)
    chunksmatching = glob.glob(f"{PATH_load}X_{inputSetTag}_test*.parquet")
    chunksmatching = [chunksm.replace(f"{PATH_load}X_{inputSetTag}_test","").replace(".parquet","").replace("_","") for chunksm in chunksmatching]

    outFolder = "emulationPlots/"+outname+"/Training_" + timestamp + "/"
    if not os.path.exists(outFolder):
        os.makedirs(outFolder, exist_ok=True)

    feature_names = dict_fields[inputSetTag]

    if test:
        import random
        chunksmatching = random.sample(chunksmatching, 5)

    print ("Loading data in all",len(chunksmatching),"chunks.")

    X_test = None
    X_test_global = None
    Y_test = None
    x_b = None
    x_b_global = None
    x_taup = None
    x_taup_global = None
    x_taum = None
    x_taum_global = None
    x_bkg = None
    x_bkg_global = None
    x_gluon = None
    x_gluon_global = None
    x_charm = None
    x_charm_global = None
    x_muon = None
    x_muon_global = None
    x_electron = None
    x_electron_global = None

    for c in chunksmatching:
        if X_test is None:
            X_test = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_test_"+c+".parquet")
            X_test_global = ak.from_parquet(PATH_load+"X_global_"+inputSetTag+"_test_"+c+".parquet")
            Y_test = ak.from_parquet(PATH_load+"Y_"+inputSetTag+"_test_"+c+".parquet")
        else:
            X_test =ak.concatenate((X_test, ak.from_parquet(PATH_load+"X_"+inputSetTag+"_test_"+c+".parquet")))
            X_test_global =ak.concatenate((X_test_global, ak.from_parquet(PATH_load+"X_global_"+inputSetTag+"_test_"+c+".parquet")))
            Y_test =ak.concatenate((Y_test, ak.from_parquet(PATH_load+"Y_"+inputSetTag+"_test_"+c+".parquet")))

        x_b_ = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_b_"+c+".parquet")
        x_b_global_ = ak.from_parquet(PATH_load+"X_global_"+inputSetTag+"_b_"+c+".parquet")
        if len(x_b_) > 0:
            if x_b is None:
                x_b = x_b_
                x_b_global = x_b_global_
            else:
                x_b =ak.concatenate((x_b, x_b_))
                x_b_global =ak.concatenate((x_b_global, x_b_global_))

        x_bkg_ = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_bkg_"+c+".parquet")
        x_bkg_global_ = ak.from_parquet(PATH_load+"X_global_"+inputSetTag+"_bkg_"+c+".parquet")
        if len(x_bkg_) > 0:
            if x_bkg is None:
                x_bkg = x_bkg_
                x_bkg_global = x_bkg_global_
            else:
                x_bkg =ak.concatenate((x_bkg, x_bkg_))
                x_bkg_global =ak.concatenate((x_bkg_global, x_bkg_global_))

        x_taup_ = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_taup_"+c+".parquet")
        x_taup_global_ = ak.from_parquet(PATH_load+"X_global_"+inputSetTag+"_taup_"+c+".parquet")
        if len(x_taup_) > 0:
            if x_taup is None:
                x_taup = x_taup_
                x_taup_global = x_taup_global_
            else:
                x_taup =ak.concatenate((x_taup, x_taup_))
                x_taup_global =ak.concatenate((x_taup_global, x_taup_global_))
        x_taum_ = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_taum_"+c+".parquet")
        x_taum_global_ = ak.from_parquet(PATH_load+"X_global_"+inputSetTag+"_taum_"+c+".parquet")
        if len(x_taum_) > 0:
            if x_taum is None:
                x_taum = x_taum_
                x_taum_global = x_taum_global_
            else:
                x_taum =ak.concatenate((x_taum, x_taum_))
                x_taum_global =ak.concatenate((x_taum_global, x_taum_global_))

        x_gluon_ = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_gluon_"+c+".parquet")
        x_gluon_global_ = ak.from_parquet(PATH_load+"X_global_"+inputSetTag+"_gluon_"+c+".parquet")
        if len(x_gluon_) > 0:
            if x_gluon is None:
                x_gluon = x_gluon_
                x_gluon_global = x_gluon_global_
            else:
                x_charm =ak.concatenate((x_gluon, x_gluon_))
                x_charm_global =ak.concatenate((x_gluon_global, x_gluon_global_))

        x_charm_ = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_charm_"+c+".parquet")
        x_charm_global_ = ak.from_parquet(PATH_load+"X_global_"+inputSetTag+"_charm_"+c+".parquet")
        if len(x_charm_) > 0:
            if x_charm is None:
                x_charm = x_charm_
                x_charm_global = x_charm_global_
            else:
                x_charm =ak.concatenate((x_charm, x_charm_))
                x_charm_global =ak.concatenate((x_charm_global, x_charm_global_))
        
        x_muon_ = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_muon_"+c+".parquet")
        x_muon_global_ = ak.from_parquet(PATH_load+"X_global_"+inputSetTag+"_muon_"+c+".parquet")
        if len(x_muon_) > 0:
            if x_muon is None:
                x_muon = x_muon_
                x_muon_global = x_muon_global_
            else:
                x_muon =ak.concatenate((x_muon, x_muon_))
                x_muon_global =ak.concatenate((x_muon_global, x_muon_global_))

        x_electron_ = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_electron_"+c+".parquet")
        x_electron_global_ = ak.from_parquet(PATH_load+"X_global_"+inputSetTag+"_electron_"+c+".parquet")
        if len(x_electron_) > 0:
            if x_electron is None:
                x_electron = x_electron_
                x_electron_global = x_electron_global_
            else:
                x_electron =ak.concatenate((x_electron, x_electron_))
                x_electron_global =ak.concatenate((x_electron_global, x_electron_global_))

    x_b = ak.to_numpy(x_b)
    x_bkg = ak.to_numpy(x_bkg)
    x_taup = ak.to_numpy(x_taup)
    x_taum = ak.to_numpy(x_taum)
    x_gluon = ak.to_numpy(x_gluon)
    x_charm = ak.to_numpy(x_charm)
    x_muon = ak.to_numpy(x_muon)
    x_electron = ak.to_numpy(x_electron)

    X_test = ak.to_numpy(X_test)
    Y_test = ak.to_numpy(Y_test)

    modelArchName = modelname

    if inputQuant:
        input_quantizer = quantized_bits(bits=16, integer=6, symmetric=0, alpha=1)
        x_b = input_quantizer(x_b.astype(np.float32)).numpy()
        x_bkg = input_quantizer(x_bkg.astype(np.float32)).numpy()
        x_tau = input_quantizer(x_tau.astype(np.float32)).numpy()
        x_gluon = input_quantizer(x_gluon.astype(np.float32)).numpy()
        x_charm = input_quantizer(x_charm.astype(np.float32)).numpy()

    print("Loaded X_test      ----> shape:", X_test.shape)
    print("Loaded Y_test      ----> shape:", Y_test.shape)

    print ("Get performance for", inputSetTag, flav, modelname)

    custom_objects_ = {
        "AAtt": AAtt,
        "QDense": QDense,
        "QActivation": QActivation,
        "quantized_bits": quantized_bits,
        "ternary": ternary,
        "binary": binary,
        "QBatchNormalization": QBatchNormalization,
        "myNLL": myNLL
        }

    ncands = 16
    nfeatures = len(feature_names)
    nbits = 8

    labels = ["Bkg", "b"]
    labels.append("Tau p")
    labels.append("Tau m")
    labels.append("Gluon")
    labels.append("Charm")
    labels.append("Muon")
    labels.append("Electron")

    # Get inference of model
    trainingBasePath = "trainings_regression_weighted/" + timestamp  + "_" + flav + "_" + inputSetTag + "_"

    modelpath = modelnamesDict[modelname]+"_nconst_"+str(ncands)+"_nfeatures_"+str(nfeatures)+"_nbits_"+str(nbits)
    modelname = 'model_'+modelnamesDict[modelname]+"_nconst_"+str(ncands)+"_nfeatures_"+str(nfeatures)+"_nbits_"+str(nbits)

    modelpath = modelpath + "_pruned"
    modelname = modelname + "_pruned"

    print ("Load model", trainingBasePath+""+modelpath+'.h5')

    modelsAndNames["model"] = tf.keras.models.load_model(trainingBasePath+""+modelpath+"/"+modelname+'.h5', custom_objects = custom_objects_)
    
    y_ =  modelsAndNames["model"].predict(X_test)
    modelsAndNames["Y_predict"] = y_[0]
    modelsAndNames["Y_predict_reg"] = y_[1]

    y_ = modelsAndNames["model"].predict(x_b)
    modelsAndNames["Y_predict_b"] = y_[0]
    modelsAndNames["Y_predict_reg_b"] = y_[1]
    X_test_global["out_b"] = modelsAndNames["Y_predict"][:,labels.index("b")]

    y_ = modelsAndNames["model"].predict(x_bkg)
    modelsAndNames["Y_predict_bkg"] = y_[0]
    modelsAndNames["Y_predict_reg_bkg"] = y_[1]
    X_test_global["out_bkg"] = modelsAndNames["Y_predict"][:,labels.index("Bkg")]

    y_ = modelsAndNames["model"].predict(x_taup)
    modelsAndNames["Y_predict_taup"] = y_[0]
    modelsAndNames["Y_predict_reg_taup"] = y_[1]
    X_test_global["out_taup"] = modelsAndNames["Y_predict"][:,labels.index("Tau p")]

    y_ = modelsAndNames["model"].predict(x_taum)
    modelsAndNames["Y_predict_taum"] = y_[0]
    modelsAndNames["Y_predict_reg_taum"] = y_[1]
    X_test_global["out_taum"] = modelsAndNames["Y_predict"][:,labels.index("Tau m")]

    y_ = modelsAndNames["model"].predict(x_gluon)
    modelsAndNames["Y_predict_gluon"] = y_[0]
    modelsAndNames["Y_predict_reg_gluon"] = y_[1]
    X_test_global["out_gluon"] = modelsAndNames["Y_predict"][:,labels.index("Gluon")]

    y_ = modelsAndNames["model"].predict(x_charm)
    modelsAndNames["Y_predict_charm"] = y_[0]
    modelsAndNames["Y_predict_reg_charm"] = y_[1]
    X_test_global["out_charm"] = modelsAndNames["Y_predict"][:,labels.index("Charm")]

    y_ = modelsAndNames["model"].predict(x_muon)
    modelsAndNames["Y_predict_muon"] = y_[0]
    modelsAndNames["Y_predict_reg_muon"] = y_[1]
    X_test_global["out_muon"] = modelsAndNames["Y_predict"][:,labels.index("Muon")]

    y_ = modelsAndNames["model"].predict(x_electron)
    modelsAndNames["Y_predict_electron"] = y_[0]
    modelsAndNames["Y_predict_reg_electron"] = y_[1]
    X_test_global["out_electron"] = modelsAndNames["Y_predict"][:,labels.index("Electron")]

    X_test_global["jet_pt_reg"] = modelsAndNames["Y_predict_reg"][:,0]
    X_test_global["jet_pt_cor_reg"] = X_test_global["jet_pt_phys"] * X_test_global["jet_pt_reg"]
    X_test_global["jet_pt_cor_reg_emu"] = X_test_global["jet_pt_phys"] * X_test_global["jet_multijetscore_regression"]

    plt.close()
    plt.clf()
    figure = plot_2d(np.array(X_test_global["jet_pt_reg"]) ,np.array(X_test_global["jet_multijetscore_regression"]) ,(0,2),(0,2),"Tensorflow","CMSSW Emulation","Jet Regression")
    plt.savefig("%s/jetRegression_2D.png" % outFolder)

    plt.close()
    plt.clf()
    figure = plot_2d(np.array(modelsAndNames["Y_predict"][:,labels.index('b')]) ,np.array(X_test_global["jet_multijetscore_b"] ),(0,1),(0,1),"Tensorflow","CMSSW Emulation","b score")
    plt.savefig("%s/b_score_2D.png" % outFolder)

    plt.close()
    plt.clf()
    figure = plot_2d(np.array(modelsAndNames["Y_predict"][:,labels.index('Bkg')]) ,np.array(X_test_global["jet_multijetscore_uds"] ),(0,1),(0,1),"Tensorflow","CMSSW Emulation","Bkg score")
    plt.savefig("%s/Bkg_score_2D.png" % outFolder)

    plt.close()
    plt.clf()
    figure = plot_2d(np.array(modelsAndNames["Y_predict"][:,labels.index('Tau p')]) ,np.array(X_test_global["jet_multijetscore_taup"] ),(0,1),(0,1),"Tensorflow","CMSSW Emulation","Tau + score")
    plt.savefig("%s/Taup_score_2D.png" % outFolder)

    plt.close()
    plt.clf()
    figure = plot_2d(np.array(modelsAndNames["Y_predict"][:,labels.index('Tau m')]) ,np.array(X_test_global["jet_multijetscore_taum"] ),(0,1),(0,1),"Tensorflow","CMSSW Emulation","Tau - score")
    plt.savefig("%s/Taum_score_2D.png" % outFolder)

    plt.close()
    plt.clf()
    figure = plot_2d(np.array(modelsAndNames["Y_predict"][:,labels.index('Gluon')]) ,np.array(X_test_global["jet_multijetscore_g"] ),(0,1),(0,1),"Tensorflow","CMSSW Emulation","Gluon score")
    plt.savefig("%s/gluon_score_2D.png" % outFolder)

    plt.close()
    plt.clf()
    figure = plot_2d(np.array(modelsAndNames["Y_predict"][:,labels.index('Charm')]) ,np.array(X_test_global["jet_multijetscore_c"] ),(0,1),(0,1),"Tensorflow","CMSSW Emulation","Charm score")
    plt.savefig("%s/charm_score_2D.png" % outFolder)

    plt.close()
    plt.clf()
    figure = plot_2d(np.array(modelsAndNames["Y_predict"][:,labels.index('Electron')]) ,np.array(X_test_global["jet_multijetscore_electron"] ),(0,1),(0,1),"Tensorflow","CMSSW Emulation","Electron score")
    plt.savefig("%s/electron_score_2D.png" % outFolder)

    plt.close()
    plt.clf()
    figure = plot_2d(np.array(modelsAndNames["Y_predict"][:,labels.index('Muon')]) ,np.array(X_test_global["jet_multijetscore_muon"] ),(0,1),(0,1),"Tensorflow","CMSSW Emulation","Muon score")
    plt.savefig("%s/muon_score_2D.png" % outFolder)


    fpr = {}
    tpr = {}
    auc1 = {}
    tresholds = {}
    wps = {}

    # Loop over classes (labels) to get metrics per class
    for i, label in enumerate(labels):
        fpr[label], tpr[label], tresholds[label] = roc_curve(Y_test[:,i], modelsAndNames["Y_predict"][:,i])
        auc1[label] = auc(fpr[label], tpr[label])

    modelsAndNames["ROCs"] = {}
    modelsAndNames["ROCs"]["tpr"] = tpr
    modelsAndNames["ROCs"]["fpr"] = fpr
    modelsAndNames["ROCs"]["auc"] = auc1

    modelsAndNames["Emulation"] = {}
    fpr = {}
    tpr = {}
    auc1 = {}
    tresholds = {}
    # Get emulation ROCs
    label = "b"
    fpr[label], tpr[label], tresholds[label] = roc_curve(Y_test[:,labels.index("b")], X_test_global["jet_multijetscore_b"])
    auc1[label] = auc(fpr[label], tpr[label])

    label = "Bkg"
    fpr[label], tpr[label], tresholds[label] = roc_curve(Y_test[:,labels.index("Bkg")], 1.-X_test_global["jet_multijetscore_uds"])
    auc1[label] = auc(fpr[label], tpr[label])

    label = "Tau p"
    fpr[label], tpr[label], tresholds[label] = roc_curve(Y_test[:,labels.index("Tau p")], X_test_global["jet_multijetscore_taup"])
    auc1[label] = auc(fpr[label], tpr[label])

    label = "Tau m"
    fpr[label], tpr[label], tresholds[label] = roc_curve(Y_test[:,labels.index("Tau m")], X_test_global["jet_multijetscore_taum"])
    auc1[label] = auc(fpr[label], tpr[label])

    label = "Gluon"
    fpr[label], tpr[label], tresholds[label] = roc_curve(Y_test[:,labels.index("Gluon")], X_test_global["jet_multijetscore_g"])
    auc1[label] = auc(fpr[label], tpr[label])

    label = "Charm"
    fpr[label], tpr[label], tresholds[label] = roc_curve(Y_test[:,labels.index("Charm")], X_test_global["jet_multijetscore_c"])
    auc1[label] = auc(fpr[label], tpr[label])

    label = "Muon"
    fpr[label], tpr[label], tresholds[label] = roc_curve(Y_test[:,labels.index("Muon")], X_test_global["jet_multijetscore_muon"])
    auc1[label] = auc(fpr[label], tpr[label])

    label = "Electron"
    fpr[label], tpr[label], tresholds[label] = roc_curve(Y_test[:,labels.index("Electron")], X_test_global["jet_multijetscore_electron"])
    auc1[label] = auc(fpr[label], tpr[label])


    modelsAndNames["Emulation"]["ROCs"] = {}
    modelsAndNames["Emulation"]["ROCs"]["tpr"] = tpr
    modelsAndNames["Emulation"]["ROCs"]["fpr"] = fpr
    modelsAndNames["Emulation"]["ROCs"]["auc"] = auc1

    #===========================#

    truthclass = "b"
    plt.figure()
    tpr = modelsAndNames["Emulation"]["ROCs"]["tpr"]
    fpr = modelsAndNames["Emulation"]["ROCs"]["fpr"]
    auc1 = modelsAndNames["Emulation"]["ROCs"]["auc"]
    plotlabel ="CMSSW Emulation"
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    tpr = modelsAndNames["ROCs"]["tpr"]
    fpr = modelsAndNames["ROCs"]["fpr"]
    auc1 = modelsAndNames["ROCs"]["auc"]
    plotlabel = "Tensorflow"
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    plt.semilogy()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Mistag rate")
    plt.xlim(0.,1.)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='best')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/ROC_Emulation_comparison_"+truthclass+".png")
    plt.savefig(outFolder+"/ROC_Emulation_comparison_"+truthclass+".pdf")
    plt.cla()


    truthclass = "Bkg"
    plt.figure()
    tpr = modelsAndNames["Emulation"]["ROCs"]["tpr"]
    fpr = modelsAndNames["Emulation"]["ROCs"]["fpr"]
    auc1 = modelsAndNames["Emulation"]["ROCs"]["auc"]
    plotlabel ="CMSSW Emulation"
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    tpr = modelsAndNames["ROCs"]["tpr"]
    fpr = modelsAndNames["ROCs"]["fpr"]
    auc1 = modelsAndNames["ROCs"]["auc"]
    plotlabel = "Tensorflow"
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    plt.semilogy()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Mistag rate")
    plt.xlim(0.,1.)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='best')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/ROC_Emulation_comparison_"+truthclass+".png")
    plt.savefig(outFolder+"/ROC_Emulation_comparison_"+truthclass+".pdf")
    plt.cla()

    truthclass = "Tau p"
    plt.figure()
    tpr = modelsAndNames["Emulation"]["ROCs"]["tpr"]
    fpr = modelsAndNames["Emulation"]["ROCs"]["fpr"]
    auc1 = modelsAndNames["Emulation"]["ROCs"]["auc"]
    plotlabel ="CMSSW Emulation"
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    tpr = modelsAndNames["ROCs"]["tpr"]
    fpr = modelsAndNames["ROCs"]["fpr"]
    auc1 = modelsAndNames["ROCs"]["auc"]
    plotlabel = "Tensorflow"
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    plt.semilogy()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Mistag rate")
    plt.xlim(0.,1.)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='best')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/ROC_Emulation_comparison_"+truthclass+".png")
    plt.savefig(outFolder+"/ROC_Emulation_comparison_"+truthclass+".pdf")
    plt.cla()

    truthclass = "Tau m"
    plt.figure()
    tpr = modelsAndNames["Emulation"]["ROCs"]["tpr"]
    fpr = modelsAndNames["Emulation"]["ROCs"]["fpr"]
    auc1 = modelsAndNames["Emulation"]["ROCs"]["auc"]
    plotlabel ="CMSSW Emulation"
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    tpr = modelsAndNames["ROCs"]["tpr"]
    fpr = modelsAndNames["ROCs"]["fpr"]
    auc1 = modelsAndNames["ROCs"]["auc"]
    plotlabel = "Tensorflow"
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    plt.semilogy()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Mistag rate")
    plt.xlim(0.,1.)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='best')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/ROC_Emulation_comparison_"+truthclass+".png")
    plt.savefig(outFolder+"/ROC_Emulation_comparison_"+truthclass+".pdf")
    plt.cla()

    truthclass = "Gluon"
    plt.figure()
    tpr = modelsAndNames["Emulation"]["ROCs"]["tpr"]
    fpr = modelsAndNames["Emulation"]["ROCs"]["fpr"]
    auc1 = modelsAndNames["Emulation"]["ROCs"]["auc"]
    plotlabel ="CMSSW Emulation"
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    tpr = modelsAndNames["ROCs"]["tpr"]
    fpr = modelsAndNames["ROCs"]["fpr"]
    auc1 = modelsAndNames["ROCs"]["auc"]
    plotlabel = "Tensorflow"
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    plt.semilogy()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Mistag rate")
    plt.xlim(0.,1.)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='best')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/ROC_Emulation_comparison_"+truthclass+".png")
    plt.savefig(outFolder+"/ROC_Emulation_comparison_"+truthclass+".pdf")
    plt.cla()

    truthclass = "Charm"
    plt.figure()
    tpr = modelsAndNames["Emulation"]["ROCs"]["tpr"]
    fpr = modelsAndNames["Emulation"]["ROCs"]["fpr"]
    auc1 = modelsAndNames["Emulation"]["ROCs"]["auc"]
    plotlabel ="CMSSW Emulation"
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    tpr = modelsAndNames["ROCs"]["tpr"]
    fpr = modelsAndNames["ROCs"]["fpr"]
    auc1 = modelsAndNames["ROCs"]["auc"]
    plotlabel = "Tensorflow"
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    plt.semilogy()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Mistag rate")
    plt.xlim(0.,1.)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='best')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/ROC_Emulation_comparison_"+truthclass+".png")
    plt.savefig(outFolder+"/ROC_Emulation_comparison_"+truthclass+".pdf")
    plt.cla()

    truthclass = "Muon"
    plt.figure()
    tpr = modelsAndNames["Emulation"]["ROCs"]["tpr"]
    fpr = modelsAndNames["Emulation"]["ROCs"]["fpr"]
    auc1 = modelsAndNames["Emulation"]["ROCs"]["auc"]
    plotlabel ="CMSSW Emulation"
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    tpr = modelsAndNames["ROCs"]["tpr"]
    fpr = modelsAndNames["ROCs"]["fpr"]
    auc1 = modelsAndNames["ROCs"]["auc"]
    plotlabel = "Tensorflow"
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    plt.semilogy()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Mistag rate")
    plt.xlim(0.,1.)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='best')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/ROC_Emulation_comparison_"+truthclass+".png")
    plt.savefig(outFolder+"/ROC_Emulation_comparison_"+truthclass+".pdf")
    plt.cla()


    truthclass = "Electron"
    plt.figure()
    tpr = modelsAndNames["Emulation"]["ROCs"]["tpr"]
    fpr = modelsAndNames["Emulation"]["ROCs"]["fpr"]
    auc1 = modelsAndNames["Emulation"]["ROCs"]["auc"]
    plotlabel ="CMSSW Emulation"
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    tpr = modelsAndNames["ROCs"]["tpr"]
    fpr = modelsAndNames["ROCs"]["fpr"]
    auc1 = modelsAndNames["ROCs"]["auc"]
    plotlabel = "Tensorflow"
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    plt.semilogy()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Mistag rate")
    plt.xlim(0.,1.)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='best')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/ROC_Emulation_comparison_"+truthclass+".png")
    plt.savefig(outFolder+"/ROC_Emulation_comparison_"+truthclass+".pdf")
    plt.cla()

    X_test_global["response_reg"] = X_test_global["jet_pt_cor_reg"] / X_test_global["jet_genmatch_pt"]
    X_test_global["response_emu"] = X_test_global["jet_pt_cor_reg_emu"] / X_test_global["jet_genmatch_pt"]

    mean_reg = np.median(X_test_global["response_reg"])
    std_reg = rms(X_test_global["response_reg"])
    mean_emu = np.median(X_test_global["response_emu"])
    std_emu = rms(X_test_global["response_emu"])

    X = np.linspace(0.0, 2.0, 100)
    histo = plt.hist(X_test_global["response_emu"], bins=X, label='Regression Emulation' ,histtype='step', density=True, color = '#ff7f0e')
    histo = plt.hist(X_test_global["response_reg"], bins=X, label='Regression Tensorflow' ,histtype='step', density=True, color = '#2ca02c')
    plt.xlabel('Jet response (reco/gen)')
    plt.ylabel('Jets')
    plt.xlim(0.,2.)
    plt.legend(prop={'size': 10})
    plt.legend(loc='upper right')
    plt.text(1.3, 1.3, "median: "+str(np.round(mean_emu,3))+" rms: "+str(np.round(std_emu,3)), color = '#ff7f0e', fontsize = 14)
    plt.text(1.3, 1.2, "median: "+str(np.round(mean_reg,3))+" rms: "+str(np.round(std_reg,3)), color = '#2ca02c', fontsize = 14)
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/response_emulation"+".png")
    plt.savefig(outFolder+"/response_emulation"+".pdf")
    plt.cla()

if __name__ == "__main__":
    from args import get_common_parser, handle_common_args
    parser = get_common_parser()
    parser.add_argument('-t','--testDataDir', default='/eos/user/s/sewuchte/L1Trigger/ForDuc/datasetsNewComplete/extendedAll200/' , help = 'input testing data directory')
    parser.add_argument('-f','--file', help = 'input model file path')
    parser.add_argument('-o','--outname', help = 'output file path')
    parser.add_argument('-c','--flav', help = 'Which flavor to run, options are b, bt, btg.')
    parser.add_argument('-i','--input', help = 'Which input to run, options are baseline, ext1, all.')
    parser.add_argument('-m','--model', help = 'Which model to evaluate, options are DeepSet, DeepSet-MHA.')
    parser.add_argument('--regression', dest = 'regression', default = False, action='store_true')
    parser.add_argument('--pruning', dest = 'pruning', default = False, action='store_true')
    parser.add_argument('--inputQuant', dest = 'inputQuant', default = False, action='store_true')
    parser.add_argument('--timestamp', dest = 'timestamp')
    parser.add_argument('--test', dest = 'test', default = False, action='store_true')


    args = parser.parse_args()
    handle_common_args(args)

    print('#'*30)
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('#'*30)


    doPlots(
        args.testDataDir,
        args.file,
        args.timestamp,
        args.flav,
        args.input,
        args.model,
        args.outname,
        args.regression,
        args.pruning,
        args.inputQuant,
        args.test,
        )
