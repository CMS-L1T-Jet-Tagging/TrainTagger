from imports import *
from dataset import *
import argparse
from models import *
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
from histbook import *

from createDataset_chunks import *


def shapPlot(shap_values, feature_names, class_names, isMLP = False):
    feature_order = np.argsort(np.sum(np.mean(np.abs(shap_values), axis=1), axis=0))
    num_features = (shap_values[0].shape[1])
    feature_inds = feature_order
    y_pos = np.arange(len(feature_inds))
    left_pos = np.zeros(len(feature_inds))

    axis_color="#333333"

    class_inds = np.argsort([-np.abs(shap_values[i]).mean() for i in range(len(shap_values))])

    for i, ind in enumerate(class_inds):
        global_shap_values = np.abs(shap_values[ind]).mean(0)
        label = class_names[ind]
        plt.barh(y_pos, global_shap_values[feature_inds], 0.7, left=left_pos, align='center',label=label)
        left_pos += global_shap_values[feature_inds]
    plt.yticks(y_pos, fontsize=15)
    plt.gca().set_yticklabels([feature_names[i] for i in feature_inds])
    plt.legend(loc='best', fontsize=18)

    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().tick_params(color=axis_color, labelcolor=axis_color)
    plt.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=15)
    plt.xlabel("mean (shapley value) - (average impact on model output magnitude)", fontsize=15)
    plt.tight_layout()

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

    tempflav = "btgc"
    PATH_load = workdir + '/datasetsNewComplete2/' + filetag + "/" + tempflav + "/"
    outFolder = "outputPlots/"+outname+"/Training_" + timestamp + "/"
    if not os.path.exists(outFolder):
        os.makedirs(outFolder, exist_ok=True)

    feature_names = dict_fields[inputSetTag]

    chunksmatching = glob.glob(PATH_load+"X_"+inputSetTag+"_test*.parquet")
    print (PATH_load+"X_"+inputSetTag+"_test*.parquet")
    chunksmatching = [chunksm.replace(PATH_load+"X_"+inputSetTag+"_test","").replace(".parquet","").replace("_","") for chunksm in chunksmatching]

    if test:
        import random
        # chunksmatching = random.sample(chunksmatching, 5)
        chunksmatching = random.sample(chunksmatching, 15)

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

    if modelname == "MLP":
        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1]*X_test.shape[2]))
        x_b = np.reshape(x_b, (x_b.shape[0],x_b.shape[1]*x_b.shape[2]))
        x_bkg = np.reshape(x_bkg, (x_bkg.shape[0],x_bkg.shape[1]*x_bkg.shape[2]))
        x_taup = np.reshape(x_taup, (x_taup.shape[0],x_taup.shape[1]*x_taup.shape[2]))
        x_taum = np.reshape(x_taum, (x_taum.shape[0],x_taum.shape[1]*x_taum.shape[2]))
        x_gluon = np.reshape(x_gluon, (x_gluon.shape[0],x_gluon.shape[1]*x_gluon.shape[2]))
        x_charm = np.reshape(x_charm, (x_charm.shape[0],x_charm.shape[1]*x_charm.shape[2]))
        x_muon = np.reshape(x_muon, (x_muon.shape[0],x_muon.shape[1]*x_muon.shape[2]))
        x_electron = np.reshape(x_electron, (x_electron.shape[0],x_electron.shape[1]*x_electron.shape[2]))

        # append global features
        X_b_global = np.stack((x_b_global["jet_pt_raw"],x_b_global["jet_eta"],x_b_global["jet_phi"],x_b_global["jet_npfcand"]), axis=1)
        X_bkg_global = np.stack((x_bkg_global["jet_pt_raw"],x_bkg_global["jet_eta"],x_bkg_global["jet_phi"],x_bkg_global["jet_npfcand"]), axis=1)
        X_taup_global = np.stack((x_taup_global["jet_pt_raw"],x_taup_global["jet_eta"],x_taup_global["jet_phi"],x_taup_global["jet_npfcand"]), axis=1)
        X_taum_global = np.stack((x_taum_global["jet_pt_raw"],x_taum_global["jet_eta"],x_taum_global["jet_phi"],x_taum_global["jet_npfcand"]), axis=1)
        X_gluon_global = np.stack((x_gluon_global["jet_pt_raw"],x_gluon_global["jet_eta"],x_gluon_global["jet_phi"],x_gluon_global["jet_npfcand"]), axis=1)
        X_charm_global = np.stack((x_charm_global["jet_pt_raw"],x_charm_global["jet_eta"],x_charm_global["jet_phi"],x_charm_global["jet_npfcand"]), axis=1)
        X_muon_global = np.stack((x_muon_global["jet_pt_raw"],x_muon_global["jet_eta"],x_muon_global["jet_phi"],x_muon_global["jet_npfcand"]), axis=1)
        X_electron_global = np.stack((x_electron_global["jet_pt_raw"],x_electron_global["jet_eta"],x_electron_global["jet_phi"],x_electron_global["jet_npfcand"]), axis=1)
        X_test_val_global = np.stack((X_test_global["jet_pt_raw"],X_test_global["jet_eta"],X_test_global["jet_phi"],X_test_global["jet_npfcand"]), axis=1)

        x_b = np.concatenate((np.array(X_b_global),x_b), axis=-1)
        x_bkg = np.concatenate((np.array(X_bkg_global),x_bkg), axis=-1)
        x_taup = np.concatenate((np.array(X_taup_global),x_taup), axis=-1)
        x_taum = np.concatenate((np.array(X_taum_global),x_taum), axis=-1)
        x_gluon = np.concatenate((np.array(X_gluon_global),x_gluon), axis=-1)
        x_charm = np.concatenate((np.array(X_charm_global),x_charm), axis=-1)
        x_muon = np.concatenate((np.array(X_muon_global),x_muon), axis=-1)
        x_electron = np.concatenate((np.array(X_electron_global),x_electron), axis=-1)
        X_test = np.concatenate((np.array(X_test_val_global),X_test), axis=-1)

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
        "QBatchNormalization": QBatchNormalization
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
    if regression:
        trainingBasePath = "trainings_regression_weighted/" + timestamp + "_" + filetag + "_" + flav + "_" + inputSetTag + "_"
    else:
        trainingBasePath = "trainings_notreduced/" + filetag + "_" + flav + "_" + inputSetTag + "_"

    modelpath = modelnamesDict[modelname]+"_nconst_"+str(ncands)+"_nfeatures_"+str(nfeatures)+"_nbits_"+str(nbits)
    modelname = 'model_'+modelnamesDict[modelname]+"_nconst_"+str(ncands)+"_nfeatures_"+str(nfeatures)+"_nbits_"+str(nbits)

    if pruning:
        modelpath = modelpath + "_pruned"
        modelname = modelname + "_pruned"
    if inputQuant:
        modelpath = modelpath + "_inputQuant"
        modelname = modelname + "_inputQuant"

    print ("Load model", trainingBasePath+""+modelpath+'.h5')



    modelsAndNames["model"] = tf.keras.models.load_model(trainingBasePath+""+modelpath+"/"+modelname+'.h5', custom_objects=custom_objects_)
    
    if regression:
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
    else:
        modelsAndNames["Y_predict"] = modelsAndNames["model"].predict(X_test)
        modelsAndNames["Y_predict_b"] = modelsAndNames["model"].predict(x_b)
        X_test_global["out_b"] = modelsAndNames["Y_predict"][:,labels.index("b")]
        modelsAndNames["Y_predict_bkg"] = modelsAndNames["model"].predict(x_bkg)
        X_test_global["out_bkg"] = modelsAndNames["Y_predict"][:,labels.index("Bkg")]
        modelsAndNames["Y_predict_tau"] = modelsAndNames["model"].predict(x_tau)
        X_test_global["out_taup"] = modelsAndNames["Y_predict"][:,labels.index("Tau p")]
        X_test_global["out_taum"] = modelsAndNames["Y_predict"][:,labels.index("Tau m")]
        modelsAndNames["Y_predict_gluon"] = modelsAndNames["model"].predict(x_gluon)
        X_test_global["out_gluon"] = modelsAndNames["Y_predict"][:,labels.index("Gluon")]
        modelsAndNames["Y_predict_charm"] = modelsAndNames["model"].predict(x_charm)
        X_test_global["out_charm"] = modelsAndNames["Y_predict"][:,labels.index("Charm")]

    if regression:
        X_test_global["jet_pt_reg"] = modelsAndNames["Y_predict_reg"][:,0]
        X_test_global["jet_pt_cor_reg"] = X_test_global["jet_pt_phys"] * X_test_global["jet_pt_reg"]

    X_test_global["b_vs_uds"] = X_test_global["out_b"]/(X_test_global["out_b"] + X_test_global["out_bkg"])
    X_test_global["b_vs_udsg"] = X_test_global["out_b"]/(X_test_global["out_b"] + X_test_global["out_bkg"] + X_test_global["out_gluon"])
    X_test_global["b_vs_g"] = X_test_global["out_b"]/(X_test_global["out_b"] + X_test_global["out_gluon"])
    X_test_global["b_vs_c"] = X_test_global["out_b"]/(X_test_global["out_b"] + X_test_global["out_charm"])
    X_test_global["b_vs_tau"] = X_test_global["out_b"]/(X_test_global["out_b"] + X_test_global["out_taup"]+X_test_global["out_taum"])
    X_test_global["b_vs_all"] = X_test_global["out_b"]

    X_test_global["c_vs_uds"] = X_test_global["out_charm"]/(X_test_global["out_charm"] + X_test_global["out_bkg"])
    X_test_global["c_vs_udsg"] = X_test_global["out_charm"]/(X_test_global["out_charm"] + X_test_global["out_bkg"] + X_test_global["out_gluon"])
    X_test_global["c_vs_g"] = X_test_global["out_charm"]/(X_test_global["out_charm"] + X_test_global["out_gluon"])
    X_test_global["c_vs_b"] = X_test_global["out_charm"]/(X_test_global["out_charm"] + X_test_global["out_b"])
    X_test_global["c_vs_tau"] = X_test_global["out_charm"]/(X_test_global["out_charm"] + X_test_global["out_taup"]+X_test_global["out_taum"])
    X_test_global["c_vs_all"] = X_test_global["out_charm"]

    X_test_global["tau_vs_uds"] = (X_test_global["out_taup"]+X_test_global["out_taum"])/(X_test_global["out_taup"]+X_test_global["out_taum"] + X_test_global["out_bkg"])
    X_test_global["tau_vs_udsg"] = (X_test_global["out_taup"]+X_test_global["out_taum"])/(X_test_global["out_taup"]+X_test_global["out_taum"] + X_test_global["out_bkg"] + X_test_global["out_gluon"])
    X_test_global["tau_vs_g"] = (X_test_global["out_taup"]+X_test_global["out_taum"])/(X_test_global["out_taup"]+X_test_global["out_taum"] + X_test_global["out_gluon"])
    X_test_global["tau_vs_b"] = (X_test_global["out_taup"]+X_test_global["out_taum"])/(X_test_global["out_taup"]+X_test_global["out_taum"] + X_test_global["out_b"])
    X_test_global["tau_vs_c"] = (X_test_global["out_taup"]+X_test_global["out_taum"])/(X_test_global["out_taup"]+X_test_global["out_taum"] + X_test_global["out_charm"])
    X_test_global["tau_vs_all"] = (X_test_global["out_taup"]+X_test_global["out_taum"])
    X_test_global["taup_vs_taum"] = (X_test_global["out_taup"])/(X_test_global["out_taup"]+X_test_global["out_taum"])

    X_test_global["g_vs_uds"] = X_test_global["out_gluon"]/(X_test_global["out_gluon"] + X_test_global["out_bkg"])
    X_test_global["g_vs_udsbc"] = X_test_global["out_gluon"]/(X_test_global["out_gluon"] + X_test_global["out_bkg"] + X_test_global["out_b"] + X_test_global["out_charm"])
    X_test_global["g_vs_b"] = X_test_global["out_gluon"]/(X_test_global["out_gluon"] + X_test_global["out_b"])
    X_test_global["g_vs_tau"] = X_test_global["out_gluon"]/(X_test_global["out_gluon"] + X_test_global["out_taup"]+X_test_global["out_taum"])
    X_test_global["g_vs_c"] = X_test_global["out_gluon"]/(X_test_global["out_gluon"] + X_test_global["out_charm"])
    X_test_global["g_vs_all"] = X_test_global["out_gluon"]

    X_test_global["muon_vs_jet"] = X_test_global["out_muon"]/(X_test_global["out_muon"] + X_test_global["out_gluon"] + X_test_global["out_bkg"] + X_test_global["out_b"] + X_test_global["out_charm"] + X_test_global["out_taup"]+X_test_global["out_taum"])
    X_test_global["muon_vs_udsgbc"] = X_test_global["out_muon"]/(X_test_global["out_muon"] + X_test_global["out_gluon"] + X_test_global["out_bkg"] + X_test_global["out_b"] + X_test_global["out_charm"])

    X_test_global["electron_vs_jet"] = X_test_global["out_electron"]/(X_test_global["out_muon"] + X_test_global["out_gluon"] + X_test_global["out_bkg"] + X_test_global["out_b"] + X_test_global["out_charm"] + X_test_global["out_taup"]+X_test_global["out_taum"])
    X_test_global["electron_vs_udsgbc"] = X_test_global["out_electron"]/(X_test_global["out_muon"] + X_test_global["out_gluon"] + X_test_global["out_bkg"] + X_test_global["out_b"] + X_test_global["out_charm"])

    X_test_global["lepton_vs_jet"] = (X_test_global["out_electron"]+X_test_global["out_muon"])/(X_test_global["out_electron"] + X_test_global["out_muon"] + X_test_global["out_gluon"] + X_test_global["out_bkg"] + X_test_global["out_b"] + X_test_global["out_charm"] + X_test_global["out_taup"]+X_test_global["out_taum"])
    X_test_global["lepton_vs_udsgbc"] = (X_test_global["out_electron"]+X_test_global["out_muon"])/(X_test_global["out_electron"] + X_test_global["out_muon"] + X_test_global["out_gluon"] + X_test_global["out_bkg"] + X_test_global["out_b"] + X_test_global["out_charm"])

    X_test_global["label_tau"] = (X_test_global["label_taup"]) | (X_test_global["label_taum"])
    X_test_global["label_lepton"] = (X_test_global["label_muon"]) | (X_test_global["label_electron"])

    # Plot the ROC curves vs ALL
    fpr = {}
    tpr = {}
    auc1 = {}
    tresholds = {}
    wps = {}

    labels_roc = [
                  "bVSuds", "bVSudsg", "bVSg", "bVSc", "bVStau","bVSall",
                  "cVSuds", "cVSudsg", "cVSg", "cVSb", "cVStau","cVSall",
                  "tauVSuds", "tauVSudsg", "tauVSg", "tauVSb", "tauVSc","tauVSall", "taupVStaum",
                  "gluonVSuds", "gluonVSudsbc", "gluonVSb", "gluonVStau", "gluonVSc","gluonVSall",
                  "muonVSudsgbc", "muonVSudsgbctauptaum",
                  "electronVSudsgbc", "electronVSudsgbctauptaum",
                  "leptonVSudsgbc", "leptonVSudsgbctauptaum",
                  ]
    masks_roc = [
                (X_test_global["label_b"] > 0) | (X_test_global["label_uds"] > 0),
                (X_test_global["label_b"] > 0) | (X_test_global["label_uds"] > 0) | (X_test_global["label_g"] > 0),
                (X_test_global["label_b"] > 0) | (X_test_global["label_g"] > 0),
                (X_test_global["label_b"] > 0) | (X_test_global["label_c"] > 0),
                (X_test_global["label_b"] > 0) | (X_test_global["label_taup"] > 0) | (X_test_global["label_taum"] > 0),
                (X_test_global["label_b"] > 0) | (X_test_global["label_uds"] > 0) | (X_test_global["label_g"] > 0) | (X_test_global["label_c"] > 0) | (X_test_global["label_taup"] > 0) | (X_test_global["label_taum"] > 0),

                (X_test_global["label_c"] > 0) | (X_test_global["label_uds"] > 0),
                (X_test_global["label_c"] > 0) | (X_test_global["label_uds"] > 0) | (X_test_global["label_g"] > 0),
                (X_test_global["label_c"] > 0) | (X_test_global["label_g"] > 0),
                (X_test_global["label_c"] > 0) | (X_test_global["label_b"] > 0),
                (X_test_global["label_c"] > 0) | (X_test_global["label_taup"] > 0) | (X_test_global["label_taum"] > 0),
                (X_test_global["label_c"] > 0) | (X_test_global["label_uds"] > 0) | (X_test_global["label_g"] > 0) | (X_test_global["label_b"] > 0) | (X_test_global["label_taup"] > 0) | (X_test_global["label_taum"] > 0),

                (X_test_global["label_taup"] > 0) | (X_test_global["label_taum"] > 0) | (X_test_global["label_uds"] > 0),
                (X_test_global["label_taup"] > 0) | (X_test_global["label_taum"] > 0) | (X_test_global["label_uds"] > 0) | (X_test_global["label_g"] > 0),
                (X_test_global["label_taup"] > 0) | (X_test_global["label_taum"] > 0) | (X_test_global["label_g"] > 0),
                (X_test_global["label_taup"] > 0) | (X_test_global["label_taum"] > 0) | (X_test_global["label_b"] > 0),
                (X_test_global["label_taup"] > 0) | (X_test_global["label_taum"] > 0) | (X_test_global["label_c"] > 0),
                (X_test_global["label_taup"] > 0) | (X_test_global["label_taum"] > 0) | (X_test_global["label_uds"] > 0) | (X_test_global["label_g"] > 0) | (X_test_global["label_c"] > 0) | (X_test_global["label_b"] > 0),
                # (X_test_global["label_taup"] > 0) | (X_test_global["label_taum"] > 0) | (X_test_global["label_uds"] > 0) | (X_test_global["label_g"] > 0) | (X_test_global["label_c"] > 0) | (X_test_global["label_b"] > 0),
                (X_test_global["label_taup"] > 0) | (X_test_global["label_taum"] > 0),

                (X_test_global["label_g"] > 0) | (X_test_global["label_uds"] > 0),
                (X_test_global["label_g"] > 0) | (X_test_global["label_uds"] > 0) | (X_test_global["label_b"] > 0) | (X_test_global["label_c"] > 0),
                (X_test_global["label_g"] > 0) | (X_test_global["label_b"] > 0),
                (X_test_global["label_g"] > 0) | (X_test_global["label_taup"] > 0) | (X_test_global["label_taum"] > 0),
                (X_test_global["label_g"] > 0) | (X_test_global["label_c"] > 0),
                (X_test_global["label_g"] > 0) | (X_test_global["label_uds"] > 0) | (X_test_global["label_b"] > 0) | (X_test_global["label_c"] > 0) | (X_test_global["label_taup"] > 0) | (X_test_global["label_taum"] > 0),
                
                (X_test_global["label_g"] > 0) | (X_test_global["label_uds"] > 0) | (X_test_global["label_b"] > 0) | (X_test_global["label_c"] > 0) | (X_test_global["label_muon"] > 0),
                (X_test_global["label_g"] > 0) | (X_test_global["label_uds"] > 0) | (X_test_global["label_b"] > 0) | (X_test_global["label_c"] > 0) | (X_test_global["label_muon"] > 0),

                (X_test_global["label_g"] > 0) | (X_test_global["label_uds"] > 0) | (X_test_global["label_b"] > 0) | (X_test_global["label_c"] > 0) | (X_test_global["label_electron"] > 0),
                (X_test_global["label_g"] > 0) | (X_test_global["label_uds"] > 0) | (X_test_global["label_b"] > 0) | (X_test_global["label_c"] > 0) | (X_test_global["label_electron"] > 0),

                (X_test_global["label_g"] > 0) | (X_test_global["label_uds"] > 0) | (X_test_global["label_b"] > 0) | (X_test_global["label_c"] > 0) | (X_test_global["label_lepton"] > 0),
                (X_test_global["label_g"] > 0) | (X_test_global["label_uds"] > 0) | (X_test_global["label_b"] > 0) | (X_test_global["label_c"] > 0) | (X_test_global["label_lepton"] > 0),
                
                ]
    truths_roc = [
                X_test_global["label_b"],
                X_test_global["label_b"],
                X_test_global["label_b"],
                X_test_global["label_b"],
                X_test_global["label_b"],
                X_test_global["label_b"],

                X_test_global["label_c"],
                X_test_global["label_c"],
                X_test_global["label_c"],
                X_test_global["label_c"],
                X_test_global["label_c"],
                X_test_global["label_c"],

                X_test_global["label_tau"],
                X_test_global["label_tau"],
                X_test_global["label_tau"],
                X_test_global["label_tau"],
                X_test_global["label_tau"],
                X_test_global["label_tau"],
                X_test_global["label_taup"],

                X_test_global["label_g"],
                X_test_global["label_g"],
                X_test_global["label_g"],
                X_test_global["label_g"],
                X_test_global["label_g"],
                X_test_global["label_g"],

                X_test_global["label_muon"],
                X_test_global["label_muon"],

                X_test_global["label_electron"],
                X_test_global["label_electron"],

                X_test_global["label_lepton"],
                X_test_global["label_lepton"],
    ]
    scores_roc = [
                X_test_global["b_vs_uds"],
                X_test_global["b_vs_udsg"],
                X_test_global["b_vs_g"],
                X_test_global["b_vs_c"],
                X_test_global["b_vs_tau"],
                X_test_global["b_vs_all"],

                X_test_global["c_vs_uds"],
                X_test_global["c_vs_udsg"],
                X_test_global["c_vs_g"],
                X_test_global["c_vs_b"],
                X_test_global["c_vs_tau"],
                X_test_global["c_vs_all"],

                X_test_global["tau_vs_uds"],
                X_test_global["tau_vs_udsg"],
                X_test_global["tau_vs_g"],
                X_test_global["tau_vs_b"],
                X_test_global["tau_vs_c"],
                X_test_global["tau_vs_all"],
                X_test_global["taup_vs_taum"],

                X_test_global["g_vs_uds"],
                X_test_global["g_vs_udsbc"],
                X_test_global["g_vs_b"],
                X_test_global["g_vs_tau"],
                X_test_global["g_vs_c"],
                X_test_global["g_vs_all"],

                X_test_global["muon_vs_udsgbc"],
                X_test_global["muon_vs_jet"],

                X_test_global["electron_vs_udsgbc"],
                X_test_global["electron_vs_jet"],

                X_test_global["lepton_vs_udsgbc"],
                X_test_global["lepton_vs_jet"],
    ]
    scores_roc_ref = [
                X_test_global["jet_bjetscore"],
                X_test_global["jet_bjetscore"],
                X_test_global["jet_bjetscore"],
                X_test_global["jet_bjetscore"],
                X_test_global["jet_bjetscore"],
                X_test_global["jet_bjetscore"],

                X_test_global["jet_bjetscore"],
                X_test_global["jet_bjetscore"],
                X_test_global["jet_bjetscore"],
                X_test_global["jet_bjetscore"],
                X_test_global["jet_bjetscore"],
                X_test_global["jet_bjetscore"],

                X_test_global["jet_tauscore"],
                X_test_global["jet_tauscore"],
                X_test_global["jet_tauscore"],
                X_test_global["jet_tauscore"],
                X_test_global["jet_tauscore"],
                X_test_global["jet_tauscore"],
                X_test_global["jet_tauscore"],

                X_test_global["jet_bjetscore"],
                X_test_global["jet_bjetscore"],
                X_test_global["jet_bjetscore"],
                X_test_global["jet_bjetscore"],
                X_test_global["jet_bjetscore"],
                X_test_global["jet_bjetscore"],

                X_test_global["jet_bjetscore"],
                X_test_global["jet_bjetscore"],

                X_test_global["jet_bjetscore"],
                X_test_global["jet_bjetscore"],

                X_test_global["jet_bjetscore"],
                X_test_global["jet_bjetscore"],
    ]

    print ("lengths of arrays for ROCs")
    print (len(labels_roc), len(masks_roc), len(truths_roc),len( scores_roc), len(scores_roc_ref))


    for i_roc, label_roc in enumerate(labels_roc):
        fpr[label_roc], tpr[label_roc], tresholds[label_roc] = roc_curve(truths_roc[i_roc][masks_roc[i_roc]], scores_roc[i_roc][masks_roc[i_roc]])
        auc1[label_roc] = auc(fpr[label_roc], tpr[label_roc])
    # get working points for 50%, 20%, 10%
    wp_b_loose = tresholds["bVSudsg"][np.argmin(np.abs(np.array(fpr["bVSudsg"])-0.5))]
    wp_b_medium = tresholds["bVSudsg"][np.argmin(np.abs(np.array(fpr["bVSudsg"])-0.2))]
    wp_b_tight = tresholds["bVSudsg"][np.argmin(np.abs(np.array(fpr["bVSudsg"])-0.1))]
    # get working points for 50%, 20%, 10% for tau
    wp_tau_loose = tresholds["tauVSall"][np.argmin(np.abs(np.array(fpr["tauVSall"])-0.5))]
    wp_tau_medium = tresholds["tauVSall"][np.argmin(np.abs(np.array(fpr["tauVSall"])-0.2))]
    wp_tau_tight = tresholds["tauVSall"][np.argmin(np.abs(np.array(fpr["tauVSall"])-0.1))]
    # print (wp_b_tight, wp_b_medium, wp_b_loose)

    for i_roc, label_roc in enumerate(labels_roc):
        fpr[label_roc+"_ref"], tpr[label_roc+"_ref"], tresholds[label_roc+"_ref"] = roc_curve(truths_roc[i_roc][masks_roc[i_roc]], scores_roc_ref[i_roc][masks_roc[i_roc]])
        auc1[label_roc+"_ref"] = auc(fpr[label_roc+"_ref"], tpr[label_roc+"_ref"])
    wp_b_loose_ref = tresholds["bVSudsg_ref"][np.argmin(np.abs(np.array(fpr["bVSudsg_ref"])-0.5))]
    wp_b_medium_ref = tresholds["bVSudsg_ref"][np.argmin(np.abs(np.array(fpr["bVSudsg_ref"])-0.2))]
    wp_b_tight_ref = tresholds["bVSudsg_ref"][np.argmin(np.abs(np.array(fpr["bVSudsg_ref"])-0.1))]
    wp_tau_loose_ref = tresholds["tauVSall_ref"][np.argmin(np.abs(np.array(fpr["tauVSall_ref"])-0.5))]
    wp_tau_medium_ref = tresholds["tauVSall_ref"][np.argmin(np.abs(np.array(fpr["tauVSall_ref"])-0.2))]
    wp_tau_tight_ref = tresholds["tauVSall_ref"][np.argmin(np.abs(np.array(fpr["tauVSall_ref"])-0.1))]

    # Loop over classes (labels) to get metrics per class
    for i, label in enumerate(labels):
        fpr[label], tpr[label], tresholds[label] = roc_curve(Y_test[:,i], modelsAndNames["Y_predict"][:,i])
        auc1[label] = auc(fpr[label], tpr[label])

    modelsAndNames["ROCs"] = {}
    modelsAndNames["ROCs"]["tpr"] = tpr
    modelsAndNames["ROCs"]["fpr"] = fpr
    modelsAndNames["ROCs"]["auc"] = auc1

    modelsAndNames["Reference"] = {}
    fpr = {}
    tpr = {}
    auc1 = {}
    tresholds = {}

    # Get reference ROCs
    label = "b"
    fpr[label], tpr[label], tresholds[label] = roc_curve(Y_test[:,labels.index("b")], X_test_global["jet_bjetscore"])
    auc1[label] = auc(fpr[label], tpr[label])

    label = "Bkg"
    fpr[label], tpr[label], tresholds[label] = roc_curve(Y_test[:,labels.index("Bkg")], 1.-X_test_global["jet_bjetscore"])
    auc1[label] = auc(fpr[label], tpr[label])

    label = "Tau p"
    fpr[label], tpr[label], tresholds[label] = roc_curve(Y_test[:,labels.index("Tau p")], X_test_global["jet_tauscore"])
    auc1[label] = auc(fpr[label], tpr[label])
    label = "Tau m"
    fpr[label], tpr[label], tresholds[label] = roc_curve(Y_test[:,labels.index("Tau m")], X_test_global["jet_tauscore"])
    auc1[label] = auc(fpr[label], tpr[label])

    modelsAndNames["Reference"]["ROCs"] = {}
    modelsAndNames["Reference"]["ROCs"]["tpr"] = tpr
    modelsAndNames["Reference"]["ROCs"]["fpr"] = fpr
    modelsAndNames["Reference"]["ROCs"]["auc"] = auc1


    # make the ROC plots
    # make one plot per truth tagger (b/tau)
    truthclass = "b"
    plt.figure()
    tpr = modelsAndNames["Reference"]["ROCs"]["tpr"]
    fpr = modelsAndNames["Reference"]["ROCs"]["fpr"]
    auc1 = modelsAndNames["Reference"]["ROCs"]["auc"]
    plotlabel ="Reference"
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    tpr = modelsAndNames["ROCs"]["tpr"]
    fpr = modelsAndNames["ROCs"]["fpr"]
    auc1 = modelsAndNames["ROCs"]["auc"]
    plotlabel =modelname + " " + flav + " " + inputSetTag
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    plt.semilogy()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Mistag rate")
    plt.xlim(0.,1.)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='best')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/ROC_comparison_"+truthclass+".png")
    plt.savefig(outFolder+"/ROC_comparison_"+truthclass+".pdf")
    plt.cla()

    truthclass = "Bkg"
    plt.figure()
    # reference tagger, only once
    tpr = modelsAndNames["Reference"]["ROCs"]["tpr"]
    fpr = modelsAndNames["Reference"]["ROCs"]["fpr"]
    auc1 = modelsAndNames["Reference"]["ROCs"]["auc"]
    plotlabel ="Reference"
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    tpr = modelsAndNames["ROCs"]["tpr"]
    fpr = modelsAndNames["ROCs"]["fpr"]
    auc1 = modelsAndNames["ROCs"]["auc"]
    plotlabel =modelname + " " + flav + " " + inputSetTag
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    plt.semilogy()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Mistag rate")
    plt.xlim(0.,1.)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='best')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/ROC_comparison_"+truthclass+".png")
    plt.savefig(outFolder+"/ROC_comparison_"+truthclass+".pdf")
    plt.cla()

    truthclass = "Tau p"
    tpr = modelsAndNames["Reference"]["ROCs"]["tpr"]
    fpr = modelsAndNames["Reference"]["ROCs"]["fpr"]
    auc1 = modelsAndNames["Reference"]["ROCs"]["auc"]
    plotlabel ="Reference"
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    tpr = modelsAndNames["ROCs"]["tpr"]
    fpr = modelsAndNames["ROCs"]["fpr"]
    auc1 = modelsAndNames["ROCs"]["auc"]
    plotlabel =modelname + " " + flav + " " + inputSetTag
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    plt.semilogy()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Mistag rate")
    plt.xlim(0.,1.)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='best')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/ROC_comparison_"+truthclass+".png")
    plt.savefig(outFolder+"/ROC_comparison_"+truthclass+".pdf")
    plt.cla()

    truthclass = "Tau m"
    tpr = modelsAndNames["Reference"]["ROCs"]["tpr"]
    fpr = modelsAndNames["Reference"]["ROCs"]["fpr"]
    auc1 = modelsAndNames["Reference"]["ROCs"]["auc"]
    plotlabel ="Reference"
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    tpr = modelsAndNames["ROCs"]["tpr"]
    fpr = modelsAndNames["ROCs"]["fpr"]
    auc1 = modelsAndNames["ROCs"]["auc"]
    plotlabel =modelname + " " + flav + " " + inputSetTag
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    plt.semilogy()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Mistag rate")
    plt.xlim(0.,1.)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='best')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/ROC_comparison_"+truthclass+".png")
    plt.savefig(outFolder+"/ROC_comparison_"+truthclass+".pdf")
    plt.cla()

    truthclass = "Gluon"
    tpr = modelsAndNames["ROCs"]["tpr"]
    fpr = modelsAndNames["ROCs"]["fpr"]
    auc1 = modelsAndNames["ROCs"]["auc"]
    plotlabel =modelname + " " + flav + " " + inputSetTag
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    plt.semilogy()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Mistag rate")
    plt.xlim(0.,1.)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='best')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/ROC_comparison_"+truthclass+".png")
    plt.savefig(outFolder+"/ROC_comparison_"+truthclass+".pdf")
    plt.cla()

    truthclass = "Charm"
    tpr = modelsAndNames["ROCs"]["tpr"]
    fpr = modelsAndNames["ROCs"]["fpr"]
    auc1 = modelsAndNames["ROCs"]["auc"]
    plotlabel =modelname + " " + flav + " " + inputSetTag
    plt.plot(tpr[truthclass],fpr[truthclass],label='%s Tagger, AUC = %.2f%%'%(plotlabel, auc1[truthclass]*100.))
    plt.semilogy()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Mistag rate")
    plt.xlim(0.,1.)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='best')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/ROC_comparison_"+truthclass+".png")
    plt.savefig(outFolder+"/ROC_comparison_"+truthclass+".pdf")
    plt.cla()

    # plot all b ROCs in one
    label_roc = "AllB"
    tpr = modelsAndNames["ROCs"]["tpr"]
    fpr = modelsAndNames["ROCs"]["fpr"]
    auc1 = modelsAndNames["ROCs"]["auc"]
    plt.plot(tpr["bVSuds"],fpr["bVSuds"],label='%s Tagger, AUC = %.2f%%'%("b vs uds", auc1["bVSuds"]*100.), color="blue")
    plt.plot(tpr["bVSg"],fpr["bVSg"],label='%s Tagger, AUC = %.2f%%'%("b vs g", auc1["bVSg"]*100.), color="orange")
    plt.plot(tpr["bVSc"],fpr["bVSc"],label='%s Tagger, AUC = %.2f%%'%("b vs c", auc1["bVSc"]*100.), color="green")
    plt.plot(tpr["bVStau"],fpr["bVStau"],label='%s Tagger, AUC = %.2f%%'%("b vs tau", auc1["bVStau"]*100.), color="red")
    plt.plot(tpr["bVSuds"+"_ref"],fpr["bVSuds"+"_ref"],label='%s Tagger, AUC = %.2f%%'%("ref b vs uds", auc1["bVSuds"+"_ref"]*100.), linestyle="--", color="blue")
    plt.plot(tpr["bVSg"+"_ref"],fpr["bVSg"+"_ref"],label='%s Tagger, AUC = %.2f%%'%("ref b vs g", auc1["bVSg"+"_ref"]*100.), linestyle="--", color="orange")
    plt.plot(tpr["bVSc"+"_ref"],fpr["bVSc"+"_ref"],label='%s Tagger, AUC = %.2f%%'%("ref b vs c", auc1["bVSc"+"_ref"]*100.), linestyle="--", color="green")
    plt.plot(tpr["bVStau"+"_ref"],fpr["bVStau"+"_ref"],label='%s Tagger, AUC = %.2f%%'%("ref b vs tau", auc1["bVStau"+"_ref"]*100.), linestyle="--", color="red")
    plt.semilogy()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Mistag rate")
    plt.xlim(0.,1.)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='best')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/ROC_"+label_roc+".png")
    plt.savefig(outFolder+"/ROC_"+label_roc+".pdf")
    plt.cla()

    # plot all c ROCs in one
    label_roc = "AllC"
    tpr = modelsAndNames["ROCs"]["tpr"]
    fpr = modelsAndNames["ROCs"]["fpr"]
    auc1 = modelsAndNames["ROCs"]["auc"]
    plt.plot(tpr["cVSuds"],fpr["cVSuds"],label='%s Tagger, AUC = %.2f%%'%("c vs uds", auc1["cVSuds"]*100.), color="blue")
    plt.plot(tpr["cVSg"],fpr["cVSg"],label='%s Tagger, AUC = %.2f%%'%("c vs g", auc1["cVSg"]*100.), color="orange")
    plt.plot(tpr["cVSb"],fpr["cVSb"],label='%s Tagger, AUC = %.2f%%'%("c vs b", auc1["cVSb"]*100.), color="green")
    plt.plot(tpr["cVStau"],fpr["cVStau"],label='%s Tagger, AUC = %.2f%%'%("c vs tau", auc1["cVStau"]*100.), color="red")
    plt.plot(tpr["cVSuds"+"_ref"],fpr["cVSuds"+"_ref"],label='%s Tagger, AUC = %.2f%%'%("ref c vs uds", auc1["cVSuds"+"_ref"]*100.), linestyle="--", color="blue")
    plt.plot(tpr["cVSg"+"_ref"],fpr["cVSg"+"_ref"],label='%s Tagger, AUC = %.2f%%'%("ref c vs g", auc1["cVSg"+"_ref"]*100.), linestyle="--", color="orange")
    plt.plot(tpr["cVSb"+"_ref"],fpr["cVSb"+"_ref"],label='%s Tagger, AUC = %.2f%%'%("ref c vs b", auc1["cVSb"+"_ref"]*100.), linestyle="--", color="green")
    plt.plot(tpr["cVStau"+"_ref"],fpr["cVStau"+"_ref"],label='%s Tagger, AUC = %.2f%%'%("ref c vs tau", auc1["cVStau"+"_ref"]*100.), linestyle="--", color="red")
    plt.semilogy()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Mistag rate")
    plt.xlim(0.,1.)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='best')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/ROC_"+label_roc+".png")
    plt.savefig(outFolder+"/ROC_"+label_roc+".pdf")
    plt.cla()

    # plot all tau ROCs in one
    label_roc = "AllTau"
    tpr = modelsAndNames["ROCs"]["tpr"]
    fpr = modelsAndNames["ROCs"]["fpr"]
    auc1 = modelsAndNames["ROCs"]["auc"]
    plt.plot(tpr["tauVSuds"],fpr["tauVSuds"],label='%s Tagger, AUC = %.2f%%'%("tau vs uds", auc1["tauVSuds"]*100.), color="blue")
    plt.plot(tpr["tauVSg"],fpr["tauVSg"],label='%s Tagger, AUC = %.2f%%'%("tau vs g", auc1["tauVSg"]*100.), color="orange")
    plt.plot(tpr["tauVSb"],fpr["tauVSb"],label='%s Tagger, AUC = %.2f%%'%("tau vs b", auc1["tauVSb"]*100.), color="green")
    plt.plot(tpr["tauVSc"],fpr["tauVSc"],label='%s Tagger, AUC = %.2f%%'%("tau vs c", auc1["tauVSc"]*100.), color="red")
    plt.plot(tpr["tauVSuds"+"_ref"],fpr["tauVSuds"+"_ref"],label='%s Tagger, AUC = %.2f%%'%("ref tau vs uds", auc1["tauVSuds"+"_ref"]*100.), linestyle="--", color="blue")
    plt.plot(tpr["tauVSg"+"_ref"],fpr["tauVSg"+"_ref"],label='%s Tagger, AUC = %.2f%%'%("ref tau vs g", auc1["tauVSg"+"_ref"]*100.), linestyle="--", color="orange")
    plt.plot(tpr["tauVSb"+"_ref"],fpr["tauVSb"+"_ref"],label='%s Tagger, AUC = %.2f%%'%("ref tau vs b", auc1["tauVSb"+"_ref"]*100.), linestyle="--", color="green")
    plt.plot(tpr["tauVSc"+"_ref"],fpr["tauVSc"+"_ref"],label='%s Tagger, AUC = %.2f%%'%("ref tau vs c", auc1["tauVSc"+"_ref"]*100.), linestyle="--", color="red")
    plt.semilogy()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Mistag rate")
    plt.xlim(0.,1.)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='best')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/ROC_"+label_roc+".png")
    plt.savefig(outFolder+"/ROC_"+label_roc+".pdf")
    plt.cla()

    # plot all tau ROCs in one
    label_roc = "AllTauCharge"
    tpr = modelsAndNames["ROCs"]["tpr"]
    fpr = modelsAndNames["ROCs"]["fpr"]
    auc1 = modelsAndNames["ROCs"]["auc"]
    plt.plot(tpr["taupVStaum"],fpr["taupVStaum"],label='%s Tagger, AUC = %.2f%%'%("taup vs taum", auc1["taupVStaum"]*100.), color="blue")
    plt.semilogy()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Mistag rate")
    plt.xlim(0.,1.)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='best')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/ROC_"+label_roc+".png")
    plt.savefig(outFolder+"/ROC_"+label_roc+".pdf")
    plt.cla()

    # plot all g ROCs in one
    label_roc = "AllGluon"
    tpr = modelsAndNames["ROCs"]["tpr"]
    fpr = modelsAndNames["ROCs"]["fpr"]
    auc1 = modelsAndNames["ROCs"]["auc"]
    plt.plot(tpr["gluonVSuds"],fpr["gluonVSuds"],label='%s Tagger, AUC = %.2f%%'%("gluon vs uds", auc1["gluonVSuds"]*100.))
    plt.plot(tpr["gluonVSudsbc"],fpr["gluonVSudsbc"],label='%s Tagger, AUC = %.2f%%'%("gluon vs udsbc", auc1["gluonVSudsbc"]*100.))
    plt.plot(tpr["gluonVSb"],fpr["gluonVSb"],label='%s Tagger, AUC = %.2f%%'%("gluon vs b", auc1["gluonVSb"]*100.))
    plt.plot(tpr["gluonVStau"],fpr["gluonVStau"],label='%s Tagger, AUC = %.2f%%'%("gluon vs tau", auc1["gluonVStau"]*100.))
    plt.plot(tpr["gluonVSc"],fpr["gluonVSc"],label='%s Tagger, AUC = %.2f%%'%("gluon vs c", auc1["gluonVSc"]*100.))
    plt.semilogy()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Mistag rate")
    plt.xlim(0.,1.)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='best')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/ROC_"+label_roc+".png")
    plt.savefig(outFolder+"/ROC_"+label_roc+".pdf")
    plt.cla()

    # plot all lepton ROCs in one
    label_roc = "AllLepton"
    tpr = modelsAndNames["ROCs"]["tpr"]
    fpr = modelsAndNames["ROCs"]["fpr"]
    auc1 = modelsAndNames["ROCs"]["auc"]
    plt.plot(tpr["muonVSudsgbc"],fpr["muonVSudsgbc"],label='%s Tagger, AUC = %.2f%%'%("muon vs udsgbc", auc1["muonVSudsgbc"]*100.))
    plt.plot(tpr["muonVSudsgbctauptaum"],fpr["muonVSudsgbctauptaum"],label='%s Tagger, AUC = %.2f%%'%("muon vs udsgbctau", auc1["muonVSudsgbctauptaum"]*100.))
    plt.plot(tpr["electronVSudsgbc"],fpr["electronVSudsgbc"],label='%s Tagger, AUC = %.2f%%'%("electron vs udsgbc", auc1["electronVSudsgbc"]*100.))
    plt.plot(tpr["electronVSudsgbctauptaum"],fpr["electronVSudsgbctauptaum"],label='%s Tagger, AUC = %.2f%%'%("electron vs udsgbctau", auc1["electronVSudsgbctauptaum"]*100.))
    plt.plot(tpr["leptonVSudsgbc"],fpr["leptonVSudsgbc"],label='%s Tagger, AUC = %.2f%%'%("lepton vs udsgbc", auc1["leptonVSudsgbc"]*100.))
    plt.plot(tpr["leptonVSudsgbctauptaum"],fpr["leptonVSudsgbctauptaum"],label='%s Tagger, AUC = %.2f%%'%("lepton vs udsgbctau", auc1["leptonVSudsgbctauptaum"]*100.))
    plt.semilogy()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Mistag rate")
    plt.xlim(0.,1.)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='best')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/ROC_"+label_roc+".png")
    plt.savefig(outFolder+"/ROC_"+label_roc+".pdf")
    plt.cla()

    for score_loop in [
        "b_vs_uds", "b_vs_udsg", "b_vs_g", "b_vs_c", "b_vs_tau", "b_vs_all",
        "c_vs_uds", "c_vs_udsg", "c_vs_g", "c_vs_b", "c_vs_tau", "c_vs_all",
        "tau_vs_uds", "tau_vs_udsg", "tau_vs_g", "tau_vs_c", "tau_vs_b", "tau_vs_all", "taup_vs_taum",
        "muon_vs_jet", "electron_vs_jet", "lepton_vs_jet",
        ]:

        X = np.linspace(0.0, 1.0, 100)
        histo = plt.hist(X_test_global[X_test_global["label_b"]>0][score_loop], bins=X, label='b' ,histtype='step', density = True)
        histo = plt.hist(X_test_global[X_test_global["label_uds"]>0][score_loop], bins=X, label='uds' ,histtype='step', density = True)
        histo = plt.hist(X_test_global[X_test_global["label_taup"]>0][score_loop], bins=X, label='Tau p' ,histtype='step', density = True)
        histo = plt.hist(X_test_global[X_test_global["label_taum"]>0][score_loop], bins=X, label='Tau m' ,histtype='step', density = True)
        histo = plt.hist(X_test_global[X_test_global["label_g"]>0][score_loop], bins=X, label='g' ,histtype='step', density = True)
        histo = plt.hist(X_test_global[X_test_global["label_c"]>0][score_loop], bins=X, label='c' ,histtype='step', density = True)
        histo = plt.hist(X_test_global[X_test_global["label_muon"]>0][score_loop], bins=X, label='muon' ,histtype='step', density = True)
        histo = plt.hist(X_test_global[X_test_global["label_electron"]>0][score_loop], bins=X, label='electron' ,histtype='step', density = True)
        plt.xlabel(score_loop.replace("_"," ")+' score')
        plt.legend(prop={'size': 10})
        plt.legend(loc='upper right')
        hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
        plt.savefig(outFolder+"/score_"+score_loop+".png")
        plt.savefig(outFolder+"/score_"+score_loop+".pdf")
        plt.cla()

    # efficiency vs pT
    # x_bins_pt = np.array([0., 15., 20., 30., 40., 50., 75., 100., 125., 150., 175., 200., 300., 500., 750., 1000.])
    x_bins_pt = np.array([15., 20., 30., 40., 50., 60., 70., 80., 90., 100., 125., 150., 175., 200., 250., 300., 400., 500., 750., 1000.])

    data_ = ak.to_pandas(X_test_global[X_test_global["label_b"]>0])
    data_["wp1"] = data_["b_vs_udsg"] > wp_b_loose
    data_["wp2"] = data_["b_vs_udsg"] > wp_b_medium
    data_["wp3"] = data_["b_vs_udsg"] > wp_b_tight
    data_["wp1_ref"] = data_["jet_bjetscore"] > wp_b_loose_ref
    data_["wp2_ref"] = data_["jet_bjetscore"] > wp_b_medium_ref
    data_["wp3_ref"] = data_["jet_bjetscore"] > wp_b_tight_ref

    h_wp1 = Hist(split("jet_pt_phys", x_bins_pt), cut("wp1"))
    h_wp2 = Hist(split("jet_pt_phys", x_bins_pt), cut("wp2"))
    h_wp3 = Hist(split("jet_pt_phys", x_bins_pt), cut("wp3"))
    h_wp1.fill(data_)
    h_wp2.fill(data_)
    h_wp3.fill(data_)
    h_wp1_ref = Hist(split("jet_pt_phys", x_bins_pt), cut("wp1_ref"))
    h_wp2_ref = Hist(split("jet_pt_phys", x_bins_pt), cut("wp2_ref"))
    h_wp3_ref = Hist(split("jet_pt_phys", x_bins_pt), cut("wp3_ref"))
    h_wp1_ref.fill(data_)
    h_wp2_ref.fill(data_)
    h_wp3_ref.fill(data_)

    df_wp1 = h_wp1.pandas("wp1", error="normal")
    df_wp2 = h_wp2.pandas("wp2", error="normal")
    df_wp3 = h_wp3.pandas("wp3", error="normal")
    df_wp1_ref = h_wp1_ref.pandas("wp1_ref", error="normal")
    df_wp2_ref = h_wp2_ref.pandas("wp2_ref", error="normal")
    df_wp3_ref = h_wp3_ref.pandas("wp3_ref", error="normal")

    df_wp1["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_wp1.index]
    df_wp2["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_wp2.index]
    df_wp3["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_wp3.index]
    df_wp1_ref["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_wp1_ref.index]
    df_wp2_ref["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_wp2_ref.index]
    df_wp3_ref["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_wp3_ref.index]

    plt.cla()
    # now scatter plot with error bars
    ax = df_wp1.plot.line(x="midpoints", y="wp1", yerr="err(wp1)", label = "b jets - loose", color="blue", linestyle='-')
    df_wp2.plot.line(x="midpoints", y="wp2", yerr="err(wp2)", label = "b jets - medium", ax = ax, color="orange", linestyle='-')
    df_wp3.plot.line(x="midpoints", y="wp3", yerr="err(wp3)", label = "b jets - tight", ax = ax, color="green", linestyle='-')
    df_wp1_ref.plot.line(x="midpoints", y="wp1_ref", yerr="err(wp1_ref)", label = "b jets - loose (ref)", ax = ax, color="blue", linestyle='--')
    df_wp2_ref.plot.line(x="midpoints", y="wp2_ref", yerr="err(wp2_ref)", label = "b jets - medium (ref)", ax = ax, color="orange", linestyle='--')
    df_wp3_ref.plot.line(x="midpoints", y="wp3_ref", yerr="err(wp3_ref)", label = "b jets - tight (ref)", ax = ax, color="green", linestyle='--')
    plt.xlabel(r'Jet $p_T$ [GeV]')
    plt.ylabel('Tagging efficiency')
    plt.ylim(0., 1.3)
    plt.xlim(0., 1000.)
    plt.legend(prop={'size': 10})
    plt.legend(loc='upper right')
    hep.cms.label("Private Work", data = False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/eff_b_pt"+".png")
    plt.savefig(outFolder+"/eff_b_pt"+".pdf")
    plt.cla()

    # efficiency vs eta
    x_bins_eta = np.array([-2.5, -2., -1.5 ,-1., -0.5, 0., 0.5, 1., 1.5, 2., 2.5])

    h_wp1_eta = Hist(split("jet_eta_phys", x_bins_eta), cut("wp1"))
    h_wp2_eta = Hist(split("jet_eta_phys", x_bins_eta), cut("wp2"))
    h_wp3_eta = Hist(split("jet_eta_phys", x_bins_eta), cut("wp3"))
    h_wp1_eta.fill(data_)
    h_wp2_eta.fill(data_)
    h_wp3_eta.fill(data_)
    h_wp1_eta_ref = Hist(split("jet_eta_phys", x_bins_eta), cut("wp1_ref"))
    h_wp2_eta_ref = Hist(split("jet_eta_phys", x_bins_eta), cut("wp2_ref"))
    h_wp3_eta_ref = Hist(split("jet_eta_phys", x_bins_eta), cut("wp3_ref"))
    h_wp1_eta_ref.fill(data_)
    h_wp2_eta_ref.fill(data_)
    h_wp3_eta_ref.fill(data_)

    df_wp1_eta = h_wp1_eta.pandas("wp1", error="normal")
    df_wp2_eta = h_wp2_eta.pandas("wp2", error="normal")
    df_wp3_eta = h_wp3_eta.pandas("wp3", error="normal")
    df_wp1_eta_ref = h_wp1_eta_ref.pandas("wp1_ref", error="normal")
    df_wp2_eta_ref = h_wp2_eta_ref.pandas("wp2_ref", error="normal")
    df_wp3_eta_ref = h_wp3_eta_ref.pandas("wp3_ref", error="normal")

    df_wp1_eta["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_wp1_eta.index]
    df_wp2_eta["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_wp2_eta.index]
    df_wp3_eta["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_wp3_eta.index]
    df_wp1_eta_ref["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_wp1_eta_ref.index]
    df_wp2_eta_ref["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_wp2_eta_ref.index]
    df_wp3_eta_ref["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_wp3_eta_ref.index]

    plt.cla()
    # now scatter plot with error bars
    ax = df_wp1_eta.plot.line(x="midpoints", y="wp1", yerr="err(wp1)", label = "b jets - loose", color="blue", linestyle='-')
    df_wp2_eta.plot.line(x="midpoints", y="wp2", yerr="err(wp2)", label = "b jets - medium", ax = ax, color="orange", linestyle='-')
    df_wp3_eta.plot.line(x="midpoints", y="wp3", yerr="err(wp3)", label = "b jets - tight", ax = ax, color="green", linestyle='-')
    df_wp1_eta_ref.plot.line(x="midpoints", y="wp1_ref", yerr="err(wp1_ref)", label = "b jets - loose (ref)", ax = ax, color="blue", linestyle='--')
    df_wp2_eta_ref.plot.line(x="midpoints", y="wp2_ref", yerr="err(wp2_ref)", label = "b jets - medium (ref)", ax = ax, color="orange", linestyle='--')
    df_wp3_eta_ref.plot.line(x="midpoints", y="wp3_ref", yerr="err(wp3_ref)", label = "b jets - tight (ref)", ax = ax, color="green", linestyle='--')
    plt.xlabel(r'Jet eta')
    plt.ylabel('Tagging efficiency')
    plt.ylim(0., 1.3)
    plt.xlim(-2.5, 2.5)
    plt.legend(prop={'size': 10})
    plt.legend(loc='upper right')
    hep.cms.label("Private Work", data = False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/eff_b_eta"+".png")
    plt.savefig(outFolder+"/eff_b_eta"+".pdf")
    plt.cla()
    

    # efficiency vs pT for taus
    x_bins_pt = np.array([0., 15., 20., 30., 40., 50., 75., 100., 125., 150., 175., 200., 300., 500., 750., 1000.])

    data_ = ak.to_pandas(X_test_global[X_test_global["label_tau"]>0])
    data_["wp1"] = data_["tau_vs_all"] > wp_tau_loose
    data_["wp2"] = data_["tau_vs_all"] > wp_tau_medium
    data_["wp3"] = data_["tau_vs_all"] > wp_tau_tight
    data_["wp1_ref"] = data_["jet_tauscore"] > wp_tau_loose_ref
    data_["wp2_ref"] = data_["jet_tauscore"] > wp_tau_medium_ref
    data_["wp3_ref"] = data_["jet_tauscore"] > wp_tau_tight_ref

    h_wp1 = Hist(split("jet_pt_phys", x_bins_pt), cut("wp1"))
    h_wp2 = Hist(split("jet_pt_phys", x_bins_pt), cut("wp2"))
    h_wp3 = Hist(split("jet_pt_phys", x_bins_pt), cut("wp3"))
    h_wp1.fill(data_)
    h_wp2.fill(data_)
    h_wp3.fill(data_)
    h_wp1_ref = Hist(split("jet_pt_phys", x_bins_pt), cut("wp1_ref"))
    h_wp2_ref = Hist(split("jet_pt_phys", x_bins_pt), cut("wp2_ref"))
    h_wp3_ref = Hist(split("jet_pt_phys", x_bins_pt), cut("wp3_ref"))
    h_wp1_ref.fill(data_)
    h_wp2_ref.fill(data_)
    h_wp3_ref.fill(data_)

    df_wp1 = h_wp1.pandas("wp1", error="normal")
    df_wp2 = h_wp2.pandas("wp2", error="normal")
    df_wp3 = h_wp3.pandas("wp3", error="normal")
    df_wp1_ref = h_wp1_ref.pandas("wp1_ref", error="normal")
    df_wp2_ref = h_wp2_ref.pandas("wp2_ref", error="normal")
    df_wp3_ref = h_wp3_ref.pandas("wp3_ref", error="normal")

    df_wp1["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_wp1.index]
    df_wp2["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_wp2.index]
    df_wp3["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_wp3.index]
    df_wp1_ref["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_wp1_ref.index]
    df_wp2_ref["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_wp2_ref.index]
    df_wp3_ref["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_wp3_ref.index]

    plt.cla()
    # now scatter plot with error bars
    ax = df_wp1.plot.line(x="midpoints", y="wp1", yerr="err(wp1)", label = "tau jets - loose", color="blue", linestyle='-')
    df_wp2.plot.line(x="midpoints", y="wp2", yerr="err(wp2)", label = "tau jets - medium", ax = ax, color="orange", linestyle='-')
    df_wp3.plot.line(x="midpoints", y="wp3", yerr="err(wp3)", label = "tau jets - tight", ax = ax, color="green", linestyle='-')
    df_wp1_ref.plot.line(x="midpoints", y="wp1_ref", yerr="err(wp1_ref)", label = "tau jets - loose (ref)", ax = ax, color="blue", linestyle='--')
    df_wp2_ref.plot.line(x="midpoints", y="wp2_ref", yerr="err(wp2_ref)", label = "tau jets - medium (ref)", ax = ax, color="orange", linestyle='--')
    df_wp3_ref.plot.line(x="midpoints", y="wp3_ref", yerr="err(wp3_ref)", label = "tau jets - tight (ref)", ax = ax, color="green", linestyle='--')
    plt.xlabel(r'Jet $p_T$ [GeV]')
    plt.ylabel('Tagging efficiency')
    plt.ylim(0., 1.3)
    plt.xlim(0., 1000.)
    plt.legend(prop={'size': 10})
    plt.legend(loc='upper right')
    hep.cms.label("Private Work", data = False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/eff_tau_pt"+".png")
    plt.savefig(outFolder+"/eff_tau_pt"+".pdf")
    plt.cla()

    # efficiency vs eta
    h_wp1_eta = Hist(split("jet_eta_phys", x_bins_eta), cut("wp1"))
    h_wp2_eta = Hist(split("jet_eta_phys", x_bins_eta), cut("wp2"))
    h_wp3_eta = Hist(split("jet_eta_phys", x_bins_eta), cut("wp3"))
    h_wp1_eta.fill(data_)
    h_wp2_eta.fill(data_)
    h_wp3_eta.fill(data_)
    h_wp1_eta_ref = Hist(split("jet_eta_phys", x_bins_eta), cut("wp1_ref"))
    h_wp2_eta_ref = Hist(split("jet_eta_phys", x_bins_eta), cut("wp2_ref"))
    h_wp3_eta_ref = Hist(split("jet_eta_phys", x_bins_eta), cut("wp3_ref"))
    h_wp1_eta_ref.fill(data_)
    h_wp2_eta_ref.fill(data_)
    h_wp3_eta_ref.fill(data_)

    df_wp1_eta = h_wp1_eta.pandas("wp1", error="normal")
    df_wp2_eta = h_wp2_eta.pandas("wp2", error="normal")
    df_wp3_eta = h_wp3_eta.pandas("wp3", error="normal")
    df_wp1_eta_ref = h_wp1_eta_ref.pandas("wp1_ref", error="normal")
    df_wp2_eta_ref = h_wp2_eta_ref.pandas("wp2_ref", error="normal")
    df_wp3_eta_ref = h_wp3_eta_ref.pandas("wp3_ref", error="normal")

    df_wp1_eta["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_wp1_eta.index]
    df_wp2_eta["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_wp2_eta.index]
    df_wp3_eta["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_wp3_eta.index]
    df_wp1_eta_ref["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_wp1_eta_ref.index]
    df_wp2_eta_ref["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_wp2_eta_ref.index]
    df_wp3_eta_ref["midpoints"] = [x[0].mid if isinstance(x[0], pandas.Interval) else numpy.nan for x in df_wp3_eta_ref.index]

    plt.cla()
    # now scatter plot with error bars
    ax = df_wp1_eta.plot.line(x="midpoints", y="wp1", yerr="err(wp1)", label = "tau jets - loose", color="blue", linestyle='-')
    df_wp2_eta.plot.line(x="midpoints", y="wp2", yerr="err(wp2)", label = "tau jets - medium", ax = ax, color="orange", linestyle='-')
    df_wp3_eta.plot.line(x="midpoints", y="wp3", yerr="err(wp3)", label = "tau jets - tight", ax = ax, color="green", linestyle='-')
    df_wp1_eta_ref.plot.line(x="midpoints", y="wp1_ref", yerr="err(wp1_ref)", label = "tau jets - loose (ref)", ax = ax, color="blue", linestyle='--')
    df_wp2_eta_ref.plot.line(x="midpoints", y="wp2_ref", yerr="err(wp2_ref)", label = "tau jets - medium (ref)", ax = ax, color="orange", linestyle='--')
    df_wp3_eta_ref.plot.line(x="midpoints", y="wp3_ref", yerr="err(wp3_ref)", label = "tau jets - tight (ref)", ax = ax, color="green", linestyle='--')
    plt.xlabel(r'Jet eta')
    plt.ylabel('Tagging efficiency')
    plt.ylim(0., 1.3)
    plt.xlim(-2.5, 2.5)
    plt.legend(prop={'size': 10})
    plt.legend(loc='upper right')
    hep.cms.label("Private Work", data = False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/eff_tau_eta"+".png")
    plt.savefig(outFolder+"/eff_tau_eta"+".pdf")
    plt.cla()


    # get response and resolution plots
    if regression:
        X_test_global["response"] = X_test_global["jet_pt_phys"] / X_test_global["jet_genmatch_pt"]
        X_test_global["response_cor"] = X_test_global["jet_pt_corr"] / X_test_global["jet_genmatch_pt"]
        X_test_global["response_tau"] = X_test_global["jet_taupt"] / X_test_global["jet_genmatch_pt"]
        X_test_global["response_reg"] = X_test_global["jet_pt_cor_reg"] / X_test_global["jet_genmatch_pt"]

        mean_uncor = np.median(np.array(X_test_global["response"]))
        std_uncor = rms(X_test_global["response"])
        mean_cor = np.median(X_test_global["response_cor"])
        std_cor = rms(X_test_global["response_cor"])
        mean_reg = np.median(X_test_global["response_reg"])
        std_reg = rms(X_test_global["response_reg"])
        X = np.linspace(0.0, 2.0, 100)
        histo = plt.hist(X_test_global["response"], bins=X, label='Uncorrected' ,histtype='step', density=True, color = '#1f77b4')
        histo = plt.hist(X_test_global["response_cor"], bins=X, label='JEC LOT' ,histtype='step', density=True, color = '#ff7f0e')
        histo = plt.hist(X_test_global["response_reg"], bins=X, label='Regression' ,histtype='step', density=True, color = '#2ca02c')
        plt.xlabel('Jet response (reco/gen)')
        plt.ylabel('Jets')
        plt.xlim(0.,2.)
        plt.legend(prop={'size': 10})
        plt.legend(loc='upper right')
        plt.text(1.3, 1.4, "median: "+str(np.round(mean_uncor,3))+" rms: "+str(np.round(std_uncor,3)), color = '#1f77b4', fontsize = 14)
        plt.text(1.3, 1.3, "median: "+str(np.round(mean_cor,3))+" rms: "+str(np.round(std_cor,3)), color = '#ff7f0e', fontsize = 14)
        plt.text(1.3, 1.2, "median: "+str(np.round(mean_reg,3))+" rms: "+str(np.round(std_reg,3)), color = '#2ca02c', fontsize = 14)
        hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
        plt.savefig(outFolder+"/response_all"+".png")
        plt.savefig(outFolder+"/response_all"+".pdf")
        plt.cla()

        def getMedianError(ar):
            if len(ar)!=0:
                return 1.253 * rms(ar) / len(ar)
            else:
                return 0

        # now make it for several pT bins
        # pt_bins = [15., 20., 30., 50., 75., 100., 150., 200., 300., 400., 500., 1000.]
        pt_bins = [15., 17., 20., 23., 27., 30., 35., 40., 45., 57., 72., 90., 120., 150., 200., 300., 400., 550., 750., 1000.]
        means_cor, means_cor_b, means_cor_c, means_cor_uds, means_cor_g, means_cor_tau = [],[],[],[],[],[]
        means_cor_err, means_cor_err_b, means_cor_err_c, means_cor_err_uds, means_cor_err_g, means_cor_err_tau = [],[],[],[],[],[]
        means_uncor, means_uncor_b, means_uncor_c, means_uncor_uds, means_uncor_g, means_uncor_tau = [],[],[],[],[],[]
        means_uncor_err, means_uncor_err_b, means_uncor_err_c, means_uncor_err_uds, means_uncor_err_g, means_uncor_err_tau = [],[],[],[],[],[]
        means_reg, means_reg_b, means_reg_c, means_reg_uds, means_reg_g, means_reg_tau = [],[],[],[],[],[]
        means_refReg_tau, means_refReg_err_tau = [], []
        means_reg_err, means_reg_err_b, means_reg_err_c, means_reg_err_uds, means_reg_err_g, means_reg_err_tau = [],[],[],[],[],[]
        stds_cor, stds_cor_b, stds_cor_c, stds_cor_uds, stds_cor_g, stds_cor_tau = [],[],[],[],[],[]
        stds_cor_err, stds_cor_err_b, stds_cor_err_c, stds_cor_err_uds, stds_cor_err_g, stds_cor_err_tau = [],[],[],[],[],[]
        stds_uncor, stds_uncor_b, stds_uncor_c, stds_uncor_uds, stds_uncor_g, stds_uncor_tau = [],[],[],[],[],[]
        stds_uncor_err, stds_uncor_err_b, stds_uncor_err_c, stds_uncor_err_uds, stds_uncor_err_g, stds_uncor_err_tau = [],[],[],[],[],[]
        stds_reg, stds_reg_b, stds_reg_c, stds_reg_uds, stds_reg_g, stds_reg_tau = [],[],[],[],[],[]
        stds_reg_err, stds_reg_err_b, stds_reg_err_c, stds_reg_err_uds, stds_reg_err_g, stds_reg_err_tau = [],[],[],[],[],[]
        centers = []
        for ptIdx in range(len(pt_bins)-1):
            ptLowEdge = pt_bins[ptIdx]
            ptHighEdge = pt_bins[ptIdx+1]
            centers.append(((ptHighEdge-ptLowEdge)/2.)+ptLowEdge)

            resp_ = np.array(X_test_global[((X_test_global["jet_genmatch_pt"] > ptLowEdge) & (X_test_global["jet_genmatch_pt"] < ptHighEdge) & (X_test_global["response"] > 0.  )& (X_test_global["response"] < 2. ))]["response"])
            resp_b = np.array(X_test_global[((X_test_global["label_b"] > 0) & (X_test_global["jet_genmatch_pt"] > ptLowEdge) & (X_test_global["jet_genmatch_pt"] < ptHighEdge) & (X_test_global["response"] > 0.  )& (X_test_global["response"] < 2. ))]["response"])
            resp_c = np.array(X_test_global[((X_test_global["label_c"] > 0) & (X_test_global["jet_genmatch_pt"] > ptLowEdge) & (X_test_global["jet_genmatch_pt"] < ptHighEdge) & (X_test_global["response"] > 0.  )& (X_test_global["response"] < 2. ))]["response"])
            resp_uds = np.array(X_test_global[((X_test_global["label_uds"] > 0) & (X_test_global["jet_genmatch_pt"] > ptLowEdge) & (X_test_global["jet_genmatch_pt"] < ptHighEdge) & (X_test_global["response"] > 0.  )& (X_test_global["response"] < 2. ))]["response"])
            resp_g = np.array(X_test_global[((X_test_global["label_g"] > 0) & (X_test_global["jet_genmatch_pt"] > ptLowEdge) & (X_test_global["jet_genmatch_pt"] < ptHighEdge) & (X_test_global["response"] > 0.  )& (X_test_global["response"] < 2. ))]["response"])
            resp_tau = np.array(X_test_global[((X_test_global["label_tau"] > 0) & (X_test_global["jet_genmatch_pt"] > ptLowEdge) & (X_test_global["jet_genmatch_pt"] < ptHighEdge) & (X_test_global["response"] > 0.  )& (X_test_global["response"] < 2. ))]["response"])
            
            resp_cor_ = np.array(X_test_global[((X_test_global["jet_genmatch_pt"] > ptLowEdge) & (X_test_global["jet_genmatch_pt"] < ptHighEdge) & (X_test_global["response_cor"] > 0.  )& (X_test_global["response_cor"] < 2. ))]["response_cor"])
            resp_cor_b = np.array(X_test_global[((X_test_global["label_b"] > 0) & (X_test_global["jet_genmatch_pt"] > ptLowEdge) & (X_test_global["jet_genmatch_pt"] < ptHighEdge) & (X_test_global["response_cor"] > 0.  )& (X_test_global["response_cor"] < 2. ))]["response_cor"])
            resp_cor_c = np.array(X_test_global[((X_test_global["label_c"] > 0) & (X_test_global["jet_genmatch_pt"] > ptLowEdge) & (X_test_global["jet_genmatch_pt"] < ptHighEdge) & (X_test_global["response_cor"] > 0.  )& (X_test_global["response_cor"] < 2. ))]["response_cor"])
            resp_cor_uds = np.array(X_test_global[((X_test_global["label_uds"] > 0) & (X_test_global["jet_genmatch_pt"] > ptLowEdge) & (X_test_global["jet_genmatch_pt"] < ptHighEdge) & (X_test_global["response_cor"] > 0.  )& (X_test_global["response_cor"] < 2. ))]["response_cor"])
            resp_cor_g = np.array(X_test_global[((X_test_global["label_g"] > 0) & (X_test_global["jet_genmatch_pt"] > ptLowEdge) & (X_test_global["jet_genmatch_pt"] < ptHighEdge) & (X_test_global["response_cor"] > 0.  )& (X_test_global["response_cor"] < 2. ))]["response_cor"])
            resp_cor_tau = np.array(X_test_global[((X_test_global["label_tau"] > 0) & (X_test_global["jet_genmatch_pt"] > ptLowEdge) & (X_test_global["jet_genmatch_pt"] < ptHighEdge) & (X_test_global["response_cor"] > 0.  )& (X_test_global["response_cor"] < 2. ))]["response_cor"])
            
            resp_reg_ = np.array(X_test_global[((X_test_global["jet_genmatch_pt"] > ptLowEdge) & (X_test_global["jet_genmatch_pt"] < ptHighEdge) & (X_test_global["response_reg"] > 0.  )& (X_test_global["response_reg"] < 2. ))]["response_reg"])
            resp_reg_b = np.array(X_test_global[((X_test_global["label_b"] > 0) & (X_test_global["jet_genmatch_pt"] > ptLowEdge) & (X_test_global["jet_genmatch_pt"] < ptHighEdge) & (X_test_global["response_reg"] > 0.  )& (X_test_global["response_reg"] < 2. ))]["response_reg"])
            resp_reg_c = np.array(X_test_global[((X_test_global["label_c"] > 0) & (X_test_global["jet_genmatch_pt"] > ptLowEdge) & (X_test_global["jet_genmatch_pt"] < ptHighEdge) & (X_test_global["response_reg"] > 0.  )& (X_test_global["response_reg"] < 2. ))]["response_reg"])
            resp_reg_uds = np.array(X_test_global[((X_test_global["label_uds"] > 0) & (X_test_global["jet_genmatch_pt"] > ptLowEdge) & (X_test_global["jet_genmatch_pt"] < ptHighEdge) & (X_test_global["response_reg"] > 0.  )& (X_test_global["response_reg"] < 2. ))]["response_reg"])
            resp_reg_g = np.array(X_test_global[((X_test_global["label_g"] > 0) & (X_test_global["jet_genmatch_pt"] > ptLowEdge) & (X_test_global["jet_genmatch_pt"] < ptHighEdge) & (X_test_global["response_reg"] > 0.  )& (X_test_global["response_reg"] < 2. ))]["response_reg"])
            resp_reg_tau = np.array(X_test_global[((X_test_global["label_tau"] > 0) & (X_test_global["jet_genmatch_pt"] > ptLowEdge) & (X_test_global["jet_genmatch_pt"] < ptHighEdge) & (X_test_global["response_reg"] > 0.  )& (X_test_global["response_reg"] < 2. ))]["response_reg"])

            resp_refReg_tau = np.array(X_test_global[((X_test_global["label_tau"] > 0) & (X_test_global["jet_genmatch_pt"] > ptLowEdge) & (X_test_global["jet_genmatch_pt"] < ptHighEdge) & (X_test_global["response_tau"] > 0.  )& (X_test_global["response_tau"] < 2. ))]["response"])

            mean_uncor = np.median(resp_)
            mean_uncor_b = np.median(resp_b)
            mean_uncor_c = np.median(resp_c)
            mean_uncor_uds = np.median(resp_uds)
            mean_uncor_g = np.median(resp_g)
            mean_uncor_tau = np.median(resp_tau)
            mean_uncor_err = getMedianError(resp_)
            mean_uncor_err_b = getMedianError(resp_b)
            mean_uncor_err_c = getMedianError(resp_c)
            mean_uncor_err_uds = getMedianError(resp_uds)
            mean_uncor_err_g = getMedianError(resp_g)
            mean_uncor_err_tau = getMedianError(resp_tau)

            std_uncor = rms(resp_)
            std_uncor_err = rms(resp_)

            mean_cor = np.median(resp_cor_)
            mean_cor_b = np.median(resp_cor_b)
            mean_cor_c = np.median(resp_cor_c)
            mean_cor_uds = np.median(resp_cor_uds)
            mean_cor_g = np.median(resp_cor_g)
            mean_cor_tau = np.median(resp_cor_tau)
            mean_cor_err = getMedianError(resp_cor_)
            mean_cor_err_b = getMedianError(resp_cor_b)
            mean_cor_err_c = getMedianError(resp_cor_c)
            mean_cor_err_uds = getMedianError(resp_cor_uds)
            mean_cor_err_g = getMedianError(resp_cor_g)
            mean_cor_err_tau = getMedianError(resp_cor_tau)

            std_cor = rms(resp_cor_)
            std_cor_err = rms(resp_cor_)

            mean_reg = np.median(resp_reg_)
            mean_reg_b = np.median(resp_reg_b)
            mean_reg_c = np.median(resp_reg_c)
            mean_reg_uds = np.median(resp_reg_uds)
            mean_reg_g = np.median(resp_reg_g)
            mean_reg_tau = np.median(resp_reg_tau)
            mean_refReg_tau = np.median(resp_refReg_tau)
            mean_reg_err = getMedianError(resp_reg_)
            mean_reg_err_b = getMedianError(resp_reg_b)
            mean_reg_err_c = getMedianError(resp_reg_c)
            mean_reg_err_uds = getMedianError(resp_reg_uds)
            mean_reg_err_g = getMedianError(resp_reg_g)
            mean_reg_err_tau = getMedianError(resp_reg_tau)
            mean_refReg_err_tau = getMedianError(resp_refReg_tau)

            std_reg = rms(resp_reg_)
            std_reg_err = rms(resp_reg_)

            means_cor.append(mean_cor)
            means_cor_b.append(mean_cor_b)
            means_cor_c.append(mean_cor_c)
            means_cor_uds.append(mean_cor_uds)
            means_cor_g.append(mean_cor_g)
            means_cor_tau.append(mean_cor_tau)

            means_cor_err.append(mean_cor_err)
            means_cor_err_b.append(mean_cor_err_b)
            means_cor_err_c.append(mean_cor_err_c)
            means_cor_err_uds.append(mean_cor_err_uds)
            means_cor_err_g.append(mean_cor_err_g)
            means_cor_err_tau.append(mean_cor_err_tau)

            means_uncor.append(mean_uncor)
            means_uncor_b.append(mean_uncor_b)
            means_uncor_c.append(mean_uncor_c)
            means_uncor_uds.append(mean_uncor_uds)
            means_uncor_g.append(mean_uncor_g)
            means_uncor_tau.append(mean_uncor_tau)

            means_uncor_err.append(mean_uncor_err)
            means_uncor_err_b.append(mean_uncor_err_b)
            means_uncor_err_c.append(mean_uncor_err_c)
            means_uncor_err_uds.append(mean_uncor_err_uds)
            means_uncor_err_g.append(mean_uncor_err_g)
            means_uncor_err_tau.append(mean_uncor_err_tau)

            means_reg.append(mean_reg)
            means_reg_b.append(mean_reg_b)
            means_reg_c.append(mean_reg_c)
            means_reg_uds.append(mean_reg_uds)
            means_reg_g.append(mean_reg_g)
            means_reg_tau.append(mean_reg_tau)
            means_refReg_tau.append(mean_refReg_tau)

            means_reg_err.append(mean_reg_err)
            means_reg_err_b.append(mean_reg_err_b)
            means_reg_err_c.append(mean_reg_err_c)
            means_reg_err_uds.append(mean_reg_err_uds)
            means_reg_err_g.append(mean_reg_err_g)
            means_reg_err_tau.append(mean_reg_err_tau)
            means_refReg_err_tau.append(mean_refReg_err_tau)

            stds_cor.append(std_cor)
            stds_cor_err.append(std_cor_err)
            stds_uncor.append(std_uncor)
            stds_uncor_err.append(std_uncor_err)
            stds_reg.append(std_reg)
            stds_reg_err.append(std_reg_err)

            X = np.linspace(0.0, 2.0, 100)
            histo = plt.hist(resp_, bins=X, label='Uncorrected' ,histtype='step', density=True, color = '#1f77b4')
            histo = plt.hist(resp_cor_, bins=X, label='JEC LOT' ,histtype='step', density=True, color = '#ff7f0e')
            histo = plt.hist(resp_reg_, bins=X, label='Regression' ,histtype='step', density=True, color = '#2ca02c')
            plt.xlabel('Jet response (reco/gen)')
            plt.ylabel('Jets')
            plt.xlim(0.,2.)
            plt.legend(prop={'size': 10})
            plt.legend(loc='upper right')
            plt.text(1.3, 1.4, "median: "+str(np.round(mean_uncor,3))+" rms: "+str(np.round(std_uncor,3)), color = '#1f77b4', fontsize = 14)
            plt.text(1.3, 1.3, "median: "+str(np.round(mean_cor,3))+" rms: "+str(np.round(std_cor,3)), color = '#ff7f0e', fontsize = 14)
            plt.text(1.3, 1.2, "median: "+str(np.round(mean_reg,3))+" rms: "+str(np.round(std_reg,3)), color = '#2ca02c', fontsize = 14)
            plt.text(0.7, 2.2, str(ptLowEdge)+" GeV < $p_T$ (gen) < "+str(ptHighEdge)+" GeV", color = 'black', fontsize = 18)
            hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
            plt.savefig(outFolder+"/response_all_"+str(ptLowEdge)+"_"+str(ptHighEdge)+".png")
            plt.savefig(outFolder+"/response_all_"+str(ptLowEdge)+"_"+str(ptHighEdge)+".pdf")
            plt.cla()

        print (centers)
        # now plot the response vs gen pT
        means_cor = np.array(means_cor)
        means_cor_err = np.array(means_cor_err)
        means_uncor = np.array(means_uncor)
        means_uncor_err = np.array(means_uncor_err)
        means_reg = np.array(means_reg)
        means_reg_err = np.array(means_reg_err)
        stds_cor = np.array(stds_cor)
        stds_cor_err = np.array(stds_cor_err)
        stds_uncor = np.array(stds_uncor)
        stds_uncor_err = np.array(stds_uncor_err)
        stds_reg = np.array(stds_reg)
        stds_reg_err = np.array(stds_reg_err)
        centers = np.array(centers)
        plt.errorbar(centers, means_uncor, yerr = means_uncor_err, label='Uncorrected', linestyle = "-", marker = "o")
        plt.errorbar(centers, means_cor, yerr = means_cor_err, label='JEC LOT', linestyle = "-", marker = "o")
        plt.errorbar(centers, means_reg, yerr = means_reg_err, label='Regression', linestyle = "-", marker = "o")
        plt.xlabel('Jet gen $p_T$')
        plt.ylabel('Response (reco/gen)')
        plt.legend(prop={'size': 10})
        plt.legend(loc='upper right')
        hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
        plt.savefig(outFolder+"/response_vs_gen_pT"+".png")
        plt.savefig(outFolder+"/response_vs_gen_pT"+".pdf")
        plt.cla()

        # now plot the response vs gen pT - for b
        means_cor = np.array(means_cor_b)
        means_cor_err = np.array(means_cor_err_b)
        means_uncor = np.array(means_uncor_b)
        means_uncor_err = np.array(means_uncor_err_b)
        means_reg = np.array(means_reg_b)
        means_reg_err = np.array(means_reg_err_b)
        centers = np.array(centers)
        plt.errorbar(centers, means_uncor, yerr = means_uncor_err, label='Uncorrected - b', linestyle = "-", marker = "o", color = '#1f77b4')
        plt.errorbar(centers, means_cor, yerr = means_cor_err, label='JEC LOT - b', linestyle = "-", marker = "o", color = '#ff7f0e')
        plt.errorbar(centers, means_reg, yerr = means_reg_err, label='Regression - b', linestyle = "-", marker = "o", color = '#2ca02c')
        plt.xlabel('Jet gen $p_T$')
        plt.ylabel('Response (reco/gen)')
        plt.legend(prop={'size': 10})
        plt.legend(loc='upper right')
        hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
        plt.savefig(outFolder+"/response_vs_gen_pT_b"+".png")
        plt.savefig(outFolder+"/response_vs_gen_pT_b"+".pdf")
        plt.cla()

        # now plot the response vs gen pT - for c
        means_cor = np.array(means_cor_c)
        means_cor_err = np.array(means_cor_err_c)
        means_uncor = np.array(means_uncor_c)
        means_uncor_err = np.array(means_uncor_err_c)
        means_reg = np.array(means_reg_c)
        means_reg_err = np.array(means_reg_err_c)
        centers = np.array(centers)
        plt.errorbar(centers, means_uncor, yerr = means_uncor_err, label='Uncorrected - c', linestyle = "-", marker = "o", color = '#1f77b4')
        plt.errorbar(centers, means_cor, yerr = means_cor_err, label='JEC LOT - c', linestyle = "-", marker = "o", color = '#ff7f0e')
        plt.errorbar(centers, means_reg, yerr = means_reg_err, label='Regression - c', linestyle = "-", marker = "o", color = '#2ca02c')
        plt.xlabel('Jet gen $p_T$')
        plt.ylabel('Response (reco/gen)')
        plt.legend(prop={'size': 10})
        plt.legend(loc='upper right')
        hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
        plt.savefig(outFolder+"/response_vs_gen_pT_c"+".png")
        plt.savefig(outFolder+"/response_vs_gen_pT_c"+".pdf")
        plt.cla()
        
        # now plot the response vs gen pT - for uds
        means_cor = np.array(means_cor_uds)
        means_cor_err = np.array(means_cor_err_uds)
        means_uncor = np.array(means_uncor_uds)
        means_uncor_err = np.array(means_uncor_err_uds)
        means_reg = np.array(means_reg_uds)
        means_reg_err = np.array(means_reg_err_uds)
        centers = np.array(centers)
        plt.errorbar(centers, means_uncor, yerr = means_uncor_err, label='Uncorrected - uds', linestyle = "-", marker = "o", color = '#1f77b4')
        plt.errorbar(centers, means_cor, yerr = means_cor_err, label='JEC LOT - uds', linestyle = "-", marker = "o", color = '#ff7f0e')
        plt.errorbar(centers, means_reg, yerr = means_reg_err, label='Regression - uds', linestyle = "-", marker = "o", color = '#2ca02c')
        plt.xlabel('Jet gen $p_T$')
        plt.ylabel('Response (reco/gen)')
        plt.legend(prop={'size': 10})
        plt.legend(loc='upper right')
        hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
        plt.savefig(outFolder+"/response_vs_gen_pT_uds"+".png")
        plt.savefig(outFolder+"/response_vs_gen_pT_uds"+".pdf")
        plt.cla()
        
        # now plot the response vs gen pT - for g
        means_cor = np.array(means_cor_g)
        means_cor_err = np.array(means_cor_err_g)
        means_uncor = np.array(means_uncor_g)
        means_uncor_err = np.array(means_uncor_err_g)
        means_reg = np.array(means_reg_g)
        means_reg_err = np.array(means_reg_err_g)
        centers = np.array(centers)
        plt.errorbar(centers, means_uncor, yerr = means_uncor_err, label='Uncorrected - g', linestyle = "-", marker = "o", color = '#1f77b4')
        plt.errorbar(centers, means_cor, yerr = means_cor_err, label='JEC LOT - g', linestyle = "-", marker = "o", color = '#ff7f0e')
        plt.errorbar(centers, means_reg, yerr = means_reg_err, label='Regression - g', linestyle = "-", marker = "o", color = '#2ca02c')
        plt.xlabel('Jet gen $p_T$')
        plt.ylabel('Response (reco/gen)')
        plt.legend(prop={'size': 10})
        plt.legend(loc='upper right')
        hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
        plt.savefig(outFolder+"/response_vs_gen_pT_g"+".png")
        plt.savefig(outFolder+"/response_vs_gen_pT_g"+".pdf")
        plt.cla()
        
        # now plot the response vs gen pT - for tau
        means_cor = np.array(means_cor_tau)
        means_cor_err = np.array(means_cor_err_tau)
        means_uncor = np.array(means_uncor_tau)
        means_uncor_err = np.array(means_uncor_err_tau)
        means_reg = np.array(means_reg_tau)
        means_refReg = np.array(means_refReg_tau)
        means_reg_err = np.array(means_reg_err_tau)
        means_refReg_err = np.array(means_refReg_err_tau)
        centers = np.array(centers)
        plt.errorbar(centers, means_uncor, yerr = means_uncor_err, label='Uncorrected - tau', linestyle = "-", marker = "o", color = '#1f77b4')
        plt.errorbar(centers, means_cor, yerr = means_cor_err, label='JEC LOT - tau', linestyle = "-", marker = "o", color = '#ff7f0e')
        plt.errorbar(centers, means_reg, yerr = means_reg_err, label='Regression - tau', linestyle = "-", marker = "o", color = '#2ca02c')
        plt.errorbar(centers, means_refReg, yerr = means_refReg_err, label='Baseline-regression - tau', linestyle = "-", marker = "o", color = 'violet')
        plt.xlabel('Jet gen $p_T$')
        plt.ylabel('Response (reco/gen)')
        plt.legend(prop={'size': 10})
        plt.legend(loc='upper right')
        hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
        plt.savefig(outFolder+"/response_vs_gen_pT_tau"+".png")
        plt.savefig(outFolder+"/response_vs_gen_pT_tau"+".pdf")
        plt.cla()

        # now plot the resolution vs gen pT
        plt.plot(centers, stds_uncor, label='Uncorrected', linestyle = "-", marker = "o", color = '#1f77b4')
        plt.plot(centers, stds_cor, label='JEC LOT', linestyle = "-", marker = "o", color = '#ff7f0e')
        plt.plot(centers, stds_reg, label='Regression', linestyle = "-", marker = "o", color = '#2ca02c')
        plt.xlabel('Jet gen $p_T$')
        plt.ylabel('RMS (Response (reco/gen))')
        plt.legend(prop={'size': 10})
        plt.legend(loc='upper right')
        hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
        plt.savefig(outFolder+"/response_rms_vs_gen_pT"+".png")
        plt.savefig(outFolder+"/response_rms_vs_gen_pT"+".pdf")
        plt.cla()

        # for b
        mean_uncor = np.median(np.array(X_test_global[X_test_global["label_b"]>0]["response"]))
        std_uncor = rms(X_test_global[X_test_global["label_b"]>0]["response"])
        mean_cor = np.median(X_test_global[X_test_global["label_b"]>0]["response_cor"])
        std_cor = rms(X_test_global[X_test_global["label_b"]>0]["response_cor"])
        mean_reg = np.median(X_test_global[X_test_global["label_b"]>0]["response_reg"])
        std_reg = rms(X_test_global[X_test_global["label_b"]>0]["response_reg"])
        X = np.linspace(0.0, 2.0, 100)
        histo = plt.hist(X_test_global[X_test_global["label_b"]>0]["response"], bins=X, label='Uncorrected' ,histtype='step', density=True, color = '#1f77b4')
        histo = plt.hist(X_test_global[X_test_global["label_b"]>0]["response_cor"], bins=X, label='JEC LOT' ,histtype='step', density=True, color = '#ff7f0e')
        histo = plt.hist(X_test_global[X_test_global["label_b"]>0]["response_reg"], bins=X, label='Regression' ,histtype='step', density=True, color = '#2ca02c')
        plt.xlabel('Jet response (reco/gen)')
        plt.ylabel('Jets')
        plt.xlim(0.,2.)
        plt.legend(prop={'size': 10})
        plt.legend(loc='upper right')
        plt.text(1.3, 1.4, "median: "+str(np.round(mean_uncor,3))+" rms: "+str(np.round(std_uncor,3)), color = '#1f77b4', fontsize = 14)
        plt.text(1.3, 1.3, "median: "+str(np.round(mean_cor,3))+" rms: "+str(np.round(std_cor,3)), color = '#ff7f0e', fontsize = 14)
        plt.text(1.3, 1.2, "median: "+str(np.round(mean_reg,3))+" rms: "+str(np.round(std_reg,3)), color = '#2ca02c', fontsize = 14)
        hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
        plt.savefig(outFolder+"/response_b"+".png")
        plt.savefig(outFolder+"/response_b"+".pdf")
        plt.cla()

        # for g
        mean_uncor = np.median(np.array(X_test_global[X_test_global["label_g"]>0]["response"]))
        std_uncor = rms(X_test_global[X_test_global["label_g"]>0]["response"])
        mean_cor = np.median(X_test_global[X_test_global["label_g"]>0]["response_cor"])
        std_cor = rms(X_test_global[X_test_global["label_g"]>0]["response_cor"])
        mean_reg = np.median(X_test_global[X_test_global["label_g"]>0]["response_reg"])
        std_reg = rms(X_test_global[X_test_global["label_g"]>0]["response_reg"])
        X = np.linspace(0.0, 2.0, 100)
        histo = plt.hist(X_test_global[X_test_global["label_g"]>0]["response"], bins=X, label='Uncorrected' ,histtype='step', density=True, color = '#1f77b4')
        histo = plt.hist(X_test_global[X_test_global["label_g"]>0]["response_cor"], bins=X, label='JEC LOT' ,histtype='step', density=True, color = '#ff7f0e')
        histo = plt.hist(X_test_global[X_test_global["label_g"]>0]["response_reg"], bins=X, label='Regression' ,histtype='step', density=True, color = '#2ca02c')
        plt.xlabel('Jet response (reco/gen)')
        plt.ylabel('Jets')
        plt.xlim(0.,2.)
        plt.legend(prop={'size': 10})
        plt.legend(loc='upper right')
        plt.text(1.3, 1.4, "median: "+str(np.round(mean_uncor,3))+" rms: "+str(np.round(std_uncor,3)), color = '#1f77b4', fontsize = 14)
        plt.text(1.3, 1.3, "median: "+str(np.round(mean_cor,3))+" rms: "+str(np.round(std_cor,3)), color = '#ff7f0e', fontsize = 14)
        plt.text(1.3, 1.2, "median: "+str(np.round(mean_reg,3))+" rms: "+str(np.round(std_reg,3)), color = '#2ca02c', fontsize = 14)
        hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
        plt.savefig(outFolder+"/response_g"+".png")
        plt.savefig(outFolder+"/response_g"+".pdf")
        plt.cla()

        # for uds
        mean_uncor = np.median(np.array(X_test_global[X_test_global["label_uds"]>0]["response"]))
        std_uncor = rms(X_test_global[X_test_global["label_uds"]>0]["response"])
        mean_cor = np.median(X_test_global[X_test_global["label_uds"]>0]["response_cor"])
        std_cor = rms(X_test_global[X_test_global["label_uds"]>0]["response_cor"])
        mean_reg = np.median(X_test_global[X_test_global["label_uds"]>0]["response_reg"])
        std_reg = rms(X_test_global[X_test_global["label_uds"]>0]["response_reg"])
        X = np.linspace(0.0, 2.0, 100)
        histo = plt.hist(X_test_global[X_test_global["label_uds"]>0]["response"], bins=X, label='Uncorrected' ,histtype='step', density=True, color = '#1f77b4')
        histo = plt.hist(X_test_global[X_test_global["label_uds"]>0]["response_cor"], bins=X, label='JEC LOT' ,histtype='step', density=True, color = '#ff7f0e')
        histo = plt.hist(X_test_global[X_test_global["label_uds"]>0]["response_reg"], bins=X, label='Regression' ,histtype='step', density=True, color = '#2ca02c')
        plt.xlabel('Jet response (reco/gen)')
        plt.ylabel('Jets')
        plt.xlim(0.,2.)
        plt.legend(prop={'size': 10})
        plt.legend(loc='upper right')
        plt.text(1.3, 1.4, "median: "+str(np.round(mean_uncor,3))+" rms: "+str(np.round(std_uncor,3)), color = '#1f77b4', fontsize = 14)
        plt.text(1.3, 1.3, "median: "+str(np.round(mean_cor,3))+" rms: "+str(np.round(std_cor,3)), color = '#ff7f0e', fontsize = 14)
        plt.text(1.3, 1.2, "median: "+str(np.round(mean_reg,3))+" rms: "+str(np.round(std_reg,3)), color = '#2ca02c', fontsize = 14)
        hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
        plt.savefig(outFolder+"/response_uds"+".png")
        plt.savefig(outFolder+"/response_uds"+".pdf")
        plt.cla()

        # for c
        mean_uncor = np.median(np.array(X_test_global[X_test_global["label_c"]>0]["response"]))
        std_uncor = rms(X_test_global[X_test_global["label_c"]>0]["response"])
        mean_cor = np.median(X_test_global[X_test_global["label_c"]>0]["response_cor"])
        std_cor = rms(X_test_global[X_test_global["label_c"]>0]["response_cor"])
        mean_reg = np.median(X_test_global[X_test_global["label_c"]>0]["response_reg"])
        std_reg = rms(X_test_global[X_test_global["label_c"]>0]["response_reg"])
        X = np.linspace(0.0, 2.0, 100)
        histo = plt.hist(X_test_global[X_test_global["label_c"]>0]["response"], bins=X, label='Uncorrected' ,histtype='step', density=True, color = '#1f77b4')
        histo = plt.hist(X_test_global[X_test_global["label_c"]>0]["response_cor"], bins=X, label='JEC LOT' ,histtype='step', density=True, color = '#ff7f0e')
        histo = plt.hist(X_test_global[X_test_global["label_c"]>0]["response_reg"], bins=X, label='Regression' ,histtype='step', density=True, color = '#2ca02c')
        plt.xlabel('Jet response (reco/gen)')
        plt.ylabel('Jets')
        plt.xlim(0.,2.)
        plt.legend(prop={'size': 10})
        plt.legend(loc='upper right')
        plt.text(1.3, 1.4, "median: "+str(np.round(mean_uncor,3))+" rms: "+str(np.round(std_uncor,3)), color = '#1f77b4', fontsize = 14)
        plt.text(1.3, 1.3, "median: "+str(np.round(mean_cor,3))+" rms: "+str(np.round(std_cor,3)), color = '#ff7f0e', fontsize = 14)
        plt.text(1.3, 1.2, "median: "+str(np.round(mean_reg,3))+" rms: "+str(np.round(std_reg,3)), color = '#2ca02c', fontsize = 14)
        hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
        plt.savefig(outFolder+"/response_c"+".png")
        plt.savefig(outFolder+"/response_c"+".pdf")
        plt.cla()

        # for tau
        mean_uncor = np.median(np.array(X_test_global[X_test_global["label_tau"]>0]["response"]))
        std_uncor = rms(X_test_global[X_test_global["label_tau"]>0]["response"])
        mean_cor = np.median(X_test_global[X_test_global["label_tau"]>0]["response_cor"])
        std_cor = rms(X_test_global[X_test_global["label_tau"]>0]["response_cor"])
        mean_reg = np.median(X_test_global[X_test_global["label_tau"]>0]["response_reg"])
        mean_refReg = np.median(X_test_global[X_test_global["label_tau"]>0]["response_tau"])
        std_reg = rms(X_test_global[X_test_global["label_tau"]>0]["response_reg"])
        std_refReg = rms(X_test_global[X_test_global["label_tau"]>0]["response_tau"])
        X = np.linspace(0.0, 2.0, 100)
        histo = plt.hist(X_test_global[X_test_global["label_tau"]>0]["response"], bins=X, label='Uncorrected' ,histtype='step', density=True, color = '#1f77b4')
        histo = plt.hist(X_test_global[X_test_global["label_tau"]>0]["response_cor"], bins=X, label='JEC LOT' ,histtype='step', density=True, color = '#ff7f0e')
        histo = plt.hist(X_test_global[X_test_global["label_tau"]>0]["response_reg"], bins=X, label='Regression' ,histtype='step', density=True, color = '#2ca02c')
        histo = plt.hist(X_test_global[X_test_global["label_tau"]>0]["response_tau"], bins=X, label='Baseline-regression' ,histtype='step', density=True, color = 'violet')
        plt.xlabel('Jet response (reco/gen)')
        plt.ylabel('Jets')
        plt.xlim(0.,2.)
        plt.legend(prop={'size': 10})
        plt.legend(loc='upper right')
        plt.text(1.3, 1.4, "median: "+str(np.round(mean_uncor,3))+" rms: "+str(np.round(std_uncor,3)), color = '#1f77b4', fontsize = 14)
        plt.text(1.3, 1.3, "median: "+str(np.round(mean_cor,3))+" rms: "+str(np.round(std_cor,3)), color = '#ff7f0e', fontsize = 14)
        plt.text(1.3, 1.2, "median: "+str(np.round(mean_reg,3))+" rms: "+str(np.round(std_reg,3)), color = '#2ca02c', fontsize = 14)
        plt.text(1.3, 1.1, "median: "+str(np.round(mean_refReg,3))+" rms: "+str(np.round(std_refReg,3)), color = 'violet', fontsize = 14)
        hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
        plt.savefig(outFolder+"/response_tau"+".png")
        plt.savefig(outFolder+"/response_tau"+".pdf")
        plt.cla()

        if modelArchName in ["DeepSet", "DeepSet-MHA"]:
            model = modelsAndNames["model"]
            model2 = keras.Model(model.input, model.output[0])
            model3 = keras.Model(model.input, model.output[1])
            for explainer, name  in [(shap.GradientExplainer(model2, X_test[:1000]), "GradientExplainer"), ]:
                print("... {0}: explainer.shap_values(X)".format(name))
                shap_values = explainer.shap_values(X_test[:1000])
                new = np.sum(shap_values, axis = 2)
                print("... shap summary_plot classification")
                plt.clf()
                labels = ["uds", "b"]
                labels.append("Tau p")
                labels.append("Tau m")
                labels.append("Gluon")
                labels.append("Charm")
                labels.append("Muon")
                labels.append("Electron")
                shapPlot(new, feature_names, labels)
                plt.savefig(outFolder+"/shap_summary_class_"+inputSetTag+"_{0}.pdf".format(name))
                plt.savefig(outFolder+"/shap_summary_class_"+inputSetTag+"_{0}.png".format(name))

            for explainer, name  in [(shap.GradientExplainer(model3, X_test[:1000]), "GradientExplainer"), ]:
                print("... {0}: explainer.shap_values(X)".format(name))
                shap_values = explainer.shap_values(X_test[:1000])
                new = np.sum(shap_values, axis = 2)
                print("... shap summary_plot regression")
                plt.clf()
                labels = ["Regression"]
                shapPlot(new, feature_names, labels)
                plt.savefig(outFolder+"/shap_summary_reg_"+inputSetTag+"_{0}.pdf".format(name))
                plt.savefig(outFolder+"/shap_summary_reg_"+inputSetTag+"_{0}.png".format(name))



if __name__ == "__main__":
    from args import get_common_parser, handle_common_args
    parser = get_common_parser()
    parser.add_argument('-f','--file', help = 'input file name part')
    parser.add_argument('-o','--outname', help = 'output file name part')
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