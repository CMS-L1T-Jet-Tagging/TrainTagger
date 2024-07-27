from imports import *
from dataset import *
import argparse
from models import *
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks, pruning_wrapper,  pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning

from utils.createDataset import *

from sklearn.metrics import roc_curve, auc,precision_recall_curve
import matplotlib.pyplot as plt
import pandas as pd
import shap
import json
import glob
from datetime import datetime

import pdb

'''
# https://www.tensorflow.org/model_optimization/guide/pruning/comprehensive_guide#prune_custom_keras_layer_or_modify_parts_of_layer_to_prune
# The code bellow prunes the model as a whole
# Prune model small weights that don't impact network overall performance. 
# Let's remove 50% of the weights (spasity=0.5).  The training step is one gradient update, or epochs*N_samples/batchsize
pruning_params = {"pruning_schedule": pruning_schedule.ConstantSparsity(0.5, begin_step=6000, frequency=10)}
model = prune.prune_low_magnitude(model, **pruning_params)
'''

def pruneFunction(layer):
#    pruning_params = {'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.0, final_sparsity=0.5, begin_step=NSTEPS*2, end_step=NSTEPS*10, frequency=NSTEPS)}
    pruning_params = {"pruning_schedule": pruning_schedule.ConstantSparsity(0.5, begin_step=6000, frequency=10)}
    # i_sparsity = 0 # initial 
    # f_sparsity = 0.6 # final
    # # num_samples = X_train.shape[0] * (1 - nnConfig["validation_split"])
    # num_samples = 4271641 * (1 - 0.25)
    # # end_step = np.ceil(num_samples / nnConfig["batch_size"]).astype(np.int32) * nnConfig["epochs"]
    # end_step = np.ceil(num_samples / 1024).astype(np.int32) * 150
    # pruning_params = {
    #           'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=i_sparsity,
    #                                                                    final_sparsity=f_sparsity,
    #                                                                    begin_step=0,
    #                                                                    end_step=end_step)
    # }
        
        # Apply prunning to Dense layers type excluding the output layer
    # if isinstance(layer, tf.keras.layers.Dense) and layer.name != 'dense_out': # exclude output_dense
    if isinstance(layer, tf.keras.layers.Dense) and not layer.name in ["qDense_out_class","qDense_out_reg"]: # exclude output_dense
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
#
#        if isinstance(layer, tf.keras.layers.Conv1D): 
#            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
#
    return layer
#


def plotInputFeatures(Xb, Xuds, Xtau, Xtaum, Xgluon, Xcharm, Xmuon, Xelectron, featureNames, outFolder, outputAddName = ""):
    print ("Plot all input features w/ add. name", outputAddName)
    plt.figure()
    for idxFeature, name in enumerate(featureNames):
        b_ = ak.flatten(Xb[:,:,idxFeature][Xb[:,:,0]!=0.],axis=-1)
        uds_ = ak.flatten(Xuds[:,:,idxFeature][Xuds[:,:,0]!=0.],axis=-1)
        g_ = ak.flatten(Xgluon[:,:,idxFeature][Xgluon[:,:,0]!=0.],axis=-1)
        tau_ = ak.flatten(Xtau[:,:,idxFeature][Xtau[:,:,0]!=0.],axis=-1)
        if Xtaum is not None:
            taum_ = ak.flatten(Xtaum[:,:,idxFeature][Xtaum[:,:,0]!=0.],axis=-1)
        charm_ = ak.flatten(Xcharm[:,:,idxFeature][Xcharm[:,:,0]!=0.],axis=-1)
        muon_ = ak.flatten(Xmuon[:,:,idxFeature][Xmuon[:,:,0]!=0.],axis=-1)
        electron_ = ak.flatten(Xelectron[:,:,idxFeature][Xelectron[:,:,0]!=0.],axis=-1)
        min_ = min(min(b_), min(uds_))
        max_ = max(max(b_), max(uds_))
        min_ = min(min_, min(tau_))
        max_ = max(max_, max(tau_))
        min_ = min(min_, min(g_))
        max_ = max(max_, max(g_))
        min_ = min(min_, min(charm_))
        max_ = max(max_, max(charm_))
        range = (min_, max_)
        plt.hist(b_, label='b', bins = 200, density = True, log = True, histtype = "step", range = range, color="blue")
        plt.hist(uds_, label='uds', bins = 200, density = True, log = True, histtype = "step", range = range, color="orange")
        plt.hist(g_, label='Gluon', bins = 200, density = True, log = True, histtype = "step", range = range, color="green")
        plt.hist(tau_, label='Tau p', bins = 200, density = True, log = True, histtype = "step", range = range, color="red")
        if Xtaum is not None:
            plt.hist(taum_, label='Tau m', bins = 200, density = True, log = True, histtype = "step", range = range, color="purple")
        plt.hist(charm_, label='Charm', bins = 200, density = True, log = True, histtype = "step", range = range, color="black")
        plt.hist(muon_, label='Muon', bins = 200, density = True, log = True, histtype = "step", range = range, color="yellow")
        plt.hist(electron_, label='Electron', bins = 200, density = True, log = True, histtype = "step", range = range, color="cyan")
        plt.legend(loc = "upper right")
        plt.xlabel(f"{name}")
        plt.ylabel('Jets (Normalized to 1)')
        hep.cms.label("Private Work", data = False, rlabel = "14 TeV (PU 200)")
        plt.savefig(outFolder+"/inputFeature_"+name+outputAddName+".png")
        plt.savefig(outFolder+"/inputFeature_"+name+outputAddName+".pdf")
        plt.cla()


def rms(array):
   return np.sqrt(np.mean(array ** 2))


def doTraining(
        filetag,
        flavs,
        inputSetTag,
        nnConfig,
        save = True,
        strstamp = "strstamp",
        test = False,
        plotFeatures = False,
        workdir = "./",):

    outFolder = "trainings/"
    if nnConfig["classweights"]:
        outFolder = "trainings_weighted/"
    if nnConfig["regression"]:
        outFolder = outFolder.replace("trainings_","trainings_regression_")

    if not os.path.exists(outFolder):
        os.makedirs(outFolder, exist_ok=True)

    feature_names = dict_fields[inputSetTag]

    nconstit = 16

    PATH_load = workdir + '/datasetsNewComplete2/' + filetag + "/" + flavs + "/"
    print (PATH_load)
    chunksmatching = glob.glob(PATH_load+"X_"+inputSetTag+"_test*.parquet")
    chunksmatching = [chunksm.replace(PATH_load+"X_"+inputSetTag+"_test","").replace(".parquet","").replace("_","") for chunksm in chunksmatching]

    import random
    if test:
        chunksmatching = random.sample(chunksmatching, 10)
    else:
        chunksmatching = random.sample(chunksmatching, len(chunksmatching))


    print ("Loading data in all",len(chunksmatching),"chunks.")

    X_train_val = None
    X_test = None
    Y_train_val = None
    Y_train_val_reg = None
    Y_test = None
    Y_test_reg = None
    X_train_global = None
    X_test_global = None
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
    x_electron = None
    x_electron_global = None
    x_muon = None
    x_muon_global = None

    for c in chunksmatching:
        if X_test is None:
            X_test = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_test_"+c+".parquet")
            X_train_global = ak.from_parquet(PATH_load+"X_global_"+inputSetTag+"_train_"+c+".parquet")
            X_test_global = ak.from_parquet(PATH_load+"X_global_"+inputSetTag+"_test_"+c+".parquet")
            X_train_val = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_train_"+c+".parquet")
            Y_test = ak.from_parquet(PATH_load+"Y_"+inputSetTag+"_test_"+c+".parquet")
            Y_test_reg = ak.from_parquet(PATH_load+"Y_target_"+inputSetTag+"_test_"+c+".parquet")
            Y_train_val = ak.from_parquet(PATH_load+"Y_"+inputSetTag+"_train_"+c+".parquet")
            Y_train_val_reg = ak.from_parquet(PATH_load+"Y_target_"+inputSetTag+"_train_"+c+".parquet")
        else:
            X_test =ak.concatenate((X_test, ak.from_parquet(PATH_load+"X_"+inputSetTag+"_test_"+c+".parquet")))
            X_test_global =ak.concatenate((X_test_global, ak.from_parquet(PATH_load+"X_global_"+inputSetTag+"_test_"+c+".parquet")))
            X_train_global =ak.concatenate((X_train_global, ak.from_parquet(PATH_load+"X_global_"+inputSetTag+"_train_"+c+".parquet")))
            X_train_val =ak.concatenate((X_train_val, ak.from_parquet(PATH_load+"X_"+inputSetTag+"_train_"+c+".parquet")))
            Y_test =ak.concatenate((Y_test, ak.from_parquet(PATH_load+"Y_"+inputSetTag+"_test_"+c+".parquet")))
            Y_test_reg =ak.concatenate((Y_test_reg, ak.from_parquet(PATH_load+"Y_target_"+inputSetTag+"_test_"+c+".parquet")))
            Y_train_val =ak.concatenate((Y_train_val, ak.from_parquet(PATH_load+"Y_"+inputSetTag+"_train_"+c+".parquet")))
            Y_train_val_reg =ak.concatenate((Y_train_val_reg, ak.from_parquet(PATH_load+"Y_target_"+inputSetTag+"_train_"+c+".parquet")))

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

    # rebalance data set
    X_train_val = ak.to_numpy(X_train_val)
    X_test = ak.to_numpy(X_test)
    Y_train_val = ak.to_numpy(Y_train_val)
    Y_train_val_reg = ak.to_numpy(Y_train_val_reg)
    Y_test = ak.to_numpy(Y_test)
    Y_test_reg = ak.to_numpy(Y_test_reg)

    print("Loaded X_train_val ----> shape:", X_train_val.shape)
    print("Loaded X_test      ----> shape:", X_test.shape)
    print("Loaded Y_train_val ----> shape:", Y_train_val.shape)
    print("Loaded Y_test      ----> shape:", Y_test.shape)

    nbits = nnConfig["nbits"]
    integ = nnConfig["integ"]
    nfeat = X_train_val.shape[-1]

    if nnConfig["model"] == "DeepSet":
        model, modelname, custom_objects = getDeepSet(nclasses = len(Y_train_val[0]), input_shape = (nconstit, nfeat),
                                                      nnodes_phi = nnConfig["nNodes"], nnodes_rho = nnConfig["nNodes"],
                                                      nbits = nbits, integ = integ, addRegression = nnConfig["regression"], nLayers = nnConfig["nLayers"])
    elif nnConfig["model"] == "MLP":
        model, modelname, custom_objects = getMLP(nclasses = len(Y_train_val[0]), input_shape = (nconstit*nfeat)+4,
                                                      nnodes_phi = nnConfig["nNodes"], nnodes_rho = nnConfig["nNodes"],
                                                      nbits = nbits, integ = integ, addRegression = nnConfig["regression"], nLayers = nnConfig["nLayers"], nFeatures = nfeat)
        # pdb.set_trace()
        X_train_val = np.reshape(X_train_val, (X_train_val.shape[0],X_train_val.shape[1]*X_train_val.shape[2]))
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
        X_train_val_global = np.stack((X_train_global["jet_pt_raw"],X_train_global["jet_eta"],X_train_global["jet_phi"],X_train_global["jet_npfcand"]), axis=1)
        X_b_global = np.stack((x_b_global["jet_pt_raw"],x_b_global["jet_eta"],x_b_global["jet_phi"],x_b_global["jet_npfcand"]), axis=1)
        X_bkg_global = np.stack((x_bkg_global["jet_pt_raw"],x_bkg_global["jet_eta"],x_bkg_global["jet_phi"],x_bkg_global["jet_npfcand"]), axis=1)
        X_taup_global = np.stack((x_taup_global["jet_pt_raw"],x_taup_global["jet_eta"],x_taup_global["jet_phi"],x_taup_global["jet_npfcand"]), axis=1)
        X_taum_global = np.stack((x_taum_global["jet_pt_raw"],x_taum_global["jet_eta"],x_taum_global["jet_phi"],x_taum_global["jet_npfcand"]), axis=1)
        X_gluon_global = np.stack((x_gluon_global["jet_pt_raw"],x_gluon_global["jet_eta"],x_gluon_global["jet_phi"],x_gluon_global["jet_npfcand"]), axis=1)
        X_charm_global = np.stack((x_charm_global["jet_pt_raw"],x_charm_global["jet_eta"],x_charm_global["jet_phi"],x_charm_global["jet_npfcand"]), axis=1)
        X_muon_global = np.stack((x_muon_global["jet_pt_raw"],x_muon_global["jet_eta"],x_muon_global["jet_phi"],x_muon_global["jet_npfcand"]), axis=1)
        X_electron_global = np.stack((x_electron_global["jet_pt_raw"],x_electron_global["jet_eta"],x_electron_global["jet_phi"],x_electron_global["jet_npfcand"]), axis=1)
        X_test_val_global = np.stack((X_test_global["jet_pt_raw"],X_test_global["jet_eta"],X_test_global["jet_phi"],X_test_global["jet_npfcand"]), axis=1)

        X_train_val = np.concatenate((np.array(X_train_val_global),X_train_val), axis=-1)
        x_b = np.concatenate((np.array(X_b_global),x_b), axis=-1)
        x_bkg = np.concatenate((np.array(X_bkg_global),x_bkg), axis=-1)
        x_taup = np.concatenate((np.array(X_taup_global),x_taup), axis=-1)
        x_taum = np.concatenate((np.array(X_taum_global),x_taum), axis=-1)
        x_gluon = np.concatenate((np.array(X_gluon_global),x_gluon), axis=-1)
        x_charm = np.concatenate((np.array(X_charm_global),x_charm), axis=-1)
        x_muon = np.concatenate((np.array(X_muon_global),x_muon), axis=-1)
        x_electron = np.concatenate((np.array(X_electron_global),x_electron), axis=-1)
        X_test = np.concatenate((np.array(X_test_val_global),X_test), axis=-1)

    elif nnConfig["model"] == "DeepSet-MHA":
        model, modelname, custom_objects = getDeepSetWAttention(nclasses = len(Y_train_val[0]), input_shape = (nconstit, nfeat),
                                                                nnodes_phi = nnConfig["nNodes"], nnodes_rho = nnConfig["nNodes"]/2,
                                                                nbits = nbits, integ = integ,
                                                                n_head = nnConfig["nHeads"], dim = nfeat, dim2 = nnConfig["nNodesHead"], addRegression = nnConfig["regression"])
    elif nnConfig["model"] == "MLP-MHA":
        model, modelname, custom_objects = getMLPWAttention(nclasses = len(Y_train_val[0]), input_shape = (nconstit*nfeat),
                                                                nnodes_phi = nnConfig["nNodes"], nnodes_rho = nnConfig["nNodes"]/2,
                                                                nbits = nbits, integ = integ,
                                                                n_head = nnConfig["nHeads"], dim = nfeat, dim2 = nnConfig["nNodesHead"], addRegression = nnConfig["regression"], nLayers = nnConfig["nLayers"],
                                                                nFeatures = nfeat)

        X_train_val = np.reshape(X_train_val, (X_train_val.shape[0],X_train_val.shape[1]*X_train_val.shape[2]))
        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1]*X_test.shape[2]))
        x_b = np.reshape(x_b, (x_b.shape[0],x_b.shape[1]*x_b.shape[2]))
        x_bkg = np.reshape(x_bkg, (x_bkg.shape[0],x_bkg.shape[1]*x_bkg.shape[2]))
        x_taup = np.reshape(x_taup, (x_taup.shape[0],x_taup.shape[1]*x_taup.shape[2]))
        x_taum = np.reshape(x_taum, (x_taum.shape[0],x_taum.shape[1]*x_taum.shape[2]))
        x_gluon = np.reshape(x_gluon, (x_gluon.shape[0],x_gluon.shape[1]*x_gluon.shape[2]))
        x_charm = np.reshape(x_charm, (x_charm.shape[0],x_charm.shape[1]*x_charm.shape[2]))
        x_muon = np.reshape(x_muon, (x_muon.shape[0],x_muon.shape[1]*x_muon.shape[2]))
        x_electron = np.reshape(x_electron, (x_electron.shape[0],x_electron.shape[1]*x_electron.shape[2]))

    if nnConfig["pruning"]:
        modelname = modelname + "_pruned"
    if nnConfig["inputQuant"]:
        modelname = modelname + "_inputQuant"

    print('Model name :', modelname)

    print ("Model summary:")
    # print the model summary
    model.summary()

    outFolder = outFolder + "/"+ strstamp + "_" + filetag + "_" + flavs + "_" + inputSetTag + "_" + modelname + "/"
    if not os.path.exists(outFolder):
        os.makedirs(outFolder, exist_ok=True)
    print ("Use the following output folder:", outFolder)

    if plotFeatures:
        plotInputFeatures(x_b, x_bkg, x_taup, x_taum, x_gluon, x_charm, x_muon, x_electron, feature_names, outFolder, outputAddName = "")

    if nnConfig["inputQuant"]:
        input_quantizer = quantized_bits(bits=16, integer=6, symmetric=0, alpha=1)
        x_b = input_quantizer(x_b.astype(np.float32)).numpy()
        x_bkg = input_quantizer(x_bkg.astype(np.float32)).numpy()
        x_taup = input_quantizer(x_taup.astype(np.float32)).numpy()
        x_taum = input_quantizer(x_taum.astype(np.float32)).numpy()
        x_gluon = input_quantizer(x_gluon.astype(np.float32)).numpy()
        x_charm = input_quantizer(x_charm.astype(np.float32)).numpy()

        plotInputFeatures(x_b, x_bkg, x_taup+x_taum, x_gluon, x_charm, feature_names, outFolder, outputAddName = "_quant")

    # calculate class weights
    class_weights = np.ones(len(Y_train_val[0]))
    # class_weights = np.array([2, 1, 1, 2, 2, 2, 1, 1])

    # get pT weights, weighting all to b spectrum
    # bins_pt_weights = np.array([15, 20, 25, 30, 38, 48, 60, 76, 97, 122, 154, 195, 246, 311, 393, 496, 627, 792, 9999999999999999999])
    # bins_pt_weights = np.array([15, 18, 22, 25, 30, 38, 48, 60, 76, 97, 122, 154, 195, 246, 311, 393, 496, 627, 792, 9999999999999999999])
    bins_pt_weights = np.array([15, 17, 19, 22, 25, 30, 35, 40, 45, 50, 60, 76, 97, 122, 154, 195, 246, 311, 393, 496, 627, 792, 9999999999999999999])
    # bins_pt_weights = np.array([15, 17, 19, 22, 25, 30, 35, 40, 45, 50, 60, 76, 97, 122, 154, 195, 246, 311, 393, 496, 9999999999999999999])
    # bins_pt_weights = np.array([15, 18, 22, 25, 30, 38, 48, 60, 76, 97, 122, 154, 200, 300, 400, 627, 9999999999999999999])
    # bins_pt_weights = np.array([15, 18, 22, 25, 30, 38, 48, 60, 76, 97, 122, 154, 300, 627, 9999999999999999999])
    counts_b, edges_b = np.histogram(X_train_global[X_train_global["label_b"]>0]["jet_pt_phys"], bins = bins_pt_weights) 
    counts_uds, edges_uds = np.histogram(X_train_global[X_train_global["label_uds"]>0]["jet_pt_phys"], bins = bins_pt_weights) 
    counts_g, edges_g = np.histogram(X_train_global[X_train_global["label_g"]>0]["jet_pt_phys"], bins = bins_pt_weights) 
    counts_c, edges_c = np.histogram(X_train_global[X_train_global["label_c"]>0]["jet_pt_phys"], bins = bins_pt_weights) 
    counts_taup, edges_taup = np.histogram(X_train_global[X_train_global["label_taup"]>0]["jet_pt_phys"], bins = bins_pt_weights) 
    counts_taum, edges_taum = np.histogram(X_train_global[X_train_global["label_taum"]>0]["jet_pt_phys"], bins = bins_pt_weights) 
    counts_muon, edges_muon = np.histogram(X_train_global[X_train_global["label_muon"]>0]["jet_pt_phys"], bins = bins_pt_weights) 
    counts_electron, edges_electron = np.histogram(X_train_global[X_train_global["label_electron"]>0]["jet_pt_phys"], bins = bins_pt_weights) 

    # print(counts_b, counts_uds, counts_g, counts_c, counts_taup, counts_taum, counts_muon, counts_electron)
    for tp in (counts_b, counts_uds, counts_g, counts_c, counts_taup, counts_taum, counts_muon, counts_electron):
        print (tp)

    w_b = np.nan_to_num(counts_b/counts_b * class_weights[0], nan = 1., posinf = 1., neginf = 1.)
    w_uds =  np.nan_to_num(counts_b/counts_uds * class_weights[1], nan = 1., posinf = 1., neginf = 1.)
    w_g =  np.nan_to_num(counts_b/counts_g * class_weights[2], nan = 1., posinf = 1., neginf = 1.)
    w_c =  np.nan_to_num(counts_b/counts_c * class_weights[3], nan = 1., posinf = 1., neginf = 1.)
    w_taup =  np.nan_to_num(counts_b/counts_taup * class_weights[4], nan = 1., posinf = 1., neginf = 1.)
    w_taum =  np.nan_to_num(counts_b/counts_taum * class_weights[5], nan = 1., posinf = 1., neginf = 1.)
    w_muon =  np.nan_to_num(counts_b/counts_muon * class_weights[6], nan = 1., posinf = 1., neginf = 1.)
    w_electron =  np.nan_to_num(counts_b/counts_electron * class_weights[7], nan = 1., posinf = 1., neginf = 1.)

    # print (w_uds)

    # do it flat in pt and overwrite the other one
    # enable this for tests - if we can do a flat pT weighting and if it helps
    # otherwise just comment this for loop block
    for iBin in range(0, len(counts_b)):
        w_b[iBin] = np.nan_to_num(counts_b[0] / counts_b[iBin], nan = 1., posinf = 1., neginf = 1.)
        w_uds[iBin] = np.nan_to_num(counts_b[0] / counts_uds[iBin], nan = 1., posinf = 1., neginf = 1.)
        w_g[iBin] = np.nan_to_num(counts_b[0] / counts_g[iBin], nan = 1., posinf = 1., neginf = 1.)
        w_c[iBin] = np.nan_to_num(counts_b[0] / counts_c[iBin], nan = 1., posinf = 1., neginf = 1.)
        w_taup[iBin] = np.nan_to_num(counts_b[0] / counts_taup[iBin], nan = 1., posinf = 1., neginf = 1.)
        w_taum[iBin] = np.nan_to_num(counts_b[0] / counts_taum[iBin], nan = 1., posinf = 1., neginf = 1.)
        w_muon[iBin] = np.nan_to_num(counts_b[0] / counts_muon[iBin], nan = 1., posinf = 1., neginf = 1.)
        w_electron[iBin] = np.nan_to_num(counts_b[0] / counts_electron[iBin], nan = 1., posinf = 1., neginf = 1.)

    # print (w_uds)

    X_train_global["weight_jetpT_binidx"] = to_categorical( np.digitize(X_train_global["jet_pt_phys"], bins_pt_weights)-1)

    X_train_global["weight_pt"] = (X_train_global["label_b"]*ak.sum(w_b*X_train_global["weight_jetpT_binidx"], axis =-1) + \
                                X_train_global["label_uds"]*ak.sum(w_uds*X_train_global["weight_jetpT_binidx"], axis =-1) + \
                                X_train_global["label_g"]*ak.sum(w_g*X_train_global["weight_jetpT_binidx"], axis =-1) + \
                                X_train_global["label_c"]*ak.sum(w_c*X_train_global["weight_jetpT_binidx"], axis =-1) + \
                                X_train_global["label_taup"]*ak.sum(w_taup*X_train_global["weight_jetpT_binidx"], axis =-1)+ \
                                X_train_global["label_taum"]*ak.sum(w_taum*X_train_global["weight_jetpT_binidx"], axis =-1)+ \
                                X_train_global["label_muon"]*ak.sum(w_muon*X_train_global["weight_jetpT_binidx"], axis =-1)+ \
                                X_train_global["label_electron"]*ak.sum(w_electron*X_train_global["weight_jetpT_binidx"], axis =-1)
                                )

    wSum_b = ak.sum(( (X_train_global["weight_pt"]) * (X_train_global["label_b"]>0) )) / ak.sum(( (X_train_global["weight_pt"]) * (X_train_global["label_b"]>0) ))
    wSum_uds = ak.sum(( (X_train_global["weight_pt"]) * (X_train_global["label_b"]>0) )) / ak.sum(( (X_train_global["weight_pt"]) * (X_train_global["label_uds"]>0) ))
    wSum_g = ak.sum(( (X_train_global["weight_pt"]) * (X_train_global["label_b"]>0) )) / ak.sum(( (X_train_global["weight_pt"]) * (X_train_global["label_g"]>0) ))
    wSum_c = ak.sum(( (X_train_global["weight_pt"]) * (X_train_global["label_b"]>0) )) / ak.sum(( (X_train_global["weight_pt"]) * (X_train_global["label_c"]>0) ))
    wSum_taup = ak.sum(( (X_train_global["weight_pt"]) * (X_train_global["label_b"]>0) )) / ak.sum(( (X_train_global["weight_pt"]) * (X_train_global["label_taup"]>0) ))
    wSum_taum = ak.sum(( (X_train_global["weight_pt"]) * (X_train_global["label_b"]>0) )) / ak.sum(( (X_train_global["weight_pt"]) * (X_train_global["label_taum"]>0) ))
    wSum_muon = ak.sum(( (X_train_global["weight_pt"]) * (X_train_global["label_b"]>0) )) / ak.sum(( (X_train_global["weight_pt"]) * (X_train_global["label_muon"]>0) ))
    wSum_electron = ak.sum(( (X_train_global["weight_pt"]) * (X_train_global["label_b"]>0) )) / ak.sum(( (X_train_global["weight_pt"]) * (X_train_global["label_electron"]>0) ))

    # weight some classes higher
    # wSum_b = 10.0 * wSum_b
    # wSum_uds = 10.0 * wSum_uds
    # wSum_g = 10.0 * wSum_g
    # wSum_c = 5.0 * wSum_c
    # wSum_taup = 1.0 * wSum_taup
    # wSum_taum = 1.0 * wSum_taum
    # wSum_muon = 0.4 * wSum_muon
    # wSum_electron = 0.4 * wSum_electron

    X_train_global["weight_pt"] = (X_train_global["weight_pt"] * X_train_global["label_b"] * wSum_b + \
                                   X_train_global["weight_pt"] * X_train_global["label_uds"] * wSum_uds + \
                                   X_train_global["weight_pt"] * X_train_global["label_g"] * wSum_g + \
                                   X_train_global["weight_pt"] * X_train_global["label_c"] * wSum_c + \
                                   X_train_global["weight_pt"] * X_train_global["label_taup"] * wSum_taup + \
                                   X_train_global["weight_pt"] * X_train_global["label_taum"] * wSum_taum + \
                                   X_train_global["weight_pt"] * X_train_global["label_muon"] * wSum_muon + \
                                   X_train_global["weight_pt"] * X_train_global["label_electron"] * wSum_electron
                                )

    # print (np.histogram(X_train_global[X_train_global["label_b"]>0]["jet_pt_phys"], bins = bins_pt_weights, weights = X_train_global[X_train_global["label_b"]>0]["weight_pt"]))
    # print (np.histogram(X_train_global[X_train_global["label_uds"]>0]["jet_pt_phys"], bins = bins_pt_weights, weights = X_train_global[X_train_global["label_uds"]>0]["weight_pt"]))
    # print (np.histogram(X_train_global[X_train_global["label_g"]>0]["jet_pt_phys"], bins = bins_pt_weights, weights = X_train_global[X_train_global["label_g"]>0]["weight_pt"]))
    # print (np.histogram(X_train_global[X_train_global["label_c"]>0]["jet_pt_phys"], bins = bins_pt_weights, weights = X_train_global[X_train_global["label_c"]>0]["weight_pt"]))
    # print (np.histogram(X_train_global[X_train_global["label_taup"]>0]["jet_pt_phys"], bins = bins_pt_weights, weights = X_train_global[X_train_global["label_taup"]>0]["weight_pt"]))
    # print (np.histogram(X_train_global[X_train_global["label_taum"]>0]["jet_pt_phys"], bins = bins_pt_weights, weights = X_train_global[X_train_global["label_taum"]>0]["weight_pt"]))
    # print (np.histogram(X_train_global[X_train_global["label_muon"]>0]["jet_pt_phys"], bins = bins_pt_weights, weights = X_train_global[X_train_global["label_muon"]>0]["weight_pt"]))
    # print (np.histogram(X_train_global[X_train_global["label_electron"]>0]["jet_pt_phys"], bins = bins_pt_weights, weights = X_train_global[X_train_global["label_electron"]>0]["weight_pt"]))

    sample_weights = ak.to_numpy(X_train_global["weight_pt"])
    sample_weights = (sample_weights/np.mean(sample_weights))

    X_train_global["weight_pt"] = X_train_global["weight_pt"] / np.mean(sample_weights)

    print ("Using sample weights:", sample_weights)
    sample_weights = np.nan_to_num(sample_weights, nan = 1., posinf = 1., neginf = 1.)
    print ("Using sample weights fixed:", sample_weights)

    print ("Check all histograms are the same")
    print (np.histogram(X_train_global[X_train_global["label_b"]>0]["jet_pt_phys"], bins = bins_pt_weights, weights = X_train_global[X_train_global["label_b"]>0]["weight_pt"])[0])
    print (np.histogram(X_train_global[X_train_global["label_uds"]>0]["jet_pt_phys"], bins = bins_pt_weights, weights = X_train_global[X_train_global["label_uds"]>0]["weight_pt"])[0])
    print (np.histogram(X_train_global[X_train_global["label_g"]>0]["jet_pt_phys"], bins = bins_pt_weights, weights = X_train_global[X_train_global["label_g"]>0]["weight_pt"])[0])
    print (np.histogram(X_train_global[X_train_global["label_c"]>0]["jet_pt_phys"], bins = bins_pt_weights, weights = X_train_global[X_train_global["label_c"]>0]["weight_pt"])[0])
    print (np.histogram(X_train_global[X_train_global["label_taup"]>0]["jet_pt_phys"], bins = bins_pt_weights, weights = X_train_global[X_train_global["label_taup"]>0]["weight_pt"])[0])
    print (np.histogram(X_train_global[X_train_global["label_taum"]>0]["jet_pt_phys"], bins = bins_pt_weights, weights = X_train_global[X_train_global["label_taum"]>0]["weight_pt"])[0])
    print (np.histogram(X_train_global[X_train_global["label_muon"]>0]["jet_pt_phys"], bins = bins_pt_weights, weights = X_train_global[X_train_global["label_muon"]>0]["weight_pt"])[0])
    print (np.histogram(X_train_global[X_train_global["label_electron"]>0]["jet_pt_phys"], bins = bins_pt_weights, weights = X_train_global[X_train_global["label_electron"]>0]["weight_pt"])[0])

    # plot the weight distributions
    plt.figure()
    bins = np.linspace(0., 20., 1000)
    plt.hist(X_train_global[X_train_global["label_b"] > 0]["weight_pt"], label='b', bins = bins)
    plt.hist(X_train_global[X_train_global["label_uds"] > 0]["weight_pt"], label='uds', bins = bins)
    plt.hist(X_train_global[X_train_global["label_g"] > 0]["weight_pt"], label='Gluon', bins = bins)
    plt.hist(X_train_global[X_train_global["label_taup"] > 0]["weight_pt"], label='Tau p', bins = bins)
    plt.hist(X_train_global[X_train_global["label_taum"] > 0]["weight_pt"], label='Tau m', bins = bins)
    plt.hist(X_train_global[X_train_global["label_c"] > 0]["weight_pt"], label='Charm', bins = bins)
    plt.hist(X_train_global[X_train_global["label_electron"] > 0]["weight_pt"], label='Electron', bins = bins)
    plt.hist(X_train_global[X_train_global["label_muon"] > 0]["weight_pt"], label='Muon', bins = bins)
    plt.legend(loc = "upper right")
    plt.xlabel('Weights')
    plt.ylabel('Counts')
    hep.cms.label("Private Work", data = False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/weights_"+inputSetTag+".png")
    plt.savefig(outFolder+"/weights_"+inputSetTag+".pdf")
    plt.cla()

    # Define the optimizer ( minimization algorithm )
    if nnConfig["optimizer"] == "adam":
        optim = Adam(learning_rate = nnConfig["learning_rate"])
        # optim = Adam(learning_rate = nnConfig["learning_rate"], weight_decay=1e-4, epsilon=1e-08, amsgrad = True)
    elif nnConfig["optimizer"] == "sgd":
        optim = SGD(learning_rate = nnConfig["learning_rate"], momentum=0.9)

    if nnConfig["pruning"]:
        # Clone model to apply "pruneFunction" to model layers 
        model = tf.keras.models.clone_model(model, clone_function=pruneFunction)



    if nnConfig["regression"] == True:
        # model.compile(optimizer = optim, loss = ['categorical_crossentropy', 'mean_squared_error'],
        #             #   metrics = ['categorical_accuracy', 'mae'],
        #               loss_weights=[1., 1.]
        # #               )
        # model.compile(optimizer=optim, loss={'output_class': 'categorical_crossentropy', 'output_reg': 'mean_squared_error'},
        #               metrics={'output_class': 'categorical_accuracy', 'output_reg': ['mae', 'mean_squared_error']}, loss_weights=[1., 1.])
        model.compile(optimizer=optim, loss={'output_class': 'categorical_crossentropy', 'output_reg': 'log_cosh'},
                      metrics={'output_class': 'categorical_accuracy', 'output_reg': ['mae', 'mean_squared_error']},
                      weighted_metrics={'output_class': 'categorical_accuracy', 'output_reg': ['mae', 'mean_squared_error']},
                      loss_weights=[1., 2.])
                    #   loss_weights=[1., 50.])
    else:
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # if nnConfig["pruning"]:
    #     # The code bellow allows pruning of selected layers of the model
        
    #     def pruneFunction(layer):
    #         pruning_params = {"pruning_schedule": pruning_schedule.ConstantSparsity(0.5, begin_step=6000, frequency=10)}
                
    #         # Apply prunning to Dense layers type excluding the output layer
    #         if isinstance(layer, tf.keras.layers.Dense) and layer.name != 'dense_out': # exclude output_dense
    #             return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
    #         return layer

    #     # Clone model to apply "pruneFunction" to model layers 
    #     model = tf.keras.models.clone_model(model, clone_function=pruneFunction)

    # if nnConfig["regression"] == False:
    #     # compile the model
    #     model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    print ("Model summary:")
    # # print the model summary
    model.summary()


    print ("Dump nnConfig to file:", outFolder+"/nnConfig.json")
    json_object = json.dumps(nnConfig, indent=4)
    with open(outFolder+"/nnConfig.json", "w") as outfile:
        outfile.write(json_object)

    # ------------------------------------------------------
    # Run training
    # Figure o merit to monitor during training
    merit = 'val_loss'

    # early stopping callback
    es = EarlyStopping(monitor=merit, patience = 10)
    # Learning rate scheduler 
    # ls = ReduceLROnPlateau(monitor=merit, factor=0.2, patience=10)
    ls = ReduceLROnPlateau(monitor=merit, factor=0.2, patience=5, min_lr=0.00001)
    # model checkpoint callback
    # this saves our model architecture + parameters into mlp_model.h5
    chkp = ModelCheckpoint(outFolder+'/model_'+modelname+'.h5', monitor = merit, 
                                    verbose = 0, save_best_only = True, 
                                    save_weights_only = False, mode = 'auto', 
                                    save_freq = 'epoch')

    # callbacks_=[es,ls,chkp])
    # callbacks_=[es,chkp])
    callbacks_ = [chkp]
    # callbacks_ = [chkp, ls]
    if nnConfig["pruning"]:
        # Prunning callback
        pr = pruning_callbacks.UpdatePruningStep()
        callbacks_.append(pr)

    # print ("sample_weights",min(sample_weights),max(sample_weights))
    # print ("Y_train_val",min(Y_train_val),max(Y_train_val))
    # print ("Y_train_val_reg",min(Y_train_val_reg),max(Y_train_val_reg))

    if nnConfig["inputQuant"]:
        input_quantizer = quantized_bits(bits=16, integer=6, symmetric=0, alpha=1)
        X_train_val = input_quantizer(X_train_val.astype(np.float32)).numpy()
        X_test = input_quantizer(X_test.astype(np.float32)).numpy()

    # Train classifier
    if nnConfig["regression"] == True:
        history = model.fit(x = X_train_val,
                            y = [Y_train_val, Y_train_val_reg], 
                            epochs = nnConfig["epochs"], 
                            batch_size = nnConfig["batch_size"], 
                            verbose = 1,
                            validation_split = nnConfig["validation_split"],
                            callbacks = callbacks_,
                            sample_weight = sample_weights,
                            shuffle = True)
    else:
        history = model.fit(X_train_val, Y_train_val, 
                            epochs = nnConfig["epochs"], 
                            batch_size = nnConfig["batch_size"], 
                            verbose = 1,
                            validation_split = nnConfig["validation_split"],
                            callbacks=callbacks_,
                            class_weight = class_weights,
                            shuffle = True)
    
    custom_objects_ = {}
    if custom_objects is not None:
        for co in custom_objects:   
            custom_objects_[co] = custom_objects[co]

    if nnConfig["pruning"]:
        custom_objects_["PruneLowMagnitude"] = pruning_wrapper.PruneLowMagnitude

    model = tf.keras.models.load_model(outFolder+'/model_'+modelname+'.h5', custom_objects=custom_objects_)


    if nnConfig["pruning"]:
        # Strip the model 
        pmodel = strip_pruning(model)
        if nnConfig["regression"] == True:
            # pmodel.compile(optimizer=optim, loss={'output_class': 'categorical_crossentropy', 'output_reg': 'mean_squared_error'},
            #           metrics={'output_class': 'categorical_accuracy', 'output_reg': ['mae', 'mean_squared_error']}, loss_weights=[1., 1.])
            pmodel.compile(optimizer=optim, loss={'output_class': 'categorical_crossentropy', 'output_reg': 'log_cosh'},
                        metrics={'output_class': 'categorical_accuracy', 'output_reg': ['mae', 'mean_squared_error']},
                        weighted_metrics={'output_class': 'categorical_accuracy', 'output_reg': ['mae', 'mean_squared_error']},
                        loss_weights=[1., 2.])
                        # loss_weights=[1., 50.])
        else:
            pmodel.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        # Print the stripped model summary
        print ("pmodel")
        pmodel.summary()
        # Save the stripped and prunned model
        # pmodel.save(outFolder+'/model_'+modelname+'.h5', custom_objects=custom_objects_)
        pmodel.save(outFolder+'/model_'+modelname+'.h5')
        # model = pmodel


    # Plot performance

    # Here, we plot the history of the training and the performance in a ROC curve using the best saved model

    # Load the best saved model
    model = tf.keras.models.load_model(outFolder+'/model_'+modelname+'.h5', custom_objects=custom_objects_)

    if nnConfig["regression"] == False:
        # Plot loss vs epoch
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.legend(loc="upper right")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
        plt.savefig(outFolder+"/loss_"+inputSetTag+".png")
        plt.savefig(outFolder+"/loss_"+inputSetTag+".pdf")
        plt.cla()

        # Plot accuracy vs epoch
        plt.plot(history.history['categorical_accuracy'], label='Accuracy')
        plt.plot(history.history['val_categorical_accuracy'], label='Validation accuracy')
        plt.legend(loc="upper left")
        plt.xlabel('epoch')
        plt.ylabel('acc')
        hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
        plt.savefig(outFolder+"/acc_"+inputSetTag+".png")
        plt.savefig(outFolder+"/acc_"+inputSetTag+".pdf")
        plt.cla()

    if nnConfig["regression"]:
        # Plot loss vs epoch
        plt.plot(history.history['output_class_loss'], label='class loss')
        plt.plot(history.history['val_output_class_loss'], label='val class loss')
        plt.legend(loc="upper right")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
        plt.savefig(outFolder+"/loss_class_"+inputSetTag+".png")
        plt.savefig(outFolder+"/loss_class_"+inputSetTag+".pdf")
        plt.cla()

        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.legend(loc="upper right")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
        plt.savefig(outFolder+"/loss_"+inputSetTag+".png")
        plt.savefig(outFolder+"/loss_"+inputSetTag+".pdf")
        plt.cla()

        # Plot accuracy vs epoch
        plt.plot(history.history['output_class_categorical_accuracy'], label='Accuracy')
        plt.plot(history.history['val_output_class_categorical_accuracy'], label='Validation accuracy')
        plt.legend(loc="upper left")
        plt.xlabel('epoch')
        plt.ylabel('acc')
        hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
        plt.savefig(outFolder+"/acc_"+inputSetTag+".png")
        plt.savefig(outFolder+"/acc_"+inputSetTag+".pdf")
        plt.cla()

        # Plot reg loss vs epoch
        plt.plot(history.history['output_reg_loss'], label='Regression loss')
        plt.plot(history.history['val_output_reg_loss'], label='Validation regression loss')
        plt.legend(loc="upper left")
        plt.xlabel('epoch')
        plt.ylabel('acc')
        hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
        plt.savefig(outFolder+"/loss_reg_"+inputSetTag+".png")
        plt.savefig(outFolder+"/loss_reg_"+inputSetTag+".pdf")
        plt.cla()

        # Plot reg loss (MAE) vs epoch
        plt.plot(history.history['output_reg_mae'], label='MAE')
        plt.plot(history.history['val_output_reg_mae'], label='Validation MAE')
        plt.legend(loc="upper left")
        plt.xlabel('epoch')
        plt.ylabel('acc')
        hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
        plt.savefig(outFolder+"/mae_reg_"+inputSetTag+".png")
        plt.savefig(outFolder+"/mae_reg_"+inputSetTag+".pdf")
        plt.cla()

        # Plot reg loss (MSE) vs epoch
        plt.plot(history.history['output_reg_mean_squared_error'], label='MSE')
        plt.plot(history.history['val_output_reg_mean_squared_error'], label='Validation MSE')
        plt.legend(loc="upper left")
        plt.xlabel('epoch')
        plt.ylabel('acc')
        hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
        plt.savefig(outFolder+"/mse_reg_"+inputSetTag+".png")
        plt.savefig(outFolder+"/mse_reg_"+inputSetTag+".pdf")
        plt.cla()

    # Plot the ROC curves
    labels = ["Bkg", "b", "Taup", "Taum", "Gluon", "Charm", "Muon", "Electron"]
    fpr = {}
    tpr = {}
    auc1 = {}
    precision = {}
    recall = {}
    NN = {}
    NP = {}
    TP = {}
    FP = {}
    TN = {}
    FN = {}
    tresholds = {}


    if nnConfig["regression"]:
        Y_predict = model.predict(X_test)
        Y_predict_reg = Y_predict[1]
        Y_predict = Y_predict[0]
    else:
        Y_predict = model.predict(X_test)
    

    # Loop over classes(labels) to get metrics per class
    for i, label in enumerate(labels):
        fpr[label], tpr[label], tresholds[label] = roc_curve(Y_test[:,i], Y_predict[:,i])
        _ , N = np.unique(Y_test[:,i], return_counts=True) # count the NEGATIVES and POSITIVES samples in your test set
        NN[label] = N[0]                   # number of NEGATIVES 
        NP[label] = N[1]                   # number of POSITIVES
        TP[label] = tpr[label]*NP[label]
        FP[label] = fpr[label]*NN[label] 
        TN[label] = NN[label] - FP[label]
        FN[label] = NP[label] - TP[label]
        plt.grid()
        auc1[label] = auc(fpr[label], tpr[label])
        plt.plot(tpr[label],fpr[label],label='%s tagger, AUC = %.1f%%'%(label,auc1[label]*100.))


    plt.semilogy()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Mistag rate")
    plt.xlim(0.,1.)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='lower right')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/ROC_"+inputSetTag+".png")
    plt.savefig(outFolder+"/ROC_"+inputSetTag+".pdf")
    plt.cla()



    # Plot DNN output 
    y_b_predict = model.predict(x_b)
    y_bkg_predict = model.predict(x_bkg)
    y_taup_predict = model.predict(x_taup)
    y_taum_predict = model.predict(x_taum)
    y_gluon_predict = model.predict(x_gluon)
    y_charm_predict = model.predict(x_charm)
    y_muon_predict = model.predict(x_muon)
    y_electron_predict = model.predict(x_electron)


    if nnConfig["regression"]:
        y_b_predict_reg = y_b_predict[1]
        y_b_predict = y_b_predict[0]
        y_bkg_predict_reg = y_bkg_predict[1]
        y_bkg_predict = y_bkg_predict[0]
        y_taup_predict_reg = y_taup_predict[1]
        y_taum_predict_reg = y_taum_predict[1]
        y_taup_predict = y_taup_predict[0]
        y_taum_predict = y_taum_predict[0]
        y_charm_predict_reg = y_charm_predict[1]
        y_charm_predict = y_charm_predict[0]
        y_gluon_predict_reg = y_gluon_predict[1]
        y_gluon_predict = y_gluon_predict[0]


    X = np.linspace(0.0, 1.0, 100)
    histo = plt.hist(y_b_predict[:,0], bins=X, label='b' ,histtype='step', density = True, color="blue")
    histo = plt.hist(y_taup_predict[:,0], bins=X, label='Tau p' ,histtype='step', density = True, color="red")
    histo = plt.hist(y_taum_predict[:,0], bins=X, label='Tau m' ,histtype='step', density = True, color="purple")
    histo = plt.hist(y_bkg_predict[:,0], bins=X, label='uds' ,histtype='step', density = True, color="orange")
    histo = plt.hist(y_gluon_predict[:,0], bins=X, label='Gluon' ,histtype='step', density = True, color="green")
    histo = plt.hist(y_charm_predict[:,0], bins=X, label='Charm' ,histtype='step', density = True, color="black")
    plt.xlabel('uds score')
    plt.legend(prop={'size': 10})
    plt.legend(loc='upper right')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/score_bkg_"+inputSetTag+".png")
    plt.savefig(outFolder+"/score_bkg_"+inputSetTag+".pdf")
    plt.cla()

    X = np.linspace(0.0, 1.0, 100)
    histo = plt.hist(y_b_predict[:,1], bins=X, label='b ' ,histtype='step', density = True, color="blue")
    histo = plt.hist(y_taup_predict[:,1], bins=X, label='Tau p' ,histtype='step', density = True, color="red")
    histo = plt.hist(y_taum_predict[:,1], bins=X, label='Tau m' ,histtype='step', density = True, color="purple")
    histo = plt.hist(y_bkg_predict[:,1], bins=X, label='uds' ,histtype='step', density = True, color="orange")
    histo = plt.hist(y_gluon_predict[:,1], bins=X, label='Gluon' ,histtype='step', density = True, color="green")
    histo = plt.hist(y_charm_predict[:,1], bins=X, label='Charm' ,histtype='step', density = True, color="black")
    plt.xlabel('b score')
    plt.legend(prop={'size': 10})
    plt.legend(loc='upper right')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/score_b_"+inputSetTag+".png")
    plt.savefig(outFolder+"/score_b_"+inputSetTag+".pdf")
    plt.cla()

    X = np.linspace(0.0, 1.0, 100)
    histo = plt.hist(y_b_predict[:,2], bins=X, label='b' ,histtype='step', density = True, color="blue")
    histo = plt.hist(y_taup_predict[:,2], bins=X, label='Tau p' ,histtype='step', density = True, color="red")
    histo = plt.hist(y_taum_predict[:,2], bins=X, label='Tau m' ,histtype='step', density = True, color="purple")
    histo = plt.hist(y_bkg_predict[:,2], bins=X, label='uds' ,histtype='step', density = True, color="orange")
    histo = plt.hist(y_gluon_predict[:,2], bins=X, label='Gluon' ,histtype='step', density = True, color="green")
    histo = plt.hist(y_charm_predict[:,2], bins=X, label='Charm' ,histtype='step', density = True, color="black")
    plt.xlabel('tau score')
    plt.legend(prop={'size': 10})
    plt.legend(loc='upper right')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/score_tau_"+inputSetTag+".png")
    plt.savefig(outFolder+"/score_tau_"+inputSetTag+".pdf")
    plt.cla()

    X = np.linspace(0.0, 1.0, 100)
    histo = plt.hist(y_b_predict[:,3], bins=X, label='b' ,histtype='step', density = True, color="blue")
    histo = plt.hist(y_taup_predict[:,3], bins=X, label='Tau p' ,histtype='step', density = True, color="red")
    histo = plt.hist(y_taum_predict[:,3], bins=X, label='Tau m' ,histtype='step', density = True, color="purple")
    histo = plt.hist(y_bkg_predict[:,3], bins=X, label='uds' ,histtype='step', density = True, color="orange")
    histo = plt.hist(y_gluon_predict[:,3], bins=X, label='Gluon' ,histtype='step', density = True, color="green")
    histo = plt.hist(y_charm_predict[:,3], bins=X, label='Charm' ,histtype='step', density = True, color="black")
    plt.xlabel('gluon score')
    plt.legend(prop={'size': 10})
    plt.legend(loc='upper right')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/score_gluon_"+inputSetTag+".png")
    plt.savefig(outFolder+"/score_gluon_"+inputSetTag+".pdf")
    plt.cla()

    X = np.linspace(0.0, 1.0, 100)
    histo = plt.hist(y_b_predict[:,4], bins=X, label='b' ,histtype='step', density = True, color="blue")
    histo = plt.hist(y_taup_predict[:,4], bins=X, label='Tau p' ,histtype='step', density = True, color="red")
    histo = plt.hist(y_taum_predict[:,4], bins=X, label='Tau m' ,histtype='step', density = True, color="purple")
    histo = plt.hist(y_bkg_predict[:,4], bins=X, label='uds' ,histtype='step', density = True, color="orange")
    histo = plt.hist(y_gluon_predict[:,4], bins=X, label='Gluon' ,histtype='step', density = True, color="green")
    histo = plt.hist(y_charm_predict[:,4], bins=X, label='Charm' ,histtype='step', density = True, color="black")
    plt.xlabel('charm score')
    plt.legend(prop={'size': 10})
    plt.legend(loc='upper right')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.savefig(outFolder+"/score_charm_"+inputSetTag+".png")
    plt.savefig(outFolder+"/score_charm_"+inputSetTag+".pdf")
    plt.cla()

    if nnConfig["regression"]:
        # a quick response plot before and after...
        X_test_global["jet_pt_reg"] = Y_predict_reg[:,0]
        X_test_global["jet_pt_cor_reg"] = X_test_global["jet_pt_phys"] * X_test_global["jet_pt_reg"]
        mean_uncor = np.mean(np.array(X_test_global["jet_pt_phys"] / X_test_global["jet_genmatch_pt"]))
        std_uncor = rms(X_test_global["jet_pt_phys"] / X_test_global["jet_genmatch_pt"])
        mean_cor = np.mean(X_test_global["jet_pt_corr"] / X_test_global["jet_genmatch_pt"])
        std_cor = rms(X_test_global["jet_pt_corr"] / X_test_global["jet_genmatch_pt"])
        mean_reg = np.mean(X_test_global["jet_pt_cor_reg"] / X_test_global["jet_genmatch_pt"])
        std_reg = rms(X_test_global["jet_pt_cor_reg"] / X_test_global["jet_genmatch_pt"])
        print("uncor", mean_uncor, std_uncor)
        print("cor", mean_cor, std_cor)
        print("reg", mean_reg, std_reg)
        X = np.linspace(0.0, 2.0, 100)
        histo = plt.hist(X_test_global["jet_pt_phys"] / X_test_global["jet_genmatch_pt"], bins=X, label='Uncorrected' ,histtype='step', density=True, color = '#1f77b4')
        histo = plt.hist(X_test_global["jet_pt_corr"] / X_test_global["jet_genmatch_pt"], bins=X, label='JEC LOT' ,histtype='step', density=True, color = '#ff7f0e')
        histo = plt.hist(X_test_global["jet_pt_cor_reg"] / X_test_global["jet_genmatch_pt"], bins=X, label='Regression' ,histtype='step', density=True, color = '#2ca02c')
        plt.xlabel('Jet response (reco/gen)')
        plt.ylabel('Jets')
        plt.xlim(0.,2.)
        plt.legend(prop={'size': 10})
        plt.legend(loc='upper right')
        plt.text(0.7, 0.7, "mean: "+str(np.round(mean_uncor,3))+" rms:"+str(np.round(std_uncor,3)), color = '#1f77b4')
        plt.text(0.7, 0.8, "mean: "+str(np.round(mean_cor,3))+" rms:"+str(np.round(std_cor,3)), color = '#ff7f0e')
        plt.text(0.7, 0.9, "mean: "+str(np.round(mean_reg,3))+" rms:"+str(np.round(std_reg,3)), color = '#2ca02c')
        hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
        plt.savefig(outFolder+"/response_"+inputSetTag+".png")
        plt.savefig(outFolder+"/response_"+inputSetTag+".pdf")
        plt.cla()


if __name__ == "__main__":
    from args import get_common_parser, handle_common_args
    parser = get_common_parser()
    parser.add_argument('-f','--file', help = 'input file name part')
    parser.add_argument('-c','--classes', help = 'Which flavors to run, options are b, bt, btg, btgc.')
    parser.add_argument('-i','--inputs', help = 'Which inputs to run, options are baseline, ext1, ext2, all.')
    parser.add_argument('--model', dest = 'model', default = "deepset")
    parser.add_argument('--train-batch-size', dest = 'batch_size', default = 1024)
    parser.add_argument('--train-epochs', dest = 'epochs', default = 50)
    parser.add_argument('--train-validation-split', dest = 'validation_split', default = .25)
    parser.add_argument('--learning-rate', dest = 'learning_rate', default = 0.0001)
    parser.add_argument('--optimizer', dest = 'optimizer', default = "adam")
    parser.add_argument('--classweights', dest = 'classweights', default = False, action='store_true')
    parser.add_argument('--regression', dest = 'regression', default = False, action='store_true')
    parser.add_argument('--pruning', dest = 'pruning', default = False, action='store_true')
    parser.add_argument('--inputQuant', dest = 'inputQuant', default = False, action='store_true')
    parser.add_argument('--test', dest = 'test', default = False, action='store_true')
    parser.add_argument('--plotFeatures', dest = 'plotFeatures', default = False, action='store_true')
    parser.add_argument('--nbits', dest = 'nbits', default = 8)
    parser.add_argument('--integ', dest = 'integ', default = 0)
    parser.add_argument('--nNodes', dest = 'nNodes', default = 15)
    parser.add_argument('--nLayers', dest = 'nLayers', default = 3)
    parser.add_argument('--nHeads', dest = 'nHeads', default = 3)
    parser.add_argument('--nNodesHead', dest = 'nNodesHead', default = 12)
    parser.add_argument('--strstamp', dest = 'strstamp', default = "")

    args = parser.parse_args()
    handle_common_args(args)

    print('#'*30)
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('#'*30)

    allowedModels = ["DeepSet", "DeepSet-MHA", "MLP", "MLP-MHA"]
    allowedClasses = ["b", "bt", "btg", "btgc"]
    allowedFiles = ["All200", "extendedAll200", "baselineAll200", "AllHIG200", "AllQCD200", "AllTT200", "TT_PU200", "TT1L_PU200", "TT2L_PU200", "ggHtt_PU200"]
    allowedInputs = dict_fields.keys()
    allowedOptimizer = ["adam", "ranger", "sgd"]

    if args.model not in allowedModels:
        raise ValueError("args.model not in allowed models! Options are", allowedModels)
    if args.classes not in allowedClasses:
        raise ValueError("args.classes not in allowed classes! Options are", allowedClasses)
    if args.file not in allowedFiles:
        raise ValueError("args.file not in allowed file! Options are", allowedFiles)
    if args.inputs not in allowedInputs:
        raise ValueError("args.inputs not in allowed inputs! Options are", allowedInputs)
    if args.optimizer not in allowedOptimizer:
        raise ValueError("args.optimizer not in allowed optimizer! Options are", allowedOptimizer)

    nnConfig = {
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "validation_split": float(args.validation_split),
        "model": args.model,
        "learning_rate": float(args.learning_rate),
        "optimizer": args.optimizer,
        "classweights": args.classweights,
        "regression": args.regression,
        "pruning": args.pruning,
        "inputQuant": args.inputQuant,
        "nbits": int(args.nbits),
        "integ": int(args.integ),
        "nNodes": int(args.nNodes),
        "nLayers": int(args.nLayers),
        "nHeads": int(args.nHeads),
        "nNodesHead": int(args.nNodesHead),
    }

    doTraining(
        args.file,
        args.classes,
        args.inputs,
        nnConfig,
        args.save,
        args.strstamp,
        args.test,
        args.plotFeatures,
        args.workdir,
        )