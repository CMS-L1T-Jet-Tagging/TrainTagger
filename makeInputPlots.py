from utils.imports import *
from utils.dataset import *
import argparse
from train.models import *
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


def plotInputFeatures(Xb, Xuds, Xtau, Xtaum, Xgluon, Xcharm, Xmuon, Xelectron, featureNames, outFolder, outputAddName = ""):
    print ("Plot all input features w/ add. name", outputAddName)
    plt.figure()
    for idxFeature, name in enumerate(featureNames):
        b_ = ak.flatten(Xb[:,:,idxFeature][Xb[:,:,-1]!=0.],axis=-1)
        uds_ = ak.flatten(Xuds[:,:,idxFeature][Xuds[:,:,-1]!=0.],axis=-1)
        g_ = ak.flatten(Xgluon[:,:,idxFeature][Xgluon[:,:,-1]!=0.],axis=-1)
        tau_ = ak.flatten(Xtau[:,:,idxFeature][Xtau[:,:,-1]!=0.],axis=-1)
        if Xtaum is not None:
            taum_ = ak.flatten(Xtaum[:,:,idxFeature][Xtaum[:,:,-1]!=0.],axis=-1)
        charm_ = ak.flatten(Xcharm[:,:,idxFeature][Xcharm[:,:,-1]!=0.],axis=-1)
        muon_ = ak.flatten(Xmuon[:,:,idxFeature][Xmuon[:,:,-1]!=0.],axis=-1)
        electron_ = ak.flatten(Xelectron[:,:,idxFeature][Xelectron[:,:,-1]!=0.],axis=-1)
        min_ = min(min(b_), min(uds_))
        max_ = max(max(b_), max(uds_))
        min_ = min(min_, min(tau_))
        max_ = max(max_, max(tau_))
        min_ = min(min_, min(taum_))
        max_ = max(max_, max(taum_))
        min_ = min(min_, min(g_))
        max_ = max(max_, max(g_))
        min_ = min(min_, min(charm_))
        max_ = max(max_, max(charm_))
        min_ = min(min_, min(muon_))
        max_ = max(max_, max(muon_))
        min_ = min(min_, min(electron_))
        max_ = max(max_, max(electron_))
        range = (min_, max_)
        print (name, range)
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

        if name == "dxy_custom":
            b_ = ak.flatten(Xb[:,:,idxFeature][Xb[:,:,-1]!=0.],axis=-1)
            uds_ = ak.flatten(Xuds[:,:,idxFeature][Xuds[:,:,-1]!=0.],axis=-1)
            g_ = ak.flatten(Xgluon[:,:,idxFeature][Xgluon[:,:,-1]!=0.],axis=-1)
            tau_ = ak.flatten(Xtau[:,:,idxFeature][Xtau[:,:,-1]!=0.],axis=-1)
            if Xtaum is not None:
                taum_ = ak.flatten(Xtaum[:,:,idxFeature][Xtaum[:,:,-1]!=0.],axis=-1)
            charm_ = ak.flatten(Xcharm[:,:,idxFeature][Xcharm[:,:,-1]!=0.],axis=-1)
            muon_ = ak.flatten(Xmuon[:,:,idxFeature][Xmuon[:,:,-1]!=0.],axis=-1)
            electron_ = ak.flatten(Xelectron[:,:,idxFeature][Xelectron[:,:,-1]!=0.],axis=-1)
            range = (-0.5, 0.5)
            print (name, range)
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
            plt.savefig(outFolder+"/inputFeature_"+name+outputAddName+"_close.png")
            plt.savefig(outFolder+"/inputFeature_"+name+outputAddName+"_close.pdf")
            plt.cla()



            b_ = ak.flatten(Xb[:,:,idxFeature][Xb[:,:,-1]!=0.],axis=-1)
            uds_ = ak.flatten(Xuds[:,:,idxFeature][Xuds[:,:,-1]!=0.],axis=-1)
            g_ = ak.flatten(Xgluon[:,:,idxFeature][Xgluon[:,:,-1]!=0.],axis=-1)
            tau_ = ak.flatten(Xtau[:,:,idxFeature][Xtau[:,:,-1]!=0.],axis=-1)
            if Xtaum is not None:
                taum_ = ak.flatten(Xtaum[:,:,idxFeature][Xtaum[:,:,-1]!=0.],axis=-1)
            charm_ = ak.flatten(Xcharm[:,:,idxFeature][Xcharm[:,:,-1]!=0.],axis=-1)
            muon_ = ak.flatten(Xmuon[:,:,idxFeature][Xmuon[:,:,-1]!=0.],axis=-1)
            electron_ = ak.flatten(Xelectron[:,:,idxFeature][Xelectron[:,:,-1]!=0.],axis=-1)

            # scale with 256 and cast to int
            # pdb.set_trace()
            input_quantizer = quantized_bits(bits=8, integer=8, symmetric=0, alpha=1)
            b_ = b_ * 256
            b_ = input_quantizer(b_).numpy()
            uds_ = uds_ * 256
            uds_ = input_quantizer(uds_).numpy()
            g_ = g_ * 256
            g_ = input_quantizer(g_).numpy()
            tau_ = tau_ * 256
            tau_ = input_quantizer(tau_).numpy()
            taum_ = taum_ * 256
            taum_ = input_quantizer(taum_).numpy()
            charm_ = charm_ * 256
            charm_ = input_quantizer(charm_).numpy()
            muon_ = muon_ * 256
            muon_ = input_quantizer(muon_).numpy()
            electron_ = electron_ * 256
            electron_ = input_quantizer(electron_).numpy()
            min_ = min(min(b_), min(uds_))
            max_ = max(max(b_), max(uds_))
            min_ = min(min_, min(tau_))
            max_ = max(max_, max(tau_))
            min_ = min(min_, min(taum_))
            max_ = max(max_, max(taum_))
            min_ = min(min_, min(g_))
            max_ = max(max_, max(g_))
            min_ = min(min_, min(charm_))
            max_ = max(max_, max(charm_))
            min_ = min(min_, min(muon_))
            max_ = max(max_, max(muon_))
            min_ = min(min_, min(electron_))
            max_ = max(max_, max(electron_))
            range = (min_, max_)
            print (name, range)
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
            plt.savefig(outFolder+"/inputFeature_"+name+outputAddName+"_quant.png")
            plt.savefig(outFolder+"/inputFeature_"+name+outputAddName+"_quant.pdf")
            plt.cla()

            range = (-0.5*256, 0.5*256)
            print (name, range)
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
            plt.savefig(outFolder+"/inputFeature_"+name+outputAddName+"_close_quant.png")
            plt.savefig(outFolder+"/inputFeature_"+name+outputAddName+"_close_quant.pdf")
            plt.cla()



def doTraining(
        filetag,
        flavs,
        inputSetTag,
        test = False,
        plotFeatures = False,
        workdir = "./",):

    outFolder = "inputFeaturePlots/"

    if not os.path.exists(outFolder):
        os.makedirs(outFolder, exist_ok=True)

    feature_names = dict_fields[inputSetTag]
    nconstit = 16

    PATH_load = workdir + '/datasetsNewComplete_plotting/' + filetag + "/" + flavs + "/"
    print (PATH_load)
    chunksmatching = glob.glob(PATH_load+"X_"+inputSetTag+"_test*.parquet")
    chunksmatching = [chunksm.replace(PATH_load+"X_"+inputSetTag+"_test","").replace(".parquet","").replace("_","") for chunksm in chunksmatching]

    import random
    if test:
        chunksmatching = random.sample(chunksmatching, 2)
    else:
        chunksmatching = random.sample(chunksmatching, len(chunksmatching))

    print ("Loading data in all",len(chunksmatching),"chunks.")

    x_b = None
    x_taup = None
    x_taum = None
    x_bkg = None
    x_gluon = None
    x_charm = None
    x_electron = None
    x_muon = None

    for c in chunksmatching:
        x_b_ = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_b_"+c+".parquet")
        if len(x_b_) > 0:
            if x_b is None:
                x_b = x_b_
            else:
                x_b =ak.concatenate((x_b, x_b_))

        x_bkg_ = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_bkg_"+c+".parquet")
        if len(x_bkg_) > 0:
            if x_bkg is None:
                x_bkg = x_bkg_
            else:
                x_bkg =ak.concatenate((x_bkg, x_bkg_))

        x_taup_ = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_taup_"+c+".parquet")
        if len(x_taup_) > 0:
            if x_taup is None:
                x_taup = x_taup_
            else:
                x_taup =ak.concatenate((x_taup, x_taup_))

        x_taum_ = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_taum_"+c+".parquet")
        if len(x_taum_) > 0:
            if x_taum is None:
                x_taum = x_taum_
            else:
                x_taum =ak.concatenate((x_taum, x_taum_))

        x_gluon_ = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_gluon_"+c+".parquet")
        if len(x_gluon_) > 0:
            if x_gluon is None:
                x_gluon = x_gluon_
            else:
                x_charm =ak.concatenate((x_gluon, x_gluon_))

        x_charm_ = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_charm_"+c+".parquet")
        if len(x_charm_) > 0:
            if x_charm is None:
                x_charm = x_charm_
            else:
                x_charm =ak.concatenate((x_charm, x_charm_))
        
        x_muon_ = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_muon_"+c+".parquet")
        if len(x_muon_) > 0:
            if x_muon is None:
                x_muon = x_muon_
            else:
                x_muon =ak.concatenate((x_muon, x_muon_))

        x_electron_ = ak.from_parquet(PATH_load+"X_"+inputSetTag+"_electron_"+c+".parquet")
        if len(x_electron_) > 0:
            if x_electron is None:
                x_electron = x_electron_
            else:
                x_electron =ak.concatenate((x_electron, x_electron_))

    x_b = ak.to_numpy(x_b)
    x_bkg = ak.to_numpy(x_bkg)
    x_taup = ak.to_numpy(x_taup)
    x_taum = ak.to_numpy(x_taum)
    x_gluon = ak.to_numpy(x_gluon)
    x_charm = ak.to_numpy(x_charm)
    x_muon = ak.to_numpy(x_muon)
    x_electron = ak.to_numpy(x_electron)

    outFolder = outFolder + "/"+ filetag + "_" + flavs + "_" + inputSetTag + "/"
    if not os.path.exists(outFolder):
        os.makedirs(outFolder, exist_ok=True)
    print ("Use the following output folder:", outFolder)

    if plotFeatures:
        plotInputFeatures(x_b, x_bkg, x_taup, x_taum, x_gluon, x_charm, x_muon, x_electron, feature_names, outFolder, outputAddName = "")

    # if nnConfig["inputQuant"]:
    #     input_quantizer = quantized_bits(bits=16, integer=6, symmetric=0, alpha=1)
    #     x_b = input_quantizer(x_b.astype(np.float32)).numpy()
    #     x_bkg = input_quantizer(x_bkg.astype(np.float32)).numpy()
    #     x_taup = input_quantizer(x_taup.astype(np.float32)).numpy()
    #     x_taum = input_quantizer(x_taum.astype(np.float32)).numpy()
    #     x_gluon = input_quantizer(x_gluon.astype(np.float32)).numpy()
    #     x_charm = input_quantizer(x_charm.astype(np.float32)).numpy()

    #     plotInputFeatures(x_b, x_bkg, x_taup+x_taum, x_gluon, x_charm, feature_names, outFolder, outputAddName = "_quant")


if __name__ == "__main__":
    from args import get_common_parser, handle_common_args
    parser = get_common_parser()
    parser.add_argument('-f','--file', help = 'input file name part')
    parser.add_argument('-c','--classes', help = 'Which flavors to run, options are b, bt, btg, btgc.')
    parser.add_argument('-i','--inputs', help = 'Which inputs to run, options are baseline, ext1, ext2, all.')
    parser.add_argument('--inputQuant', dest = 'inputQuant', default = False, action='store_true')
    parser.add_argument('--test', dest = 'test', default = False, action='store_true')

    args = parser.parse_args()
    handle_common_args(args)

    print('#'*30)
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('#'*30)

    allowedClasses = ["b", "bt", "btg", "btgc"]
    allowedFiles = ["All200", "extendedAll200", "baselineAll200", "AllHIG200", "AllQCD200", "AllTT200", "TT_PU200", "TT1L_PU200", "TT2L_PU200", "ggHtt_PU200"]
    allowedInputs = dict_fields.keys()

    if args.classes not in allowedClasses:
        raise ValueError("args.classes not in allowed classes! Options are", allowedClasses)
    if args.file not in allowedFiles:
        raise ValueError("args.file not in allowed file! Options are", allowedFiles)
    if args.inputs not in allowedInputs:
        raise ValueError("args.inputs not in allowed inputs! Options are", allowedInputs)

    doTraining(
        args.file,
        args.classes,
        args.inputs,
        args.test,
        args.workdir,
        )