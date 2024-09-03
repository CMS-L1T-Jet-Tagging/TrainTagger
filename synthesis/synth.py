from datatools.imports import *
from datatools.dataset import *

import argparse
from train.models import *
from synthesis.profiling import *
from sklearn.metrics import roc_curve, auc,precision_recall_curve
import matplotlib.pyplot as plt
import glob
import pandas
from histbook import *
import sys, os, time
import hls4ml
import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time
import argparse
from hls_node_edge_projection import *

__tf_profiling_enabled__ = True
__torch_profiling_enabled__ = True

from datatools.createDataset import *

import pydot
pydot.Dot.create(pydot.Dot())

modelnamesDict = {
    "DeepSet": "QDeepSets_PermutationInv",
    "DeepSet-MHA": "QDeepSetsWithAttention_PermutationInv",
    "MLP": "QMLP",
    "MLP-MHA": "QMLPWithAttention",
}

nconstit = 16

def synthesize(
        filetag,
        timestamp,
        flav,
        inputSetTag,
        modelname,
        outname,
        regression,
        pruning,
        save = True,
        workdir = "./", build=False, trace=True):

    tempflav = "btgc"
    PATH = workdir + filetag + "/" + tempflav + "/"
    outFolder = "outputSynthesis/"+outname+"/Training_" + timestamp + "/"
    if not os.path.exists(outFolder):
        os.makedirs(outFolder, exist_ok=True)

    feature_names = dict_fields[inputSetTag]

    chunksmatching = glob.glob(PATH+"X_"+inputSetTag+"_test*.parquet")
    print (PATH+"X_"+inputSetTag+"_test*.parquet")
    chunksmatching = [chunksm.replace(PATH+"X_"+inputSetTag+"_test","").replace(".parquet","").replace("_","") for chunksm in chunksmatching]

    import random
    chunksmatching = random.sample(chunksmatching, 5)

    filter = "/(jet)_(eta|eta_phys|phi|pt|pt_phys|pt_raw|bjetscore|tauscore|pt_corr|genmatch_lep_vis_pt|genmatch_pt|label_b|label_uds|label_g|label_c|label_tau/label_taup/label_taum/label_muon/label_electron"

    print ("Loading data in all",len(chunksmatching),"chunks.")


    X_test = None
    X_test_global = None
    Y_test = None
    x_b = None
    x_taup = None
    x_taum = None
    x_bkg = None
    x_gluon = None
    x_charm = None
    x_muon = None
    x_electron = None

    for c in chunksmatching:
        if X_test is None:
            X_test = ak.from_parquet(PATH+"X_"+inputSetTag+"_test_"+c+".parquet")
            X_test_global = ak.from_parquet(PATH+"X_global_"+inputSetTag+"_test_"+c+".parquet")
            Y_test = ak.from_parquet(PATH+"Y_"+inputSetTag+"_test_"+c+".parquet")
        else:
            X_test =ak.concatenate((X_test, ak.from_parquet(PATH+"X_"+inputSetTag+"_test_"+c+".parquet")))
            X_test_global =ak.concatenate((X_test_global, ak.from_parquet(PATH+"X_global_"+inputSetTag+"_test_"+c+".parquet")))
            Y_test =ak.concatenate((Y_test, ak.from_parquet(PATH+"Y_"+inputSetTag+"_test_"+c+".parquet")))

        x_b_ = ak.from_parquet(PATH+"X_"+inputSetTag+"_b_"+c+".parquet")
        if len(x_b_) > 0:
            if x_b is None:
                x_b = x_b_
            else:
                x_b =ak.concatenate((x_b, x_b_))

        x_bkg_ = ak.from_parquet(PATH+"X_"+inputSetTag+"_bkg_"+c+".parquet")
        if len(x_bkg_) > 0:
            if x_bkg is None:
                x_bkg = x_bkg_
            else:
                x_bkg =ak.concatenate((x_bkg, x_bkg_))

        x_taup_ = ak.from_parquet(PATH+"X_"+inputSetTag+"_taup_"+c+".parquet")
        if len(x_taup_) > 0:
            if x_taup is None:
                x_taup = x_taup_
            else:
                x_taup =ak.concatenate((x_taup, x_taup_))
        x_taum_ = ak.from_parquet(PATH+"X_"+inputSetTag+"_taum_"+c+".parquet")
        if len(x_taum_) > 0:
            if x_taum is None:
                x_taum = x_taum_
            else:
                x_taum =ak.concatenate((x_taum, x_taum_))

        x_gluon_ = ak.from_parquet(PATH+"X_"+inputSetTag+"_gluon_"+c+".parquet")
        if len(x_gluon_) > 0:
            if x_gluon is None:
                x_gluon = x_gluon_
            else:
                x_charm =ak.concatenate((x_gluon, x_gluon_))

        x_charm_ = ak.from_parquet(PATH+"X_"+inputSetTag+"_charm_"+c+".parquet")
        if len(x_charm_) > 0:
            if x_charm is None:
                x_charm = x_charm_
            else:
                x_charm =ak.concatenate((x_charm, x_charm_))
        
        x_muon_ = ak.from_parquet(PATH+"X_"+inputSetTag+"_muon_"+c+".parquet")
        if len(x_muon_) > 0:
            if x_muon is None:
                x_muon = x_muon_
            else:
                x_muon =ak.concatenate((x_muon, x_muon_))

        x_electron_ = ak.from_parquet(PATH+"X_"+inputSetTag+"_electron_"+c+".parquet")
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

    X_test = ak.to_numpy(X_test)
    Y_test = ak.to_numpy(Y_test)

    modelArchName = modelname

    if modelArchName == "MLP":
        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1]*X_test.shape[2]))
        x_b = np.reshape(x_b, (x_b.shape[0],x_b.shape[1]*x_b.shape[2]))
        x_bkg = np.reshape(x_bkg, (x_bkg.shape[0],x_bkg.shape[1]*x_bkg.shape[2]))
        x_taup = np.reshape(x_taup, (x_taup.shape[0],x_taup.shape[1]*x_taup.shape[2]))
        x_taum = np.reshape(x_taum, (x_taum.shape[0],x_taum.shape[1]*x_taum.shape[2]))
        x_gluon = np.reshape(x_gluon, (x_gluon.shape[0],x_gluon.shape[1]*x_gluon.shape[2]))
        x_charm = np.reshape(x_charm, (x_charm.shape[0],x_charm.shape[1]*x_charm.shape[2]))
        x_muon = np.reshape(x_muon, (x_muon.shape[0],x_muon.shape[1]*x_muon.shape[2]))
        x_electron = np.reshape(x_electron, (x_electron.shape[0],x_electron.shape[1]*x_electron.shape[2]))

    X_test_small = X_test[:1000]
    Y_test_small = Y_test[:1000]

    print("Loaded X_test      ----> shape:", X_test.shape)
    print("Loaded Y_test      ----> shape:", Y_test.shape)

    print ("Get performance for", inputSetTag, flav, modelname)

    ncands = 16 
    nfeatures = len(feature_names)
    nbits = 8

    labels = ["Bkg", "b", "Taup", "Taum", "Gluon", "Charm", "Muon", "Electron"]

    # Get inference of model
    if regression:
        trainingBasePath = "trainings_regression_weighted/" + timestamp + "_" + flav + "_" + inputSetTag + "_"
    else:
        trainingBasePath = "trainings_notreduced/" + filetag + "_" + flav + "_" + inputSetTag + "_"
    modelpath = modelnamesDict[modelname]+"_nconst_"+str(ncands)+"_nfeatures_"+str(nfeatures)+"_nbits_"+str(nbits)
    modelname = 'model_'+modelnamesDict[modelname]+"_nconst_"+str(ncands)+"_nfeatures_"+str(nfeatures)+"_nbits_"+str(nbits)
    if pruning:
        modelpath = modelpath + "_pruned"
        modelname = modelname + "_pruned"

    print ("Load model", trainingBasePath+""+modelpath+'.h5')


    custom_objects_ = {
        "AAtt": AAtt,
        "QDense": QDense,
        "QActivation": QActivation,
        "quantized_bits": quantized_bits,
        "ternary": ternary,
        "binary": binary,
        "QBatchNormalization": QBatchNormalization
        }

    model = tf.keras.models.load_model(
        trainingBasePath+""+modelpath+"/"+modelname+'.h5',
        custom_objects={
            "QDense": QDense,
            "QActivation": QActivation,
            "quantized_bits": quantized_bits,
            "ternary": ternary,
            "binary": binary,
        },
    )
    model.summary()
    print("ncands: ", ncands)
    print("nfeatures: ", nfeatures)

    register_custom_layer()
    # remove unncessary linear layers by explicitly specifying layer names
    hls4ml.model.optimizer.get_optimizer("output_rounding_saturation_mode").configure(
        layers=[
#            "qrelu_n1",
#            "qrelu_g1",
#            "softmax_g2",
        ],
        rounding_mode="AP_RND",
        saturation_mode="AP_SAT",
    )
    config = hls4ml.utils.config_from_keras_model(
        model, granularity="name",
        # default_precision="ap_fixed<16,6>"
        # default_precision="ap_fixed<20,9>"
    )
    # config = hls4ml.utils.config_from_keras_model(model, granularity='name', default_precision='ap_fixed<32,16>')
    config["Model"]["Strategy"] = "Latency"

    # Handle large span of numerical values in input
    # inputPrecision = "ap_fixed<12,4,AP_RND,AP_SAT>"
    # inputPrecision = "ap_fixed<16,7,AP_RND,AP_SAT>"
    # inputPrecision = "ap_fixed<18,8,AP_RND,AP_SAT>"
    # inputPrecision = "ap_fixed<12,9,AP_RND,AP_SAT>"
    # inputPrecision = "ap_fixed<14,6,AP_RND,AP_SAT>" #DeepSet
    # inputPrecision = "ap_fixed<16,6,AP_RND,AP_SAT>"
    # inputPrecision = "ap_fixed<16,7,AP_RND,AP_SAT>"
    # inputPrecision = "ap_fixed<16,9,AP_RND,AP_SAT>"
    inputPrecision = "ap_fixed<20,9,AP_RND,AP_SAT>"

    print ("Default generated config")
    print (config)

    for layer in model.layers:
        if layer.__class__.__name__ in ["BatchNormalization", "InputLayer"]:
        # if layer.__class__.__name__ in ["InputLayer"]:
        # if layer.__class__.__name__ in ["QBatchNormalization","BatchNormalization", "InputLayer"]:
        # if layer.__class__.__name__ in ["InputLayer"]:
            config["LayerName"][layer.name]["Precision"] = inputPrecision
            # config["LayerName"][layer.name]["accum"] = inputPrecision
            config["LayerName"][layer.name]["result"] = inputPrecision
            config["LayerName"][layer.name]["Trace"] = trace
        # elif layer.__class__.__name__ in ["QBatchNormalization","BatchNormalization"]:
        #     config["LayerName"][layer.name]["Precision"]["accum"] = "ap_fixed<16,7,AP_RND,AP_SAT>" 
        #     config["LayerName"][layer.name]["Precision"]["result"] = "ap_fixed<16,7,AP_RND,AP_SAT>" 
        elif layer.__class__.__name__ in [
            "Permute",
            "Concatenate",
            "Flatten",
            "Reshape",
            # "GlobalAveragePooling1D",
            # "AveragePooling1D",
            "UpSampling1D",
            "Add",
        ]:
            print("Skipping trace for:", layer.name)
        else:
            config["LayerName"][layer.name]["Trace"] = trace

    for layerName in config["LayerName"]:
        config["LayerName"][layerName]["Trace"] = True

    config["LayerName"]["output_class"]["Precision"]["result"] = inputPrecision
    config["LayerName"]["output_reg"]["Precision"]["result"] = inputPrecision
    config["LayerName"]["output_class"]["Implementation"] = "latency"
    config["LayerName"]["output_reg"]["Implementation"] = "latency"

    print ("Changed  config")
    print (config)

    # for layer in model.layers:
    #     if "qDense_phi" in layer.name:
    #         if modelArchName in ["DeepSet"]:
    #             print ("Add custom pointwise implementation for layer", layer.name)
    #             config["LayerName"][layer.name]["ConvImplementation"] = "Pointwise"
    #         config["LayerName"][layer.name]["Strategy"] = "Latency"


    layerNames = [layer.name for layer in model.layers]


    # if "qDense_phi1" in layerNames: config["LayerName"]["qDense_phi1"]["ReuseFactor"] = 2
    # if "qDense_phi2" in layerNames: config["LayerName"]["qDense_phi2"]["ReuseFactor"] = 2
    # if "qDense_phi3" in layerNames: config["LayerName"]["qDense_phi3"]["ReuseFactor"] = 6
    # if "qDense_phi4" in layerNames: config["LayerName"]["qDense_phi4"]["ReuseFactor"] = 6

        # config["LayerName"]["qDense_phi1"]["ConvImplementation"] = "Pointwise"
        # config["LayerName"]["qDense_phi2"]["ConvImplementation"] = "Pointwise"
        # config["LayerName"]["qDense_phi3"]["ConvImplementation"] = "Pointwise"
        # config["LayerName"]["qDense_phi1"]["ReuseFactor"] = 2
        # config["LayerName"]["qDense_phi2"]["ReuseFactor"] = 4
        # config["LayerName"]["qDense_phi3"]["ReuseFactor"] = 4
        # config["LayerName"]["qDense_rho1"]["ReuseFactor"] = 1

    for layer in model.layers:
        config["LayerName"][layer.name]["Strategy"] = "latency"

    print("Converting the Keras Model !")


    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=outFolder,
        io_type="io_parallel",
        part="xcvu13p-flga2577-2-e", #real one
        clock_period=2.777777778,
        backend='Vitis',
    )

    print("Compiling the Model !")

    hls_model.compile()

    # Do plots
    hls4ml.utils.plot_model(
        hls_model,
        show_shapes=True,
        show_precision=True,
        to_file=f"{outFolder}/hls4ml_in_plot_{modelname}.png",
    )
    tf.keras.utils.plot_model(model, to_file=f"{outFolder}/keras_in_plot_{modelname}.png")
    tf.keras.utils.plot_model(model, to_file=f"{outFolder}/keras_in_plot_{modelname}.pdf")

    y_keras , y_ptreg_keras= model.predict(X_test)
    y_hls, y_ptreg_hls = hls_model.predict(np.ascontiguousarray(X_test))

    accuracy_keras = float(
        accuracy_score(np.argmax(Y_test, axis=-1), np.argmax(y_keras, axis=-1))
    )
    accuracy_hls4ml = float(
        accuracy_score(np.argmax(Y_test, axis=-1), np.argmax(y_hls, axis=-1))
    )

    accs = {}
    accs["cpu"] = accuracy_keras
    accs["fpga"] = accuracy_hls4ml

    with open("{}/{}_acc.txt".format(outFolder, modelname), "wb") as fp:
        pickle.dump(accs, fp)
    print("Keras:\n", accuracy_keras)
    print("hls4ml:\n", accuracy_hls4ml)

    accs_log = np.zeros(2)
    accs_log[0] = accuracy_keras
    accs_log[1] = accuracy_hls4ml
    np.savetxt("{}/acc.log".format(outFolder), accs_log, fmt="%.6f")

    
    # Plot the ROC curves
    fig = plt.figure()
    ax = fig.add_subplot()

    fpr, tpr, threshold = roc_curve(Y_test[:,0], y_keras[:,0])
    auc1 = auc(fpr, tpr)
    ax.plot(
            tpr,
            fpr,
            label="%s, auc = %.1f%%" % ('b tag', auc1 * 100.0),
           )
    fpr, tpr, threshold = roc_curve(Y_test[:,0], y_hls[:,0])
    auc1 = auc(fpr, tpr)
    ax.plot(
            tpr,
            fpr,
            label="%s HLS, auc = %.1f%%" % ('b tag', auc1 * 100.0),
            linestyle="dotted",
           )
    ax.semilogy()
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Bkg. mistag rate")
    plt.xlim(0.,1.)
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='lower right')
    hep.cms.label("Private Work", data=False, rlabel = "14 TeV (PU 200)")
    plt.figtext(0.2, 0.83, r"{}".format(modelname))
    plt.savefig(f"{outFolder}/ROC_keras_{modelname}.png")
    plt.savefig(f"{outFolder}/ROC_keras_{modelname}.pdf")

    if trace:
        print("Running tracing!")
        profile_plots = numerical(model, hls_model, X_test_small)
        for i, p in enumerate(profile_plots):
            p.savefig(f"{outFolder}/profile_{modelname}_{i}.png")
            p.savefig(f"{outFolder}/profile_{modelname}_{i}.pdf")
        plt.cla()

        y_hls, hls4ml_trace = hls_model.trace(X_test_small)
        keras_trace = get_ymodel_keras(model, X_test_small, ignoreLayer = False)

        for layer in hls4ml_trace.keys():
            print ("Doing profiling 2d for layer", layer)
            plt.figure()
            plt.scatter(hls4ml_trace[layer].flatten(), keras_trace[layer].flatten(), s=0.2)
            min_x = min(np.amin(hls4ml_trace[layer]), np.amin(keras_trace[layer]))
            max_x = max(np.amax(hls4ml_trace[layer]), np.amax(keras_trace[layer]))
            plt.plot([min_x, max_x], [min_x, max_x], c="gray")
            plt.xlabel("hls4ml {}".format(layer))
            plt.ylabel("QKeras {}".format(layer))
            plt.savefig(f"{outFolder}/profile_2d_{layer}.png")
            plt.savefig(f"{outFolder}/profile_2d_{layer}.pdf")
            plt.cla()

    

    if build:
        print("Running synthesis!")
        report = hls_model.build(csim=False, synth=True, vsynth=True)
        print(report["CSynthesisReport"])


def getReports(indir, modelname):

    with open("{}/{}_acc.txt".format(indir, modelname), "rb") as fp:
        acc = pickle.load(fp)

    data_ = {}
    if "DeepSet" in modelname:
        data_["architecture"] = "DeepSet"  
    elif "DeepSet-MHA" in modelname:
        data_["architecture"] = "DeepSet-MHA"  
    else:
        data_["architecture"] = "Unknown"

    data_["precision"] = str(indir.split("_")[-1].replace("bit", "")).replace("/", "")
    data_["acc_ratio"] = round(acc["fpga"] / acc["cpu"], 2)
    report_vsynth = Path("{}/vivado_synth.rpt".format(indir))
    report_csynth = Path(
        "{}/myproject_prj/solution1/syn/report/myproject_csynth.rpt".format(indir)
    )

    if report_vsynth.is_file() and report_csynth.is_file():
        # Get the resources from the logic synthesis report
        with report_vsynth.open() as report:
            lines = np.array(report.readlines())
            lut = int(
                lines[np.array(["CLB LUTs*" in line for line in lines])][0].split("|")[
                    2
                ]
            )
            ff = int(
                lines[np.array(["CLB Registers" in line for line in lines])][0].split(
                    "|"
                )[2]
            )
            bram = float(
                lines[np.array(["Block RAM Tile" in line for line in lines])][0].split(
                    "|"
                )[2]
            )
            dsp = int(
                lines[np.array(["DSPs" in line for line in lines])][0].split("|")[2]
            )
            lut_rel = round(
                float(
                    lines[np.array(["CLB LUTs*" in line for line in lines])][0]
                    .split("|")[5]
                    .replace("<", "")
                ),
                1,
            )
            ff_rel = round(
                float(
                    lines[np.array(["CLB Registers" in line for line in lines])][
                        0
                    ].split("|")[5]
                ),
                1,
            )
            bram_rel = round(
                float(
                    lines[np.array(["Block RAM Tile" in line for line in lines])][
                        0
                    ].split("|")[5]
                ),
                1,
            )
            dsp_rel = round(
                float(
                    lines[np.array(["DSPs" in line for line in lines])][0].split("|")[5]
                ),
                1,
            )

            data_["lut"] = "{} ({}\%)".format(lut, lut_rel)
            data_["ff"] = "{} ({}\%)".format(ff, ff_rel)
            data_["bram"] = "{} ({}\%)".format(bram, bram_rel)
            data_["dsp"] = "{} ({}\%)".format(dsp, dsp_rel)

        with report_csynth.open() as report:
            lines = np.array(report.readlines())
            lat_line = lines[
                np.argwhere(
                    np.array(["Latency (cycles)" in line for line in lines])
                ).flatten()[0]
                + 3
            ]
            data_["latency_clks"] = round(int(lat_line.split("|")[2]))
            data_["latency_ns"] = round(int(lat_line.split("|")[2]) * 5.0)
            data_["latency_ii"] = round(int(lat_line.split("|")[6]))

    return data_


# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("-C", "--create", help="Create projects", action="store_true")
parser.add_argument("-T", "--trace", help="Trace", action="store_true")
parser.add_argument("-B", "--build", help="Build projects", action="store_true")
parser.add_argument("-D", "--debug", help="High verbose", action="store_true")

parser.add_argument('-f','--file', help = 'input file name part')
parser.add_argument('-o','--outname', help = 'output file name part')
parser.add_argument('-c','--flav', help = 'Which flavor to run, options are b, bt, btg.')
parser.add_argument('-i','--input', help = 'Which input to run, options are baseline, ext1, all.')
parser.add_argument('-m','--model', help = 'Which model to evaluate, options are DeepSet, DeepSet-MHA.')
parser.add_argument('--splitTau', dest = 'splitTau', default = False, action='store_true')
parser.add_argument('--splitGluon', dest = 'splitGluon', default = False, action='store_true')
parser.add_argument('--splitCharm', dest = 'splitCharm', default = False, action='store_true')
parser.add_argument('--regression', dest = 'regression', default = False, action='store_true')
parser.add_argument('--pruning', dest = 'pruning', default = False, action='store_true')
parser.add_argument('--timestamp', dest = 'timestamp')

args = parser.parse_args()

print('#'*30)
for arg in vars(args):
    print('%s: %s' %(arg, getattr(args, arg)))
print('#'*30)

if __name__ == "__main__":
    # Generate projects and produce firmware
    if args.create or args.build:
        start = time.time()
        synthesize(
            args.file,
            args.timestamp,
            args.flav,
            args.input,
            args.model,
            args.outname,
            args.regression,
            args.pruning,
            build=args.build
        )


        end = time.time()
        print("Ended after {:.4f} s".format(end - start))

    # Only read projects
    else:
        
        if args.input == "baselineHW":
            feature_names = pfcand_fields_baselineHW
        if args.input == "baselineEmulator":
            feature_names = pfcand_fields_baselineEmulator
        elif args.input == "ext1":
            feature_names = pfcand_fields_ext1
        elif args.input == "ext2":
            feature_names = pfcand_fields_ext2
        elif args.input == "all":
            feature_names = pfcand_fields_all

        ncands = 16 
        nfeatures = len(feature_names)
        nbits = 8

        outFolder = "outputSynthesis/"+args.outname+"/Training_" + args.timestamp + "/"

        modelname = 'model_'+modelnamesDict[args.model]+"_nconst_"+str(ncands)+"_nfeatures_"+str(nfeatures)+"_nbits_"+str(nbits)
        if args.pruning:
            modelname = modelname + "_pruned"

        import pandas

        dataMap = {
            "architecture": [],
            "precision": [],
            "acc_ratio": [],
            "dsp": [],
            "lut": [],
            "ff": [],
            "bram": [],
            "latency_clks": [],
            "latency_ns": [],
            "latency_ii": [],
        }

        # for mname in models:
        print("Reading hls project {}/".format(outFolder))

        datai = getReports("{}/".format(outFolder), modelname)
        for key in datai.keys():
            dataMap[key].append(datai[key])

        dataPandas = pandas.DataFrame(dataMap)
        print(dataPandas)
        print(
            dataPandas.to_latex(
                columns=[
                    "architecture",
                    "precision",
                    "acc_ratio",
                    "latency_ns",
                    "latency_clks",
                    "latency_ii",
                    "dsp",
                    "lut",
                    "ff",
                    "bram",
                ],
                header=[
                    "Architecture",
                    "Precision ( \# bits )",
                    "Accuracy Ratio (FPGA/CPU)",
                    "Latency [ns]",
                    "Latency [clock cycles]",
                    "II [clock cycles]",
                    "DSP",
                    "LUT",
                    "FF",
                    "BRAM",
                ],
                index=False,
                escape=False,
            )
        )


