import os
from argparse import ArgumentParser

# Third parties
import numpy as np
from sklearn.metrics import roc_curve, auc

# Plotting
import matplotlib.pyplot as plt

# Import from other modules
from tagger.plot import style
from tagger.data.tools import make_data, load_data, to_ML
from tagger.model.common import fromFolder
from tagger.plot import common

style.set_style()


def rms(array):
    return np.sqrt(np.mean(array ** 2))


def doPlots(model, outputdir, inputdir):
    os.makedirs(outputdir, exist_ok=True)

    modelsAndNames = {"model": model}

    data, _, class_labels, input_vars, extra_vars = load_data(
        inputdir, percentage=100, test_ratio=0.0)
    X_test, Y_test, pt_target, truth_pt, _ = to_ML(
        data, class_labels)  # Last thing was reconstructed pt

    labels = list(class_labels.keys())
    model.hls4ml_convert("temp", build=False)

    y_hls, y_ptreg_hls = model.hls_jet_model.predict(np.ascontiguousarray(X_test))
    y_class, y_ptreg = model.jet_model.predict(np.ascontiguousarray(X_test))
    jet_pt_phys = np.array(data['jet_pt_phys'])

    modelsAndNames["Y_predict"] = y_class
    modelsAndNames["Y_predict_reg"] = y_ptreg

    modelsAndNames["Y_hls_predict"] = y_hls
    modelsAndNames["Y_hls_predict_reg"] = y_ptreg_hls

    for iJet in range(y_hls.shape[0]):
        print_class = False
        for i, label in enumerate(labels):
            if abs(np.array(data['jet_SC4NGJet_score_'+label])[iJet] - y_hls[iJet][i]) > 0.001:
                print_class = True
        if print_class == True:
            print("=== " + str(iJet) + " ===")
            print("Inputs: " + str(X_test[iJet]))
            for i, label in enumerate(labels):
                print(label + ": cmssw : " +
                      str(np.array(data['jet_SC4NGJet_score_'+label])[iJet]))
                print(label + ": hls : " + str(y_hls[iJet][i]))
                print(label + ": tf : " + str(y_class[iJet][i]))

            if abs(np.array(data['jet_SC4NGJet_score_regression'])[iJet] - y_ptreg_hls[iJet]) > 0.001:
                print("pt reg cmssw : " +
                      str(np.array(data['jet_SC4NGJet_score_regression'])[iJet]))
                print("pt reg hls : " + str(y_ptreg_hls[iJet]))
                print("pt reg tf : " + str(y_ptreg[iJet]))

    jet_pt_cor_reg = jet_pt_phys * modelsAndNames["Y_predict_reg"][:, 0]
    jet_pt_cor_reg_hls = jet_pt_phys * \
        modelsAndNames["Y_hls_predict_reg"][:, 0]
    jet_pt_cor_reg_emu = jet_pt_phys * \
        np.array(data['jet_SC4NGJet_score_regression'])

    figure = common.plot_2d(np.array(modelsAndNames["Y_predict_reg"][:, 0]), np.array(
        data['jet_SC4NGJet_score_regression']), (0, 2), (0, 2), "Tensorflow", "CMSSW Emulation", "Jet Regression")
    plt.savefig("%s/jetRegression_2D.png" % outputdir, bbox_inches='tight')
    plt.savefig("%s/jetRegression_2D.pdf" % outputdir, bbox_inches='tight')

    plt.clf()
    figure = common.plot_histo([modelsAndNames["Y_predict_reg"][:, 0], np.array(data['jet_SC4NGJet_score_regression']), np.array(
        modelsAndNames["Y_hls_predict_reg"][:, 0])], ["Tensorflow", "CMSSW Emulation", "hls4ml"], "", 'Regression Output', 'a.u.', range=(0, 2))
    bit_accurate = np.count_nonzero((np.array(
        data['jet_SC4NGJet_score_regression']) - np.array(modelsAndNames['Y_hls_predict_reg'][:, 0])))
    print("Percent bit accuracy between CMSSW emulator and hls4ml for regression is",
          100 - 100*bit_accurate/len(np.array(data['jet_SC4NGJet_score_regression'])), "%")
    plt.savefig("%s/jetRegression_1D.png" % outputdir, bbox_inches='tight')
    plt.savefig("%s/jetRegression_1D.pdf" % outputdir, bbox_inches='tight')

    for i, label in enumerate(labels):
        plt.close()
        plt.clf()
        figure = common.plot_histo([np.array(modelsAndNames['Y_predict'][:, i]), np.array(data['jet_SC4NGJet_score_'+label]), np.array(
            modelsAndNames['Y_hls_predict'][:, i])], ["Tensorflow", "CMSSW Emulation", "hls4ml"], "", style.CLASS_LABEL_STYLE[label]+' score', 'a.u.', range=(0, 1))
        bit_accurate = np.count_nonzero((np.array(
            data['jet_SC4NGJet_score_'+label]) - np.array(modelsAndNames['Y_hls_predict'][:, i])))
        print("Percent bit accuracy between CMSSW emulator and hls4ml for " + label + " classification is",
              100 - 100*bit_accurate/len(np.array(data['jet_SC4NGJet_score_'+label])), "%")

        plt.savefig("%s/%s_score_1D.png" %
                    (outputdir, label), bbox_inches='tight')
        plt.savefig("%s/%s_score_1D.pdf" %
                    (outputdir, label), bbox_inches='tight')

        plt.clf()
        figure = common.plot_2d(np.array(modelsAndNames['Y_predict'][:, i]), np.array(
            data['jet_SC4NGJet_score_'+label]), (0, 1), (0, 1), "Tensorflow", "CMSSW Emulation", style.CLASS_LABEL_STYLE[label]+" score")
        plt.savefig("%s/%s_score_2D.png" %
                    (outputdir, label), bbox_inches='tight')
        plt.savefig("%s/%s_score_2D.pdf" %
                    (outputdir, label), bbox_inches='tight')

    fpr = {}
    tpr = {}
    auc1 = {}
    thresholds = {}
    # Loop over classes (labels) to get metrics per class
    for i, label in enumerate(labels):
        fpr[label], tpr[label], thresholds[label] = roc_curve(
            Y_test[:, i], modelsAndNames["Y_predict"][:, i])
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
        fpr[label], tpr[label], thresholds[label] = roc_curve(
            Y_test[:, i], modelsAndNames["Y_hls_predict"][:, i])
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
        fpr[label], tpr[label], thresholds[label] = roc_curve(
            Y_test[:, i], data['jet_SC4NGJet_score_'+label])
        auc1[label] = auc(fpr[label], tpr[label])

    modelsAndNames["Emulation"] = {}
    modelsAndNames["Emulation"]["ROCs"] = {}
    modelsAndNames["Emulation"]["ROCs"]["tpr"] = tpr
    modelsAndNames["Emulation"]["ROCs"]["fpr"] = fpr
    modelsAndNames["Emulation"]["ROCs"]["auc"] = auc1

    # ===========================#

    for i, label in enumerate(labels):
        plt.close()
        common.plot_roc(modelsAndNames, label, keys=["Tensorflow", "Emulation", "hls4ml"], labels=[
                        "Tensorflow", "CMSSW Emulation", "hls4ml"], title=style.CLASS_LABEL_STYLE[label]+" ROC Comparison")
        plt.savefig(outputdir+"/ROC_Emulation_comparison_" +
                    label+".png", bbox_inches='tight')
        plt.savefig(outputdir+"/ROC_Emulation_comparison_" +
                    label+".pdf", bbox_inches='tight')

    response_reg = jet_pt_cor_reg / data['jet_genmatch_pt']
    response_emu = jet_pt_cor_reg_emu / data['jet_genmatch_pt']
    response_hls = jet_pt_cor_reg_hls / data['jet_genmatch_pt']

    figure = common.plot_histo([response_reg, response_emu, response_hls],
                               ["Emulation" + " median: "+str(np.round(np.median(response_emu), 3))+" rms: "+str(np.round(rms(response_emu), 3)),
                                "Tensorflow" + " median: " +
                                str(np.round(np.median(response_reg), 3)) +
                                " rms: "+str(np.round(rms(response_reg), 3)),
                                "hls4ml" + " median: "+str(np.round(np.median(response_hls), 3))+" rms: "+str(np.round(rms(response_hls), 3)),],
                               "Jet Regression", 'Jet Response (L1/Gen)', 'a.u.', range=(0, 2))
    plt.savefig(outputdir+"/response_emulation"+".png", bbox_inches='tight')
    plt.savefig(outputdir+"/response_emulation"+".pdf", bbox_inches='tight')
    plt.close()
    return


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m', '--model_path', default='output/baseline',
                        help='Input model path for comparison')
    parser.add_argument('-o', '--outpath', default='output/baseline/plots/emulation',
                        help='Jet tagger plotting directory')
    parser.add_argument('-i', '--input', default='data/jetTuple_extended_5.root',
                        help='Path to emulation data rootfile')
    parser.add_argument('-r', '--remake', default=False,
                        help='Remake emulation data? ')

    args = parser.parse_args()

    # Load the model
    model = fromFolder(args.model_path)

    if args.remake:
        make_data(infile=args.input, outdir="emulation_data/",
                  extras='extra_emulation_fields', tree="outnano/Jets")

    doPlots(model, args.outpath, "emulation_data/")
