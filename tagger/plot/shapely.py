from argparse import ArgumentParser
import os, shutil, json

#Import from other modules
from tagger.data.tools import make_data, load_data, to_ML
from tagger.firmware.hls4ml_convert import convert
import tagger.train.models
import style
#Third parties
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import hls4ml
from qkeras.utils import load_qmodel
from sklearn.metrics import roc_curve, auc,precision_recall_curve
import shap
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

style.set_style()

def shapPlot(shap_values, feature_names, class_names):
    fig,ax = plt.subplots(1,1,figsize=(18,15))
    
    feature_order = np.argsort(np.sum(np.mean(np.abs(shap_values), axis=1), axis=0))
    num_features = (shap_values[0].shape[1])
    feature_inds = feature_order
    y_pos = np.arange(len(feature_inds))
    left_pos = np.zeros(len(feature_inds))

    axis_color="#333333"

    class_inds = np.argsort([-np.abs(shap_values[i]).mean() for i in range(len(shap_values))])
    colormap = cm.get_cmap('Set1', len(class_names))  # Use 'tab10' with enough colors

    for i, ind in enumerate(class_inds):
        global_shap_values = np.abs(shap_values[ind]).mean(0)
        label = class_names[ind]
        ax.barh(y_pos, global_shap_values[feature_inds], 0.7, left=left_pos, align='center',label=label,color=colormap(class_inds[i]))
        left_pos += global_shap_values[feature_inds]

    ax.set_yticklabels([feature_names[i] for i in feature_inds])
    ax.legend(loc='best',fontsize=30)

    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color)
    ax.set_yticks(range(len(feature_order)), [feature_names[i] for i in feature_order],fontsize=30)
    ax.set_xlabel("mean (shapley value) - (average impact on model output magnitude)",fontsize=30)
    plt.tight_layout()

def doPlots(model,outputdir,inputdir):
    os.makedirs(outputdir, exist_ok=True)

    modelsAndNames = {"model":model}
    
    data, _, class_labels, input_vars, extra_vars = load_data(inputdir, percentage=0.1,test_ratio=0.0)
    X_test, Y_test, pt_target, truth_pt, _ = to_ML(data, class_labels)

    labels = list(class_labels.keys())

    model2 = tf.keras.Model(model.input, model.output[0])
    model3 = tf.keras.Model(model.input, model.output[1])

    for explainer, name  in [(shap.GradientExplainer(model2, X_test[:1000]), "GradientExplainer"), ]:
        print("... {0}: explainer.shap_values(X)".format(name))
        shap_values = explainer.shap_values(X_test[:1000])
        new = np.sum(shap_values, axis = 2)
        print("... shap summary_plot classification")
        plt.clf()

        new = np.transpose(new, (2, 0, 1))
        shapPlot(new, input_vars, labels)
        plt.savefig(outputdir+"/shap_summary_class.pdf")
        plt.savefig(outputdir+"/shap_summary_class.png")

    for explainer, name  in [(shap.GradientExplainer(model3, X_test[:1000]), "GradientExplainer"), ]:
        print("... {0}: explainer.shap_values(X)".format(name))
        shap_values = explainer.shap_values(X_test[:1000])
        new = np.sum(shap_values, axis = 2)
        print("... shap summary_plot regression")
        plt.clf()
        labels = ["Regression"]
        new = np.transpose(new, (2, 0, 1))
        shapPlot(new, input_vars, labels)
        plt.savefig(outputdir+"/shap_summary_reg.pdf")
        plt.savefig(outputdir+"/shap_summary_reg.png")


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='output/baseline/model/saved_model.h5' , help = 'Input model path for comparison')    
    parser.add_argument('-o','--outpath', default='output/baseline/plots/profile' , help = 'Jet tagger plotting directory')    
    parser.add_argument('-i','--input', default='data/jetTuple.root' , help = 'Path to profiling data rootfile')
    parser.add_argument('-r','--remake', default=False , help = 'Remake profiling data? ')

    args = parser.parse_args()

    #Load the model
    model=load_qmodel(args.model)

    if args.remake:
        make_data(infile=args.input,outdir="profiling_data/")

    doPlots(model,args.outpath,"profiling_data/")