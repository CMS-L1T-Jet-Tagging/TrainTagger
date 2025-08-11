import json

#Third parties
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import mplhep as hep
import tagger.plot.style as style
import awkward as ak

import os
import sys
import pickle
style.set_style()


from os import listdir
from os.path import isfile, join


if __name__ == "__main__":
    onlyfiles = [f for f in listdir(sys.argv[1]) if isfile(join(sys.argv[1], f))]

    fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax, fontsize=style.CMSHEADER_SIZE)
    colormap = cm.get_cmap('Set1', len(8))
    linestyles = ["-","--","."]
    for j,i_file in enumerate(onlyfiles):
        print(i_file)
        with open(os.path.join(sys.argv[1], i_file), "r") as f: plotting_dict = pickle.load(f)
        
        for i,class_label in enumerate(plotting_dict['basic_ROC'].keys()):
            ax.plot(plotting_dict['basic_ROC'][class_label]['tpr'], plotting_dict['basic_ROC'][class_label]['fpr'], label=f'{i_file_+' '+style.CLASS_LABEL_STYLE[class_label]} (AUC = {plotting_dict['basic_ROC'][class_label]['roc_auc']:.2f})',
                    color=colormap(i), linewidth=style.LINEWIDTH,linestyle=linestyles[j])
            
        # Plot formatting
    ax.grid(True)
    ax.set_ylabel('Mistag Rate')
    ax.set_xlabel('Signal Efficiency')

    # auc_list = [value for key,value in ROC_dict.items()]
    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = np.argsort(auc_list)
    # ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper left',ncol=2,fontsize=style.SMALL_SIZE-3)

    ax.set_yscale('log')
    ax.set_ylim([1e-3, 1.1])

    # Save the plot
    save_path = os.path.join(sys.argv[1], "basic_ROC_merged")
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close()
        
    