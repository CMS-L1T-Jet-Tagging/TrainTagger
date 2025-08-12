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
    list_of_dirs = next(os.walk(sys.argv[1]))[1]
    print(list_of_dirs)
    plot_dir = "/latest/plots/plotting_dict.pkl"
    
    plot_dir = "/plotting_dict.pkl"
    
    meta_plotting_dict = {}
    
    
    
    
    for j,i_folder in enumerate(list_of_dirs):
        print(i_folder)
        
        with open(os.path.join(sys.argv[1], i_folder+plot_dir), "rb") as f: plotting_dict = pickle.load(f)
        
        meta_plotting_dict[i_folder] = plotting_dict
        class_labels = plotting_dict['basic_ROC'].keys()
        
    
    linestyles = ["-","--","."]
    
    fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax, fontsize=style.CMSHEADER_SIZE)
    colormap = matplotlib.colormaps.get_cmap('Set1')
    for j,i_folder in enumerate(meta_plotting_dict.keys()):
        
        for i,class_label in enumerate(meta_plotting_dict[i_folder]['basic_ROC'].keys()):
            ax.plot(meta_plotting_dict[i_folder]['basic_ROC'][class_label]['tpr'], meta_plotting_dict[i_folder]['basic_ROC'][class_label]['fpr'], label=f"{i_folder} {style.CLASS_LABEL_STYLE[class_label]} (AUC = {meta_plotting_dict[i_folder]['basic_ROC'][class_label]['roc_auc']:.2f})",
                    color=colormap(i), linewidth=style.LINEWIDTH,linestyle=linestyles[j])
            
        # Plot formatting
    ax.grid(True)
    ax.set_ylabel('Mistag Rate')
    ax.set_xlabel('Signal Efficiency')

    # auc_list = [value for key,value in ROC_dict.items()]
    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = np.argsort(auc_list)
    ax.legend(loc='upper left',ncol=2,fontsize=style.SMALL_SIZE-3)

    ax.set_yscale('log')
    ax.set_ylim([1e-3, 1.1])

    # Save the plot
    save_path = os.path.join(sys.argv[1], "basic_ROC_merged")
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close()
    
    for i,class_label in enumerate(class_labels):
        fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
        hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax, fontsize=style.CMSHEADER_SIZE)
        
        for j,i_folder in enumerate(meta_plotting_dict.keys()):
            print(i)
            ax.plot(meta_plotting_dict[i_folder]['basic_ROC'][class_label]['tpr'], meta_plotting_dict[i_folder]['basic_ROC'][class_label]['fpr'], label=f"{i_folder} {style.CLASS_LABEL_STYLE[class_label]} (AUC = {meta_plotting_dict[i_folder]['basic_ROC'][class_label]['roc_auc']:.2f})",
                    color=colormap(i), linewidth=style.LINEWIDTH,linestyle=linestyles[j])
                
            # Plot formatting
        ax.grid(True)
        ax.set_ylabel('Mistag Rate')
        ax.set_xlabel('Signal Efficiency')

        ax.legend(loc='upper left',ncol=2,fontsize=style.SMALL_SIZE-3)

        ax.set_yscale('log')
        ax.set_ylim([1e-3, 1.1])

        # Save the plot
        save_path = os.path.join(sys.argv[1], "basic_ROC_merged_"+class_label)
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.close()
    
    
        
    