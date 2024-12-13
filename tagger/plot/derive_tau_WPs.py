'''
Scripts for deriving the working points for taus.

Usage: 

python derive_tau_WPs.py
'''
import sys, gc
import uproot4
import numpy as np
import awkward as ak
from argparse import ArgumentParser
from qkeras.utils import load_qmodel

# Add path so the script sees other modules
sys.path.append('../')
from datatools.createDataset import dict_fields
from datatools import helpers
from hist import Hist
import hist

#Plotting
import matplotlib.pyplot as plt
import mplhep as hep
import tagger.plot.style as style

style.set_style()

def delta_r(eta1, phi1, eta2, phi2):
    """
    Calculate the delta R between two sets of eta and phi values.
    """
    delta_eta = eta1 - eta2
    delta_phi = phi1 - phi2

    # Ensure delta_phi is within -pi to pi
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(delta_eta**2 + delta_phi**2)

def find_rate(rate_list, target_rate = 28):
    
    RateRange = 1 #kHz
    
    idx_list = []
    
    for i in range(len(rate_list)):
        if target_rate-RateRange <= rate_list[i] <= target_rate + RateRange:
            idx_list.append(i)
            
    return idx_list

def plot_rate(rate_list, pt_list, nn_list, target_rate = 28, correct_pt=True, mult=False):
    
    fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax)
    im = ax.scatter(nn_list, pt_list, c=rate_list, s=500, marker='s',
                    cmap='Spectral_r',
                    linewidths=0,
                    norm=matplotlib.colors.LogNorm())

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'Di-$\tau_h$ rate [kHZ]')

    ax.set_ylabel(r"Min reco $p_T$ [GeV]")
    ax.set_xlabel(r"Min NN Score")
    
    if mult:
        ax.set_xlim([0,0.06])
        ax.set_ylim([10,100])
    else:
        ax.set_xlim([0,0.6])
        ax.set_ylim([10,100])
    
    #Find the target rate points, plot them and print out some info as well
    target_rate_idx = find_rate(rate_list, target_rate = target_rate)
    
    legend_count = 0
    for i in target_rate_idx:
        print("Rate: ", rate_list[i])
        print("NN Cut: ", nn_list[i])
        print("pt Cut: ", pt_list[i])
        print("------")
        
        if legend_count == 0:
            ax.scatter(nn_list[i], pt_list[i], s=600, marker='*',
                        color ='firebrick', label = r"${} \pm 1$ kHz".format(target_rate))
        else:
            ax.scatter(nn_list[i], pt_list[i], s=600, marker='*',
                        color ='firebrick')
            
        legend_count += 1
    
    ax.legend(loc='upper right')
    if mult:
        plot_name = 'tau_rate_scan_ptcorrected_mult' if correct_pt else 'tau_rate_scan_ptuncorrected_mult' 
    else:
        plot_name = 'tau_rate_scan_ptcorrected' if correct_pt else 'tau_rate_scan_ptuncorrected' 
    plt.savefig(f'plots/{plot_name}'+'.png', bbox_inches='tight')
    plt.savefig(f'plots/{plot_name}'+'.pdf', bbox_inches='tight')

def derive_tau_rate(model, minbias_path, mult=True, input_tag='ext7', tree='jetntuple/Jets', n_entries=500000, correct_pt=True):
    '''
    Derive the tau rate, using n_entries minbias events 
    '''

    minbias_rate = 32e+3 #32 kHZ

    #Load the minbias data
    minbias = uproot4.open(minbias_path)[tree]

    raw_event_id = helpers.extract_array(minbias, 'event', n_entries)
    raw_jet_pt = helpers.extract_array(minbias, 'jet_pt', n_entries)
    raw_jet_eta = helpers.extract_array(minbias, 'jet_eta_phys', n_entries)
    raw_jet_phi = helpers.extract_array(minbias, 'jet_phi_phys', n_entries)
    raw_inputs = helpers.extract_nn_inputs(minbias, input_fields_tag=input_tag, nconstit=16, n_entries=n_entries)

    #Count number of total event
    n_events = len(np.unique(raw_event_id))
    print("Total number of minbias events: ", n_events)

    #Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays  = helpers.group_id_values(raw_event_id, raw_jet_pt, raw_jet_eta, raw_jet_phi, raw_inputs, num_elements=2)

    # Extract the grouped arrays
    # Jet pt is already sorted in the producer, no need to do it here
    jet_pt, jet_eta, jet_phi, jet_nn_inputs = grouped_arrays

    #calculate delta_r
    eta1, eta2 = jet_eta[:, 0], jet_eta[:, 1]
    phi1, phi2 = jet_phi[:, 0], jet_phi[:, 1]
    delta_r_values = delta_r(eta1, phi1, eta2, phi2)

    # Additional cuts recommended here:
    # https://indico.cern.ch/event/1380964/contributions/5852368/attachments/2841655/4973190/AnnualReview_2024.pdf
    # Slide 7
    cuts = (np.abs(eta1) < 2.172) & (np.abs(eta2) < 2.172) & (delta_r_values > 0.5)

    #Get inputs and pts for processing
    pt1_uncorrected, pt2_uncorrected = np.asarray(jet_pt[:,0][cuts]), np.asarray(jet_pt[:,1][cuts])
    input1, input2 = np.asarray(jet_nn_inputs[:, 0][cuts]).transpose(0, 2, 1), np.asarray(jet_nn_inputs[:, 1][cuts]).transpose(0, 2, 1) #Flip the last two axes

    #Get the NN predictions
    tau_index = [2,3] #2=tau positive, 3=tau_negative
    pred_score1, ratio1 = model.predict(input1)
    pred_score2, ratio2 = model.predict(input2)

    if correct_pt:
        pt1 = pt1_uncorrected*(ratio1.flatten())
        pt2 = pt2_uncorrected*(ratio2.flatten())
    else:
        pt1 = pt1_uncorrected
        pt2 = pt2_uncorrected

    if mult:
        tau_score1 = np.multiply(pred_score1[:,tau_index[0]], pred_score1[:,tau_index[1]])
        tau_score2 = np.multiply(pred_score2[:,tau_index[0]], pred_score2[:,tau_index[1]])
    else: #Just add them normally
        tau_score1=pred_score1[:,tau_index[0]] + pred_score1[:,tau_index[1]]
        tau_score2=pred_score2[:,tau_index[0]] + pred_score2[:,tau_index[1]]
    
    # #Put them together
    NN_score = np.vstack([tau_score1, tau_score2]).transpose()
    NN_score_min = np.min(NN_score, axis=1)

    pt = np.vstack([pt1, pt2]).transpose()
    pt_min = np.min(pt, axis=1)

    #Define the histograms (pT edge and NN Score edge)
    pT_edges = list(np.arange(0,100,2)) + [1500] #Make sure to capture everything
    NN_edges = list([round(i,3) for i in np.arange(0, 0.1, 0.001)]) if mult else list([round(i,2) for i in np.arange(0, 1.01, 0.01)])

    RateHist = Hist(hist.axis.Variable(pT_edges, name="pt", label="pt"),
                    hist.axis.Variable(NN_edges, name="nn", label="nn"))

    RateHist.fill(pt = pt_min, nn = NN_score_min)

    #Derive the rate
    rate_list = []
    pt_list = []
    nn_list = []

    #Loop through the edges and integrate
    for pt in pT_edges[:-1]:
        for NN in NN_edges[:-1]:
            
            #Calculate the rate
            rate = RateHist[{"pt": slice(pt*1j, pT_edges[-1]*1.0j, sum)}][{"nn": slice(NN*1.0j,1.0j, sum)}]/n_events
            rate_list.append(rate*minbias_rate)

            #Append the results   
            pt_list.append(pt)
            nn_list.append(NN)

    plot_rate(rate_list, pt_list, nn_list, target_rate=28, correct_pt=correct_pt, mult=mult)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='/eos/home-s/sewuchte/www/L1T/trainings_regression_weighted/2024_08_27_v4_extendedAll200_btgc_ext7_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned/model_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned.h5' , help = 'Input model for plotting')    
    parser.add_argument('--uncorrect_pt', action='store_true', help='Enable pt correction in plot_bkg_rate_tau')
    parser.add_argument('--mult', action='store_true', help='whether to multiply the tau probablity together or not')

    args = parser.parse_args()

    model=load_qmodel(args.model)
    print(model.summary())

    #These paths are default to evaluate some of the rate
    minbias_path = '/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_ntuples_v131Xv9/extendedTRK_HW_260824/MinBias_PU200.root'

    #Parse the options a bit here
    correct_pt = not args.uncorrect_pt
    prob_multiply=args.mult

    derive_tau_rate(model, minbias_path, mult=prob_multiply, n_entries=3000000, input_tag='ext7', correct_pt=correct_pt)