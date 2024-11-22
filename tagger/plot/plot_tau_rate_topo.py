import sys
import uproot4
import numpy as np
import awkward as ak
from argparse import ArgumentParser
from qkeras.utils import load_qmodel

#Import the calculated working points
from official_WPs import WPs, WPs_CMSSW
sys.path.append('../')
from datatools.createDataset import dict_fields
from datatools import helpers

#Plotting
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'medium',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)

#line thickness
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 5


def calculate_topo_score(tau_plus, tau_minus):
    #2=tau positive, 3=tau_negative

    p_pos = tau_plus[:,0] + tau_plus[:,1]
    p_neg = tau_minus[:,0] + tau_minus[:,1]

    return np.multiply(p_pos, p_neg)

def delta_r(eta1, phi1, eta2, phi2):
    """
    Calculate the delta R between two sets of eta and phi values.
    """
    delta_eta = eta1 - eta2
    delta_phi = phi1 - phi2

    # Ensure delta_phi is within -pi to pi
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(delta_eta**2 + delta_phi**2)

def group_id_values_nn(event_id, raw_tau_score_sum, *arrays, num_elements = 2):
    '''
    Group values according to event id.
    Filter out events that has less than num_elements
    '''

    # Use ak.argsort to sort based on event_id
    sorted_indices = ak.argsort(event_id)
    sorted_event_id = event_id[sorted_indices]

    # Find unique event_ids and counts manually
    unique_event_id, counts = np.unique(sorted_event_id, return_counts=True)
    
    # Use ak.unflatten to group the arrays by counts
    grouped_id = ak.unflatten(sorted_event_id, counts)
    grouped_arrays = [ak.unflatten(arr[sorted_indices], counts) for arr in arrays]

    #Sort by tau score
    tau_score = ak.unflatten(raw_tau_score_sum[sorted_indices],counts)
    tau_sort_index = ak.argsort(tau_score, ascending=False)
    grouped_arrays_sorted = [arr[tau_sort_index] for arr in grouped_arrays]

    #Filter out groups that don't have at least 2 elements
    mask = ak.num(grouped_id) >= num_elements
    filtered_grouped_arrays = [arr[mask] for arr in grouped_arrays_sorted]

    return grouped_id[mask], filtered_grouped_arrays

def cmssw_pt_score(raw_event_id, raw_jet_pt, raw_jet_eta, raw_jet_phi, raw_cmssw_tau, raw_cmssw_taupt):

    #Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays  = helpers.group_id_values(raw_event_id, raw_jet_pt, raw_jet_eta, raw_jet_phi, raw_cmssw_tau, raw_cmssw_taupt, num_elements=2)

    # Extract the grouped arrays
    # Jet pt is already sorted in the producer, no need to do it here
    jet_pt, jet_eta, jet_phi, cmssw_tau, cmssw_taupt = grouped_arrays

    #calculate delta_r
    eta1, eta2 = jet_eta[:, 0], jet_eta[:, 1]
    phi1, phi2 = jet_phi[:, 0], jet_phi[:, 1]
    delta_r_values = delta_r(eta1, phi1, eta2, phi2)

    # Additional cuts recommended here:
    # https://indico.cern.ch/event/1380964/contributions/5852368/attachments/2841655/4973190/AnnualReview_2024.pdf
    # Slide 7
    cuts = (np.abs(eta1) < 2.172) & (np.abs(eta2) < 2.172) & (delta_r_values > 0.5)

    #Get cmssw attribubtes to calculate the rate
    cmssw_pt1, cmssw_pt2 = np.asarray(cmssw_taupt[:,0][cuts]), np.asarray(cmssw_taupt[:,1][cuts])
    cmssw_pt = np.vstack([cmssw_pt1, cmssw_pt2]).transpose()
    cmssw_pt_min = np.min(cmssw_pt, axis=1)

    #Do similar thing for the tau score
    cmssw_tau1, cmssw_tau2 = np.asarray(cmssw_tau[:,0][cuts]), np.asarray(cmssw_tau[:,1][cuts])
    cmssw_tau = np.vstack([cmssw_tau1, cmssw_tau2]).transpose()
    cmssw_tau_min =  np.min(cmssw_tau, axis=1)

    pt1, pt2 = np.asarray(jet_pt[:,0][cuts]), np.asarray(jet_pt[:,1][cuts])
    pt = np.vstack([pt1, pt2]).transpose()
    pt_min = np.min(pt, axis=1)

    return event_id[cuts], pt_min, cmssw_pt_min, cmssw_tau_min


def model_pt_score(raw_event_id, raw_tau_score_sum, raw_tau_plus, raw_tau_minus, raw_jet_pt, raw_pt_correction, raw_jet_eta, raw_jet_phi):

    event_id, grouped_arrays  = group_id_values_nn(raw_event_id, raw_tau_score_sum, raw_tau_plus, raw_tau_minus, raw_jet_pt, raw_pt_correction.flatten(), raw_jet_eta, raw_jet_phi, num_elements=2)

    # Extract the grouped arrays
    tau_plus, tau_minus, jet_pt, jet_pt_correction, jet_eta, jet_phi = grouped_arrays

    #calculate delta_r
    eta1, eta2 = jet_eta[:, 0], jet_eta[:, 1]
    phi1, phi2 = jet_phi[:, 0], jet_phi[:, 1]
    delta_r_values = delta_r(eta1, phi1, eta2, phi2)

    # Additional cuts recommended here:
    # https://indico.cern.ch/event/1380964/contributions/5852368/attachments/2841655/4973190/AnnualReview_2024.pdf
    # Slide 7
    cuts = (np.abs(eta1) < 2.172) & (np.abs(eta2) < 2.172) & (delta_r_values > 0.5)

    tau_topo_score = calculate_topo_score(tau_plus, tau_minus)

    #correct for pt
    pt1_uncorrected, pt2_uncorrected = np.asarray(jet_pt[:,0][cuts]), np.asarray(jet_pt[:,1][cuts])
    ratio1, ratio2 = np.asarray(jet_pt_correction[:,0][cuts]), np.asarray(jet_pt_correction[:,1][cuts])

    pt1 = pt1_uncorrected*ratio1
    pt2 = pt2_uncorrected*ratio2

    pt = np.vstack([pt1, pt2]).transpose()
    pt_min = np.min(pt, axis=1)

    return event_id[cuts], pt_min, tau_topo_score[cuts]


def plot_bkg_rate_tau(model, minbias_path, correct_pt=True, input_tag='ext7', tree='jetntuple/Jets', n_entries=600000):
    """
    Plot the background (mimbias) rate w.r.t pT cuts.
    """

    minbias_rate = 32e+3 #32 kHZ

    #Load the minbias data
    minbias = uproot4.open(minbias_path)[tree]

    raw_event_id = helpers.extract_array(minbias, 'event', n_entries)
    raw_jet_pt = helpers.extract_array(minbias, 'jet_pt', n_entries)
    raw_jet_eta = helpers.extract_array(minbias, 'jet_eta_phys', n_entries)
    raw_jet_phi = helpers.extract_array(minbias, 'jet_phi_phys', n_entries)
    raw_cmssw_tau = helpers.extract_array(minbias, 'jet_tauscore', n_entries)
    raw_cmssw_taupt = helpers.extract_array(minbias, 'jet_taupt', n_entries)

    #NN related 
    raw_inputs = np.asarray(helpers.extract_nn_inputs(minbias, input_fields_tag=input_tag, nconstit=16, n_entries=n_entries)).transpose(0, 2, 1)
    raw_pred_score, raw_pt_correction = model.predict(raw_inputs)

    tau_index = [2,3] #2=tau positive, 3=tau_negative
    raw_tau_score_sum = raw_pred_score[:,tau_index[0]] + raw_pred_score[:,tau_index[1]]
    raw_tau_plus = raw_pred_score[:,tau_index[0]]
    raw_tau_minus = raw_pred_score[:,tau_index[1]]

    #Count number of total event
    n_events = len(np.unique(raw_event_id))
    print("Total number of minbias events: ", n_events)

    #Extract the minpt and tau score from cmssw
    cmssw_event_id, pt_min, cmssw_pt_min, cmssw_tau_min = cmssw_pt_score(raw_event_id, raw_jet_pt, raw_jet_eta, raw_jet_phi, raw_cmssw_tau, raw_cmssw_taupt)
    model_event_id, model_pt_min, model_tau_topo = model_pt_score(raw_event_id, raw_tau_score_sum, raw_tau_plus, raw_tau_minus, raw_jet_pt, raw_pt_correction, raw_jet_eta, raw_jet_phi)

    event_id_cmssw = cmssw_event_id[cmssw_tau_min > WPs_CMSSW['tau']]
    event_id_model = model_event_id[model_tau_topo > WPs['tau_topo']]

    #Total number of unique event
    n_event = len(np.unique(raw_event_id))
    minbias_rate_no_nn = []
    minbias_rate_cmssw = []
    minbias_rate_model = []

    # Initialize lists for uncertainties (Poisson)
    uncertainty_no_nn = []
    uncertainty_cmssw = []
    uncertainty_model = []

    pt_cuts =  list(np.arange(0,100,2))
    for pt_cut in pt_cuts:

        print("pT Cut: ", pt_cut)
        n_pass_no_nn = len(np.unique(ak.flatten(cmssw_event_id[pt_min > pt_cut])))
        n_pass_cmssw = len(np.unique(ak.flatten(event_id_cmssw[cmssw_pt_min[cmssw_tau_min > WPs_CMSSW['tau']] > pt_cut])))
        n_pass_model = len(np.unique(ak.flatten(event_id_model[model_pt_min[model_tau_topo > WPs['tau_topo']] > pt_cut])))
        print('------------')

        minbias_rate_no_nn.append((n_pass_no_nn/n_event)*minbias_rate)
        minbias_rate_cmssw.append((n_pass_cmssw/n_event)*minbias_rate)
        minbias_rate_model.append((n_pass_model/n_event)*minbias_rate)

        # Poisson uncertainty is sqrt(N) where N is the number of events passing the cut
        uncertainty_no_nn.append(np.sqrt(n_pass_no_nn) / n_event * minbias_rate)
        uncertainty_cmssw.append(np.sqrt(n_pass_cmssw) / n_event * minbias_rate)
        uncertainty_model.append(np.sqrt(n_pass_model) / n_event * minbias_rate)

    plt.plot(pt_cuts, minbias_rate_no_nn, label=r'No ID/$p_T$ correction', linewidth = 5)
    plt.plot(pt_cuts, minbias_rate_cmssw, label=r'CMSSW PuppiTau Emulator', linewidth = 5)
    plt.plot(pt_cuts, minbias_rate_model, label=r'SeedCone Tau Topology', linewidth = 5)
    
    # Add uncertainty bands
    plt.fill_between(pt_cuts, np.array(minbias_rate_no_nn) - np.array(uncertainty_no_nn),
                     np.array(minbias_rate_no_nn) + np.array(uncertainty_no_nn), alpha=0.3)
    plt.fill_between(pt_cuts, np.array(minbias_rate_cmssw) - np.array(uncertainty_cmssw),
                     np.array(minbias_rate_cmssw) + np.array(uncertainty_cmssw), alpha=0.3)
    plt.fill_between(pt_cuts, np.array(minbias_rate_model) - np.array(uncertainty_model),
                     np.array(minbias_rate_model) + np.array(uncertainty_model), alpha=0.3)
    
    hep.cms.text("Phase 2 Simulation")
    hep.cms.lumitext("PU 200 (14 TeV)")
    plt.yscale('log')
    plt.ylabel(r"VBF H$\to \tau_h \tau_h$ trigger rate [kHz]")
    plt.xlabel(r"Min($p^1_T$,$p^2_T$) [GeV]")
    plt.legend(loc = 'upper right',fontsize = 15)

    figname='bkg_rate_vbfhtautau_ptCorrected' if correct_pt else 'bkg_rate_vbfhtautau_ptUncorrected'
    plt.savefig(f'plots/{figname}.pdf', bbox_inches='tight')


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='/eos/home-s/sewuchte/www/L1T/trainings_regression_weighted/2024_08_27_v4_extendedAll200_btgc_ext7_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned/model_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned.h5' , help = 'Input model for plotting')    
    parser.add_argument('--uncorrect_pt', action='store_true', help='Enable pt correction in plot_bkg_rate_tau')
    parser.add_argument('--eta_cut', action='store_true', help='Enable eta cut (|eta| < 2.5)')


    args = parser.parse_args()

    model=load_qmodel(args.model)
    print(model.summary())

    #These paths are default to evaluate some of the rate
    minbias_path = '/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_ntuples_v131Xv9/extendedTRK_HW_260824/MinBias_PU200.root'

    #Parse the options a bit here
    correct_pt = not args.uncorrect_pt

    plot_bkg_rate_tau(model, minbias_path, correct_pt=correct_pt)
