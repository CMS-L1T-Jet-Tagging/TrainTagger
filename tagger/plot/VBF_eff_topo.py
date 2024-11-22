"""
Plot tau efficiencies, usage:

python plot_tau_eff.py <see more arguments below>
"""
import sys
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

#Import the calculated working points
from official_WPs import WPs, WPs_CMSSW

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
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


def delta_r(eta1, phi1, eta2, phi2):
    """
    Calculate the delta R between two sets of eta and phi values.
    """
    delta_eta = eta1 - eta2
    delta_phi = phi1 - phi2

    # Ensure delta_phi is within -pi to pi
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(delta_eta**2 + delta_phi**2)

def ratio_2D(nume, deno):
    ratio = np.divide(nume, deno, where=deno != 0)
    ratio[deno == 0] = np.nan
    return ratio

def plot_2D_ratio(ratio, pt_edges, figname="VBF_eff_CMSSW"):
    extent = [pt_edges[0], pt_edges[-1], pt_edges[0], pt_edges[-1]]
    plt.imshow(ratio.T, origin='lower', extent=extent, vmin=0, vmax=0.5, aspect='auto')
    plt.colorbar()

    hep.cms.text("")
    hep.cms.lumitext("PU 200 (14 TeV)")

    plt.xlabel(r"Gen. $p_T^1$ [GeV]")
    plt.ylabel(r"Gen. $p_T^2$ [GeV]")
    plt.savefig(f'plots/{figname}.pdf', bbox_inches='tight')
    plt.show(block=False)

def VBF_eff_CMSSW(model, tau_eff_filepath,  tree='jetntuple/Jets',  correct_pt=True, input_tag='ext7', n_entries=100000):

    #Load the signal data
    signal = uproot4.open(tau_eff_filepath)[tree]

    raw_event_id = helpers.extract_array(signal, 'event', n_entries)
    raw_jet_pt = helpers.extract_array(signal, 'jet_pt', n_entries)
    raw_jet_genpt = helpers.extract_array(signal, 'jet_genmatch_pt', n_entries)
    raw_jet_eta = helpers.extract_array(signal, 'jet_eta_phys', n_entries)
    raw_jet_phi = helpers.extract_array(signal, 'jet_phi_phys', n_entries)
    raw_inputs = helpers.extract_nn_inputs(signal, input_fields_tag=input_tag, nconstit=16, n_entries=n_entries)
    raw_cmssw_tau = helpers.extract_array(signal, 'jet_tauscore', n_entries)
    raw_cmssw_taupt = helpers.extract_array(signal, 'jet_taupt', n_entries)


    #Count number of total event
    n_events = len(np.unique(raw_event_id))
    print("Total number of signal events: ", n_events)

    #Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays  = helpers.group_id_values(raw_event_id, raw_jet_pt, raw_jet_genpt, raw_jet_eta, raw_jet_phi, raw_inputs, raw_cmssw_tau, raw_cmssw_taupt, num_elements=2)

    # Extract the grouped arrays
    # Jet pt is already sorted in the producer, no need to do it here
    jet_pt, jet_genpt, jet_eta, jet_phi, jet_nn_inputs, cmssw_tau, cmssw_taupt = grouped_arrays

    #calculate delta_r
    eta1, eta2 = jet_eta[:, 0], jet_eta[:, 1]
    phi1, phi2 = jet_phi[:, 0], jet_phi[:, 1]
    delta_r_values = delta_r(eta1, phi1, eta2, phi2)

    # Additional cuts recommended here:
    # https://indico.cern.ch/event/1380964/contributions/5852368/attachments/2841655/4973190/AnnualReview_2024.pdf
    # Slide 7
    cuts = (np.abs(eta1) < 2.172) & (np.abs(eta2) < 2.172) & (delta_r_values > 0.5)

    #Get genpt
    genpt1, genpt2 = np.asarray(jet_genpt[:,0]), np.asarray(jet_genpt[:,1])

    #Get cmssw attribubtes to calculate the rate
    cmssw_pt1, cmssw_pt2 = np.asarray(cmssw_taupt[:,0][cuts]), np.asarray(cmssw_taupt[:,1][cuts])
    cmssw_pt = np.vstack([cmssw_pt1, cmssw_pt2]).transpose()
    cmssw_pt_min = np.min(cmssw_pt, axis=1)

    #Do similar thing for the tau score
    cmssw_tau1, cmssw_tau2 = np.asarray(cmssw_tau[:,0][cuts]), np.asarray(cmssw_tau[:,1][cuts])
    cmssw_tau = np.vstack([cmssw_tau1, cmssw_tau2]).transpose()
    cmssw_tau_min =  np.min(cmssw_tau, axis=1)

    #Get inputs and pts for processing model
    pt1_uncorrected, pt2_uncorrected = np.asarray(jet_pt[:,0][cuts]), np.asarray(jet_pt[:,1][cuts])
    input1, input2 = np.asarray(jet_nn_inputs[:, 0][cuts]).transpose(0, 2, 1), np.asarray(jet_nn_inputs[:, 1][cuts]).transpose(0, 2, 1) #Flip the last two axes

    #Get the NN predictions
    pred_score1, ratio1 = model.predict(input1)
    pred_score2, ratio2 = model.predict(input2)

    if correct_pt:
        pt1_model = pt1_uncorrected*(ratio1.flatten())
        pt2_model = pt2_uncorrected*(ratio2.flatten())
    else:
        pt1_model = pt1_uncorrected
        pt2_model = pt2_uncorrected


    pt_model = np.vstack([pt1_model, pt2_model]).transpose()
    pt_min_model = np.min(pt_model, axis=1)

    pt=np.vstack([pt1_uncorrected, pt2_uncorrected]).transpose()
    pt_min=np.min(pt, axis=1)

    tau_topo_score = calculate_topo_score(pred_score1, pred_score2)

    #Create histograms to contain the gen pts
    pt_edges = list(np.arange(0,200,10)) #Make sure to capture everything

    all_genpt = Hist(hist.axis.Variable(pt_edges, name="genpt1", label="genpt1"),
                    hist.axis.Variable(pt_edges, name="genpt2", label="genpt2"))
    cmssw_pt = Hist(hist.axis.Variable(pt_edges, name="genpt1", label="genpt1"),
                    hist.axis.Variable(pt_edges, name="genpt2", label="genpt2"))
    model_pt = Hist(hist.axis.Variable(pt_edges, name="genpt1", label="genpt1"),
                    hist.axis.Variable(pt_edges, name="genpt2", label="genpt2"))
    
    all_genpt.fill(genpt1=genpt1, genpt2=genpt2)

    cmssw_selection = (cmssw_tau_min > WPs_CMSSW['tau']) & (cmssw_pt_min > WPs_CMSSW['tau_l1_pt'])
    cmssw_pt.fill(genpt1=genpt1[cuts][cmssw_selection], genpt2=genpt2[cuts][cmssw_selection])

    model_selection = (tau_topo_score > WPs['tau_topo']) & (pt_min_model > WPs['tau_pt_topo'])
    model_pt.fill(genpt1=genpt1[cuts][model_selection], genpt2=genpt2[cuts][model_selection])

    cmssw_ratio = ratio_2D(cmssw_pt, all_genpt)
    model_ratio = ratio_2D(model_pt, all_genpt)

    # plot_2D_ratio(cmssw_ratio, pt_edges, figname="VBF_topo_eff_CMSSW")
    plot_2D_ratio(model_ratio, pt_edges, figname="VBF_topo_eff_Model")

def calculate_topo_score(tau_plus, tau_minus):
    #2=tau positive, 3=tau_negative

    p_pos = tau_plus[:,0] + tau_plus[:,1]
    p_neg = tau_minus[:,0] + tau_minus[:,1]

    return np.multiply(p_pos, p_neg)

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

def VBF_eff_model(model, tau_eff_filepath,  tree='jetntuple/Jets',  correct_pt=True, input_tag='ext7', n_entries=1000):

    #Load the signal data
    signal = uproot4.open(tau_eff_filepath)[tree]

    raw_event_id = helpers.extract_array(signal, 'event', n_entries)
    raw_jet_pt = helpers.extract_array(signal, 'jet_pt', n_entries)
    raw_jet_genpt = helpers.extract_array(signal, 'jet_genmatch_pt', n_entries)
    raw_jet_eta = helpers.extract_array(signal, 'jet_eta_phys', n_entries)
    raw_jet_phi = helpers.extract_array(signal, 'jet_phi_phys', n_entries)

    #NN related 
    raw_inputs = np.asarray(helpers.extract_nn_inputs(signal, input_fields_tag=input_tag, nconstit=16, n_entries=n_entries)).transpose(0, 2, 1)
    raw_pred_score, raw_pt_correction = model.predict(raw_inputs)

    tau_index = [2,3] #2=tau positive, 3=tau_negative
    raw_tau_score_sum = raw_pred_score[:,tau_index[0]] + raw_pred_score[:,tau_index[1]]
    raw_tau_plus = raw_pred_score[:,tau_index[0]]
    raw_tau_minus = raw_pred_score[:,tau_index[1]]

    #Count number of total event
    n_events = len(np.unique(raw_event_id))
    print("Total number of signal events: ", n_events)

    #Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays  = group_id_values_nn(raw_event_id, raw_tau_score_sum, raw_tau_plus, raw_tau_minus, raw_jet_pt, raw_jet_genpt, raw_pt_correction.flatten(), raw_jet_eta, raw_jet_phi, num_elements=2)

    # Extract the grouped arrays
    tau_plus, tau_minus, jet_pt, jet_genpt, jet_pt_correction, jet_eta, jet_phi = grouped_arrays
    genpt1, genpt2 = np.asarray(jet_genpt[:,0]), np.asarray(jet_genpt[:,1])


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
    pt_min_model = np.min(pt, axis=1)

    #Create histograms to contain the gen pts
    pt_edges = list(np.arange(0,200,10)) #Make sure to capture everything

    all_genpt = Hist(hist.axis.Variable(pt_edges, name="genpt1", label="genpt1"),
                    hist.axis.Variable(pt_edges, name="genpt2", label="genpt2"))
    model_pt = Hist(hist.axis.Variable(pt_edges, name="genpt1", label="genpt1"),
                    hist.axis.Variable(pt_edges, name="genpt2", label="genpt2"))
    
    all_genpt.fill(genpt1=genpt1, genpt2=genpt2)


    model_selection = (pt_min_model > WPs['tau_pt_topo']) & (tau_topo_score[cuts] > WPs['tau_topo']) 
    model_pt.fill(genpt1=genpt1[cuts][model_selection], genpt2=genpt2[cuts][model_selection])

    model_ratio = ratio_2D(model_pt, all_genpt)

    # plot_2D_ratio(cmssw_ratio, pt_edges, figname="VBF_topo_eff_CMSSW")
    plot_2D_ratio(model_ratio, pt_edges, figname="VBF_topo_eff_Model")
    

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='/eos/home-s/sewuchte/www/L1T/trainings_regression_weighted/2024_08_27_v4_extendedAll200_btgc_ext7_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned/model_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned.h5', help = 'Input model for plotting')    
    args = parser.parse_args()

    #Load the model defined above
    model=load_qmodel(args.model)

    #These paths are default to evaluate some of the efficiency
    tau_eff_filepath = '/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_ntuples_v131Xv9/extendedTRK_HW_260824/VBFHtt_PU200.root'

    #Barrel
    VBF_eff_model(model, tau_eff_filepath, n_entries=500000)
