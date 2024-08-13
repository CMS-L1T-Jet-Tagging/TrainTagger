import sys
import uproot4
import numpy as np
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

def eta_region_selection(eta_array, eta_region):
    """
    eta range for barrel: |eta| < 1.5
    eta range for endcap: 1.5 < |eta| < 2.5

    return eta array selection
    """

    if eta_region == 'barrel': return np.abs(eta_array) < 1.5
    elif eta_region == 'endcap': return (np.abs(eta_array) > 1.5) & (np.abs(eta_array) < 2.5)
    else: return True #Select everything

def plot_ratio(all_tau, selected_tau, num_label = r"Selected CMSSW Emulator Taus", figname='plots/cmssw_eff.pdf'):

    fig = plt.figure(figsize=(10, 12))
    _, eff = selected_tau.plot_ratio(all_tau,
                                              rp_num_label=num_label, rp_denom_label=r"All Taus",
                                              rp_uncert_draw_type="bar", rp_uncertainty_type="efficiency")
    plt.savefig(figname, bbox_inches='tight')
    plt.show(block=False)

    return eff

def get_model_prediction(signal, model, l1_pt_raw, n_entries,  tau_index = [2,3]):

    nn_inputs = np.asarray(helpers.extract_nn_inputs(signal, input_fields_tag='ext3', nconstit=16, n_entries=n_entries)).transpose(0, 2, 1)
    pred_score, ratio = model.predict(nn_inputs)

    tau_score = pred_score[:,tau_index[0]] + pred_score[:,tau_index[1]]
    tau_pt = l1_pt_raw*ratio.flatten()

    return tau_score, tau_pt

def get_bar_patch_data(artists):
    x_data = [artists.bar.patches[i].get_x() for i in range(len(artists.bar.patches))]
    y_data = [artists.bar.patches[i].get_y() for i in range(len(artists.bar.patches))]
    err_data = [artists.bar.patches[i].get_height() for i in range(len(artists.bar.patches))]
    return x_data, y_data, err_data

def eff_pt_tau(model, signal_path, eta_region='barrel', tree='jetntuple/Jets', n_entries=10000):
    """
    Plot the single tau efficiency for signal in signal_path w.r.t pt
    eta range for barrel: |eta| < 1.5
    eta range for endcap: 1.5 < |eta| < 2.5
    """
    pT_egdes = [0,10,15,20,25,30,35,40,45,50,55,60,70,80,100,125,150,175,200]

    signal = uproot4.open(signal_path)[tree]

    #Select out the taus
    tau_flav = helpers.extract_array(signal, 'jet_tauflav', n_entries)
    gen_pt_raw = helpers.extract_array(signal, 'jet_genmatch_pt', n_entries)
    gen_eta_raw = helpers.extract_array(signal, 'jet_genmatch_eta', n_entries)
    gen_dr_raw = helpers.extract_array(signal, 'jet_taumatch_dR', n_entries)

    l1_pt_raw = helpers.extract_array(signal, 'jet_pt', n_entries)
    jet_taupt_raw= helpers.extract_array(signal, 'jet_taupt', n_entries)
    jet_tauscore_raw = helpers.extract_array(signal, 'jet_tauscore', n_entries)

    #Get nn predictions
    nn_tauscore_raw, nn_taupt_raw = get_model_prediction(signal, model, l1_pt_raw, n_entries)

    #selecting the eta region
    gen_eta_selection = eta_region_selection(gen_eta_raw, eta_region)

    #Denominator & numerator selection for efficiency
    tau_deno = (tau_flav==1) & (gen_pt_raw > 1.) & gen_eta_selection
    tau_nume_seedcone = tau_deno & (np.abs(gen_dr_raw) < 0.4) & (l1_pt_raw > WPs['tau_l1_pt'])
    tau_nume_nn = tau_deno & (np.abs(gen_dr_raw) < 0.4) & (nn_taupt_raw > WPs['tau_l1_pt']) & (nn_tauscore_raw > WPs['tau'])
    tau_nume_cmssw = tau_deno & (np.abs(gen_dr_raw) < 0.4) & (jet_taupt_raw > WPs_CMSSW['tau_l1_pt']) & (jet_tauscore_raw > WPs_CMSSW['tau'])
    
    #Get the needed attributes
    #Basically we want to bin the selected truth pt and divide it by the overall count
    gen_pt = gen_pt_raw[tau_deno]
    seedcone_pt = gen_pt_raw[tau_nume_seedcone]
    cmssw_pt = gen_pt_raw[tau_nume_cmssw]
    nn_pt = gen_pt_raw[tau_nume_nn]
    
    #Constructing the histograms
    pT_axis = hist.axis.Variable(pT_egdes, name = r"$ \tau_h$ $p_T^{gen}$")

    all_tau = Hist(pT_axis)
    seedcone_tau = Hist(pT_axis)
    cmssw_tau = Hist(pT_axis)
    nn_tau = Hist(pT_axis)

    #Fill the histogram using the values above
    all_tau.fill(gen_pt)
    seedcone_tau.fill(seedcone_pt)
    cmssw_tau.fill(cmssw_pt)
    nn_tau.fill(nn_pt)

    #Plot and get the artist objects
    eff_seedcone = plot_ratio(all_tau, seedcone_tau, num_label="Selected SeededCone Taus", figname='plots/seedcone_eff.pdf')
    eff_cmssw = plot_ratio(all_tau, cmssw_tau, num_label="Selected CMSSW Taus", figname='plots/cmssw_eff.pdf')
    eff_nn = plot_ratio(all_tau, nn_tau, num_label="Selected NN Taus", figname='plots/nn_eff.pdf')

    #Extract data from the artists
    sc_x, sc_y, sc_err = get_bar_patch_data(eff_seedcone)
    cmssw_x, cmssw_y, cmssw_err = get_bar_patch_data(eff_cmssw)
    nn_x, nn_y, nn_err = get_bar_patch_data(eff_nn)
    
    #Plot the efficiencies together
    fig = plt.figure()
    plt.errorbar(sc_x, sc_y, yerr=sc_err, fmt='o', linewidth=2, label=r'SeededCone PuppiJet (L1 $p_T$ > {})'.format(WPs['tau_l1_pt']))
    plt.errorbar(cmssw_x, cmssw_y, yerr=cmssw_err, fmt='o', linewidth=2, label=r'Tau CMSSW Emulator (L1 $p_T$ > {}, NN > {})'.format(WPs_CMSSW['tau_l1_pt'], WPs_CMSSW['tau']))
    plt.errorbar(nn_x, nn_y, yerr=nn_err, fmt='o', linewidth=2, label=r'SeededCone Tau Tagger (L1 $p_T$ > {}, NN > {})'.format(WPs['tau_l1_pt'], WPs['tau']))
    
    #Plot other labels
    plt.hlines(1, 0, 150, linestyles='dashed', color='black', linewidth=3)
    plt.ylim([0., 1.1])
    plt.xlim([0, 150])
    hep.cms.text("Phase 2 Simulation")
    hep.cms.lumitext("PU 200 (14 TeV)")
    plt.xlabel(r"$\tau_h$ $p_T^{gen}$ [GeV]")
    plt.ylabel(r"$\epsilon$(Di-$\tau_h$ trigger rate at 28 kHz)")
    plt.legend(loc='lower right', fontsize=15)
    plt.savefig('plots/tau_eff_all.pdf')
    plt.show(block=False)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='/eos/user/s/sewuchte/L1Trigger/ForDuc/trainings_regression_weighted/2024_07_25_v10_extendedAll200_btgc_ext3_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned/model_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned.h5', help = 'Input model for plotting')    
    args = parser.parse_args()

    #Load the model defined above
    model=load_qmodel(args.model)

    #These paths are default to evaluate some of the efficiency
    tau_eff_filepath = '/eos/user/s/sewuchte/L1Trigger/ForDuc/nTuples/VBFHtt_PU200.root'

    #Barrel
    eff_pt_tau(model, tau_eff_filepath, n_entries=100000)
