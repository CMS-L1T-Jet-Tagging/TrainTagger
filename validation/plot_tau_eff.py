"""
Plot tau efficiencies, usage:

python plot_tau_eff.py <see more arguments below>
"""
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
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

def eta_region_selection(eta_array, eta_region):
    """
    eta range for barrel: |eta| < 1.5
    eta range for endcap: 1.5 < |eta| < 2.5

    return eta array selection
    """

    if eta_region == 'barrel': return np.abs(eta_array) < 1.5
    elif eta_region == 'endcap': return (np.abs(eta_array) > 1.5) & (np.abs(eta_array) < 2.5)
    else: return np.abs(eta_array) > 0.0 #Select everything

def plot_ratio(all_tau, selected_tau, num_label = r"Selected CMSSW Emulator Taus", figname='plots/cmssw_eff.pdf'):

    fig = plt.figure(figsize=(10, 12))
    _, eff = selected_tau.plot_ratio(all_tau,
                                              rp_num_label=num_label, rp_denom_label=r"All Taus",
                                              rp_uncert_draw_type="bar", rp_uncertainty_type="efficiency")
    # plt.savefig(figname, bbox_inches='tight')
    plt.show(block=False)

    return eff

def get_model_prediction(signal, model, l1_pt_raw, n_entries, uncorrect_pt=False, tau_index = [2,3]):

    nn_inputs = np.asarray(helpers.extract_nn_inputs(signal, input_fields_tag='ext7', nconstit=16, n_entries=n_entries)).transpose(0, 2, 1)
    pred_score, ratio = model.predict(nn_inputs)

    tau_score = pred_score[:,tau_index[0]] + pred_score[:,tau_index[1]]

    if uncorrect_pt: tau_pt = l1_pt_raw
    else: tau_pt = l1_pt_raw*ratio.flatten()
    

    return tau_score, tau_pt

def get_bar_patch_data(artists):
    x_data = [artists.bar.patches[i].get_x() for i in range(len(artists.bar.patches))]
    y_data = [artists.bar.patches[i].get_y() for i in range(len(artists.bar.patches))]
    err_data = [artists.bar.patches[i].get_height() for i in range(len(artists.bar.patches))]
    return x_data, y_data, err_data

def eff_pt_tau(model, signal_path, uncorrect_pt=False, eta_region='barrel', tree='jetntuple/Jets', n_entries=10000):
    """
    Plot the single tau efficiency for signal in signal_path w.r.t pt
    eta range for barrel: |eta| < 1.5
    eta range for endcap: 1.5 < |eta| < 2.5
    """

    model_pt_WP =  WPs['tau_l1_pt_ptUncorrected'] if uncorrect_pt else WPs['tau_l1_pt_ptCorrected']
    model_NN_WP =  WPs['tau_ptUncorrected'] if uncorrect_pt else WPs['tau_ptCorrected']

    pT_egdes = [0,10,15,20,25,30,35,40,45,50,55,60,70,80,100,125,150,175,200]

    signal = uproot4.open(signal_path)[tree]

    #Select out the taus
    tau_flav = helpers.extract_array(signal, 'jet_tauflav', n_entries)
    gen_pt_raw = helpers.extract_array(signal, 'jet_genmatch_pt', n_entries)
    gen_eta_raw = helpers.extract_array(signal, 'jet_genmatch_eta', n_entries)
    gen_dr_raw = helpers.extract_array(signal, 'jet_genmatch_dR', n_entries)

    l1_pt_raw = helpers.extract_array(signal, 'jet_pt', n_entries)
    jet_taupt_raw= helpers.extract_array(signal, 'jet_taupt', n_entries)
    jet_tauscore_raw = helpers.extract_array(signal, 'jet_tauscore', n_entries)

    #Get nn predictions
    nn_tauscore_raw, nn_taupt_raw = get_model_prediction(signal, model, l1_pt_raw, n_entries, uncorrect_pt=uncorrect_pt)

    #selecting the eta region
    gen_eta_selection = eta_region_selection(gen_eta_raw, eta_region)

    #Denominator & numerator selection for efficiency
    tau_deno = (tau_flav==1) & (gen_pt_raw > 1.) & gen_eta_selection
    tau_nume_seedcone = tau_deno & (np.abs(gen_dr_raw) < 0.4) & (l1_pt_raw > model_pt_WP)
    tau_nume_nn = tau_deno & (np.abs(gen_dr_raw) < 0.4) & (nn_taupt_raw > model_pt_WP) & (nn_tauscore_raw > model_NN_WP)
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
    eff_seedcone = plot_ratio(all_tau, seedcone_tau, num_label="Selected SeededCone Taus", figname=f'plots/seedcone_eff_{eta_region}.pdf')
    eff_cmssw = plot_ratio(all_tau, cmssw_tau, num_label="Selected CMSSW Taus", figname=f'plots/cmssw_eff_{eta_region}.pdf')
    eff_nn = plot_ratio(all_tau, nn_tau, num_label="Selected NN Taus", figname=f'plots/nn_eff_{eta_region}.pdf')

    #Extract data from the artists
    sc_x, sc_y, sc_err = get_bar_patch_data(eff_seedcone)
    cmssw_x, cmssw_y, cmssw_err = get_bar_patch_data(eff_cmssw)
    nn_x, nn_y, nn_err = get_bar_patch_data(eff_nn)
    
    #Plot the efficiencies together
    fig = plt.figure()
    eta_label = r'Barrel ($|\eta| < 1.5$)' if eta_region == 'barrel' else r'EndCap (1.5 < $|\eta|$ < 2.5)'
    plt.plot([], [], 'none', label=eta_label)
    plt.errorbar(sc_x, sc_y, yerr=sc_err, fmt='o', c=color_cycle[0], linewidth=2, label=r'SeededCone PuppiJet (L1 $p_T$ > {})'.format(model_pt_WP))
    plt.errorbar(cmssw_x, cmssw_y, yerr=cmssw_err, c=color_cycle[1], fmt='o', linewidth=2, label=r'Tau CMSSW Emulator (L1 $p_T$ > {}, NN > {})'.format(WPs_CMSSW['tau_l1_pt'], WPs_CMSSW['tau']))
    plt.errorbar(nn_x, nn_y, yerr=nn_err, fmt='o', c=color_cycle[2], linewidth=2, label=r'SeededCone Tau Tagger (L1 $p_T$ > {}, NN > {})'.format(model_pt_WP, model_NN_WP))
    
    #Plot other labels
    plt.hlines(1, 0, 150, linestyles='dashed', color='black', linewidth=3)
    plt.ylim([0., 1.1])
    plt.xlim([0, 150])
    hep.cms.text("Phase 2 Simulation")
    hep.cms.lumitext("PU 200 (14 TeV)")
    plt.xlabel(r"$\tau_h$ $p_T^{gen}$ [GeV]")
    plt.ylabel(r"$\epsilon$(Di-$\tau_h$ trigger rate at 28 kHz)")
    plt.legend(loc='lower right', fontsize=15)
    figname = f'tau_eff_all_{eta_region}_ptUncorrected' if uncorrect_pt else f'tau_eff_all_{eta_region}_ptCorrected'
    plt.savefig(f'plots/{figname}.pdf')
    plt.show(block=False)

def eff_sc_and_tau(model, signal_path, eta_region='barrel', tree='jetntuple/Jets', n_entries=10000):
    """
    Plot the raw seedcone and tau efficiencies without any pt or nn cuts
    """

    pT_egdes = [0,10,15,20,25,30,35,40,45,50,55,60,70,80,100,125,150,175,200]

    signal = uproot4.open(signal_path)[tree]

    #Select out the taus
    tau_flav = helpers.extract_array(signal, 'jet_tauflav', n_entries)
    gen_pt_raw = helpers.extract_array(signal, 'jet_genmatch_pt', n_entries)
    gen_eta_raw = helpers.extract_array(signal, 'jet_genmatch_eta', n_entries)
    gen_dr_raw = helpers.extract_array(signal, 'jet_genmatch_dR', n_entries)
    tau_dr_raw = helpers.extract_array(signal, 'jet_taumatch_dR', n_entries)


    l1_pt_raw = helpers.extract_array(signal, 'jet_pt', n_entries)
    jet_taupt_raw= helpers.extract_array(signal, 'jet_taupt', n_entries)
    jet_tauscore_raw = helpers.extract_array(signal, 'jet_tauscore', n_entries)

    #Denominator & numerator selection for efficiency
    tau_deno = (tau_flav==1) & (gen_pt_raw > 1.)
    tau_nume_seedcone = tau_deno & (np.abs(gen_dr_raw) < 0.4) & (l1_pt_raw > 1.)
    tau_nume_cmssw = tau_deno & (np.abs(gen_dr_raw) < 0.4) & (np.abs(tau_dr_raw) < 0.4) & (jet_taupt_raw > 1.)
    
    #Get the needed attributes
    #Basically we want to bin the selected truth pt and divide it by the overall count
    gen_pt = gen_pt_raw[tau_deno]
    seedcone_pt = gen_pt_raw[tau_nume_seedcone]
    cmssw_pt = gen_pt_raw[tau_nume_cmssw]
    
    #Constructing the histograms
    pT_axis = hist.axis.Variable(pT_egdes, name = r"$ \tau_h$ $p_T^{gen}$")

    all_tau = Hist(pT_axis)
    seedcone_tau = Hist(pT_axis)
    cmssw_tau = Hist(pT_axis)

    #Fill the histogram using the values above
    all_tau.fill(gen_pt)
    seedcone_tau.fill(seedcone_pt)
    cmssw_tau.fill(cmssw_pt)

    #Plot and get the artist objects
    eff_seedcone = plot_ratio(all_tau, seedcone_tau, num_label="Selected SeededCone Taus", figname=f'plots/seedcone_eff_{eta_region}.pdf')
    eff_cmssw = plot_ratio(all_tau, cmssw_tau, num_label="Selected CMSSW Taus", figname=f'plots/cmssw_eff_{eta_region}.pdf')

    #Extract data from the artists
    sc_x, sc_y, sc_err = get_bar_patch_data(eff_seedcone)
    cmssw_x, cmssw_y, cmssw_err = get_bar_patch_data(eff_cmssw)
    
    #Plot the efficiencies together
    fig = plt.figure()
    eta_label = r'Barrel ($|\eta| < 1.5$)' if eta_region == 'barrel' else r'EndCap (1.5 < $|\eta|$ < 2.5)'
    if eta_region != 'none':
        plt.plot([], [], 'none', label=eta_label)

    plt.errorbar(sc_x, sc_y, yerr=sc_err, fmt='o', c=color_cycle[0], linewidth=2, label=r'SeededCone CMSSW Emulator')
    plt.errorbar(cmssw_x, cmssw_y, yerr=cmssw_err, c=color_cycle[1], fmt='o', linewidth=2, label=r'Tau CMSSW Emulator')
    
    #Plot other labels
    plt.hlines(1, 0, 150, linestyles='dashed', color='black', linewidth=3)
    plt.ylim([0., 1.1])
    plt.xlim([0, 150])
    hep.cms.text("Phase 2 Simulation")
    hep.cms.lumitext("PU 200 (14 TeV)")
    plt.xlabel(r"$\tau_h$ $p_T^{gen}$ [GeV]")
    plt.ylabel(r"$\epsilon$ (VBF H $\rightarrow$ $\tau\tau$)")
    plt.legend(loc='lower right', fontsize=15)
    figname = f'sc_and_tau_eff_{eta_region}'
    plt.savefig(f'plots/{figname}.pdf')
    plt.show(block=False)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='/eos/user/s/sewuchte/L1Trigger/ForDuc/trainings_regression_weighted/2024_08_17_v6_extendedAll200_btgc_ext7_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned/model_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned.h5', help = 'Input model for plotting')    
    parser.add_argument('--uncorrect_pt', action='store_true', help='Enable pt correction in plot_bkg_rate_tau')
    parser.add_argument('--eta_region', choices=['barrel', 'endcap','none'], default='none', help='Select the eta region: "barrel", "endcap" or "none"')

    args = parser.parse_args()

    #Load the model defined above
    model=load_qmodel(args.model)

    #These paths are default to evaluate some of the efficiency
    tau_eff_filepath = '/eos/user/s/sewuchte/L1Trigger/ForDuc/nTuples/VBFHtt_PU200.root'

    #Barrel
    eff_pt_tau(model, tau_eff_filepath, eta_region=args.eta_region, n_entries=100000)
