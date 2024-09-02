"""
Plot HH efficiencies, usage:

python HH_eff.py <see more arguments below>
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

#Other custom plotting tools
from plot_utils import plot_ratio, get_bar_patch_data

def nn_bscore_sum(model, jet_nn_inputs, n_jets=4, b_index = 1):

    print(jet_nn_inputs)
    #Get the inputs for the first n_jets
    btag_inputs = [np.asarray(jet_nn_inputs[:, i]).transpose(0, 2, 1) for i in range(0, n_jets)]

    #Get the nn outputs
    nn_outputs = [model.predict(nn_input) for nn_input in btag_inputs]

    #Sum them together
    bscore_sum = sum([pred_score[0][:, b_index] for pred_score in nn_outputs])

    return bscore_sum



####### Define functions
def HH_eff_HT(model, hh_file_path, input_tag='ext7', n_entries=100000, tree='jetntuple/Jets'):
    """
    Main efficiency funtion that pulls everything together
    """

    ht_egdes = list(np.arange(0,800,20))
    ht_axis = hist.axis.Variable(ht_egdes, name = r"$HT^{gen}$")

    #Define working points
    cmssw_btag_ht =  WPs_CMSSW['btag_l1_ht']
    cmssw_btag = WPs_CMSSW['btag']

    btag_wp = WPs['btag']
    btag_ht_wp = WPs['btag_l1_ht']
    
    signal = uproot4.open(hh_file_path)[tree]

    #Calculate the truth HT
    raw_event_id = helpers.extract_array(signal, 'event', n_entries)
    raw_jet_genpt = helpers.extract_array(signal, 'jet_genmatch_pt', n_entries)
    raw_jet_pt = helpers.extract_array(signal, 'jet_pt_phys', n_entries)
    raw_cmssw_bscore = helpers.extract_array(signal, 'jet_bjetscore', n_entries)

    raw_inputs = helpers.extract_nn_inputs(signal, input_fields_tag=input_tag, nconstit=16, n_entries=n_entries)

    #Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays  = helpers.group_id_values(raw_event_id, raw_jet_genpt, raw_jet_pt, raw_cmssw_bscore, raw_inputs, num_elements=4)
    jet_genpt, jet_pt, cmssw_bscore, jet_nn_inputs = grouped_arrays

    #Calculate the ht
    jet_genht = ak.sum(jet_genpt, axis=1)
    jet_ht = ak.sum(jet_pt, axis=1)

    #B score from cmssw emulator
    cmsssw_bscore_sum = ak.sum(cmssw_bscore[:,:4], axis=1) #Only sum up the first four
    model_bscore_sum = nn_bscore_sum(model, jet_nn_inputs)

    cmssw_selection = (jet_ht > cmssw_btag_ht) & (cmsssw_bscore_sum > cmssw_btag)
    model_selection = (jet_ht > btag_ht_wp) & (model_bscore_sum > btag_wp)

    #PLot the efficiencies
    #Basically we want to bin the selected truth ht and divide it by the overall count
    all_events = Hist(ht_axis)
    cmssw_selected_events = Hist(ht_axis)
    model_selected_events = Hist(ht_axis)

    all_events.fill(jet_genht)
    cmssw_selected_events.fill(jet_genht[cmssw_selection])
    model_selected_events.fill(jet_genht[model_selection])

    #Plot the ratio
    eff_cmssw = plot_ratio(all_events, cmssw_selected_events)
    eff_model = plot_ratio(all_events, model_selected_events)


    #Get data from handles
    cmssw_x, cmssw_y, cmssw_err = get_bar_patch_data(eff_cmssw)
    model_x, model_y, model_err = get_bar_patch_data(eff_model)

    #Now plot all
    fig = plt.figure()
    plt.errorbar(cmssw_x, cmssw_y, yerr=cmssw_err, c=color_cycle[0], fmt='o', linewidth=2, label=r'Btag CMSSW Emulator (L1 $HT$ > {} GeV, $\sum$ 4b > {})'.format(cmssw_btag_ht, cmssw_btag))
    plt.errorbar(model_x, model_y, yerr=model_err, c=color_cycle[1], fmt='o', linewidth=2, label=r'Improved Btag (L1 $HT$ > {} GeV, $\sum$ 4b > {})'.format(btag_ht_wp, btag_wp))

    #Plot other labels
    plt.hlines(1, 0, 800, linestyles='dashed', color='black', linewidth=3)
    plt.ylim([0., 1.1])
    plt.xlim([0, 800])
    hep.cms.text("Phase 2 Simulation")
    hep.cms.lumitext("PU 200 (14 TeV)")
    plt.xlabel(r"$HT^{gen}$ [GeV]")
    plt.ylabel(r"$\epsilon$(HH $\to$ 4b trigger rate at 14 kHz)")
    plt.legend(loc='lower right', fontsize=15)
    plt.savefig(f'plots/HH_eff_HT.pdf')
    plt.show(block=False)



def HH_eff_mass(model, hh_file_path, n_entries=100000, tree='jetntuple/Jets'):
    return

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='/eos/user/s/sewuchte/L1Trigger/ForDuc/trainings_regression_weighted/2024_08_17_v6_extendedAll200_btgc_ext7_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned/model_QDeepSets_PermutationInv_nconst_16_nfeatures_21_nbits_8_pruned.h5', help = 'Input model for plotting')    
    parser.add_argument('--mode', choices=['ht', 'mass'], default='ht', help='Select the efficiency to be calculated with. Either ht or mass.')

    args = parser.parse_args()

    #Load the model defined above
    model=load_qmodel(args.model)

    #These paths are default to evaluate some of the efficiency
    hh_file_path = '/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_ntuples_v131Xv9/extendedTRK_HW_260824/ggHHbbbb_PU200.root'

    if args.mode == 'ht':
        HH_eff_HT(model, hh_file_path, n_entries=600000)
    elif args.mode == 'mass':
        HH_eff_mass(model, hh_file_path, n_entries=600000)
