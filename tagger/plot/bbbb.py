from qkeras.utils import load_qmodel
from argparse import ArgumentParser

#Imports from other modules
from tagger.data.tools import extract_array

def derive_bbbb_WPs(model, minbias_path, target_rate=14, n_entries=10000, tree='jetntuple/Jets'):
    """
    Derive the HH->4b working points
    """

    #Load the minbias data
    minbias = uproot4.open(minbias_path)[tree]

    raw_event_id = extract_array(minbias, 'event', n_entries)
    raw_jet_pt = extract_array(minbias, 'jet_pt', n_entries)
    raw_inputs = extract_nn_inputs(minbias, input_fields_tag='ext7', nconstit=16, n_entries=n_entries)

    #Count number of total event
    n_events = len(np.unique(raw_event_id))
    print("Total number of minbias events: ", n_events)

    #Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays  = helpers.group_id_values(raw_event_id, raw_jet_pt, raw_inputs, num_elements=4)

    # Extract the grouped arrays
    # Jet pt is already sorted in the producer, no need to do it here
    jet_pt, jet_nn_inputs = grouped_arrays

    #Btag input list for first 4 jets
    btag_inputs = [np.asarray(jet_nn_inputs[:, i]).transpose(0, 2, 1) for i in range(0,4)]
    nn_outputs = [model.predict(nn_input) for nn_input in btag_inputs]

    b_index = 1

    bscore_sum = sum([pred_score[0][:, b_index] for pred_score in nn_outputs])
    ht = ak.sum(jet_pt, axis=1)

    assert(len(bscore_sum) == len(ht))

    #Define the histograms (pT edge and NN Score edge)
    ht_edges = list(np.arange(0,500,2)) + [10000] #Make sure to capture everything
    NN_edges = list([round(i,2) for i in np.arange(0, 1.51, 0.01)])

    RateHist = Hist(hist.axis.Variable(ht_edges, name="ht", label="ht"),
                    hist.axis.Variable(NN_edges, name="nn", label="nn"))

    RateHist.fill(ht = ht, nn = bscore_sum)

    #Derive the rate
    rate_list = []
    ht_list = []
    nn_list = []

    #Loop through the edges and integrate
    for ht in ht_edges[:-1]:
        for NN in NN_edges[:-1]:
            
            #Calculate the rate
            rate = RateHist[{"ht": slice(ht*1j, ht_edges[-1]*1.0j, sum)}][{"nn": slice(NN*1.0j,4.0j, sum)}]/n_events
            rate_list.append(rate*minbias_rate)

            #Append the results   
            ht_list.append(ht)
            nn_list.append(NN)

    plot_rate(rate_list, ht_list, nn_list, target_rate=target_rate,  plot_name="btag_rate_scan.pdf", correct_pt=False)

    return

def bbbb_eff_HT(model, n_entries=100000):

    return 

    """
    Plot HH->4b efficiency w.r.t HT
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


if __name__ == "__main__":
    """
    2 steps:

    1. Derive working points: python bbbb.py --deriveWPs
    2. Run efficiency based on the derived working points: python bbbb.py --eff
    """

    parser = ArgumentParser()
    parser.add_argument('-m','--model', default='output/baseline/saved_model.h5' , help = 'Input model')
    parser.add_argument('-s', '--sample', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_ntuples_v131Xv9/baselineTRK_4param_021024/ggHHbbbb_PU200.root' , help = 'Signal sample for HH->bbbb') 
    parser.add_argument('--minbias', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_ntuples_v131Xv9/baselineTRK_4param_021024/MinBias_PU200.root' , help = 'Minbias sample for deriving rates')    

    #Different modes
    parser.add_argument('--deriveWPs', action='store_true', help='derive the working points for b-tagging')
    parser.add_argument('--eff', action='store_true', help='plot efficiency for HH->4b')

    #Other controls
    parser.add_argument('-n','--n_entries', type=int, default=100000, help = 'Number of data entries in root file to run over, can speed up run time, set to None to run on all data entries')
    args = parser.parse_args()

    model=load_qmodel(args.model)
    print(model.summary())

    if args.deriveWPs:
        derive_bbbb_WPs(model, args.minbias)
    # elif args.eff:
    #     bbbb_eff_HT(model, args.sample, n_entries=args.n_entries)
