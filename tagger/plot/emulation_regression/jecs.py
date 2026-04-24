import uproot
import numpy as np
import awkward as ak
import os
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.stats import norm
from coffea.nanoevents.methods import vector
from argparse import ArgumentParser
from scipy.interpolate import interp1d

# plotting imports
from physics import get_rate_wps, turn_on_curve, plot_mjj, match_to_reco, find_daughter_jets, to_coffea, plot_resonance, find_top_daughters, plot_tt
from basic import response, rms, plot_distribution, get_response, get_rms, rms, plot_response_bin, distribution_heatmaps
from extras import N_BUNCHES, REVOLUTION_FREQUENCY, MINBIAS_RATE, PT_BINS, LABELS_DICT, COLORS_DICT, PROCS_DICT, COLLECTION_KEYS

import tagger.plot.style
style.set_style()

style.set_style()

# Helpers
# def apply_jecs(jets_pt, jecs_x, jecs_y):
#     interp = interp1d(jecs_x, jecs_y, kind="linear", fill_value="extrapolate")
#     pt_corr = interp(jets_pt)
#     return pt_corr

def apply_jecs(jets_pt, jecs_x, jecs_y):
    p1, p0 = np.polyfit(jecs_x, jecs_y, deg=1)
    corr_pt = jets_pt * p1 + p0
    return corr_pt

def extract_genjets(reco, genjets):
    matches = reco.genpt[:, :, None] == genjets.pt[:, None, :]
    matches_idx = ak.argmax(matches, axis=2)
    matched_genjets = genjets[matches_idx]
    return matched_genjets

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', help='Model output path')
    parser.add_argument('-i', '--input', help='FP files output path')
    args = parser.parse_args()
    version = f"{args.output}/plots/emulation_regression"
    os.makedirs(version, exist_ok=True)
    procs = ['TT_PU200','QCD_Pt15To3000_PU200', 'MinBias_PU200', 'VBFHToBB_PU200', 'VBFHToCC_PU200', 'VBFHToInvisible_PU200']
    proc_collections = {}
    for proc in procs:
        jecs = uproot.open(os.path.join(args.input, 'jecs.root'))
        jets = uproot.open(os.path.join(args.input, f'{proc}_perfNano.root'))['Events']
        eta_bins = [0, 1.3, 1.7, 1.9, 2.1, 2.4]
        jet_collections = {'scPuppiL1TSC4NGJetJets': {'raw': {}, 'jecs': {}}, 'scPuppiExtendedJets': {'raw': {}, 'jecs': {}}}
        for coll in COLLECTION_KEYS:
            genjets = jets.arrays(filter_name='/(GenJets)_(pt|eta|phi|mass|partonFlavour)/', how='zip')['GenJets']
            genparts = jets.arrays(filter_name='/(GenPart)_(pt|eta|phi|mass|status|pdgId|genPartIdxMother)/', how='zip')['GenPart']
            reco = jets.arrays(filter_name=f'/({coll})_(pt|eta|phi|mass|genpt|gendr)/', how='zip')[coll]
            eta_mask = abs(reco.eta) < 2.399
            pt_mask = reco.pt > 15 if (coll == 'scPuppiExtendedJets') else reco.pt > 0
            mask = pt_mask & eta_mask
            reco = reco[mask]
            matched_genjets = extract_genjets(reco, genjets)
            reco_eta, reco_pt = ak.flatten(abs(reco.eta)), ak.flatten(reco.pt)
            corr_pt = ak.to_numpy(np.zeros_like(reco_pt))
            jec = jecs[coll]
            for ieta in range(1, 6): # up to, ie. eta 2.4
                lower_eta, upper_eta = eta_bins[ieta - 1], eta_bins[ieta]
                jecs_x, jecs_y = jec[f'eta_bin{ieta}'].values()
                bin_mask = (reco_eta >= lower_eta) & (reco_eta < upper_eta)
                jpt = reco_pt[bin_mask]
                corr_pt[bin_mask] = apply_jecs(jpt, jecs_x, jecs_y)
            corr_pt = ak.unflatten(corr_pt, ak.num(reco.pt))
            pt_ratio = corr_pt / reco.pt

            # write info to dict
            jet_collections['genjets'] = genjets
            jet_collections['genparts'] = genparts

            # raw and jecs applied, can be used with coffea
            jet_collections[coll]['gen'] = matched_genjets
            jet_collections[coll]['pt_ratio'] = pt_ratio
            jet_collections[coll]['mask'] = mask
            jet_collections[coll]['raw'] = ak.copy(reco)
            reco['pt'] = corr_pt
            jet_collections[coll]['jecs'] = reco

        proc_collections[proc] = jet_collections

    # START PLOTTIMG
    for p in proc_collections.keys():
        jet_collections = proc_collections[p]
        plot_dir = f"{version}/{p}/distributions"
        os.makedirs(plot_dir, exist_ok=True)
        plot_distribution(jet_collections['scPuppiL1TSC4NGJetJets']['raw'].pt,
                    jet_collections['scPuppiExtendedJets']['raw'].pt,
                    jet_collections['scPuppiExtendedJets']['jecs'].pt,
                    jet_collections['genjets'].pt,
                    'scPuppiL1TSC4NGJetJets', 'scPuppiExtendedJets',
                    p, plot_dir=plot_dir)

        # response and rms
        response(ak.to_numpy(ak.flatten(jet_collections['scPuppiL1TSC4NGJetJets']['raw'].genpt)),
            ak.to_numpy(ak.flatten(jet_collections['scPuppiL1TSC4NGJetJets']['raw'].pt)),
            ak.to_numpy(ak.flatten(jet_collections['scPuppiL1TSC4NGJetJets']['pt_ratio'])),
            'scPuppiL1TSC4NGJetJets',
            ak.to_numpy(ak.flatten(jet_collections['scPuppiExtendedJets']['raw'].genpt)),
            ak.to_numpy(ak.flatten(jet_collections['scPuppiExtendedJets']['raw'].pt)),
            ak.to_numpy(ak.flatten(jet_collections['scPuppiExtendedJets']['pt_ratio'])),
            'scPuppiExtendedJets',
            f"{version}/{p}"
            )
        rms(ak.to_numpy(ak.flatten(jet_collections['scPuppiL1TSC4NGJetJets']['raw'].genpt)),
            ak.to_numpy(ak.flatten(jet_collections['scPuppiL1TSC4NGJetJets']['raw'].pt)),
            ak.to_numpy(ak.flatten(jet_collections['scPuppiL1TSC4NGJetJets']['pt_ratio'])),
            'scPuppiL1TSC4NGJetJets',
            ak.to_numpy(ak.flatten(jet_collections['scPuppiExtendedJets']['raw'].genpt)),
            ak.to_numpy(ak.flatten(jet_collections['scPuppiExtendedJets']['raw'].pt)),
            ak.to_numpy(ak.flatten(jet_collections['scPuppiExtendedJets']['pt_ratio'])),
            'scPuppiExtendedJets',
            f"{version}/{p}"
            )

        # 2d heatmaps
        distribution_heatmaps(jet_collections, plot_dir)

    # Derive Rates
    target_rates =  [10, 20, 50, 100, 150]
    for coll in COLLECTION_KEYS:
        for obj in ['jet1', 'jet2', 'ht15', 'ht30', 'dijet']:
            wps_raw = get_rate_wps(proc_collections['MinBias_PU200'][coll]['raw'], target_rates, obj)
            wps_jecs = get_rate_wps(proc_collections['MinBias_PU200'][coll]['jecs'], target_rates, obj)
            for r in target_rates:
                proc_collections['MinBias_PU200'][coll][f'wp_{obj}_{r}_raw'] = wps_raw[r]
                proc_collections['MinBias_PU200'][coll][f'wp_{obj}_{r}_jecs'] = wps_jecs[r]

    # Plot Turn on cuves
    turn_on_curve(proc_collections['TT_PU200'], proc_collections['MinBias_PU200'], 'TT_PU200', ['jet1', 'jet2', 'ht15', 'ht30', 'dijet'], plot_dir=f"{version}/TT_PU200")
    turn_on_curve(proc_collections['QCD_Pt15To3000_PU200'], proc_collections['MinBias_PU200'], 'QCD_Pt15To3000_PU200', ['jet1', 'jet2', 'ht15', 'ht30'], plot_dir=f"{version}/QCD_Pt15To3000_PU200")

    # Invariant masses
    # Top and W
    plot_tt(proc_collections['TT_PU200'], ['scPuppiL1TSC4NGJetJets', 'scPuppiExtendedJets'], f'{version}/TT_PU200')

    # Higgs
    for vbf, pdgId in {'VBFHToBB_PU200': 5, 'VBFHToCC_PU200': 4}.items():
        mjjs, mHH = match_to_reco(proc_collections[vbf], ['scPuppiL1TSC4NGJetJets', 'scPuppiExtendedJets'], pdgId)
        plot_mjj(mjjs, mHH, ['scPuppiL1TSC4NGJetJets', 'scPuppiExtendedJets'], vbf, version)

