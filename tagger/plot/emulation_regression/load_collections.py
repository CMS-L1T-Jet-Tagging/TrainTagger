import uproot
import numpy as np
import awkward as ak
import os
from coffea.nanoevents.methods import vector

# plotting imports
import tagger.plot.style as style
from tagger.plot.style import LABELS_DICT, COLORS_DICT, PROCESS_STYLE, LINESTYLES_DICT, COLLECTION_KEYS
from tagger.plot.common import to_coffea, PT_CUT, ETA_CUT, ETA_BINS

def apply_jecs(jets_pt, jecs_x, jecs_y):
    p1, p0 = np.polyfit(jecs_x, jecs_y, deg=1)
    corr_pt = jets_pt * p1 + p0
    return corr_pt

def load_collections(procs, base_path):
    proc_collections = {}
    jecs = uproot.open(os.path.join(base_path, 'jecs.root'))

    for proc in procs:
        jets = uproot.open(os.path.join(base_path, f"{proc}_perfNano.root"))['Events']
        jet_collections = {
            'scPuppiL1TSC4NGJetJets': {'raw': {}, 'jecs': {}},
            'scPuppiExtendedJets': {'raw': {}, 'jecs': {}}
        }

        # save gen info
        genjets = jets.arrays(filter_name='/(GenJets)_(pt|eta|phi|mass|partonFlavour)/', how='zip')['GenJets']
        genparts = jets.arrays(filter_name='/(GenPart)_(pt|eta|phi|mass|status|pdgId|genPartIdxMother)/', how='zip')['GenPart']
        jet_collections['genjets'] = genjets
        jet_collections['genparts'] = genparts

        # run individual collections
        for coll in COLLECTION_KEYS:
            reco = jets.arrays(filter_name=f'/({coll})_(pt|eta|phi|mass|nDau|genpt|gendr)/', how='zip')[coll]
            eta_mask = abs(reco.eta) < ETA_CUT
            pt_mask = reco.pt > PT_CUT if (coll == 'scPuppiExtendedJets') else reco.pt > 0 # mimic NG jets training conditions
            mask = pt_mask & eta_mask
            reco = reco[mask]
            reco_eta, reco_pt = ak.flatten(abs(reco.eta)), ak.flatten(reco.pt)
            corr_pt = ak.to_numpy(np.zeros_like(reco_pt))
            jec = jecs[coll]

            # run eta bins up to 2.4
            for ieta in range(1, 6):
                lower_eta, upper_eta = ETA_BINS[ieta - 1], ETA_BINS[ieta]
                jecs_x, jecs_y = jec[f'eta_bin{ieta}'].values()
                bin_mask = (reco_eta >= lower_eta) & (reco_eta < upper_eta)
                jpt = reco_pt[bin_mask]
                corr_pt[bin_mask] = apply_jecs(jpt, jecs_x, jecs_y)
            corr_pt = ak.unflatten(corr_pt, ak.num(reco.pt))
            pt_ratio = corr_pt / reco.pt

            # raw and jecs applied, can be used with coffea
            jet_collections[coll]['pt_ratio'] = pt_ratio
            jet_collections[coll]['mask'] = mask
            jet_collections[coll]['raw'] = ak.copy(reco)
            reco['pt'] = corr_pt
            jet_collections[coll]['jecs'] = reco

        proc_collections[proc] = jet_collections
    return proc_collections
