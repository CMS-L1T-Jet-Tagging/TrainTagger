import uproot
import numpy as np
import awkward as ak
import os
from coffea.nanoevents.methods import vector

# plotting imports
from extras import COLLECTION_KEYS, ETA_BINS

def apply_jecs(jets_pt, jecs_x, jecs_y):
    p1, p0 = np.polyfit(jecs_x, jecs_y, deg=1)
    corr_pt = jets_pt * p1 + p0
    return corr_pt

def load_collections(procs, base_path):
    proc_collections = {}
    for proc in procs:
        jecs = uproot.open(os.path.join(base_path, 'jecs.root'))
        jets = uproot.open(os.path.join(base_path, f"{proc}_perfNano.root"))['Events']
        jet_collections = {
            'scPuppiL1TSC4NGJetJets': {'raw': {}, 'jecs': {}},
            'scPuppiExtendedJets': {'raw': {}, 'jecs': {}}
        }
        for coll in COLLECTION_KEYS:
            print(f"Processing collection: {coll}")
            genjets = jets.arrays(filter_name='/(GenJets)_(pt|eta|phi|mass|partonFlavour)/', how='zip')['GenJets']
            genparts = jets.arrays(filter_name='/(GenPart)_(pt|eta|phi|mass|status|pdgId|genPartIdxMother)/', how='zip')['GenPart']
            print(f'N events for {proc}: {len(genjets)}')
            reco = jets.arrays(filter_name=f'/({coll})_(pt|eta|phi|mass|genpt|gendr)/', how='zip')[coll]
            eta_mask = abs(reco.eta) < 2.399
            pt_mask = reco.pt > 15 if (coll == 'scPuppiExtendedJets') else reco.pt > 0
            mask = pt_mask & eta_mask
            reco = reco[mask]
            reco_eta, reco_pt = ak.flatten(abs(reco.eta)), ak.flatten(reco.pt)
            corr_pt = ak.to_numpy(np.zeros_like(reco_pt))
            jec = jecs[coll]
            for ieta in range(1, 6):
                lower_eta, upper_eta = ETA_BINS[ieta - 1], ETA_BINS[ieta]
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
            jet_collections[coll]['pt_ratio'] = pt_ratio
            jet_collections[coll]['mask'] = mask
            jet_collections[coll]['raw'] = ak.copy(reco)
            reco['pt'] = corr_pt
            jet_collections[coll]['jecs'] = reco

        proc_collections[proc] = jet_collections
    return proc_collections
