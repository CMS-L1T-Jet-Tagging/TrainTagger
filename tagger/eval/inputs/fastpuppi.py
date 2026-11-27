"""FastPUPPI performance ntuples of the CMSSW emulation
(<process>_perfNano.root, written by the CI job run_processes with the
model's emulator) and the jet energy corrections derived from them
(jecs.root).

Per process: the gen jets, the gen particles, and per L1 jet collection
the jets as read (``raw``), with the standard jet energy corrections
applied (``jecs``), and the correction factor (``pt_ratio``).
"""

import os

import awkward as ak
import numpy as np
import uproot

# L1 jet collections: the seeded cone jets with the model's pT
# regression (SC4NN) and the standard ones
COLLECTIONS = ("scPuppiL1TSC4NGJetJets", "scPuppiExtendedJets")


def to_coffea(array):
    """The array as coffea Lorentz vectors (fields pt, eta, phi, mass)."""
    # imported here: a slow import, needed only by this function
    from coffea.nanoevents.methods import vector

    return ak.Array(
        array, behavior=vector.behavior, with_name="PtEtaPhiMLorentzVector"
    )


def _apply_jecs(jets_pt, jecs_x, jecs_y):
    p1, p0 = np.polyfit(jecs_x, jecs_y, deg=1)
    return jets_pt * p1 + p0


def load_process(folder, process, config):
    """The collections of one process: {"genjets", "genparts",
    <collection>: {"raw", "jecs", "pt_ratio", "mask"}}. Reco jets: |eta|
    < JET_ETA_CUT, the standard jets also pT > JET_PT_CUT (as in the
    SC4NN jet training). The corrections are linear fits of jecs.root
    per |eta| bin of JEC_ETA_BINS (up to 2.4).
    """
    path = os.path.join(folder, f"{process}_perfNano.root")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing: set config FASTPUPPI_DIR (in CI the FPOutput"
            " folder of run_processes)"
        )
    jecs = uproot.open(os.path.join(folder, "jecs.root"))
    events = uproot.open(path)["Events"]
    eta_bins = config["JEC_ETA_BINS"]
    out = {
        "genjets": events.arrays(
            filter_name="/(GenJets)_(pt|eta|phi|mass|partonFlavour)/",
            how="zip",
        )["GenJets"],
        "genparts": events.arrays(
            filter_name=(
                "/(GenPart)_(pt|eta|phi|mass|status|pdgId|genPartIdxMother)/"
            ),
            how="zip",
        )["GenPart"],
    }
    for coll in COLLECTIONS:
        reco = events.arrays(
            filter_name=f"/({coll})_(pt|eta|phi|mass|nDau|genpt|gendr)/",
            how="zip",
        )[coll]
        eta_mask = abs(reco.eta) < config["JET_ETA_CUT"]
        pt_mask = (
            reco.pt > config["JET_PT_CUT"]
            if coll == "scPuppiExtendedJets"
            else reco.pt > 0
        )
        mask = pt_mask & eta_mask
        reco = reco[mask]
        reco_eta, reco_pt = ak.flatten(abs(reco.eta)), ak.flatten(reco.pt)
        corr_pt = ak.to_numpy(np.zeros_like(reco_pt))
        # the JEC_ETA_BINS bins that start below JET_ETA_CUT (eta_bin1,
        # eta_bin2, ... of jecs.root)
        n_bins = sum(edge < config["JET_ETA_CUT"] for edge in eta_bins)
        for ieta in range(1, n_bins + 1):
            jecs_x, jecs_y = jecs[coll][f"eta_bin{ieta}"].values()
            in_bin = (reco_eta >= eta_bins[ieta - 1]) & (
                reco_eta < eta_bins[ieta]
            )
            corr_pt[in_bin] = _apply_jecs(reco_pt[in_bin], jecs_x, jecs_y)
        corr_pt = ak.unflatten(corr_pt, ak.num(reco.pt))
        corrected = ak.copy(reco)
        corrected["pt"] = corr_pt
        out[coll] = {
            "raw": ak.copy(reco),
            "jecs": corrected,
            "pt_ratio": corr_pt / reco.pt,
            "mask": mask,
        }
    return out
