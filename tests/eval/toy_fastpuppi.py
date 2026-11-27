"""Synthetic FastPUPPI performance ntuples (<process>_perfNano.root) and
jecs.root, with the branches the FastPUPPI regression collection reads.
Crude physics, but every code path is exercised:

- TT_PU200: t tbar with W -> qq' (or W -> l nu) and b, as status 22/23
  gen particles with mother indices;
- VBFHToBB_PU200, VBFHToCC_PU200: H -> b bbar / c cbar;
- QCD_Pt15To3000_PU200 and MinBias_PU200: gen jets only (MinBias soft,
  with a hard tail for the rates).

Every parton of a decay gets a gen jet close to it; every gen jet in
|eta| < 3 gets a reco jet in both collections (scPuppiL1TSC4NGJetJets:
response ~1, scPuppiExtendedJets: ~0.8) with genpt = the gen jet pT,
plus pileup jets with genpt = 0. jecs.root holds a linear correction
(TGraph) per collection and |eta| bin.
"""

import os

import awkward as ak
import numpy as np
import pandas as pd
import uproot

PROCESSES = {
    "TT_PU200": 1500,
    "QCD_Pt15To3000_PU200": 1500,
    "MinBias_PU200": 20000,
    "VBFHToBB_PU200": 1000,
    "VBFHToCC_PU200": 1000,
}
COLLECTIONS = ("scPuppiL1TSC4NGJetJets", "scPuppiExtendedJets")
RESPONSE = {
    "scPuppiL1TSC4NGJetJets": (1.0, 0.15),
    "scPuppiExtendedJets": (0.8, 0.2),
}


def _parton(pdg, status, mother, rng, pt_scale=60.0):
    return dict(
        pdgId=pdg,
        status=status,
        genPartIdxMother=mother,
        pt=rng.uniform(25, 25 + 3 * pt_scale),
        eta=rng.uniform(-2.2, 2.2),
        phi=rng.uniform(-np.pi, np.pi),
        mass=0.0,
    )


def _gen_particles(process, rng):
    """Gen particles of one event, and the indices of the partons that
    make jets."""
    if process == "TT_PU200":
        parts = [_parton(6, 22, -1, rng), _parton(-6, 22, -1, rng)]
        jets = []
        for top, w in ((0, 24), (1, -24)):
            sign = 1 if w > 0 else -1
            w_idx = len(parts)
            parts += [
                _parton(w, 22, top, rng),
                _parton(5 * sign, 23, top, rng),
            ]
            jets.append(w_idx + 1)
            if rng.uniform() < 0.7:  # hadronic W
                parts += [
                    _parton(2 * sign, 23, w_idx, rng),
                    _parton(-1 * sign, 23, w_idx, rng),
                ]
                jets += [len(parts) - 2, len(parts) - 1]
            else:
                parts += [
                    _parton(-11 * sign, 23, w_idx, rng),
                    _parton(12 * sign, 23, w_idx, rng),
                ]
        return parts, jets
    if process.startswith("VBFHTo"):
        q = 5 if process == "VBFHToBB_PU200" else 4
        parts = [
            _parton(25, 22, -1, rng),
            _parton(q, 23, 0, rng),
            _parton(-q, 23, 0, rng),
        ]
        return parts, [1, 2]
    return [_parton(21, 71, -1, rng)], []


def _gen_jets(process, parts, jet_partons, rng):
    jets = []
    for i in jet_partons:
        p = parts[i]
        jets.append(
            dict(
                pt=p["pt"] * rng.normal(1.0, 0.1),
                eta=p["eta"] + rng.normal(0, 0.05),
                phi=p["phi"] + rng.normal(0, 0.05),
                mass=rng.uniform(2, 15),
                partonFlavour=abs(p["pdgId"]),
            )
        )
    if process == "MinBias_PU200":
        n = rng.integers(0, 7)
        hard = rng.uniform() < 0.03
        pts = (
            20 + rng.exponential(35, n) if hard else 5 + rng.exponential(12, n)
        )
    elif process == "QCD_Pt15To3000_PU200":
        n = rng.integers(2, 7)
        pts = np.exp(rng.uniform(np.log(15), np.log(1000), n))
    else:
        n = rng.integers(0, 4)
        pts = rng.uniform(20, 200, n)
    for pt in pts:
        jets.append(
            dict(
                pt=pt,
                eta=rng.uniform(-3, 3),
                phi=rng.uniform(-np.pi, np.pi),
                mass=rng.uniform(1, 10),
                partonFlavour=0,
            )
        )
    return jets


def _reco_jets(coll, gen_jets, rng):
    mean, sigma = RESPONSE[coll]
    out = []
    for g in gen_jets:
        if abs(g["eta"]) > 3:
            continue
        out.append(
            dict(
                pt=g["pt"] * max(rng.normal(mean, sigma), 0.05),
                eta=g["eta"] + rng.normal(0, 0.02),
                phi=g["phi"] + rng.normal(0, 0.02),
                mass=g["mass"] * rng.uniform(0.7, 1.3),
                nDau=int(rng.integers(1, 30)),
                genpt=g["pt"],
                gendr=abs(rng.normal(0, 0.05)),
            )
        )
    for _ in range(rng.integers(0, 4)):  # pileup jets
        out.append(
            dict(
                pt=rng.uniform(3, 40),
                eta=rng.uniform(-3, 3),
                phi=rng.uniform(-np.pi, np.pi),
                mass=rng.uniform(0, 5),
                nDau=int(rng.integers(1, 10)),
                genpt=0.0,
                gendr=0.0,
            )
        )
    return out


_TYPES = {
    "pdgId": np.int32,
    "status": np.int32,
    "genPartIdxMother": np.int32,
    "partonFlavour": np.int32,
    "nDau": np.int32,
}


def _record(rows, fields):
    """Jagged record array (events x objects) of the given fields,
    NanoAOD types."""
    counts = [len(ev) for ev in rows]
    columns = {}
    for f in fields:
        flat = np.array(
            [r[f] for ev in rows for r in ev], dtype=_TYPES.get(f, np.float32)
        )
        columns[f] = ak.unflatten(flat, counts)
    return ak.zip(columns)


def write_process(path, process, n_events, seed):
    rng = np.random.default_rng(seed)
    gen_parts, gen_jets, reco = [], [], {c: [] for c in COLLECTIONS}
    for _ in range(n_events):
        parts, jet_partons = _gen_particles(process, rng)
        jets = _gen_jets(process, parts, jet_partons, rng)
        gen_parts.append(parts)
        gen_jets.append(jets)
        for c in COLLECTIONS:
            reco[c].append(_reco_jets(c, jets, rng))
    branches = {
        "GenPart": _record(
            gen_parts,
            (
                "pt",
                "eta",
                "phi",
                "mass",
                "status",
                "pdgId",
                "genPartIdxMother",
            ),
        ),
        "GenJets": _record(
            gen_jets, ("pt", "eta", "phi", "mass", "partonFlavour")
        ),
    }
    for c in COLLECTIONS:
        branches[c] = _record(
            reco[c], ("pt", "eta", "phi", "mass", "nDau", "genpt", "gendr")
        )
    with uproot.recreate(path) as f:
        tree = f.mktree(
            "Events",
            {k: v.type for k, v in branches.items()},
            field_name=lambda outer, inner: f"{outer}_{inner}",
            counter_name=lambda c: f"n{c}",
        )
        tree.extend(branches)


def write_jecs(path, seed=0):
    """A linear correction per collection and |eta| bin (eta_bin1 ...
    eta_bin5), as TGraphs."""

    rng = np.random.default_rng(seed)
    x = np.linspace(10, 500, 25)
    with uproot.recreate(path) as f:
        for c in COLLECTIONS:
            mean = RESPONSE[c][0]
            for i in range(1, 6):
                slope, offset = 1 / mean * rng.uniform(
                    0.95, 1.05
                ), rng.uniform(-3, 3)
                f[f"{c}/eta_bin{i}"] = uproot.as_TGraph(
                    pd.DataFrame({"x": x, "y": x * slope + offset})
                )


def write(folder, processes=None, seed=0):
    """All <process>_perfNano.root files and jecs.root in ``folder``."""
    os.makedirs(folder, exist_ok=True)
    for i, (proc, n) in enumerate((processes or PROCESSES).items()):
        write_process(
            os.path.join(folder, f"{proc}_perfNano.root"), proc, n, seed + i
        )
    write_jecs(os.path.join(folder, "jecs.root"), seed)
    return folder


if __name__ == "__main__":
    import sys

    print(write(sys.argv[1]))
