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
from extras import LABELS_DICT, COLORS_DICT, PROCS_DICT, COLLECTION_KEYS
from load_collections import load_collections

import tagger.plot.style as style
from tagger.plot.common import to_coffea

style.set_style()

# Helpers
def smallest_interval(data, fraction=0.68):
    data = np.sort(data)
    n = len(data)
    k = int(np.floor(fraction * n))

    min_width = np.inf

    for i in range(n - k):
        low = data[i]
        high = data[i + k]
        width = high - low

        if width < min_width:
            min_width = width

    return min_width / 2

# Matching and plotting
# match genjets to partons
def match_genjets(daughters, genjets, dr_max=0.4, rel_pt_max=0.5):
    matched_parts = []
    matched_jets = []
    dR_matrix = daughters.metric_table(genjets)

    for r, (parts, jets) in enumerate(zip(daughters, genjets)):

        if len(parts) == 0 or len(jets) == 0:
            matched_parts.append([])
            matched_jets.append([])
            continue
        dr = dR_matrix[r]

        candidates = []

        for i in range(len(parts)):
            for j in range(len(jets)):

                if dr[i, j] > dr_max:
                    continue

                # relative pt difference
                rel_pt = abs(jets.pt[j] - parts.pt[i]) / parts.pt[i]

                if rel_pt > rel_pt_max:
                    continue

                candidates.append((dr[i, j], i, j))

        candidates.sort()

        used_parts = set()
        used_jets = set()

        evt_parts, evt_jets = [], []

        for dR, i, j in candidates:
            if len(used_jets) == len(jets):
                break
            if i in used_parts or j in used_jets:
                continue

            used_parts.add(i)
            used_jets.add(j)

            evt_parts.append(parts[i])
            evt_jets.append(jets[j])

        matched_parts.append(evt_parts)
        matched_jets.append(evt_jets)

    return matched_parts, matched_jets

# find Higgs daughters and match to genjets
def find_daughter_jets(pdgId_mother, genparts, genjets, pdgId_daughter=0,
                 dr_max=0.4, rel_pt_max=0.5):

    genPartMothers = genparts.pdgId[genparts.genPartIdxMother]

    if pdgId_daughter == 0:
        daughters = genparts[
            (genparts.status == 23)
            & (genPartMothers == pdgId_mother)
        ]
    else:
        daughters = genparts[
            (abs(genparts.pdgId) == pdgId_daughter)
            & (genparts.status == 23)
            & (genPartMothers == pdgId_mother)
        ]
    matched_parts, matched_jets = match_genjets(daughters, genjets)

    return ak.Array(matched_parts), ak.Array(matched_jets), daughters

# match Higgs genjets to reco jets
def match_to_reco(proc_coll, daughter_pdgId=0):
    # Compute the delta R between each reco jet and each filtered genjet
    genparts = to_coffea(proc_coll['genparts'])
    genjets = to_coffea(proc_coll['genjets'])
    _, matched_genjets, daughters = find_daughter_jets(25, genparts, genjets, pdgId_daughter=daughter_pdgId)
    recos = {}
    for coll in COLLECTION_KEYS:
        for c in ['raw', 'jecs']:
            reco = to_coffea(proc_coll[coll][c])
            matches = ak.any(reco.genpt[:, :, None] == matched_genjets.pt[:, None, :], axis=2)
            reco = reco[matches]
            recos[f'h_{coll}_{c}'] = reco[ak.num(reco) == 2]

    # GenJets mHH
    matched_genjets = matched_genjets[ak.num(matched_genjets) == 2]
    matched_genjets = to_coffea(matched_genjets)
    recos['h_genjets'] = matched_genjets

    return recos

# Match top and W partons to genjets and then to reco jets, use only hadronic decays
def find_top_daughters(gen, colls, dr_max=0.4, rel_pt_max=0.5):
    genparts = to_coffea(gen['genparts'])
    genjets = to_coffea(gen['genjets'])
    genPartMothers = genparts.pdgId[genparts.genPartIdxMother]

    # --- Select top quarks ---
    tops = genparts[
        (abs(genparts.pdgId) == 6)
        & (genparts.status == 22)
    ]

    def get_daughters(particles, mothers, pdgId_mother):
        mask = (((particles.status == 22) | (particles.status == 23))
            & (mothers == pdgId_mother))

        return particles[mask], mask

    def decay_mode(particles):
        pdg = abs(particles.pdgId)

        def is_quark(id):
            return (pdg >= 1) & (pdg <= 6)
        def is_lepton(id):
            return (pdg == 11) | (pdg == 13) | (pdg == 15)

        decay_mode = ['unknown'] * len(particles)
        decay_mode = np.where(ak.sum(is_quark(pdg), axis=1) == 2, ['hadronic'] * len(decay_mode), decay_mode)
        decay_mode = np.where(ak.sum(is_lepton(pdg), axis=1) == 2, ['leptonic'] * len(decay_mode), decay_mode)
        return decay_mode

    top_results = []
    w_results = []

    # concatenate top and antitop afterwards
    for top in [6, -6]:
        # --- First level: find W and b ---
        W_pdgId = 24 if top > 0 else -24

        daughters, mask = get_daughters(genparts, genPartMothers, top)
        all_indices = ak.local_index(genparts.pt)
        W_indices = all_indices[mask & (genparts.pdgId == W_pdgId)]

        Ws = daughters[abs(daughters.pdgId) == 24]
        bs = daughters[abs(daughters.pdgId) == 5]

        # compare W mother index
        W_daughters, _ = get_daughters(genparts, genPartMothers, W_pdgId)
        W_mask = (ak.num(W_indices) == 1) & (ak.num(W_daughters) == 2)
        daughter_mask = W_mask & (ak.num(bs) == 1)
        decay_type = decay_mode(W_daughters[daughter_mask])

        # --- GenJet matching ---
        all_daughters = ak.concatenate([W_daughters[daughter_mask], bs[daughter_mask]], axis=1)
        matched_parts, matched_genjets = match_genjets(all_daughters, genjets[daughter_mask])
        matched_w_parts, matched_w_genjets = match_genjets(W_daughters[W_mask], genjets[W_mask])

        decay_type_top = ak.Array(decay_type)
        matched_genjets =  ak.Array(matched_genjets)
        top_results.append((decay_type_top, matched_genjets, daughter_mask))

        decay_type_w = ak.Array(decay_type)
        matched_w_genjets =  ak.Array(matched_w_genjets)
        w_results.append((decay_type_w, matched_w_genjets, W_mask))

    # Matching to reco
    recos = {} # collect results
    for (particle, results) in [('top', top_results), ('w', w_results)]:
        jet_count = 3 if particle == 'top' else 2
        decay_type = ak.concatenate([r[0] for r in results], axis=0)
        matched_genjets = ak.concatenate([r[1] for r in results], axis=0)
        event_mask = ak.concatenate([r[2] for r in results], axis=0)

        # only hadronic decay and correct number of matched genjets
        gen_mask = (ak.num(matched_genjets) == jet_count) & (decay_type == 'hadronic')
        recos[f'{particle}_genjets'] = to_coffea(matched_genjets[gen_mask])
        a = to_coffea(matched_genjets[gen_mask]).sum(axis=1).mass

        # Match to reco
        for coll in colls:
            for c in ['raw', 'jecs']:
                reco = to_coffea(ak.concatenate((gen[coll][c], gen[coll][c]), axis=0)[event_mask])
                matches = ak.any(reco.genpt[:, :, None] == matched_genjets.pt[:, None, :], axis=2)
                reco = reco[matches]
                recos[f'{particle}_{coll}_{c}'] = reco[ak.num(reco) == jet_count]
    return recos

# Plot invarinat masses
def plot_mjj(mjjs, gen, p, p_info, proc, plot_dir):
    print(f'Plotting mJJ distribution for {proc}')
    os.makedirs(plot_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    mJJ_bins = p_info['bins']
    mJJ_centers = 0.5 * (mJJ_bins[:-1] + mJJ_bins[1:])

    ax.axvline(p_info['mass'], color='black', linestyle='--', linewidth=1.5, label=p_info['label'])

    # GenJets step histogram and mean and std
    mGen_array = ak.to_numpy(gen.sum(axis=1).mass)
    hist_gen_counts, edges_gen = np.histogram(mGen_array, bins=mJJ_bins)
    hist_gen = hist_gen_counts / hist_gen_counts.sum()

    sigma_gen = smallest_interval(mGen_array)
    mean_gen = np.mean(mGen_array)

    ax.step(edges_gen[:-1], hist_gen, where='post', label=f'GenJets' + "\n" + f'($\mu$ = {mean_gen:.1f}, $\sigma_{{eff}}$ = {sigma_gen:.1f})', color='gray', linewidth=2)

    # reco collections with error bars and mean and std
    for coll in COLLECTION_KEYS:
        for t in ['raw', 'jecs']:
            if t == 'jecs' and coll == 'scPuppiL1TSC4NGJetJets':
                continue
            reco = mjjs[f'{p}_{coll}_{t}']
            mjj_array = ak.to_numpy(reco.sum(axis=1).mass)
            hist_counts, edges = np.histogram(mjj_array, bins=mJJ_bins)
            hist = hist_counts / hist_counts.sum()  # fraction

            # gauss fit
            sigma = smallest_interval(mjj_array)
            mean = np.mean(mjj_array)

            # statistical error for normalized histogram
            hist_err = np.sqrt(hist_counts) / hist_counts.sum()
            ax.errorbar(
                mJJ_centers, hist, yerr=hist_err,
                label=LABELS_DICT[f'{coll}_{t}']+ "\n" + f" ($\mu$ = {mean:.1f}, $\sigma_{{eff}}$ = {sigma:.1f})",
                color=COLORS_DICT[f"{coll}_{t}"],
                fmt='o-',         # circle marker
                markersize=6.5,
                linewidth=2,
                capsize=3
            )
            if coll == 'scPuppiL1TSC4NGJetJets' and t == 'raw':
                np.savez(f"{plot_dir}/arr_hist.npz", x=mJJ_centers, y=hist, yerr=hist_err, sigma=sigma, mu=mean)

    ax.set_xlabel(p_info['x_label'])
    ax.set_ylabel("Fraction")
    ax.legend(title=PROCS_DICT[proc], fontsize=28, title_fontsize=28)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT,
                ax=ax, fontsize=style.CMSHEADER_SIZE)

    plt.savefig(f"{plot_dir}/{p}_mass.pdf", bbox_inches='tight')
    plt.savefig(f"{plot_dir}/{p}_mass.png", bbox_inches='tight')
    return

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help='Whether to reprocess the data and regenerate the plots')
    parser.add_argument('-p', '--particles', nargs='+', default=['top', 'h'], help='List of particles to plot masses for (e.g. top, h)')
    parser.add_argument('-m', '--model', default='output/baseline', help='model output directory')
    args = parser.parse_args()
    plot_dir = f"{args.model}/plots/emulation_regression"
    os.makedirs(plot_dir, exist_ok=True)

    # particles infos for potting and extract required samples
    particle_info = {
        'top': {
            'mass': 173,
            'label': r"$m_{t}$ = 173 GeV",
            'x_label': r"Trijet invariant mass [GeV]",
            'bins': np.linspace(0, 310, 40),
            'proc': ['TT_PU200'],
        },
        'w': {
            'mass': 80,
            'label': r"$m_{W}$ = 80 GeV",
            'x_label': r"Dijet invariant mass [GeV]",
            'bins': np.linspace(0, 180, 40),
            'proc': ['TT_PU200'],
        },
        'h': {
            'mass': 125,
            'label': r"$m_{H}$ = 125 GeV",
            'x_label': r"Dijet invariant mass [GeV]",
            'bins': np.linspace(0, 230, 40),
            'proc': ['VBFHToBB_PU200', 'VBFHToCC_PU200'],
        }
    }

    # neccessary proccesses for the selected particles
    processes = ak.flatten([particle_info[p]['proc'] for p in args.particles], axis=None).tolist()
    proc_collections = load_collections(processes, args.input)

    # Top and W
    if 'top' in args.particles:
        masses_tt = find_top_daughters(proc_collections['TT_PU200'], COLLECTION_KEYS)
        for p in ['top', 'w']:
            plot_dir = f"{plot_dir}/{particle_info[p]['proc']}/masses"
            plot_mjj(masses_tt, masses_tt[f'{p}_genjets'], p, particle_info[p], particle_info[p]['proc'], plot_dir)

    # Higgs
    if 'h' in args.particles:
        for vbf, pdgId in {'VBFHToBB_PU200': 5, 'VBFHToCC_PU200': 4}.items():
            masses_h = match_to_reco(proc_collections[vbf], pdgId)
            plot_dir = f"{plot_dir}/{vbf}/masses"
            plot_mjj(masses_h, masses_h['h_genjets'], 'h', particle_info['h'], vbf, plot_dir)
