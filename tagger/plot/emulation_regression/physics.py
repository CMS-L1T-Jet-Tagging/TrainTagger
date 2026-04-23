import uproot
import numpy as np
import awkward as ak
import os
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.stats import norm
from coffea.nanoevents.methods import vector
from extras import N_BUNCHES, REVOLUTION_FREQUENCY, MINBIAS_RATE, PT_BINS, LABELS_DICT, COLORS_DICT, PROCS_DICT, COLLECTION_KEYS
from scipy.interpolate import make_interp_spline

# style from tagger
import sys
sys.path.append("/afs/cern.ch/user/s/stella/TaggerFork/TrainTagger/tagger/plot")  # The directory *containing* style.py
import style
style.set_style()

# Helpers
def to_coffea(array):
    return ak.Array(array, behavior=vector.behavior, with_name="PtEtaPhiMLorentzVector")

def inv_helper(jet1, jet2):
    return (jet1 + jet2).mass

def collapse_all(sel):
    if sel.ndim > 1:
        return np.all(sel, axis=1)
    return sel

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

# Physics starts here
# Turn-On Curves and Working Points
def get_obj(coll, obj):
    coll = to_coffea(coll)
    if obj == 'jet1':
        return coll.pt[ak.num(coll.pt) > 0][:, 0], np.arange(80, 500, 0.25)
    elif obj == 'jet2':
        return coll.pt[ak.num(coll.pt) > 1][:, 1],  np.arange(50, 320, 0.25)
    elif obj == 'jet3':
        return coll.pt[ak.num(coll.pt) > 2][:, 2],  np.arange(5, 100, 0.25)
    elif obj == 'ht15':
        return ak.sum(coll.pt[coll.pt > 15], axis=1), np.arange(100, 550, 0.25)
    elif obj == 'ht30':
        return ak.sum(coll.pt[coll.pt > 30], axis=1), np.arange(100, 550, 0.25)

    # invarinat masses
    elif obj == 'mjj':
        return (coll[ak.num(coll.pt) > 1][:, 0] + coll[ak.num(coll.pt) > 1][:, 1]).mass, np.arange(400, 2200, 0.25)
    elif obj == 'max_mjj':
        mjjs = coll[ak.num(coll.pt) > 1].metric_table(coll[ak.num(coll.pt) > 1], metric=inv_helper)
        return ak.max(ak.max(mjjs, axis=-1), axis=-1), np.arange(400, 2800, 0.25)

    # symmetric di- and quad jet seeds
    elif obj == 'dijet':
        return coll[ak.num(coll.pt) > 1][:, :2], np.arange(50, 300, 0.25)
    elif obj == 'quadjet':
        return coll[ak.num(coll.pt) > 3][:, :4], np.arange(50, 200, 0.25)

def get_rate_wps(reco, target_rate, obj):
    rates, pt_cuts = [], []
    total_events = len(reco)
    sort_idx = ak.argsort(reco.pt[ak.num(reco.pt) > 0], axis=1, ascending=False, stable=True)
    reco = reco[ak.num(reco.pt) > 0][sort_idx] # sort by pt
    reco_obj, bins = get_obj(reco, obj)
    for pt_cut in bins:
        selection = reco_obj > pt_cut
        selection = collapse_all(selection)
        rate = (np.sum(selection) / total_events) * MINBIAS_RATE
        rates.append(rate)
        pt_cuts.append(pt_cut)
    wp_idx = np.argmin(np.abs(np.array(rates) - target_rate))
    wp_rate = rates[wp_idx]
    wp = pt_cuts[wp_idx]
    if abs(wp_rate - target_rate) > 2:
        raise ValueError(f"Could not find a working point close to the target rate of {target_rate} kHz for {obj}. Closest rate: {wp_rate} kHz at pt cut {wp} GeV.")
    return wp

def turn_on_curve(tt_collection, minbias_collection, proc, turn_on_quantity, plot_dir):
    for r in [10, 20, 50, 100, 150]:
        for t in turn_on_quantity:
            fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
            hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.CMSHEADER_SIZE)
            bins = np.linspace(0, 2500, 125) if t in ['mjj', 'max_mjj'] else np.linspace(0, 255, 90)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            xerr = (bins[1:] - bins[:-1]) / 2
            for coll in COLLECTION_KEYS:
                plateau_start = 0
                for coll_type in ['raw', 'jecs']:
                    if coll_type == 'jecs' and coll == 'scPuppiL1TSC4NGJetJets':
                        continue
                    reco = to_coffea(tt_collection[coll][coll_type])
                    genjets = to_coffea(tt_collection[coll]['gen'])
                    total_events = len(reco)
                    reco, genjets = reco[ak.num(reco) > 0], genjets[ak.num(reco) > 0]  # Only consider events with at least one jet
                    arg_sort = ak.argsort(reco.pt, axis=1, ascending=False)
                    reco, genjets = reco[arg_sort], genjets[arg_sort]  # Sort jets by pt
                    gen, _ = get_obj(genjets, t)
                    reco, _ = get_obj(reco, t)
                    effs, y_errs = [], []
                    wp = minbias_collection[coll][f'wp_{t}_{r}_{coll_type}']
                    for lower, upper in zip(bins[:-1], bins[1:]):
                        bin_selection = collapse_all((gen > lower) & (gen <= upper))
                        bin_eff = np.sum(collapse_all(reco[bin_selection] > wp)) / np.sum(bin_selection) if np.sum(bin_selection) > 0 else 0
                        y_err = np.sqrt(bin_eff * (1 - bin_eff) / np.sum(bin_selection)) if np.sum(bin_selection) > 0 else 0
                        effs.append(bin_eff)
                        y_errs.append(y_err)

                    # dynamically set plotting range
                    if np.max(effs) >= 0.96:
                        plateau_start = max(plateau_start, np.where(np.array(effs) > 0.96)[0][0]) + 8 # identify plateau start
                    else:
                        plateau_start = len(bins) - 1

                    # Plot turn-on curve with error bars and spline interpolation
                    color = COLORS_DICT[f"{coll}_{coll_type}"]
                    label = '{}, {} GeV'.format(LABELS_DICT[f"{coll}_{coll_type}"], np.round(wp))
                    ax.errorbar(
                        bin_centers, effs,
                        xerr=xerr, yerr=y_err,
                        fmt='o', capsize=3, color=color, label=label
                    )
                    spl = make_interp_spline(bin_centers, effs, k=3)
                    x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 500)
                    y_smooth = spl(x_smooth)
                    ax.plot(x_smooth, y_smooth, color=color)

            # Unify all collections in one plot
            plateau_start = min(plateau_start, len(bins) - 1) # ensure plateau index is within bounds
            ax.legend(title=PROCS_DICT[proc], fontsize=11)
            ax.set_xlim(0, bins[plateau_start])
            ax.set_ylim(0, 1.05)
            ax.set_xlabel(f"{LABELS_DICT[t]} [GeV]")
            ax.set_ylabel(f"Eff (L1 rate at {r} kHz)")
            fig.savefig(f"{plot_dir}/turn_on_curve_{t}_{r}kHz.pdf", bbox_inches='tight')
            fig.savefig(f"{plot_dir}/turn_on_curve_{t}_{r}kHz.png", bbox_inches='tight')
            plt.close(fig)

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

        evt_parts = []
        evt_jets = []

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

# Invariant Masses
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

def match_to_reco(proc_coll, reco_colls, daughter_pdgId=0):
    # Compute the delta R between each reco jet and each filtered genjet
    genparts = to_coffea(proc_coll['genparts'])
    genjets = to_coffea(proc_coll['genjets'])
    _, matched_genjets, daughters = find_daughter_jets(25, genparts, genjets, pdgId_daughter=daughter_pdgId)
    recos = {}
    for coll in reco_colls:
        for c in ['raw', 'jecs']:
            reco = to_coffea(proc_coll[coll][c])
            matches = ak.any(reco.genpt[:, :, None] == matched_genjets.pt[:, None, :], axis=2)
            reco = reco[matches]
            recos[f'{coll}_{c}'] = reco

    # GenJets mHH
    matched_genjets = matched_genjets[ak.num(matched_genjets) == 2]
    matched_genjets = to_coffea(matched_genjets)

    return recos, matched_genjets


def plot_mjj(mjjs, gen, colls, proc, version):
    print(f'Plotting mJJ distribution for {proc}')
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    mHH_bins = np.linspace(0, 230, 30)
    mHH_centers = 0.5 * (mHH_bins[:-1] + mHH_bins[1:])
    sigmas = {}
    eff_helper = "eff"
    for coll in colls:
        for t in ['raw', 'jecs']:
            if t == 'jecs' and coll == 'scPuppiL1TSC4NGJetJets':
                continue
            reco = mjjs[f'{coll}_{t}']
            reco_mask = ak.num(reco) == 2
            mjj_array = ak.to_numpy((reco[reco_mask][:, 0] + reco[reco_mask][:, 1]).mass)
            hist_counts, edges = np.histogram(mjj_array, bins=mHH_bins)
            hist = hist_counts / hist_counts.sum()  # fraction

            # gauss fit
            sigma = smallest_interval(mjj_array)
            sigmas[f'H_{coll}_{t}'] = sigma
            mean = np.mean(mjj_array)

            # statistical error for normalized histogram
            hist_err = np.sqrt(hist_counts) / hist_counts.sum()
            ax.errorbar(
                mHH_centers, hist, yerr=hist_err,
                label=LABELS_DICT[f'{coll}_{t}']+ f" ($\mu$ = {mean:.1f}, $\sigma_{eff}$ = {sigma:.1f})",
                color=COLORS_DICT[f"{coll}_{t}"],
                fmt='o-',         # circle marker
                markersize=5,
                linewidth=1.5,
                capsize=3
            )
            if coll == 'scPuppiL1TSC4NGJetJets' and t == 'raw':
                np.savez(f"{version}/{proc}/arr_hist.npz", x=mHH_centers, y=hist, yerr=hist_err, sigma=sigma, mu=mean)

    # GenJets step histogram
    mHH_array = ak.to_numpy(gen[ak.num(gen) == 2].sum().mass)
    hist_gen_counts, edges_gen = np.histogram(mHH_array, bins=mHH_bins)
    hist_gen = hist_gen_counts / hist_gen_counts.sum()

    # mean and sigma of gen
    sigma_gen = smallest_interval(mHH_array)
    mean_gen = np.mean(mHH_array)

    ax.step(edges_gen[:-1], hist_gen, where='post', label=f'GenJets ($\mu$ = {mean_gen:.1f}, $\sigma_{eff_helper}$ = {sigma_gen:.1f})', color='grey', linewidth=2)
    ax.axvline(125, color='black', linestyle='--', linewidth=1.5, label=r"$m_H = 125 GeV$")

    ax.set_xlabel(r"$M_{JJ}$ [GeV]")
    ax.set_ylabel("Fraction")
    ax.set_xlim(0, 230)
    ax.legend(title=PROCS_DICT[proc], fontsize=11)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT,
                ax=ax, fontsize=style.CMSHEADER_SIZE)

    plt.savefig(f"{version}/{proc}/mjj_comparison.pdf", bbox_inches='tight')
    plt.savefig(f"{version}/{proc}/mjj_comparison.png", bbox_inches='tight')
    return sigmas

def plot_resonance(mjjs, gen, colls, proc, version):
    fig, ax = plt.subplots(1, 1, figsize=(30, 17))
    mHH_bins = np.linspace(0, 5000, 60)
    mHH_centers = 0.5 * (mHH_bins[:-1] + mHH_bins[1:])
    for coll in colls:
        for t in ['raw', 'jecs']:
            if t == 'jecs' and coll == 'scPuppiL1TSC4NGJetJets':
                continue
            reco = mjjs[f'{coll}_{t}']
            reco_mask = ak.num(reco) == 4
            mjj_array = ak.to_numpy(reco[reco_mask].sum(axis=1).mass)
            hist_counts, edges = np.histogram(mjj_array, bins=mHH_bins)
            hist = hist_counts / hist_counts.sum()  # fraction

            # statistical error for normalized histogram
            hist_err = np.sqrt(hist_counts) / hist_counts.sum()
            ax.errorbar(
                mHH_centers, hist, yerr=hist_err,
                label=LABELS_DICT[f'{coll}_{t}'],
                color=COLORS_DICT[f"{coll}_{t}"],
                fmt='o-',         # circle marker
                markersize=5,
                linewidth=1.5,
                capsize=3
            )

    # GenJets step histogram
    mHH_array = ak.to_numpy(gen[ak.num(gen) == 4].sum(axis=1).mass)
    hist_gen_counts, edges_gen = np.histogram(mHH_array, bins=mHH_bins)
    hist_gen = hist_gen_counts / hist_gen_counts.sum()
    ax.step(edges_gen[:-1], hist_gen, where='post', label='GenJets', color='grey', linewidth=2)

    ax.text(
        0.02, 0.95, PROCS_DICT[proc],                # x, y in axes fraction (0-1)
        transform=ax.transAxes,           # coordinates relative to axes
        verticalalignment='top',          # align top of text at y=0.95
        horizontalalignment='left',       # align left at x=0.02
        color='black'                     # choose color
    )
    ax.set_xlabel(r"$M_{JJ}$")
    ax.set_ylabel("Fraction")
    ax.legend(title=PROCS_DICT[proc], fontsize=11)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT,
                ax=ax, fontsize=style.CMSHEADER_SIZE)

    plt.savefig(f"{version}/{proc}/mjj_resonance_comparison.pdf", bbox_inches='tight')
    plt.savefig(f"{version}/{proc}/mjj_resonance_comparison.png", bbox_inches='tight')

# Top
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
    for top in [6, -6]:

        # --- First level: W and b ---
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

        # --- Jet matching ---
        all_daughters = ak.concatenate([W_daughters[daughter_mask], bs[daughter_mask]], axis=1)
        matched_parts, matched_genjets = match_genjets(all_daughters, genjets[daughter_mask])
        matched_w_parts, matched_w_genjets = match_genjets(W_daughters[W_mask], genjets[W_mask])

        jets_mask = ak.num(matched_genjets) == 3
        decay_type_top = ak.Array(decay_type)[jets_mask]
        matched_genjets =  ak.Array(matched_genjets)[jets_mask]
        top_results.append((decay_type_top, matched_genjets, daughter_mask, jets_mask))

        w_jets_mask = ak.num(matched_w_genjets) == 2
        decay_type_w = ak.Array(decay_type)[w_jets_mask]
        matched_w_genjets =  ak.Array(matched_w_genjets)[w_jets_mask]
        w_results.append((decay_type_w, matched_w_genjets, W_mask, w_jets_mask))

    # Match to reco jets
    recos = {}
    for (particle, results) in [('top', top_results), ('w', w_results)]:
        matched_genjets = ak.concatenate([r[1] for r in results], axis=0)
        decay_type = ak.concatenate([r[0] for r in results], axis=0)
        recos[f'{particle}_genjets'] = matched_genjets
        recos[f'{particle}_decay_type'] = decay_type
        for coll in colls:
            for c in ['raw', 'jecs']:
                masks = [(r[2], r[3]) for r in results]
                reco = to_coffea(ak.concatenate([gen[coll][c][m[0]][m[1]] for m in masks], axis=0))
                matches = ak.any(reco.genpt[:, :, None] == matched_genjets.pt[:, None, :], axis=2)
                reco = reco[matches]
                recos[f'{particle}_{coll}_{c}'] = reco

    return recos

def plot_tt(proc_colls, colls, plot_path):
    print('Plotting top and W mass peaks...')
    recos = find_top_daughters(proc_colls, colls)
    sigmas = {}
    eff_helper = "eff"
    for (p, p_mass) in [('top', 173), ('w', 80)]:
        bins = np.arange(0, p_mass + 250, 10)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        genjets = to_coffea(recos[f'{p}_genjets'])
        decay_modes = to_coffea(recos[f'{p}_decay_type'])
        for t in np.unique(decay_modes):
            fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
            decay_mask = (decay_modes == t)
            gen_masses = ak.to_numpy(genjets[decay_mask].sum(axis=1).mass)
            hist_gen_counts, edges_gen = np.histogram(gen_masses, bins=bins)
            hist_gen = hist_gen_counts / hist_gen_counts.sum()
            mean_gen = np.mean(gen_masses)
            sigma_gen = smallest_interval(gen_masses)
            ax.step(edges_gen[:-1], hist_gen, where='post', label=f'GenJets ($\mu$ = {mean_gen:.1f}, $\sigma_{eff_helper}$ = {sigma_gen:.1f})', color='grey', linewidth=2)
            for coll in colls:
                for c in ['raw', 'jecs']:
                    if c == 'jecs' and coll == 'scPuppiL1TSC4NGJetJets':
                        continue
                    reco = recos[f'{p}_{coll}_{c}'][decay_mask]
                    reco = reco[ak.num(reco) == ak.num(genjets)[0]]  # Only consider events where all 3 jets are matched
                    reco_masses = ak.to_numpy(reco.sum(axis=1).mass)
                    hist_counts, edges = np.histogram(reco_masses, bins=bins)
                    hist = hist_counts / hist_counts.sum()
                    hist_err = np.sqrt(hist_counts) / hist_counts.sum()

                    # gauss fit
                    sigma = smallest_interval(reco_masses)
                    sigmas[f'{p}_{coll}_{c}'] = sigma
                    mean = np.mean(reco_masses)
                    x_hist = 0.5 * (edges[:-1] + edges[1:])
                    ax.errorbar(
                        x_hist, hist, yerr=hist_err,
                        label=LABELS_DICT[f'{coll}_{c}'] + f" ($\mu$ = {mean:.1f}, $\sigma_{eff_helper}$ = {sigma:.1f})",
                        color=COLORS_DICT[f"{coll}_{c}"],
                        fmt='o-',         # circle marker
                        markersize=5,
                        linewidth=1.5,
                        capsize=3
                    )
                    if coll == 'scPuppiL1TSC4NGJetJets' and c == 'raw':
                        np.savez(f"{plot_path}/arr_{p}_{t}_hist.npz", x=x_hist, y=hist, yerr=hist_err, sigma=sigma, mu=mean)
            x_label = r"$M_{JJJ}$ [GeV]" if p == 'top' else r"$M_{JJ}$ [GeV]"
            p_label = r"$m_{t}$ = 173 GeV" if p == 'top' else r"$m_{W}$ = 80 GeV"
            ax.axvline(p_mass, color='black', linestyle='--', linewidth=1.5, label=p_label)
            ax.set_xlabel(x_label)
            ax.set_xlim(0, np.max(bins))
            ax.set_ylabel("Fraction")
            ax.legend(title=f"{PROCS_DICT['TT_PU200']} ({t})", fontsize=11)
            hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT,
                        ax=ax, fontsize=style.CMSHEADER_SIZE)
            plt.savefig(f"{plot_path}/tt_{p}_{t}_comparison.pdf", bbox_inches='tight')
            plt.savefig(f"{plot_path}/tt_{p}_{t}_comparison.png", bbox_inches='tight')
            plt.close(fig)
    return sigmas


