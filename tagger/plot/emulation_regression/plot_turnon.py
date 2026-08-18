import uproot
import numpy as np
import awkward as ak
import os
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.stats import norm
from coffea.nanoevents.methods import vector
from argparse import ArgumentParser
from scipy.interpolate import make_interp_spline

# plotting imports
from extras import LABELS_DICT, COLORS_DICT, PROCS_DICT, LINESTYLES_DICT, COLLECTION_KEYS
from load_collections import load_collections

# style from tagger
import tagger.plot.style as style
from tagger.plot.common import to_coffea, MINBIAS_RATE

style.set_style()

# Turn-On Curves and Working Points
def inv_helper(jet1, jet2):
    return (jet1 + jet2).mass

def get_obj(coll, n_reco, n_gen, obj):
    coll = to_coffea(coll)
    if obj == 'jet1':
        num_cut = (n_reco > 0) & (n_gen > 0)
        return ak.max(coll.pt[num_cut], axis=1), np.arange(80, 500, 0.25), 500
    elif obj == 'jet2':
        num_cut = (n_reco > 1) & (n_gen > 1)
        return ak.sort(coll.pt[num_cut], ascending=False)[:,1], np.arange(50, 320, 0.25), 500
    elif obj == 'jet3':
        num_cut = (n_reco > 2) & (n_gen > 2)
        return ak.sort(coll.pt[num_cut], ascending=False)[:,2], np.arange(5, 100, 0.25), 300
    elif obj == 'ht15':
        return ak.sum(coll.pt[coll.pt > 15], axis=1), np.arange(100, 550, 0.25), 1200
    elif obj == 'ht30':
        return ak.sum(coll.pt[coll.pt > 30], axis=1), np.arange(100, 550, 0.25), 1000

    # invarinat masses
    elif obj == 'mjj':
        num_cut = (n_reco > 1) & (n_gen > 1)
        return (coll[num_cut][:, 0] + coll[num_cut][:, 1]).mass, np.arange(400, 2200, 0.25), 1500
    elif obj == 'max_mjj':
        num_cut = (n_reco > 1) & (n_gen > 1)
        mjjs = coll[num_cut].metric_table(coll[num_cut], metric=inv_helper)
        return ak.max(ak.max(mjjs, axis=-1), axis=-1), np.arange(400, 2800, 0.25), 2000

    # symmetric di- and quad jet seeds
    elif obj == 'dijet':
        num_cut = (n_reco > 1) & (n_gen > 1)
        return ak.min(ak.sort(coll[num_cut], axis=1, ascending=False)[:, :2].pt, axis=1), np.arange(10, 500, 0.1), 400
    elif obj == 'quadjet':
        num_cut = (n_reco > 3) & (n_gen > 3)
        return ak.min(ak.sort(coll[num_cut], axis=1, ascending=False)[:, :4].pt, axis=1), np.arange(1, 200, 0.05), 200

def get_rate_wps(reco, target_rates, obj):
    wps = {}
    rates, pt_cuts = [], []
    total_events = len(reco)
    sort_idx = ak.argsort(reco.pt[ak.num(reco.pt) > 0], axis=1, ascending=False, stable=True)
    reco = reco[ak.num(reco.pt) > 0][sort_idx] # sort by pt
    n_reco = ak.num(reco)
    reco_obj, bins, _ = get_obj(reco, n_reco, n_reco, obj)
    for pt_cut in bins:
        selection = reco_obj > pt_cut
        rate = (np.sum(selection) / total_events) * MINBIAS_RATE
        rates.append(rate)
        pt_cuts.append(pt_cut)

    # find wps for target rates
    for target_rate in target_rates:
        wp_idx = np.argmin(np.abs(np.array(rates) - target_rate))
        wp_rate = rates[wp_idx]
        wp = pt_cuts[wp_idx]
        if abs(wp_rate - target_rate) > 2:
            raise ValueError(f"Could not find a working point close to the target rate of {target_rate} kHz for {obj}. Closest rate: {wp_rate} kHz at pt cut {wp} GeV.")
        wps[target_rate] = wp
    return wps

def turn_on_curve(tt_collection, minbias_collection, proc, turn_on_quantity, rates, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    for r in rates:
        for t in turn_on_quantity:
            fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
            hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.CMSHEADER_SIZE)
            for coll in COLLECTION_KEYS:
                for coll_type in ['raw', 'jecs']:
                    if coll_type == 'jecs' and coll == 'scPuppiL1TSC4NGJetJets':
                        continue # no need for jecs on top of model for now
                    reco = to_coffea(tt_collection[coll][coll_type])
                    genjets = to_coffea(tt_collection['genjets'])
                    total_events = len(reco)
                    reco, genjets = reco[ak.num(reco) > 0], genjets[ak.num(reco) > 0]  # Only consider events with at least one jet
                    n_reco, n_gen = ak.num(reco), ak.num(genjets)
                    gen, bins, bin_cut = get_obj(genjets, n_reco, n_gen, t)
                    reco, _, _ = get_obj(reco, n_reco, n_gen, t)
                    bin_centers = 0.5 * (bins[:-1] + bins[1:])
                    x_errs = (bins[1:] - bins[:-1]) / 2
                    effs, y_errs = [], []
                    wp = minbias_collection[coll][f'wp_{t}_{r}_{coll_type}']
                    for lower, upper in zip(bins[:-1], bins[1:]):
                        bin_selection = (gen > lower) & (gen <= upper)
                        bin_eff = np.sum(reco[bin_selection] > wp) / np.sum(bin_selection) if np.sum(bin_selection) > 0 else 0
                        y_err = np.sqrt(bin_eff * (1 - bin_eff) / np.sum(bin_selection)) if np.sum(bin_selection) > 0 else 0
                        effs.append(bin_eff)
                        y_errs.append(y_err)

                    # Plot turn-on curve with error bars and spline interpolation
                    color = COLORS_DICT[f"{coll}_{coll_type}"]
                    linestyle = LINESTYLES_DICT[f"{coll}_{coll_type}"]
                    label = '{}, {} GeV'.format(LABELS_DICT[f"{coll}_{coll_type}"], np.round(wp))
                    ax.errorbar(
                        bin_centers, effs,
                        xerr=x_errs, yerr=y_errs,
                        fmt='o', capsize=3, color=color, label=label
                    )
                    spl = make_interp_spline(bin_centers, effs, k=3)
                    x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 500)
                    y_smooth = spl(x_smooth)
                    ax.plot(x_smooth, y_smooth, color=color, linestyle=linestyle)

            # Unify all collections in one plot
            x_lim = bins[bins >= bin_cut][0] if np.max(bins) >= bin_cut else bins[-1]
            ax.legend(title=PROCS_DICT[proc], fontsize=40, title_fontsize=40)
            ax.set_xlim(0, x_lim)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel(f"{LABELS_DICT[t]} [GeV]", fontsize=44)
            ax.set_ylabel(f"Eff (L1 rate at {r} kHz)", fontsize=44)
            fig.savefig(f"{plot_dir}/turn_on_curve_{t}_{r}kHz.pdf", bbox_inches='tight')
            fig.savefig(f"{plot_dir}/turn_on_curve_{t}_{r}kHz.png", bbox_inches='tight')
            plt.close(fig)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help='path to fast puppi files')
    parser.add_argument('-m', '--model', default='output/baseline', help='model output directory')
    parser.add_argument('-o', '--observables', nargs='+', default=['jet1', 'jet2', 'ht30'], help='List of observables to plot turn on curves for')
    parser.add_argument('-s', '--signal', nargs='+', default=['TT_PU200', 'QCD_Pt15To3000_PU200'], help='Whether to include signal processes in the turn on curves')
    args = parser.parse_args()

    procs = ['MinBias_PU200'] + args.signal
    proc_collections = load_collections(procs, args.input)
    plot_path = f"{args.model}/plots/emulation_regression"
    os.makedirs(plot_path, exist_ok=True)

    # Derive Rates
    target_rates =  [10, 20, 50, 100]
    for coll in COLLECTION_KEYS:
        for obj in args.observables:
            wps_raw = get_rate_wps(proc_collections['MinBias_PU200'][coll]['raw'], target_rates, obj)
            wps_jecs = get_rate_wps(proc_collections['MinBias_PU200'][coll]['jecs'], target_rates, obj)
            for r in target_rates:
                proc_collections['MinBias_PU200'][coll][f'wp_{obj}_{r}_raw'] = wps_raw[r]
                proc_collections['MinBias_PU200'][coll][f'wp_{obj}_{r}_jecs'] = wps_jecs[r]

    # Plot Turn on cuves
    for s in args.signal:
        turn_on_curve(proc_collections[s], proc_collections['MinBias_PU200'], s, args.observables, target_rates, plot_dir=f"{plot_path}/{s}/turn_ons")


