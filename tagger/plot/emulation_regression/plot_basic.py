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
from load_collections import load_collections
from matplotlib.colors import LogNorm

# plotting imports
from extras import LABELS_DICT, COLORS_DICT, PROCS_DICT, COLLECTION_KEYS

# style from tagger
import tagger.plot.style as style
from tagger.plot.common import to_coffea, PT_BINS

style.set_style()

# Plotting functions
# Residuals
def get_rms(truth_pt, reco_pt, pt_ratio):

    # Calculate the regressed pt
    regressed_pt = np.multiply(reco_pt, pt_ratio)

    # Get the residuals
    uncorrected_res = reco_pt - truth_pt
    regressed_res = regressed_pt - truth_pt

    rms_uncorr = []
    rms_reg = []
    rms_uncorr_err = []
    rms_reg_err = []

    # Loop over the pT ranges
    for i in range(len(PT_BINS) - 1):
        pt_min = PT_BINS[i]
        pt_max = PT_BINS[i + 1]

        selection = (truth_pt > pt_min) & (truth_pt < pt_max)

        sigma_uncorr = np.std(uncorrected_res[selection]) / truth_pt[selection].mean()
        sigma_reg = np.std(regressed_res[selection]) / truth_pt[selection].mean()

        # Get the errors for the standard deviation
        # Standard error of the standard deviation for a normal distribution
        n_uncorr = len(uncorrected_res[selection])
        n_reg = len(regressed_res[selection])

        if n_uncorr <= 1 or n_reg <= 1:
            sigma_uncorr_err = sigma_uncorr
            sigma_reg_err = sigma_reg
        else:
            sigma_uncorr_err = sigma_uncorr / np.sqrt(2 * (n_uncorr - 1))
            sigma_reg_err = sigma_reg / np.sqrt(2 * (n_reg - 1))

        rms_uncorr.append(sigma_uncorr)
        rms_reg.append(sigma_reg)
        rms_uncorr_err.append(sigma_uncorr_err)
        rms_reg_err.append(sigma_reg_err)

    return rms_uncorr, rms_reg, rms_uncorr_err, rms_reg_err


def rms(procs_dict, proc, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    jet_coll1, jet_coll2 = procs_dict[COLLECTION_KEYS[0]], procs_dict[COLLECTION_KEYS[1]]
    jet_coll1 = [jet_coll1['raw'].genpt, jet_coll1['raw'].pt, jet_coll1['pt_ratio']]
    coll1 = [ak.to_numpy(ak.flatten(i)) for i in jet_coll1]
    jet_coll2 = [jet_coll2['raw'].genpt, jet_coll2['raw'].pt, jet_coll2['pt_ratio']]
    coll2 = [ak.to_numpy(ak.flatten(i)) for i in jet_coll2]

    # pT coordinate points for plotting
    pt_points = [np.mean((PT_BINS[i], PT_BINS[i + 1])) for i in range(len(PT_BINS) - 1)]
    os.makedirs(plot_dir, exist_ok=True)

    def plot_rms(rms_colls, proc, outpath):

        for l1 in ['log', 'linear']:
            for l2 in ['log', 'linear']:
                # Plot the response
                fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
                hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.CMSHEADER_SIZE)
                ax.errorbar(
                    pt_points,
                    rms_colls['scPuppiL1TSC4NGJetJets'][0],
                    yerr=rms_colls['scPuppiL1TSC4NGJetJets'][2],
                    fmt='o',
                    label=r"{}".format(LABELS_DICT["scPuppiL1TSC4NGJetJets_raw"]),
                    capsize=4,
                    ms=10,
                    elinewidth=3,
                    color=COLORS_DICT["scPuppiL1TSC4NGJetJets_raw"],
                )
                ax.errorbar(
                    pt_points,
                    rms_colls['scPuppiExtendedJets'][0],
                    yerr=rms_colls['scPuppiExtendedJets'][2],
                    fmt='o',
                    label=LABELS_DICT["scPuppiExtendedJets_raw"],
                    capsize=4,
                    ms=10,
                    elinewidth=3,
                    color=COLORS_DICT["scPuppiExtendedJets_raw"],
                )
                ax.errorbar(
                    pt_points,
                    rms_colls['scPuppiExtendedJets'][1],
                    yerr=rms_colls['scPuppiExtendedJets'][3],
                    fmt='o',
                    label=LABELS_DICT["scPuppiExtendedJets_jecs"],
                    capsize=4,
                    ms=10,
                    elinewidth=3,
                    color=COLORS_DICT["scPuppiExtendedJets_jecs"],
                )

                ax.set_xlabel(r"Jet $p_T^{Gen}$ [GeV]")
                ax.set_ylabel(r"$\sigma(p_T^{\mathrm{L1}} - p_T^{\mathrm{Gen}})\, / \, \mathrm{Mean}(p_T^{\mathrm{Gen}})$")
                ax.set_xscale(l1)
                ax.set_yscale(l2)
                ax.legend(title=PROCS_DICT[proc], fontsize=35, title_fontsize=35, loc='upper center')
                ax.grid(True, alpha=1, linestyle='-', lw=0.75)

                # Save the plot
                plt.savefig(f"{outpath}/residual_rms_x{l1}_y{l2}.pdf", bbox_inches='tight')
                plt.savefig(f"{outpath}/residual_rms_x{l1}_y{l2}.png", bbox_inches='tight')
                plt.close()

    # Inclusive rms
    colls_rms = {}
    coll1_rms = get_rms(coll1[0], coll1[1], coll1[2])
    coll2_rms = get_rms(coll2[0], coll2[1], coll2[2])
    colls_rms[COLLECTION_KEYS[0]] = coll1_rms
    colls_rms[COLLECTION_KEYS[1]] = coll2_rms
    plot_rms(colls_rms, proc, plot_dir)
    return

# Distributions
def plot_distribution(proc_collection, proc, plot_dir):
    bins = np.linspace(0., 1350, 55)
    for i in ['Leading', 'Full']:
        fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
        if i == 'Leading':
            data1 = ak.sort(proc_collection['scPuppiL1TSC4NGJetJets']['raw'], axis=1, ascending=False)[:, :1]
            data2 = ak.sort(proc_collection['scPuppiExtendedJets']['raw'], axis=1, ascending=False)[:, :1]
            data2_corr = ak.sort(proc_collection['scPuppiExtendedJets']['jecs'], axis=1, ascending=False)[:, :1]
            gen = ak.sort(proc_collection['genjets'], axis=1, ascending=False)[:, :1]
        else:
            data1 = proc_collection['scPuppiL1TSC4NGJetJets']['raw']
            data2 = proc_collection['scPuppiExtendedJets']['raw']
            data2_corr = proc_collection['scPuppiExtendedJets']['jecs']
            gen = proc_collection['genjets']
        for data, c in [
            (data1, 'scPuppiL1TSC4NGJetJets_raw'),
            (data2, 'scPuppiExtendedJets_raw'),
            (data2_corr, 'scPuppiExtendedJets_jecs'),
            (gen, 'genjets')
        ]:
            # histogram
            data = ak.to_numpy(ak.flatten(data, axis=None))
            data = np.clip(data, 0, np.max(bins) - 0.5)  # clip to bin edges to avoid outliers dominating the plot
            ax.hist(data, bins=bins, weights=np.ones_like(data)/len(data),
                histtype='step', label=LABELS_DICT[c], color=COLORS_DICT[c], linewidth=2)

        ax.set_xlabel(f"{i} Jet $p_T$ [GeV]")
        ax.set_ylabel("Fraction")
        ax.set_yscale('log')
        ax.set_xlim(0, np.max(bins) + 50)
        ax.legend(fontsize=35, title=PROCS_DICT[proc], title_fontsize=35)
        ax.grid(True, alpha=1, linestyle='-', lw=0.75)

        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(f"{plot_dir}/{i}_pt_distribution.pdf", bbox_inches='tight')
        fig.savefig(f"{plot_dir}/{i}_pt_distribution.png", bbox_inches='tight')

# Response
def get_response(truth_pt, reco_pt, pt_ratio, reduce):

    # Calculate the regressed pt
    regressed_pt = np.multiply(reco_pt, pt_ratio)

    # to calculate response
    uncorrected_response = []
    regressed_response = []
    uncorrected_errors = []
    regressed_errors = []

    # Loop over the pT ranges
    for i in range(len(PT_BINS) - 1):
        pt_min = PT_BINS[i]
        pt_max = PT_BINS[i + 1]

        selection = (truth_pt > pt_min) & (truth_pt < pt_max)

        # Compute responses
        uncorrected_response_bin = reco_pt[selection] / truth_pt[selection]
        regressed_response_bin = regressed_pt[selection] / truth_pt[selection]

        # Append the mean response
        if reduce == "mean":
            uncorrected_response.append(np.mean(uncorrected_response_bin))
            regressed_response.append(np.mean(regressed_response_bin))
        elif reduce == "median":
            uncorrected_response.append(np.median(uncorrected_response_bin))
            regressed_response.append(np.median(regressed_response_bin))

        # Compute the standard deviation and uncertainty in the mean
        n_events = len(truth_pt[selection])

        if n_events > 0:
            uncorrected_std = np.std(uncorrected_response_bin)
            regressed_std = np.std(regressed_response_bin)

            uncorrected_errors.append(uncorrected_std / np.sqrt(n_events))
            regressed_errors.append(regressed_std / np.sqrt(n_events))
        else:
            # No events in bin
            uncorrected_errors.append(0)
            regressed_errors.append(0)

    return uncorrected_response, regressed_response, uncorrected_errors, regressed_errors

def response(procs_dict, proc, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    jet_coll1, jet_coll2 = procs_dict[COLLECTION_KEYS[0]], procs_dict[COLLECTION_KEYS[1]]
    jet_coll1 = jet_coll1['raw'].genpt, jet_coll1['raw'].pt, jet_coll1['pt_ratio']
    coll1 = [ak.to_numpy(ak.flatten(i)) for i in jet_coll1]
    jet_coll2 = jet_coll2['raw'].genpt, jet_coll2['raw'].pt, jet_coll2['pt_ratio']
    coll2 = [ak.to_numpy(ak.flatten(i)) for i in jet_coll2]

    # pT coordinate points for plotting
    pt_points = [np.mean((PT_BINS[i], PT_BINS[i + 1])) for i in range(len(PT_BINS) - 1)]

    def plot_response(responses_coll, reduce, plot_name, proc):

        # Plot the response
        for l1 in ['log', 'linear']:
            for l2 in ['log', 'linear']:
                fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
                hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.CMSHEADER_SIZE)
                ax.errorbar(
                    pt_points,
                    responses_coll['scPuppiL1TSC4NGJetJets'][0],
                    yerr=responses_coll['scPuppiL1TSC4NGJetJets'][2],
                    fmt='o',
                    label=LABELS_DICT['scPuppiL1TSC4NGJetJets_raw'],
                    capsize=4,
                    ms=10,
                    elinewidth=3,
                    color=COLORS_DICT['scPuppiL1TSC4NGJetJets_raw'],
                )
                ax.errorbar(
                    pt_points,
                    responses_coll['scPuppiExtendedJets'][0],
                    yerr=responses_coll['scPuppiExtendedJets'][2],
                    fmt='o',
                    label=LABELS_DICT['scPuppiExtendedJets_raw'],
                    capsize=4,
                    ms=10,
                    elinewidth=3,
                    color=COLORS_DICT['scPuppiExtendedJets_raw'],
                )
                ax.errorbar(
                    pt_points,
                    responses_coll['scPuppiExtendedJets'][1],
                    yerr=responses_coll['scPuppiExtendedJets'][3],
                    fmt='o',
                    label=LABELS_DICT['scPuppiExtendedJets_jecs'],
                    capsize=4,
                    ms=10,
                    elinewidth=3,
                    color=COLORS_DICT['scPuppiExtendedJets_jecs'],
                )

                ax.set_xlabel(r"Jet $p_T^{Gen}$ [GeV]")
                ax.set_ylabel(f"Response {reduce.capitalize()}(L1/Gen)")
                ax.legend(title=PROCS_DICT[proc], fontsize=40, title_fontsize=35)
                ax.set_xscale(l1)
                ax.set_yscale(l2)
                ax.grid(alpha=1, linestyle='-', lw=0.75)
                ax.set_xlim(0, 1000)
                ax.set_ylim(0.6, 1.7)

                # Save the plot
                plt.savefig(f"{plot_name}_x{l1}_y{l2}.pdf", bbox_inches='tight', transparent=True)
                plt.savefig(f"{plot_name}_x{l1}_y{l2}.png", bbox_inches='tight', transparent=True)
                plt.close()

    # Inclusive response
    for reduce in ["mean", "median"]:
        responses_coll = {}
        coll1_response = get_response(coll1[0], coll1[1], coll1[2], reduce)
        coll2_response = get_response(coll2[0], coll2[1], coll2[2], reduce)
        responses_coll[COLLECTION_KEYS[0]] = coll1_response
        responses_coll[COLLECTION_KEYS[1]] = coll2_response

        plot_response(responses_coll, reduce, plot_name=os.path.join(plot_dir, f"response_{reduce}"), proc=proc)

    return

# Heatmaps
def distribution_heatmaps(l1jets, plot_dir):
    # 2D histogram of gen vs reco pt
    os.makedirs(plot_dir, exist_ok=True)
    ratios = []
    for coll in COLLECTION_KEYS:
        t = 'raw' if coll == 'scPuppiL1TSC4NGJetJets' else 'jecs'
        jets = l1jets[coll][t]
        pt_bins = np.linspace(0, 1000, 50)
        matched_jets = (jets.genpt != 0)
        gen_pt = ak.to_numpy(np.clip(ak.flatten(jets.genpt[matched_jets], axis=None), 0, 1000))
        reco_pt = ak.to_numpy(np.clip(ak.flatten(jets.pt[matched_jets], axis=None), 0, 1000))
        fig, ax = plt.subplots(figsize=(22, 25))
        h = ax.hist2d(gen_pt, reco_pt, bins=pt_bins)
        ratios.append(h)
        plt.close()
        y_label = LABELS_DICT[f'{coll}_{t}']
        plot_heatmap(h[0][::-1], y_label, 'Entries', f"{plot_dir}/{coll}_{t}_heatmap")

    ratio = np.where(ratios[1][0][::-1] != 0, ratios[0][0][::-1] / ratios[1][0][::-1], 0)
    plot_heatmap(ratio,
        r'Reco Jet $p_{T}$',
        f'Ratio {LABELS_DICT[COLLECTION_KEYS[0]]} vs {LABELS_DICT[COLLECTION_KEYS[1]+"_jecs"]}',
        f"{plot_dir}/ratio_heatmap", plot_ratio=True)

def plot_heatmap(ratio, y_label, label, plot_dir, plot_ratio=False):
    from matplotlib.colors import TwoSlopeNorm
    fig, ax = plt.subplots(figsize=(22, 25))
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.CMSHEADER_SIZE)

    if plot_ratio:
        max_dev = max(abs(ratio.min() - 1), abs(ratio.max() - 1))
        norm = TwoSlopeNorm(vmin=1 - max_dev, vcenter=1, vmax=1 + max_dev)
        cmap = plt.cm.coolwarm.copy()
        cmap.set_bad(color='lightgray')
    else:
        plot_data = np.where(ratio <= 0, np.nan, ratio)
        norm = LogNorm()
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='lightgray')

    im = ax.imshow(ratio, cmap=cmap, norm=norm, extent=[0, 1000, 0, 1000])
    ax.set_aspect('equal')
    ax.set_xlabel('GenJet $p_T$')
    ax.set_ylabel(y_label)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label(label, fontsize=40)

    plt.savefig(f"{plot_dir}.pdf", bbox_inches='tight')
    plt.savefig(f"{plot_dir}.png", bbox_inches='tight')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help='path to fast puppi files')
    parser.add_argument('-m', '--model', default='output/baseline', help='model output directory')
    parser.add_argument('-p', '--processes', nargs='+', default=['TT_PU200','QCD_Pt15To3000_PU200'], help='Processes to plot')
    args = parser.parse_args()

    proc_collections = load_collections(args.processes, args.input)
    plot_path = f"{args.model}/plots/emulation_regression"
    os.makedirs(plot_path, exist_ok=True)

    # Basic plots
    for p in args.processes:
        jet_collections = proc_collections[p]

        # 2d heatmaps
        distribution_heatmaps(jet_collections, plot_dir=f"{plot_path}/{p}/heatmaps")

        # # Plot pT distributions
        plot_distribution(jet_collections, p, plot_dir=f"{plot_path}/{p}/distributions")

        # # response and rms
        response(proc_collections[p], p, f"{plot_path}/{p}/response")
        rms(proc_collections[p], p, f"{plot_path}/{p}/rms")
