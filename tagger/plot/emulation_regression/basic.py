import uproot
import numpy as np
import awkward as ak
import os
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.colors import LogNorm
from extras import N_BUNCHES, REVOLUTION_FREQUENCY, MINBIAS_RATE, PT_BINS, LABELS_DICT, COLORS_DICT, PROCS_DICT, COLLECTION_KEYS
from scipy.stats import norm
from coffea.nanoevents.methods import vector

# style from tagger
import tagger.plot.style as style
style.set_style()

def get_rms(truth_pt, reco_pt, pt_ratio):

    # Calculate the regressed pt
    regressed_pt = np.multiply(reco_pt, pt_ratio)

    # Get the residuals
    uncorrected_res = reco_pt - truth_pt
    regressed_res = regressed_pt - truth_pt

    uncorr_response = reco_pt / truth_pt
    reg_response = regressed_pt / truth_pt

    rms_uncorr = []
    rms_reg = []
    rms_uncorr_err = []
    rms_reg_err = []

    # Loop over the pT ranges
    for i in range(len(PT_BINS) - 1):
        pt_min = PT_BINS[i]
        pt_max = PT_BINS[i + 1]

        selection = (truth_pt > pt_min) & (truth_pt < pt_max)

        sigma_uncorr = np.std(uncorrected_res[selection]) / uncorr_response[selection].mean()
        sigma_reg = np.std(regressed_res[selection]) / reg_response[selection].mean()

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


def rms(truth_pt_test1, reco_pt_test1, pt_ratio1, jet_collection1,
        truth_pt_test2, reco_pt_test2, pt_ratio2, jet_collection2,
        plot_dir):

    # pT coordinate points for plotting
    proc = plot_dir.split("/")[-1]
    pt_points = [np.mean((PT_BINS[i], PT_BINS[i + 1])) for i in range(len(PT_BINS) - 1)]

    def plot_rms(uncorrected_rms1, regressed_rms1, uncorrected_rms_err1, regressed_rms_err1, jet_collection1,
                 uncorrected_rms2, regressed_rms2, uncorrected_rms_err2, regressed_rms_err2, jet_collection2,
                 proc):

        for l1 in ['log', 'linear']:
            for l2 in ['log', 'linear']:
                # Plot the response
                fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
                hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.CMSHEADER_SIZE)
                ax.errorbar(
                    pt_points,
                    uncorrected_rms1,
                    yerr=uncorrected_rms_err1,
                    fmt='o',
                    label=r"{}".format(LABELS_DICT[jet_collection1]),
                    capsize=4,
                    ms=10,
                    elinewidth=3,
                    color="mediumpurple",
                )
                ax.errorbar(
                    pt_points,
                    uncorrected_rms2,
                    yerr=uncorrected_rms_err2,
                    fmt='o',
                    label=r"{}".format(LABELS_DICT[jet_collection2]),
                    capsize=4,
                    ms=10,
                    elinewidth=3,
                    color="coral",
                )
                ax.errorbar(
                    pt_points,
                    regressed_rms2,
                    yerr=regressed_rms_err2,
                    fmt='o',
                    label=r"JECs - {}".format(LABELS_DICT[jet_collection2]),
                    capsize=4,
                    ms=10,
                    elinewidth=3,
                    color="orangered",
                )

                ax.set_xlabel(r"Jet $p_T^{Gen}$ [GeV]")
                ax.set_ylabel(r"$\sigma\left(\frac{p_T^{\mathrm{Reco}} - p_T^{\mathrm{Gen}}}{p_T^{\mathrm{Gen}}}\right)$")
                ax.set_xscale(l1)
                ax.set_yscale(l2)
                ax.legend(title=PROCS_DICT[proc], fontsize=26)
                ax.grid(True, alpha=1, linestyle='-', lw=0.75)

                # Save the plot
                plt.savefig(f"{plot_dir}/residual_rms_{l1}_{l2}.pdf", bbox_inches='tight')
                plt.savefig(f"{plot_dir}/residual_rms_{l1}_{l2}.png", bbox_inches='tight')
                plt.close()

    # Inclusive rms
    uncorrected_rms1, regressed_rms1, uncorrected_rms_err1, regressed_rms_err1 = get_rms(truth_pt_test1, reco_pt_test1, pt_ratio1)
    uncorrected_rms2, regressed_rms2, uncorrected_rms_err2, regressed_rms_err2 = get_rms(truth_pt_test2, reco_pt_test2, pt_ratio2)
    plot_rms(
        uncorrected_rms1, regressed_rms1, uncorrected_rms_err1, regressed_rms_err1, jet_collection1,
        uncorrected_rms2, regressed_rms2, uncorrected_rms_err2, regressed_rms_err2, jet_collection2,
        proc
    )

    return

def plot_distribution(uncorrected1, uncorrected2, corrected2, genjets, coll1, coll2, proc, plot_dir):
    bins = np.linspace(0., 1200, 50)
    for i in ['Leading', 'Full']:
        fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
        if i == 'Leading':
            data1 = ak.sort(uncorrected1, axis=1, ascending=False)[:, :1]
            data2 = ak.sort(uncorrected2, axis=1, ascending=False)[:, :1]
            data2_corr = ak.sort(corrected2, axis=1, ascending=False)[:, :1]
            gen = ak.sort(genjets, axis=1, ascending=False)[:, :1]
        else:
            data1 = uncorrected1
            data2 = uncorrected2
            data2_corr = corrected2
            gen = genjets
        for data, c, t, color in [
            (data1, coll1, 'raw', "mediumpurple"),
            (data2, coll2, 'raw', "coral"),
            (data2_corr, coll2, 'jecs', "orangered"),
            (gen, "GenJets", '', "gray")
        ]:
            # histogram
            data = ak.to_numpy(ak.flatten(data, axis=None))
            data = np.clip(data, 0, np.max(bins) - 0.5)  # clip to bin edges to avoid outliers dominating the plot
            hist, edges = np.histogram(data, bins=bins)
            hist = hist / hist.sum()  # normalize to sum = 1
            label = LABELS_DICT[coll2 + '_' + t] if t else c
            ax.hist(data, bins=bins, weights=np.ones_like(data)/len(data),
                histtype='step', label=label, color=color, linewidth=2)

        ax.set_xlabel(f"{i} Jet $p_T$ [GeV]")
        ax.set_ylabel("Fraction")
        ax.set_yscale('log')
        ax.set_xlim(0, np.max(bins) + 50)
        ax.legend(fontsize=26, title=PROCS_DICT[proc])
        ax.grid(True, alpha=1, linestyle='-', lw=0.75)

        outpath = f"{plot_dir}"
        os.makedirs(outpath, exist_ok=True)
        fig.savefig(f"{outpath}/{i}_pt_distribution.pdf", bbox_inches='tight')
        fig.savefig(f"{outpath}/{i}_pt_distribution.png", bbox_inches='tight')

def plot_response_bin(uncorrected_response1, regressed_response1,
                      uncorrected_response2, regressed_response2,
                      coll1, coll2, bins, xlabel,
                      plot_dir):

    for i in range(len(uncorrected_response1)):
        plt.figure(figsize=style.FIGURE_SIZE)

        # Add a dummy line for legend explaining mean/median
        plt.plot([], [], color='k', linestyle='-', label='Mean')
        plt.plot([], [], color='k', linestyle=':', label='Median')
        if 'pt' in plot_dir.lower():
            bins = np.linspace(0, PT_BINS[i+1] + 200, 40)

        for data, label, color in [
            (uncorrected_response1[i], coll1, "mediumpurple"),
            (regressed_response1[i], f"JECs - {coll1}", "indigo"),
            (uncorrected_response2[i], coll2, "coral"),
            (regressed_response2[i], f"JECs - {coll2}", "orangered")
        ]:
            # histogram
            data = np.clip(data, bins[2], bins[-3])  # clip to bin edges to avoid outliers dominating the plot
            hist, edges = np.histogram(data, bins=bins)
            hist = hist / hist.sum()  # normalize to sum = 1
            plt.step(edges[:-1], hist, where='post', label=label, color=color, linewidth=2)

            # mean and median
            mean_val = np.mean(data)
            median_val = np.median(data)

            plt.axvline(mean_val, color=color, linestyle='-', linewidth=2)
            plt.axvline(median_val, color=color, linestyle=':', linewidth=2)

        plt.xlabel(xlabel)
        plt.ylabel("Fraction")
        if 'pt' in plot_dir.lower() and len(data) > 1:
            plt.yscale('log')
        plt.title(f"{PT_BINS[i]} < pT_truth < {PT_BINS[i+1]}")
        plt.legend(fontsize=26)
        plt.grid(True, alpha=1, linestyle='-', lw=0.75)

        outpath = f"{plot_dir}"
        os.makedirs(outpath, exist_ok=True)
        plt.savefig(f"{outpath}/response_bin_{PT_BINS[i]}_{PT_BINS[i+1]}.pdf", bbox_inches='tight')
        plt.savefig(f"{outpath}/response_bin_{PT_BINS[i]}_{PT_BINS[i+1]}.png", bbox_inches='tight')
        plt.close()


def plot_response_ridge_outline(uncorrected_response1, regressed_response1,
                               uncorrected_response2, regressed_response2,
                               coll1, coll2, bins, xlabel, plot_dir):

    n_bins = len(uncorrected_response1)
    y_offsets = np.arange(n_bins)

    plt.figure(figsize=(12, 20))

    for i in range(n_bins):
        offset = y_offsets[i]

        datasets = [
            (uncorrected_response1[i], coll1, "mediumpurple"),
            (regressed_response1[i], f"JECs - {coll1}", "indigo"),
            (uncorrected_response2[i], coll2, "coral"),
            (regressed_response2[i], f"JECs - {coll2}", "orangered"),
        ]

        hists = []
        max_height = 0

        # Compute histograms first (for shared normalization)
        for data, _, _ in datasets:
            data = np.clip(data, bins[2], bins[-3])
            hist, edges = np.histogram(data, bins=bins)
            hist = hist / hist.sum() if hist.sum() > 0 else hist
            hists.append(hist)
            max_height = max(max_height, hist.max())

        centers = edges[:-1]  # for step plotting
        scale = 1.0 / (max_height + 1e-6)

        # Plot each distribution
        for (data, label, color), hist in zip(datasets, hists):
            hist = hist * scale

            plt.step(centers, hist + offset,
                     where='post',
                     color=color,
                     linewidth=2,
                     label=label if i == 0 else None)

            # Mean / median
            mean_val = np.mean(data)
            median_val = np.median(data)

            plt.plot([mean_val, mean_val], [offset, offset + 1],
                     color=color, linestyle='-', linewidth=1)
            plt.plot([median_val, median_val], [offset, offset + 1],
                     color=color, linestyle=':', linewidth=1)

    # Vertical reference at 0
    plt.axvline(0, color='k', linestyle='--', linewidth=1)

    # Y-axis labels
    plt.yticks(y_offsets,
               [f"{PT_BINS[i]}–{PT_BINS[i+1]}" for i in range(n_bins)])

    plt.xlabel(xlabel)
    plt.ylabel(r"Gen $p_T$ bins")

    plt.legend(loc='lower right',
           fontsize=26,
           frameon=True,
           facecolor='white',
           edgecolor='black',
           framealpha=1.0)
    plt.grid(alpha=1, linestyle='-', lw=0.75)

    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f"{plot_dir}/response_ridge_outline.png", bbox_inches='tight')
    plt.savefig(f"{plot_dir}/response_ridge_outline.pdf", bbox_inches='tight')
    plt.close()

def get_response(truth_pt, reco_pt, pt_ratio, reduce):

    # Calculate the regressed pt
    regressed_pt = np.multiply(reco_pt, pt_ratio)

    # to calculate response
    uncorrected_response = []
    regressed_response = []
    uncorrected_errors = []
    regressed_errors = []
    uncorrected_response_bins = []
    regressed_response_bins = []
    raw_pt_bins = []
    corrected_pt_bins = []

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

        uncorrected_response_bins.append(uncorrected_response_bin)
        regressed_response_bins.append(regressed_response_bin)
        raw_pt_bins.append(reco_pt[selection])
        corrected_pt_bins.append(regressed_pt[selection])

    return uncorrected_response, regressed_response, uncorrected_errors, regressed_errors, [uncorrected_response_bins, regressed_response_bins], [raw_pt_bins, corrected_pt_bins]

def response(truth_pt_test1, reco_pt_test1, pt_ratio1, jet_collection1,
             truth_pt_test2, reco_pt_test2, pt_ratio2, jet_collection2,
             outpath):
    # pT coordinate points for plotting
    proc = outpath.split("/")[-1]
    pt_points = [np.mean((PT_BINS[i], PT_BINS[i + 1])) for i in range(len(PT_BINS) - 1)]

    def plot_response(uncorrected_response1, regressed_response1, uncorrected_errors1, regressed_errors1, jet_collection1,
                      uncorrected_response2, regressed_response2, uncorrected_errors2, regressed_errors2, jet_collection2,
                      plot_name, proc):

        # Plot the response
        for l1 in ['log', 'linear']:
            for l2 in ['log', 'linear']:
                fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
                hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.CMSHEADER_SIZE)
                bin_type = "Mean" if "mean" in plot_name else "Median"
                ax.errorbar(
                    pt_points,
                    uncorrected_response1,
                    yerr=uncorrected_errors1,
                    fmt='o',
                    label=LABELS_DICT[jet_collection1],
                    capsize=4,
                    ms=10,
                    elinewidth=3,
                    color="mediumpurple",
                )
                ax.errorbar(
                    pt_points,
                    uncorrected_response2,
                    yerr=uncorrected_errors2,
                    fmt='o',
                    label=LABELS_DICT[jet_collection2],
                    capsize=4,
                    ms=10,
                    elinewidth=3,
                    color="coral",
                )
                ax.errorbar(
                    pt_points,
                    regressed_response2,
                    yerr=regressed_errors2,
                    fmt='o',
                    label="JECs - " + LABELS_DICT[jet_collection2],
                    capsize=4,
                    ms=10,
                    elinewidth=3,
                    color="orangered",
                )

                # Add eta range label
                ax.text(
                    0.03, 0.98,
                    rf"0 < |$\eta$| < 2.4",
                    transform=ax.transAxes,
                    fontsize=32,
                    verticalalignment="top",
                )

                ax.set_xlabel(r"Jet $p_T^{Gen}$ [GeV]")
                ax.set_ylabel(f"Response {bin_type}(L1/Gen)")
                ax.legend(title=PROCS_DICT[proc], fontsize=26)
                ax.set_xscale(l1)
                ax.set_yscale(l2)
                ax.grid(alpha=1, linestyle='-', lw=0.75)
                ax.set_xlim(0, 1000)
                ax.set_ylim(0.6, 1.7)

                # Save the plot
                plt.savefig(f"{plot_name}_{l1}_{l2}.pdf", bbox_inches='tight', transparent=True)
                plt.savefig(f"{plot_name}_{l1}_{l2}.png", bbox_inches='tight', transparent=True)
                plt.close()

    # Inclusive response
    for reduce in ["mean", "median"]:
        uncorrected_response1, regressed_response1, uncorrected_errors1, regressed_errors1, hist_info1, pt_info1 = get_response(
            truth_pt_test1, reco_pt_test1, pt_ratio1, reduce)

        uncorrected_response2, regressed_response2, uncorrected_errors2, regressed_errors2, hist_info2, pt_info2 = get_response(
            truth_pt_test2, reco_pt_test2, pt_ratio2, reduce)

        plot_response_bin(
            hist_info1[0], hist_info1[1],
            hist_info2[0], hist_info2[1],
            LABELS_DICT[jet_collection1], LABELS_DICT[jet_collection2],
            bins=np.linspace(0., 5, 20), xlabel="Response (L1/Gen)",
            plot_dir=os.path.join(outpath, f"response_hists_{reduce}"))

        plot_response_bin(
            pt_info1[0], pt_info1[1],
            pt_info2[0], pt_info2[1],
            LABELS_DICT[jet_collection1], LABELS_DICT[jet_collection2],
            bins=np.linspace(0., 1000, 50), xlabel=r"Jet $p_{T}$ [GeV]",
            plot_dir=os.path.join(outpath, f"response_pt_hists_{reduce}"))

        plot_response_ridge_outline(
            hist_info1[0], hist_info1[1],
            hist_info2[0], hist_info2[1],
            LABELS_DICT[jet_collection1], LABELS_DICT[jet_collection2],
            bins=np.linspace(0., 5, 20), xlabel="Response (L1/Gen)",
            plot_dir=os.path.join(outpath, f"response_hists_{reduce}"))

        plot_response_ridge_outline(
            pt_info1[0], pt_info1[1],
            pt_info2[0], pt_info2[1],
            LABELS_DICT[jet_collection1], LABELS_DICT[jet_collection2],
            bins=np.linspace(0., 1000, 50), xlabel=r"Jet $p_{T}$ [GeV]",
            plot_dir=os.path.join(outpath, f"response_pt_hists_{reduce}"))

        plot_response(
            uncorrected_response1,
            regressed_response1,
            uncorrected_errors1,
            regressed_errors1,
            jet_collection1,
            uncorrected_response2,
            regressed_response2,
            uncorrected_errors2,
            regressed_errors2,
            jet_collection2,
            plot_name=os.path.join(outpath, f"response_{reduce}"),
            proc=proc
        )

    return

def distribution_heatmaps(l1jets, plot_dir):
    # 2D histogram of gen vs reco pt
    ratios = []
    for coll in COLLECTION_KEYS:
        t = 'raw' if coll == 'scPuppiL1TSC4NGJetJets' else 'jecs'
        jets = l1jets[coll][t]
        pt_bins = np.linspace(0, 1000, 50)
        gen_pt = ak.to_numpy(np.clip(ak.flatten(jets.genpt, axis=None), 0, 1000))
        reco_pt = ak.to_numpy(np.clip(ak.flatten(jets.pt, axis=None), 0, 1000))
        fig, ax = plt.subplots(figsize=(22, 25))
        h = ax.hist2d(gen_pt, reco_pt, bins=pt_bins, norm=LogNorm())
        ratios.append(h)
        plt.close()
        y_label = LABELS_DICT[f'{coll}_{t}']
        plot_heatmap(h[0][::-1], y_label, 'Entries', plot_dir)
    plot_heatmap(ratios[0][0][::-1] / ratios[1][0][::-1],
        r'Reco Jet $p_{T}$',
        f'Ratio {LABELS_DICT[COLLECTION_KEYS[0]]} vs {LABELS_DICT[COLLECTION_KEYS[1]+"_jecs"]}',
        plot_dir)

def plot_heatmap(ratio, y_label, label, plot_dir):
    from matplotlib.colors import TwoSlopeNorm

    fig, ax = plt.subplots(figsize=(22, 25))
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.CMSHEADER_SIZE)

    max_dev = max(abs(ratio.min() - 1), abs(ratio.max() - 1))
    norm = TwoSlopeNorm(vmin=1 - max_dev, vcenter=1, vmax=1 + max_dev)

    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color='lightgray')

    im = ax.imshow(ratio, cmap=cmap, norm=norm)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_aspect('equal')
    ax.set_xlabel('GenJet $p_T$')
    ax.set_ylabel(y_label)

    cbar = fig.colorbar(im, ax=ax, shrink=0.686)
    cbar.set_label(label, fontsize=40)

    plt.savefig(f"{plot_dir}/ratio_heatmap.pdf", bbox_inches='tight')





