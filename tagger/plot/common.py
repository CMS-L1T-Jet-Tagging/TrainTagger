# flake8: noqa
# Plotting
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from tagger.plot import style

style.set_style()
# GLOBAL VARIABLES TO USE ACROSS PLOTTING TOOLS

N_BUNCHES = 2760
REVOLUTION_FREQUENCY = 11246
MINBIAS_RATE = N_BUNCHES * REVOLUTION_FREQUENCY / 1000  # in kHz

# Define pT bins
PT_BINS = np.array([15, 17, 19, 22, 25, 30, 35, 40, 45, 50, 60, 76, 97, 122, 154, 195, 246, 311, 393, 496, 627, 792, 1000])

WPs_CMSSW = {
    # Tau working points as defined here
    # https://github.com/cms-sw/cmssw/blob/e9b58bef8b37e6113ba31a03429ffc4c300adb12/DataFormats/L1TParticleFlow/interface/PFTau.h#L16-L17
    'tau': 0.22,
    'tau_l1_pt': 34,
    # Seededcone reco pt cut
    # From these slides: https://indico.cern.ch/event/1380964/contributions/5852368/attachments/2841655/4973190/AnnualReview_2024.pdf
    'l1_pt_sc_barrel': 164,  # GeV
    'l1_pt_sc_endcap': 121,  # GeV
    # Slide 19 here: https://indico.cern.ch/event/1380964/contributions/5852368/attachments/2841655/4973190/AnnualReview_2024.pdf
    'btag': 2.32,
    'btag_l1_ht': 220,
}

# FUNCTIONS


def eta_region_selection(eta_array, eta_region):
    """
    eta range for barrel: |eta| < 1.5
    eta range for endcap: 1.5 < |eta| < 2.5

    return eta array selection
    """

    if eta_region == 'barrel': return np.abs(eta_array) < 1.5
    elif eta_region == 'endcap': return (np.abs(eta_array) > 1.5) & (np.abs(eta_array) < 2.5)
    #additional eta restriction on taus
    #slide 7 here https://indico.cern.ch/event/1380964/contributions/5852368/attachments/2841655/4973190/AnnualReview_2024.pdfa
    elif eta_region == 'tau-endcap' or eta_region =='tau_endcap': return (np.abs(eta_array) > 1.5) & (np.abs(eta_array) < 2.172)
    else: return np.abs(eta_array) > 0.0 #Select everything

def delta_r(eta1, phi1, eta2, phi2):
    """
    Calculate the delta R between two sets of eta and phi values.
    """
    delta_eta = eta1 - eta2
    delta_phi = phi1 - phi2

    # Ensure delta_phi is within -pi to pi
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(delta_eta**2 + delta_phi**2)

def find_rate(rate_list, target_rate = 14, RateRange = 0.05):

    idx_list = []

    for i in range(len(rate_list)):
        if target_rate - RateRange <= rate_list[i] <= target_rate + RateRange:
            idx_list.append(i)

    return idx_list

def plot_ratio(all_events, selected_events, plot=False):
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.CMSHEADER_SIZE)
    _, eff = selected_events.plot_ratio(
        all_events,
        rp_num_label="Selected events",
        rp_denom_label=r"All",
        rp_uncert_draw_type="bar",
        rp_uncertainty_type="efficiency",
    )

    return eff


def get_bar_patch_data(artists):
    x_data = [artists.bar.patches[i].get_x() for i in range(len(artists.bar.patches))]
    y_data = [artists.bar.patches[i].get_y() for i in range(len(artists.bar.patches))]
    err_data = [artists.bar.patches[i].get_height() for i in range(len(artists.bar.patches))]
    return x_data, y_data, err_data


def plot_2d(variable_one, variable_two, range_one, range_two, name_one, name_two, title):
    fig, ax = plt.subplots(1, 1, figsize=(style.FIGURE_SIZE[0] + 2, style.FIGURE_SIZE[1]))
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.CMSHEADER_SIZE)

    hist2d = ax.hist2d(
        variable_one, variable_two, range=(range_one, range_two), bins=50, norm=matplotlib.colors.LogNorm(), cmap='jet'
    )
    ax.set_xlabel(name_one)
    ax.set_ylabel(name_two)
    cbar = plt.colorbar(hist2d[3], ax=ax)
    cbar.set_label('a.u.')
    plt.suptitle(title)
    return fig


def plot_histo(variable, name, title, xlabel, ylabel, log = 'log', x_range=(0, 1), bins = 50):
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.CMSHEADER_SIZE)
    ## If we are histogramming by class and so want class colours
    if len(variable) > len(style.colours):
        colours = style.color_cycle
        linestyle = ['-' for i in range(len(variable))]
    else:
        colours = style.colours
        linestyle = style.LINESTYLES
    colour_list = []
    for i, histo in enumerate(variable):
        colour_list.append(colours[i])

    ax.hist(
            variable,
            bins=bins,
            range=x_range,
            histtype="step",
            stacked=False,
            color=[colours[i] for i in range(len(variable))],
            label=name,
            linewidth=style.LINEWIDTH - 1.5,
            linestyle=linestyle,
            density=True,
        )
    ax.grid(True)
    ax.set_yscale(log)
    ax.set_xlabel(xlabel, ha="right", x=1)
    ax.set_ylabel(ylabel, ha="right", y=1)
    ax.legend(loc='upper right')
    return fig


def plot_roc(
    modelsAndNames,
    truthclass,
    keys=["Emulation", "Tensorflow", "hls4ml"],
    labels=["CMSSW Emulation", "Tensorflow", "hls4ml"],
    title="None",
):
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.CMSHEADER_SIZE)

    for i, key in enumerate(keys):
        tpr = modelsAndNames[key]["ROCs"]["tpr"]
        fpr = modelsAndNames[key]["ROCs"]["fpr"]
        auc1 = modelsAndNames[key]["ROCs"]["auc"]
        ax.plot(
            tpr[truthclass],
            fpr[truthclass],
            label='%s Tagger, AUC = %.2f%%' % (labels[i], auc1[truthclass] * 100.0),
            color=style.colours[i],
            linestyle=style.LINESTYLES[i],
        )
    ax.semilogy()
    ax.set_xlabel("Signal Efficiency")
    ax.set_ylabel("Mistag Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.001, 1)
    ax.grid(True)
    ax.legend(loc='best')
    return fig


def x_vs_y(x, y, apply_light=True):
    if apply_light:
        s = x / (x + y)
        return s
    else:
        return x
