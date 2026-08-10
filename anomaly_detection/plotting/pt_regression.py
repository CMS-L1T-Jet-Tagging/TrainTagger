import numpy as np
import matplotlib.pyplot as plt

def response_vs_var(y_pred, y_true, var, edges):

    bin_centers = 0.5 * (edges[1:] + edges[:-1])
    bin_indices = np.digitize(var, edges) - 1
    bin_indices[bin_indices < 0] = 0
    bin_indices[bin_indices >= len(bin_centers)] = len(bin_centers) - 1

    r = np.divide(
        y_pred, 
        y_true, 
        out=np.full_like(y_pred, np.nan, dtype=float), 
        where=(y_true != 0)
    )
    response = np.zeros(len(bin_centers))
    response_unc = np.zeros(len(bin_centers))
    resolution = np.zeros(len(bin_centers))

    for i in range(len(bin_centers)):
        mask = bin_indices == i
        if np.any(mask):

            response[i] = np.mean(r[mask])
            response_unc[i] = np.std(r[mask]) / np.sqrt(np.sum(mask))
            resolution[i] = (np.percentile(r[mask], 84.1) - np.percentile(r[mask], 15.9)) / 2.0
        else:
            response[i] = np.nan
            response_unc[i] = np.nan
            resolution[i] = np.nan

    metrics = {
        'response': response,
        'response_unc': response_unc,
        'resolution': resolution,
    }

    return bin_centers, metrics


def plot_response_resolution_vs_var(
        class_info: dict,
        jet_pt_pred: np.ndarray,
        jet_pt_reco: np.ndarray,
        jet_pt_true: np.ndarray,
        class_array: np.ndarray,
        var: np.ndarray,
        var_edges: np.ndarray,
        var_name: str,
        output_file: str,
        logx: bool = False,
    ):
    fig, ax = plt.subplots(2, 1, figsize=(5.5, 5), sharex=True)

    for cls, info in class_info.items():

        if cls == "QCD":
            ls = ':'
        elif cls == "TT":
            ls = '-'
        else:
            continue

        cls_mask = (class_array == info['class'])
        jet_pt_reco_proc = jet_pt_reco[cls_mask]
        jet_pt_pred_proc = jet_pt_pred[cls_mask]
        jet_pt_true_proc = jet_pt_true[cls_mask]
        var_proc = var[cls_mask]

        var_centers, metrics_reg = response_vs_var(jet_pt_pred_proc, jet_pt_true_proc, var_proc, var_edges)
        _, metrics_l1 = response_vs_var(jet_pt_reco_proc, jet_pt_true_proc, var_proc, var_edges)

        ax[0].errorbar(var_centers, metrics_l1['response'], yerr=metrics_l1['response_unc'], marker='.', label=f'L1 ({info["label"]})', color='blue', ls=ls)
        ax[0].errorbar(var_centers, metrics_reg['response'], yerr=metrics_reg['response_unc'], marker='.', label=f'ML ({info["label"]})', color='red', ls=ls)

        ax[1].plot(var_centers, 100*metrics_l1['resolution'], marker='.', label=f'L1 ({info["label"]})', color='blue', ls=ls)
        ax[1].plot(var_centers, 100*metrics_reg['resolution'], marker='.', label=f'ML ({info["label"]})', color='red', ls=ls)

    if logx:
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')

    ax[0].hlines(1.0, var_edges[0], var_edges[-1], color='black', linestyle='--', lw=1)
    ax[0].set_xlim(var_edges[0], var_edges[-1])
    ax[0].set_ylim(0.5, 1.5)
    ax[0].set_ylabel('Response')
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[0].legend(fontsize=8, loc='upper right')

    ax[1].set_xlim(var_edges[0], var_edges[-1])
    ax[1].set_ylim(0, 50)
    ax[1].set_xlabel(var_name)
    ax[1].set_ylabel('Resolution (%)')
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()