import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import pandas as pd
from collections import defaultdict
import os

# All information takend from:
# https://docs.google.com/spreadsheets/d/18Mt8yD6HuATaX-fOHkpTQqPARcF0Nvk069d46eVvplY/edit?gid=937856800#gid=937856800

def singlejet_pt_trigger(jet_pt, threshold=155):
    """
    L1_SingleJet180er2p5
    Rate = 33.51 kHz
    """
    return (jet_pt > threshold).astype(int)

def dijet_pt_deta_trigger(jet_pt1, jet_pt2, jet_eta1, jet_eta2, pt_threshold=105, max_eta=2.5, max_deta=1.6):
    """
    L1_DoubleJet150er2p5_dEtaMax1p6
    Rate = 16.57 kHz
    """
    return (
        (jet_pt1 > pt_threshold) & 
        (jet_pt2 > pt_threshold) &
        (np.abs(jet_eta1) < max_eta) &
        (np.abs(jet_eta2) < max_eta) &
        (np.abs(jet_eta1 - jet_eta2) < max_deta)
    ).astype(int)

def dijet_pt_trigger(jet_pt1, jet_pt2, jet_eta1, jet_eta2, threshold=133, max_eta=2.5):
    """
    L1_DoubleJet150er2p5
    Rate = 11.15 kHz
    """
    return (
        (jet_pt1 > threshold) & 
        (jet_pt2 > threshold) &
        (np.abs(jet_eta1) < max_eta) &
        (np.abs(jet_eta2) < max_eta)
    ).astype(int)

def ht_trigger(ht, threshold=290):
    """
    L1_HTT280er
    Rate = 10.00 kHz
    """
    accept = (ht > threshold).astype(int)
    return accept

def quadjet_ht_trigger(ht, jet_pt1, jet_pt2, jet_pt3, jet_pt4, jet_eta1, jet_eta2, jet_eta3, jet_eta4, thresholds=(320, 50, 40, 30, 30), max_eta=2.4):
    """
    From https://indico.cern.ch/event/1516922/contributions/6507127/attachments/3088463/5471242/annual-review-algos_v3.pdf slide 8:
    Offline = (400, 70, 55, 40, 40), Rate = 10kHz
    """
    accept = (
        (ht > thresholds[0]) &
        (jet_pt1 > thresholds[1]) &
        (jet_pt2 > thresholds[2]) &
        (jet_pt3 > thresholds[3]) &
        (jet_pt4 > thresholds[4]) &
        (np.abs(jet_eta1) < max_eta) &
        (np.abs(jet_eta2) < max_eta) &
        (np.abs(jet_eta3) < max_eta) &
        (np.abs(jet_eta4) < max_eta)
    ).astype(int)
    return accept


def get_trigger_turn_on(
        accept: np.ndarray, 
        x_values: np.ndarray,
        x_edges: np.ndarray = np.array([0, 20, 40, 60, 80, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500, 600, 700, 800, 1000]), 
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the trigger turn on curve for a given trigger function and x_edges.
    Returns the bin centers and the turn on values.
    """
    total = np.histogram(x_values, bins=x_edges)[0]
    passed = np.histogram(x_values[accept == 1], bins=x_edges)[0]
    efficiency = np.divide(passed, total, out=np.zeros_like(passed, dtype=float), where=total != 0)

    return efficiency, x_edges

def plot_turn_ons(
        trigger_infos: dict[str, dict],
        x_values: np.ndarray,
        x_edges: np.ndarray,
        x_variable_name: str = "Variable",
        output_file: str = None
    ):
    """"""
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

    fig, ax = plt.subplots(figsize=(5.5, 5))

    for trigger_name, trigger_info in trigger_infos.items():
        effs = trigger_info['efficiencies']
        ax.plot(
            x_centers, 
            effs, 
            label=f"{trigger_name}" + (f" ({trigger_info['rate']:.1f} kHz)" if 'rate' in trigger_info else ""), 
            color=trigger_info.get('color', None), 
            linestyle=trigger_info.get('linestyle', '-'),
            marker=trigger_info.get('marker', None)
        )

    ax.set_xlabel(x_variable_name)
    ax.set_ylabel('Efficiency')
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    ax2 = ax.twinx()
    ax2.hist(x_values, bins=x_edges, histtype='stepfilled', color='gray', alpha=0.3, density=True)
    ax2.set_yticks([], labels=[])

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)

    plt.close()