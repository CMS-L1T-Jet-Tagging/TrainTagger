r"""Make the plotter gallery of docs/plotting.md: one thumbnail per
plotter, from plots that tagger.plot drew.

Draw the plot folders first, then pass them; each thumbnail comes from
the first folder that has its source plot (THUMBNAILS below). The
gallery uses the baseline model with the cmssw, hls4ml and emulator
results and the FastPUPPI collection, and a short test training of
WeightedAverageSimpleModel for the loss history and the pT weights:

    python -m tagger.plot -m output/baseline -c jetClassifier -c jetPt \
        -c seeds -c emulation -c emulationRegression
    python -m tagger.plot -m output/wavg_short -c jetPt -c train
    python docs/make_gallery.py output/baseline/plots \
        output/wavg_short/plots

A thumbnail whose source plot is in none of the folders keeps its old
image.
"""

import argparse
import os
import sys

from PIL import Image

GALLERY_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "img", "plotters"
)

# Thumbnail size (pixels): each plot is scaled to fit and centred on a
# white square
SIZE = 360

# Gallery image (plotter name, or plotter and variant) -> source plot,
# relative to a plot folder
THUMBNAILS = {
    "curve_overlay": "training/roc_taus/ROC_taus_combined",
    "curve_overlay_errors": "training/response/inclusive_response",
    "histogram": "training/pt_hist",
    "grid_2d": "training/confusion_matrix",
    "stacked_bars": "training/shap_summary_class",
    "rate_grid": "physics/bbtt/bbtt_rate_raw_tau_14",
    "eff_points": "physics/bbbb/HH_eff_HT_raw_all",
    "ratio_map": "physics/tautau_topo/topo_vbf_eff_model_cmssw_ratio",
    "band_overlay": "physics/tautau_topo/tautau_topo_BkgRate",
    "scalar_vs_x": "training/eff_at_mistag/b_vs_qcd",
    "loss_history": (
        "training/loss_prune_low_magnitude_jet_id_output_loss_history"
    ),
    "hist_2d": "training/pt_weights/pt_weights_vs_pt",
    "mesh_panels": "physics/tautau_topo/topo_vbf_eff",
    "pairwise_2d": "emulation/b_score_2D",
    "turn_on": (
        "emulation_regression/TT_PU200/turn_ons/turn_on_curve_jet1_50kHz"
    ),
    "pt_map": (
        "emulation_regression/TT_PU200/heatmaps/"
        "scPuppiL1TSC4NGJetJets_raw_heatmap"
    ),
    "mass_peak": "emulation_regression/TT_PU200/masses/w_mass",
}


def thumbnail(path, size=SIZE):
    """A plot scaled to fit a white square, centred.

    Args:
        path: PNG file of the plot.
        size: Side of the square (pixels).

    Returns:
        PIL.Image.Image: The thumbnail.
    """
    image = Image.open(path).convert("RGB")
    scale = min(size / image.width, size / image.height)
    image = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.LANCZOS,
    )
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(
        image, ((size - image.width) // 2, (size - image.height) // 2)
    )
    return canvas


def find_source(plot_dirs, source):
    """The PNG of a source plot in the first plot folder that has it, or
    None."""
    for plot_dir in plot_dirs:
        path = os.path.join(plot_dir, source + ".png")
        if os.path.exists(path):
            return path
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "plot_dirs",
        nargs="+",
        help="plot folders written by tagger.plot, searched in order",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=GALLERY_DIR,
        help="gallery folder (default: docs/img/plotters)",
    )
    args = parser.parse_args(argv)

    os.makedirs(args.output, exist_ok=True)
    missing = []
    for name, source in THUMBNAILS.items():
        path = find_source(args.plot_dirs, source)
        if path is None:
            missing.append(f"{name} ({source})")
            continue
        thumbnail(path).save(
            os.path.join(args.output, name + ".png"), optimize=True
        )
    # empty cells of the last gallery row
    Image.new("RGB", (SIZE, SIZE), "white").save(
        os.path.join(args.output, "blank.png"), optimize=True
    )
    print(
        f"wrote {len(THUMBNAILS) - len(missing)} thumbnails to {args.output}"
    )
    if missing:
        print(
            "kept the old image of "
            + ", ".join(missing)
            + ": no such plot in the plot folders"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
