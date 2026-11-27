"""Compare what the old seed scripts draw with what tagger.eval +
tagger.plot draw, file by file.

Temporary (deleted in the last commit of the refactor). Run from the repo root:
    python tests/migration/compare_seeds.py -m output/baseline -n 300000 --seeds singleTau diTau diTauTopo
Old and new read the same number of entries (-n) from every sample. The old scripts run in-process
in the order CI runs them, with model.output_directory pointed to a scratch folder.
Writes the report to <model>/eval/migration_seeds.txt.
"""

import argparse
import filecmp
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import capture  # noqa: E402

TREE = "outnano/Jets"
MINBIAS = "data/MinBias_PU200.root"
VBF = "data/VBFHToTauTau_PU200.root"
BBTT = "data/GluGluHHTo2B2Tau_PU200.root"
BBBB = "data/GluGluHHTo4B_PU200.root"

COLLECTION = {
    "singleTau": "seedSingleTau",
    "diTau": "seedDiTau",
    "diTauTopo": "seedDiTauTopo",
    "bbbb": "seedBbbb",
    "bbtt": "seedBbtt",
}


def run_old(model_dir, seeds, n, out_root):
    sys.path.insert(
        0, os.path.join(HERE, "..", "..", "tagger", "plot")
    )  # the old scripts import "common"
    from tagger.model.common import fromFolder

    model = fromFolder(model_dir)
    model.output_directory = out_root
    capture.install(os.path.join(out_root, "plots"))
    for seed in seeds:
        if seed == "singleTau":
            import singleTau as m

            m.derive_tau_WPs(
                model, MINBIAS, n_entries=n, tree=TREE, cmssw_model=True
            )
            m.derive_tau_WPs(model, MINBIAS, n_entries=n, tree=TREE)
            m.eff_tau(model, VBF, n_entries=n, tree=TREE)
        elif seed == "diTau":
            import diTaus as m

            m.derive_diTaus_WPs(model, MINBIAS, n_entries=n, tree=TREE)
            m.plot_bkg_rate_ditau(model, MINBIAS, n_entries=n, tree=TREE)
            m.eff_ditau(
                model, VBF, n_entries=n, eta_region="barrel", tree=TREE
            )
            m.eff_ditau(
                model, VBF, n_entries=n, eta_region="tau_endcap", tree=TREE
            )
        elif seed == "diTauTopo":
            import diTaus_topo as m

            m.derive_diTaus_topo_WPs(model, MINBIAS, n_entries=n, tree=TREE)
            m.plot_bkg_rate_ditau_topo(model, MINBIAS, n_entries=n, tree=TREE)
            m.topo_eff(model, VBF, n_entries=n, tree=TREE, tag="vbf")
        elif seed == "bbbb":
            import bbbb as m

            m.derive_bbbb_WPs(
                model, MINBIAS, False, False, n_entries=n, tree=TREE
            )  # as CI: raw, all
            m.bbbb_eff(
                model, BBBB, MINBIAS, False, False, n_entries=n, tree=TREE
            )
        elif seed == "bbtt":
            import bbtt as m

            m.derive_rate(MINBIAS, model, n_entries=n, tree=TREE)
            for sel in ("tau", "all"):
                m.derive_bbtt_WPs(
                    model,
                    MINBIAS,
                    220,
                    sel,
                    BBTT,
                    target_rate=14,
                    n_entries=n,
                    tree=TREE,
                )
            for score, sel in (
                ("raw", "tau"),
                ("qg", "tau"),
                ("raw", "all"),
                ("qg", "all"),
            ):
                m.bbtt_eff_HT(
                    model,
                    BBTT,
                    score,
                    sel,
                    target_rate=14,
                    n_entries=n,
                    tree=TREE,
                )
        import matplotlib.pyplot as plt

        plt.close("all")


def run_new(model_dir, seeds, n, out_root):
    from tagger.eval import config as config_mod
    from tagger.eval.inputs.sources import Sources
    from tagger.eval.utils.cli import evaluate
    from tagger.plot.cli import ModelResults, draw
    from tagger.plot.registry import load_all

    # the CMSSW curves come from the -m cmssw results, at the same
    # entries
    cmssw_dir = os.path.join(out_root, "cmssw")
    config = config_mod.load(
        overrides=[
            f"SEED_N_ENTRIES={{'minbias': {n}, 'signal': {n}}}",
            f"CMSSW_MODEL_DIR={cmssw_dir}",
        ]
    )
    names = [COLLECTION[s] for s in seeds]
    evaluate(model_dir, names, config)
    evaluate(
        cmssw_dir,
        names,
        config,
        sources=Sources(cmssw_dir, config=config, pseudo="cmssw"),
    )
    load_all()
    capture.install(os.path.join(out_root, "plots"))
    for name in names:
        draw(
            name,
            [ModelResults(model_dir, "model", [name])],
            os.path.join(out_root, "plots"),
        )


def text_files(root):
    out = {}
    for dirpath, _, files in os.walk(os.path.join(root, "plots")):
        for f in files:
            if f.endswith(".txt"):
                path = os.path.join(dirpath, f)
                out[os.path.relpath(path, os.path.join(root, "plots"))] = path
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model", default="output/baseline")
    p.add_argument("-n", type=int, default=300000)
    p.add_argument(
        "--seeds", nargs="+", default=["singleTau", "diTau", "diTauTopo"]
    )
    p.add_argument("--scratch", default=None)
    args = p.parse_args()

    scratch = args.scratch or tempfile.mkdtemp(prefix="compare_seeds_")
    old_root, new_root = os.path.join(scratch, "old"), os.path.join(
        scratch, "new"
    )
    run_old(args.model, args.seeds, args.n, old_root)
    old = dict(capture.CAPTURED)
    capture.CAPTURED.clear()
    run_new(args.model, args.seeds, args.n, new_root)
    new = dict(capture.CAPTURED)

    details = {}
    import pickle

    with open(os.path.join(scratch, "captured.pkl"), "wb") as f:
        pickle.dump({"old": old, "new": new}, f)
    report = capture.compare(old, new)
    for key in report:  # how much was compared: drawn items and values
        if key in old and key in new and report[key] == "same":
            n_items = len(old[key])
            n_values = sum(a.size for _, arrays, _ in old[key] for a in arrays)
            report[key] = "same" if n_items else "same (nothing drawn)"
            details[key] = f"{n_items} items, {n_values} values"
    old_txt, new_txt = text_files(old_root), text_files(new_root)
    for key in sorted(set(old_txt) | set(new_txt)):
        if key not in new_txt or key not in old_txt:
            report[key] = "only old" if key in old_txt else "only new"
        else:
            report[key] = (
                "same"
                if filecmp.cmp(old_txt[key], new_txt[key], shallow=False)
                else "text differs"
            )
    lines = [
        f"{v:24s} {details.get(k, ''):26s} {k}"
        for k, v in sorted(
            report.items(), key=lambda kv: (kv[1] != "same", kv[0])
        )
    ]
    n_same = sum(v == "same" for v in report.values())
    summary = (
        f"{n_same}/{len(report)} files identical (n = {args.n} entries per"
        f" sample, seeds {' '.join(args.seeds)})"
    )
    text = "\n".join(
        lines + ["", summary, f"old: {old_root}", f"new: {new_root}"]
    )
    print(text)
    os.makedirs(os.path.join(args.model, "eval"), exist_ok=True)
    with open(
        os.path.join(args.model, "eval", "migration_seeds.txt"), "w"
    ) as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
