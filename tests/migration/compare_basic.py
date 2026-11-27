"""Compare what the old basic() draws with what tagger.eval +
tagger.plot draw, file by file.

Temporary (deleted in the last commit of the refactor). Run from the repo root:
    python tests/migration/compare_basic.py -m output/baseline -sig TT_PU200
Writes the old plots to <scratch>/old and the new ones to <scratch>/new, prints one line per file and
a summary, and writes the report to <model>/eval/migration_basic.txt.
"""

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import capture  # noqa: E402


def pin_shap_seed(seed):
    """The old plot_shaply lets shap draw a random seed per call; pin
    it so old and new can be compared."""
    from shap.explainers._gradient import GradientExplainer

    original = GradientExplainer.shap_values

    def seeded(
        self,
        X,
        nsamples=200,
        ranked_outputs=None,
        output_rank_order="max",
        rseed=None,
        return_variances=False,
    ):
        return original(
            self,
            X,
            nsamples,
            ranked_outputs,
            output_rank_order,
            seed,
            return_variances,
        )

    GradientExplainer.shap_values = seeded


def run_old(model_dir, signals, out_root):
    from tagger.model.common import fromFolder
    from tagger.plot.basic import basic

    model = fromFolder(model_dir)
    os.makedirs(out_root, exist_ok=True)
    link = os.path.join(out_root, "testing_data")
    if not os.path.exists(link):
        os.symlink(
            os.path.abspath(os.path.join(model_dir, "testing_data")), link
        )
    os.makedirs(os.path.join(out_root, "plots/training"), exist_ok=True)
    model.output_directory = out_root
    capture.install(os.path.join(out_root, "plots"))
    basic(model, signals)


def run_new(model_dir, signals, out_root):
    from tagger.eval import config as config_mod
    from tagger.eval.utils.cli import evaluate
    from tagger.plot.cli import ModelResults, draw
    from tagger.plot.registry import load_all

    config = config_mod.load(overrides=[f"SIGNAL_PROCESSES={signals!r}"])
    evaluate(model_dir, ["jetClassifier", "jetPt"], config)
    load_all()
    capture.install(os.path.join(out_root, "plots"))
    for name in ("jetClassifier", "jetPt"):
        draw(
            name,
            [ModelResults(model_dir, "model", [name])],
            os.path.join(out_root, "plots"),
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model", default="output/baseline")
    p.add_argument("-sig", "--signal-processes", nargs="*", default=[])
    p.add_argument("--scratch", default=None)
    args = p.parse_args()
    scratch = args.scratch or tempfile.mkdtemp(prefix="compare_basic_")
    from tagger.eval import config as config_mod

    pin_shap_seed(config_mod.load()["SHAP_SEED"])

    run_old(args.model, args.signal_processes, os.path.join(scratch, "old"))
    old = dict(capture.CAPTURED)
    capture.CAPTURED.clear()
    run_new(args.model, args.signal_processes, os.path.join(scratch, "new"))
    new = dict(capture.CAPTURED)

    report = capture.compare(old, new)
    lines = [f"{status:40s} {key}" for key, status in report.items()]
    counts = {}
    for status in report.values():
        counts[
            status if status in ("same", "only old", "only new") else "differs"
        ] = (
            counts.get(
                (
                    status
                    if status in ("same", "only old", "only new")
                    else "differs"
                ),
                0,
            )
            + 1
        )
    lines.append(
        f"summary: {counts}  (old plots in {scratch}/old, new in"
        f" {scratch}/new)"
    )
    text = "\n".join(lines)
    print(text)
    with open(
        os.path.join(args.model, "eval", "migration_basic.txt"), "w"
    ) as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
