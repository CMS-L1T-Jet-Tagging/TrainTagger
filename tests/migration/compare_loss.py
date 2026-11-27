"""Compare the old loss history plot (basic.loss_history, drawn by
JetTagModel.plot_loss) with the train collection, for one trained model.

Temporary (deleted in the last commit of the refactor). Redraws the old plot from
training_history.json and compares the drawn data with the new plotter, as compare_basic.py does.
    python tests/migration/compare_loss.py -m output/wavg_short
"""

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import capture  # noqa: E402


class History:
    def __init__(self, history):
        self.history = history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model", required=True)
    args = p.parse_args()
    lines = []
    from tagger.eval import config as config_mod
    from tagger.eval.utils.cli import evaluate
    from tagger.plot.basic import loss_history
    from tagger.plot.cli import ModelResults, draw
    from tagger.plot.registry import load_all

    with open(os.path.join(args.model, "training_history.json")) as f:
        history = json.load(f)
    losses = [
        k for k in history if k.endswith("_loss") and not k.startswith("val_")
    ]
    scratch = tempfile.mkdtemp(prefix="compare_loss_")
    os.makedirs(os.path.join(scratch, "old", "training"))
    capture.install(os.path.join(scratch, "old"))
    loss_history(
        os.path.join(scratch, "old", "training"),
        [k[: -len("_loss")] for k in losses],
        History(history),
    )
    old = dict(capture.CAPTURED)
    capture.CAPTURED.clear()
    evaluate(args.model, ["train"], config_mod.load())
    load_all()
    capture.install(os.path.join(scratch, "new"))
    draw(
        "train",
        [ModelResults(args.model, "model", ["train"])],
        os.path.join(scratch, "new"),
    )
    report = capture.compare(old, dict(capture.CAPTURED))
    lines += [f"data    {status:16s} {key}" for key, status in report.items()]
    text = "\n".join(lines)
    print(text)
    with open(
        os.path.join(args.model, "eval", "migration_loss.txt"), "w"
    ) as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
