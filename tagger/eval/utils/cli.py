"""python -m tagger.eval: compute collections of variables and write
them to output/<model>/eval/, or write docs/variables.md (--docs)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Sequence
from typing import Any

from tagger.eval import config as config_mod
from tagger.eval.inputs import PSEUDO_MODELS, resolve_model
from tagger.eval.inputs.sources import Sources
from tagger.eval.utils import output
from tagger.eval.utils.docs import DOCS_PATH, write
from tagger.eval.variable_manager import EvalRun, ModelInfo, vm

# Classes used for --list <collection> without -m (the baseline model's
# classes)
DEFAULT_CLASSES = [
    "b",
    "charm",
    "light",
    "gluon",
    "taup",
    "taum",
    "muon",
    "electron",
]


def parser():
    p = argparse.ArgumentParser(
        prog="python -m tagger.eval", description=__doc__
    )
    p.add_argument(
        "-m",
        "--model",
        action="append",
        default=[],
        help=(
            "model folder; cmssw for the CMSSW L1 baseline (pseudo-model);"
            " hls4ml:<model> or emulator:<model> for the hls4ml build or the"
            " CMSSW emulator of a model (collection emulation); repeatable"
        ),
    )
    p.add_argument(
        "-c",
        "--collection",
        action="append",
        default=[],
        help="collection to compute, repeatable",
    )
    p.add_argument(
        "-e",
        "--expr",
        action="append",
        default=[],
        help="compute and print single expressions",
    )
    p.add_argument(
        "--config",
        action="append",
        default=[],
        help="YAML config file overriding the defaults in config/defaults/",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one config key",
    )
    p.add_argument(
        "--list",
        nargs="?",
        const="",
        metavar="COLLECTION",
        help=(
            "list variables and collections, "
            "or the expressions of one collection"
        ),
    )
    p.add_argument(
        "--docs",
        nargs="?",
        const=DOCS_PATH,
        metavar="PATH",
        help="write docs/variables.md",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print every value and its compute time",
    )
    return p


def model_info(sources: Sources) -> ModelInfo:
    """What the collections may know about a model, without running it.

    Args:
        sources: Sources of the model, pseudo-model or model version.

    Returns:
        ModelInfo: Classes, particle features, losses and flags.
    """
    pseudo = getattr(sources, "pseudo", "")
    if pseudo in PSEUDO_MODELS:
        return ModelInfo([], [], [], pseudo=pseudo)
    # hls4ml or emulator version of a model: its classes, the emulation
    # jets only
    if pseudo:
        return ModelInfo(list(sources.class_labels()), [], [], pseudo=pseudo)
    losses = sources.loss_names() if hasattr(sources, "loss_names") else []
    has_eta = sources.has_eta() if hasattr(sources, "has_eta") else False
    pt_weights = (
        sources.has_pt_weights()
        if hasattr(sources, "has_pt_weights")
        else False
    )
    return ModelInfo(
        list(sources.class_labels()),
        sources.particle_input_vars(),
        losses,
        has_eta=has_eta,
        pt_weights=pt_weights,
    )


def evaluate(
    model_dir: str,
    collections: Sequence[str],
    config: dict[str, Any],
    verbose: bool = False,
    sources: Sources | None = None,
) -> dict[str, str]:
    """Compute collections for one model and write their results.

    Args:
        model_dir: Model folder; the results go to <model_dir>/eval.
        collections: Collection names.
        config: Eval config.
        verbose: Whether to print every value and its compute time.
        sources: Sources to read the inputs from; by default those of
            model_dir.

    Returns:
        dict: Collection -> path of its json file.
    """
    sources = sources or Sources(model_dir, config=config)
    run = EvalRun(sources, config, verbose=verbose)
    info = model_info(sources)
    written = {}
    for name in collections:
        exprs = vm.get_collection(name, config, info)
        print(f"{model_dir}: {name}, {len(exprs)} variables")
        start = time.time()
        values = {e: run.get(e) for e in exprs}
        calls = [vm.parse(e) for e in exprs]
        provenance = {
            "model_dir": os.path.abspath(model_dir),
            "git_commit": output.git_commit(),
            "config_hash": config_mod.config_hash(config),
            "config": config,
            "classes": info.classes,
            "particle_features": info.particle_features,
            "losses": info.losses,
            "pseudo_model": info.pseudo,
            "has_eta": info.has_eta,
            "pt_weights": info.pt_weights,
            "inputs": (
                sorted(set().union(*(c.variable.inputs for c in calls)))
                if calls
                else []
            ),
            "n_entries": {
                k[0] + (f"({', '.join(map(str, k[1]))})" if k[1] else ""): len(
                    v
                )
                for k, v in run._inputs.items()
                if hasattr(v, "__len__")
            },
            "samples": {
                k[0] + f"({', '.join(map(str, k[1]))})": v.describe()
                for k, v in run._inputs.items()
                if hasattr(v, "describe")
            },
            "seconds": round(time.time() - start, 1),
        }
        written[name] = output.write(model_dir, name, values, provenance)
        print(f"  wrote {written[name]}")
    return written


def main(argv=None):
    args = parser().parse_args(argv)
    config = config_mod.load(args.config, args.set)

    if args.docs:
        print(f"wrote {write(config, args.docs)}")
        return 0
    if args.list is not None:
        if args.list == "":
            vm.print_variables()
            print("\ncollections")
            vm.print_collections(config)
        else:
            info = (
                model_info(_sources(args.model[0], config))
                if args.model
                else ModelInfo(DEFAULT_CLASSES)
            )
            for e in vm.get_collection(args.list, config, info):
                print(e)
        return 0
    if not args.model or not (args.collection or args.expr):
        parser().error(
            "need -m and at least one -c or -e (or --list / --docs)"
        )

    for name in args.model:
        model_dir = resolve_model(name, config)[0]
        if args.expr:
            run = EvalRun(_sources(name, config), config, verbose=args.verbose)
            for e in args.expr:
                value = run.get(e)
                if isinstance(value, dict):  # a curve: the shape of each field
                    value = {
                        k: getattr(v, "shape", v) for k, v in value.items()
                    }
                print(f"{vm.canonical(e)} = {value}")
        if args.collection:
            evaluate(
                model_dir,
                args.collection,
                config,
                verbose=args.verbose,
                sources=_sources(name, config),
            )
    return 0


def _sources(name, config):
    results_dir, pseudo, model_dir = resolve_model(name, config)
    return Sources(model_dir or results_dir, config=config, pseudo=pseudo)


if __name__ == "__main__":
    sys.exit(main())
