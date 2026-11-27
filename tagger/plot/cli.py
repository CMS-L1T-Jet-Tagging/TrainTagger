"""python -m tagger.plot: draw plot collections from eval results (no
evaluation happens here).

Repeat -m to overlay several models. Missing results are an error that
prints the eval command.

Series whose expression starts with a reference name come from that
reference's results: ``cmssw/`` from the CMSSW L1 baseline pseudo-model
(config CMSSW_MODEL_DIR of the model's eval run, written by
``python -m tagger.eval -m cmssw``), ``hls4ml/`` and ``emulator/`` from
the hls4ml build and the CMSSW emulator of the model (<model>/hls4ml and
<model>/emulator, written by ``-m hls4ml:<model>`` and
``-m emulator:<model>``). Without them those series, and text lines that
need them, are left out.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from collections.abc import Sequence

from tagger.eval import config as config_mod
from tagger.eval.inputs import resolve_model
from tagger.eval.utils import output as output_mod
from tagger.eval.variable_manager import ModelInfo
from tagger.eval.variable_manager import vm
from tagger.plot import plotters, style
from tagger.plot.registry import (
    PLOT_COLLECTIONS,
    REFERENCES,
    PlotSpec,
    format_label,
    load_all,
    needs_missing_reference,
)

# Plotters that compare their series with each other: drawn only if
# every series is available
COMPARISONS = {"pairwise_2d", "agreement", "ratio_map"}


def split_reference(expr: str) -> tuple[str | None, str]:
    """Split a reference prefix off an expression.

    Args:
        expr: Expression, possibly with a reference prefix such as
            cmssw/.

    Returns:
        tuple: The reference name or None, and the expression without
            the prefix.
    """
    head, sep, rest = expr.partition("/")
    return (head, rest) if sep and head in REFERENCES else (None, expr)


class MissingExpression(KeyError):
    """A plot needs a result that this model's eval run did not compute."""


class ModelResults:
    """Results of one model across the eval collections a plot
    collection reads."""

    def __init__(self, model_dir, label, collections, optional=False):
        self.model_dir = model_dir
        self.label = label
        self.collections = list(collections)
        self.results = []
        for c in collections:
            try:
                self.results.append(output_mod.read(model_dir, c))
            except output_mod.MissingResult:
                if not optional:
                    raise
        self._references = {}

    @property
    def provenance(self):
        return self.results[0].provenance

    def _reference_dir(self, name):
        if name == "cmssw":
            return self.provenance["config"].get(
                "CMSSW_MODEL_DIR", os.path.join("output", name)
            )
        return os.path.join(self.model_dir, name)

    def reference(self, name="cmssw"):
        """Results of a reference (cmssw, hls4ml, emulator) for the same
        collections, or None."""
        if name not in self._references:
            if self.provenance.get("pseudo_model") == name:
                self._references[name] = self
            else:
                ref = ModelResults(
                    self._reference_dir(name),
                    name,
                    self.collections,
                    optional=True,
                )
                self._references[name] = ref if ref.results else None
        return self._references[name]

    def get(self, expr):
        ref_name, rest = split_reference(expr)
        if ref_name:
            ref = self.reference(ref_name)
            if ref is None:
                model = (
                    "cmssw"
                    if ref_name == "cmssw"
                    else f"{ref_name}:{self.model_dir}"
                )
                raise MissingExpression(
                    f"{expr}: no {ref_name} results; compute them with\n   "
                    f" python -m tagger.eval -m {model} -c"
                    f" {' -c '.join(self.collections)}"
                )
            return ref.get(rest)
        name = vm.canonical(expr)
        for res in self.results:
            if name in res:
                return res[name]
        colls = ", ".join(r.collection for r in self.results)
        raise MissingExpression(
            f"{name} is not in the {colls} results of {self.model_dir}."
            f" Compute it with\n    python -m tagger.eval -m {self.model_dir}"
            f" -e '{name}'  (or add it to a collection)"
        )

    def scalar(self, expr):
        value = self.get(expr)
        if isinstance(value, dict):
            raise TypeError(f"{expr} is a curve, a scalar is needed here")
        return value


def specs_for(name: str, model: ModelResults) -> list[PlotSpec]:
    """The plots of a plot collection for a model's results.

    Args:
        name: Plot collection name.
        model: ModelResults; its provenance gives the config and the
            model's classes and features.

    Returns:
        list: PlotSpecs.
    """
    coll = PLOT_COLLECTIONS[name]
    prov = model.provenance
    # results of older eval runs store them as "features"
    particle_features = prov.get("particle_features", prov.get("features", []))
    info = ModelInfo(
        prov["classes"],
        particle_features,
        prov.get("losses", []),
        has_eta=prov.get("has_eta", False),
        pt_weights=prov.get("pt_weights", False),
    )
    return coll.build(prov["config"], info)


def _without_missing_reference(spec, models):
    """The spec without the series of references a model has no results
    for; None if nothing is left."""
    series = [
        s
        for s in spec.series
        if not any(needs_missing_reference(s.expr, m) for m in models)
    ]
    if len(series) == len(spec.series):
        return spec
    if (
        spec.plotter in COMPARISONS
    ):  # compares its series with each other: all are needed
        return None
    if not series and spec.series:
        return None
    return dataclasses.replace(spec, series=series)


def draw(
    collection: str,
    models: Sequence[ModelResults],
    out_dir: str,
    only: str | None = None,
    verbose: bool = False,
) -> int:
    """Draw the plots of one plot collection and save them.

    Plots that need reference results (cmssw/, hls4ml/, emulator/) the
    models lack are left out, and plots that miss a result of one model
    are skipped with a message.

    Args:
        collection: Plot collection name.
        models: ModelResults, one per model; several are overlaid.
        out_dir: Plot folder.
        only: Draw only plots whose name contains this text.
        verbose: Whether to print the path of every plot.

    Returns:
        int: Number of plots saved.
    """
    style.set_style()
    specs = specs_for(collection, models[0])
    n, skipped = 0, []
    for spec in specs:
        if only and only not in spec.name:
            continue
        spec = _without_missing_reference(spec, models)
        if spec is None:
            continue
        try:
            fig = plotters.PLOTTERS[spec.plotter](spec, models)
        except MissingExpression as e:
            # e.g. an overlay with a model evaluated with other
            # settings: skip the plot, say why
            plotters.close_all()
            skipped.append(f"{spec.name}: {e.args[0]}")
            continue
        path = os.path.join(
            out_dir, format_label(spec.name, models[0].scalar)
        )  # names may hold [[expr]]
        plotters.save(fig, path)
        n += 1
        if verbose:
            print(f"  {path}")
    print(f"{collection}: {n} plots in {out_dir}")
    if skipped:
        print(
            f"  skipped {len(skipped)} plots with missing results, first:"
            f" {skipped[0]}"
        )
    return n


def parser():
    p = argparse.ArgumentParser(
        prog="python -m tagger.plot", description=__doc__
    )
    p.add_argument(
        "-m",
        "--model",
        action="append",
        default=[],
        help=(
            "model folder, or cmssw for the CMSSW L1 baseline (pseudo-model),"
            " or hls4ml:<model> / emulator:<model>; repeat to overlay"
        ),
    )
    p.add_argument(
        "-l",
        "--label",
        action="append",
        default=[],
        help="legend label per -m (default: folder name)",
    )
    p.add_argument(
        "-c",
        "--collection",
        action="append",
        default=[],
        help="plot collection, repeatable",
    )
    p.add_argument(
        "-o", "--output", help="plot folder (default: <first model>/plots)"
    )
    p.add_argument(
        "--only", help="draw only plots whose path contains this text"
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="list plot collections, or with -m -c the plot names",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print the path of every plot",
    )
    return p


def plot(
    model_dirs: Sequence[str],
    collections: Sequence[str],
    out_dir: str | None = None,
    labels: Sequence[str] = (),
    only: str | None = None,
    verbose: bool = False,
) -> int:
    """Draw plot collections from the eval results of one or more model
    folders.

    Args:
        model_dirs: Model folders (or results folders); several are
            overlaid.
        collections: Plot collection names.
        out_dir: Plot folder, by default <first model>/plots.
        labels: Legend label per model; the folder names for the rest.
        only: Draw only plots whose name contains this text.
        verbose: Whether to print the path of every plot.

    Returns:
        int: Number of plots saved.

    Raises:
        KeyError: An unknown plot collection.
        output.MissingResult: A model lacks the results a collection
            reads.
    """
    load_all()
    labels = list(labels) + [
        os.path.basename(os.path.normpath(m))
        for m in model_dirs[len(labels) :]
    ]
    out_dir = out_dir or os.path.join(model_dirs[0], "plots")
    n = 0
    for name in collections:
        if name not in PLOT_COLLECTIONS:
            raise KeyError(
                f"unknown plot collection {name}, known:"
                f" {', '.join(sorted(PLOT_COLLECTIONS))}"
            )
        models = [
            ModelResults(m, l, PLOT_COLLECTIONS[name].reads)
            for m, l in zip(model_dirs, labels)
        ]
        n += draw(name, models, out_dir, only=only, verbose=verbose)
    return n


def main(argv=None):
    args = parser().parse_args(argv)
    load_all()
    if args.list and not (args.model and args.collection):
        for name, coll in sorted(PLOT_COLLECTIONS.items()):
            print(
                f"  {name:16s} reads {', '.join(coll.reads):24s}"
                f" {coll.doc.splitlines()[0]}"
            )
        return 0
    if not args.model or not args.collection:
        parser().error("need -m and -c")
    config = config_mod.load()
    model_dirs = [resolve_model(m, config)[0] for m in args.model]
    unknown = [
        name for name in args.collection if name not in PLOT_COLLECTIONS
    ]
    if unknown:
        parser().error(
            f"unknown plot collection {unknown[0]}, known:"
            f" {', '.join(sorted(PLOT_COLLECTIONS))}"
        )
    try:
        if args.list:
            for name in args.collection:
                for spec in specs_for(
                    name,
                    ModelResults(
                        model_dirs[0], "", PLOT_COLLECTIONS[name].reads
                    ),
                ):
                    print(spec.name)
            return 0
        plot(
            model_dirs,
            args.collection,
            args.output,
            args.label,
            only=args.only,
            verbose=args.verbose,
        )
    except output_mod.MissingResult as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
