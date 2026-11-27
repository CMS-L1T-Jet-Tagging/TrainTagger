"""Plot specs and plot collections.

A plot is declared, not coded: a :class:`PlotSpec` names a generic
plotter (tagger/plot/plotters.py), the result expressions it draws as
:class:`Series`, and axis options. Plot collections build lists of specs
from the config and classes stored with the eval results, so a plot
collection draws exactly what its eval collection computed.

Labels may reference scalar results:
``"b vs light (AUC = [[auc(b, light):.2f]])"``. Double brackets, because
LaTeX in labels uses braces.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tagger.plot.cli import ModelResults

# Reference results a series can draw from with a name/ prefix (see
# tagger.plot.cli)
REFERENCES = ("cmssw", "hls4ml", "emulator")


@dataclass
class Series:
    expr: str
    label: str = ""
    x: str = "x"  # curve field on the x axis (curve_overlay)
    y: str = "y"  # curve field on the y axis
    yerr: str | None = None  # curve field with y errors, draws error bars
    style: dict = field(default_factory=dict)  # passed to matplotlib


@dataclass
class PlotSpec:
    name: str  # output path without extension, relative to the plot folder
    plotter: str
    series: list
    options: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PlotCollection:
    name: str
    doc: str
    reads: tuple  # eval collections whose results it needs
    build: Callable[..., list[PlotSpec]] = field(
        repr=False
    )  # (config, ModelInfo) -> list of PlotSpec


PLOT_COLLECTIONS: dict[str, PlotCollection] = {}


def plot_collection(
    name: str | None = None, *, reads: Sequence[str]
) -> Callable[[Callable], Callable]:
    """Decorator that registers a plot collection.

    The decorated function takes (config, ModelInfo) and returns a list
    of PlotSpecs; its docstring is the collection's description.

    Args:
        name: Collection name, by default the function name.
        reads: Eval collections whose results the plots need.

    Returns:
        The decorator.

    Raises:
        ValueError: The function has no docstring.
    """

    def wrap(func):
        cname = name or func.__name__
        doc = inspect.getdoc(func)
        if not doc:
            raise ValueError(f"plot collection {cname} needs a docstring")
        PLOT_COLLECTIONS[cname] = PlotCollection(
            cname, doc, tuple(reads), func
        )
        return func

    return wrap


def load_all() -> dict[str, PlotCollection]:
    """Import the modules of tagger/plot/specs, so their plot
    collections register.

    Returns:
        dict: PLOT_COLLECTIONS, name -> PlotCollection.
    """
    # imported here: the spec modules import from this module
    import tagger.plot.specs as pkg

    for mod in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"{pkg.__name__}.{mod.name}")
    return PLOT_COLLECTIONS


_FIELD = re.compile(r"\[\[(.+?)\]\]")


def needs_missing_reference(text: str, model: ModelResults) -> bool:
    """Whether a label or expression refers to reference results the
    model does not have.

    Args:
        text: Label template or expression.
        model: ModelResults.

    Returns:
        bool: True if a [[cmssw/...]], [[hls4ml/...]] or
            [[emulator/...]] field or a name/ prefix
        refers to results the model lacks.
    """
    return any(
        ("[[" + name + "/" in text or text.startswith(name + "/"))
        and model.reference(name) is None
        for name in REFERENCES
    )


def format_label(template: str, lookup: Callable[[str], Any]) -> str:
    """Replace the [[expr]] and [[expr:fmt]] fields of a label with
    scalar results.

    Args:
        template: Label with [[...]] fields; fmt is a format spec after
            the last colon outside parentheses.
        lookup: Function of an expression that gives its value.

    Returns:
        str: The label.
    """
    return _FIELD.sub(partial(_format_field, lookup), template)


def _format_field(lookup, m):
    """One ``[[expr:fmt]]`` match of format_label, formatted: the format
    follows the last colon outside parentheses."""
    text = m.group(1)
    depth, split = 0, None
    for i, ch in enumerate(text):
        depth += ch == "("
        depth -= ch == ")"
        if ch == ":" and depth == 0:
            split = i
    expr, fmt = (
        (text, "") if split is None else (text[:split], text[split + 1 :])
    )
    return format(lookup(expr.strip()), fmt)
