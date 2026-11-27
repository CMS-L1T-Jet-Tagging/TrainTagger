"""The VariableManager: every variable and collection of the evaluation,
in one place, and the EvalRun that computes them for one model.

    from tagger.eval import vm

    vm.get_variable("auc").doc  # a variable and its documentation
    vm.canonical("effAtMistag(b,light,.1)")  # 'effAtMistag(b, light, 0.1)'
    vm.get_collection("seeds", config, info)  # its expressions
    vm.add_collection("mine", ["auc(taus, jets)", "jetPt"])

Variables register with the ``@vm.variable(...)`` decorator in the
modules of tagger/eval/variables, built-in collections with
``@vm.collection()`` in tagger/eval/variables/collections.py. Both are
loaded on the first lookup. An :class:`EvalRun` computes the variables
for one model, caches them and checks the declared inputs.
"""

import importlib
import inspect
import pkgutil
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from tagger.eval.inputs import INPUTS
from tagger.eval.utils.parse import (
    KINDS,
    Call,
    Expr,
    Variable,
    _parse,
    choices,
)  # noqa: F401 (Expr, KINDS: for variable modules)


@dataclass(frozen=True)
class ModelInfo:
    """What a collection may know about the model without loading it."""

    classes: list
    particle_features: list = field(
        default_factory=list
    )  # input features of the candidates (not the jet)
    losses: list = field(default_factory=list)
    pseudo: str = ""  # "cmssw" for the CMSSW pseudo-model
    has_eta: bool = (
        False  # the test split has the jet eta (efficiency versus eta)
    )
    # the model has per-constituent pT weights (layer pt_weights_output)
    pt_weights: bool = False


@dataclass(frozen=True)
class Collection:
    name: str
    doc: str
    build: object = field(
        repr=False
    )  # (config, ModelInfo) -> list of expressions and collection names


class VariableManager:
    """Holds the variables and collections. Use the shared instance ``vm``."""

    def __init__(self):
        self._variables = {}
        self._collections = {}
        self._loaded = False

    # -- variables -----------------------------------------------------
    def variable(self, name=None, *, inputs, kind):
        """Decorator that registers a variable:
        ``@vm.variable(inputs={"test_jets"}, kind="scalar")``. The
        function receives the EvalRun and the converted arguments; its
        docstring is required.
        """

        def wrap(func):
            self.add_variable(func, name, inputs=inputs, kind=kind)
            return func

        return wrap

    def add_variable(self, func, name=None, *, inputs, kind):
        """Register ``func`` as a variable. ``inputs``: the inputs it
        reads (see INPUTS); ``kind``: scalar, curve or inherit
        (metavariables). Argument types come from the annotations: str,
        float, int, Expr, or Literal["a", "b"] (a name out of a fixed
        set, checked when an expression is parsed).
        """
        vname = name or func.__name__
        if vname in self._variables:
            raise ValueError(f"variable {vname} registered twice")
        doc = inspect.getdoc(func)
        if not doc:
            raise ValueError(f"variable {vname} needs a docstring")
        unknown = set(inputs) - set(INPUTS)
        if unknown:
            raise ValueError(
                f"variable {vname}: unknown inputs {sorted(unknown)}, allowed"
                f" {sorted(INPUTS)}"
            )
        if kind not in KINDS:
            raise ValueError(f"variable {vname}: kind must be one of {KINDS}")
        params = []
        for pname, p in list(inspect.signature(func).parameters.items())[
            1:
        ]:  # the first one is run
            ptype = (
                p.annotation
                if p.annotation is not inspect.Parameter.empty
                else str
            )
            if ptype not in (str, float, int, Expr) and not choices(ptype):
                raise ValueError(
                    f"variable {vname}: argument {pname} has unsupported type"
                    f" {ptype}"
                )
            params.append((pname, ptype))
        var = Variable(
            vname,
            func,
            frozenset(inputs),
            kind,
            doc,
            func.__module__,
            tuple(params),
        )
        self._variables[vname] = var
        return var

    def remove_variable(self, name):
        """Unregister a variable (to redefine it interactively, or in
        tests)."""
        self._variables.pop(name, None)

    def get_variable(self, name):
        """The Variable of a name (without arguments), with a spelling
        hint if unknown."""
        self.load_all()
        if name not in self._variables:
            close = [n for n in self._variables if n.lower() == name.lower()]
            hint = (
                f", did you mean {close[0]}?"
                if close
                else ", see python -m tagger.eval --list"
            )
            raise KeyError(f"unknown variable {name!r}{hint}")
        return self._variables[name]

    def get_variables(self):
        """All variables, sorted by name."""
        self.load_all()
        return [self._variables[n] for n in sorted(self._variables)]

    def parse(self, expr):
        """Parse an expression like ``effAtMistag(b, light, 0.1)`` into
        a Call (variable + arguments)."""
        return _parse(expr, self.get_variable)

    def canonical(self, expr):
        """Canonical spelling of an expression:
        ``effAtMistag(b,light,.1)`` -> ``effAtMistag(b, light, 0.1)``.
        It is the key of the result files and the cache."""
        return self.parse(expr).name

    # -- collections ---------------------------------------------------
    def collection(self, name=None):
        """Decorator that registers a built-in collection: a function
        ``(config, info) -> list`` of expressions and collection names.
        Its docstring is required; the text before the first colon is
        its short description."""

        def wrap(func):
            cname = name or func.__name__
            doc = inspect.getdoc(func)
            if not doc:
                raise ValueError(f"collection {cname} needs a docstring")
            self._collections[cname] = Collection(cname, doc, func)
            return func

        return wrap

    def add_collection(self, name, members, doc="User collection."):
        """Register a fixed collection of expressions and collection names."""
        self.load_all()
        if name in self._collections:
            raise ValueError(f"collection {name} already exists")
        self._collections[name] = Collection(
            name, doc, lambda config, info, m=tuple(members): list(m)
        )

    def remove_collection(self, name):
        """Unregister a collection added with add_collection (or a
        built-in one)."""
        self._collections.pop(name, None)

    def get_collections(self, config):
        """All collections by name: the registered ones plus those of
        config COLLECTIONS."""
        self.load_all()
        out = dict(self._collections)
        for cname, members in config.get("COLLECTIONS", {}).items():
            if cname in out:
                raise ValueError(
                    f"config COLLECTIONS: {cname} is already a collection"
                )
            out[cname] = Collection(
                cname,
                "User collection from the config.",
                lambda c, i, m=tuple(members): list(m),
            )
        return out

    def get_collection(self, name, config, info, _seen=()):
        """The canonical expressions of a collection for a model
        (ModelInfo), nested collections resolved, in order and without
        repeats."""
        colls = self.get_collections(config)
        if name not in colls:
            raise KeyError(
                f"unknown collection {name!r}, known:"
                f" {', '.join(sorted(colls))}"
            )
        if name in _seen:
            raise ValueError(
                f"collection cycle: {' -> '.join(_seen + (name,))}"
            )
        out = []
        for member in colls[name].build(config, info):
            if member in colls:
                out += self.get_collection(
                    member, config, info, _seen + (name,)
                )
            else:
                out.append(self.canonical(member))
        return list(dict.fromkeys(out))

    # -- loading and printing ------------------------------------------
    def load_all(self):
        """Import the modules of tagger/eval/variables (variables and
        built-in collections), once."""
        if self._loaded:
            return
        self._loaded = True
        # imported here: the variable modules import vm from this module
        import tagger.eval.variables as pkg

        for mod in pkgutil.iter_modules(pkg.__path__):
            importlib.import_module(f"{pkg.__name__}.{mod.name}")

    def print_variables(self, out=sys.stdout):
        """Print every variable by module: signature, kind and the first
        line of its docstring."""
        modules = {}
        for var in self.get_variables():
            modules.setdefault(var.module.rsplit(".", 1)[-1], []).append(var)
        for module, variables in sorted(modules.items()):
            print(f"\n{module}", file=out)
            for var in variables:
                print(
                    f"  {var.signature():40s} {var.kind:7s}"
                    f" {var.doc.splitlines()[0]}",
                    file=out,
                )

    def print_collections(self, config, out=sys.stdout):
        """Print every collection with the first line of its docstring."""
        for name, coll in sorted(self.get_collections(config).items()):
            print(f"  {name:16s} {coll.doc.splitlines()[0]}", file=out)


vm = VariableManager()


# ====================================================================
# The EvalRun: computes the variables for one model
# ======================================================================
class InputNotDeclared(RuntimeError):
    pass


class EvalRun:
    """Passed to every variable function.

    ``run.get("auc(b, light)")`` computes (or returns the cached) value
    of another variable. ``run.jets()`` returns the jet sample (input
    ``test_jets``), ``run.input(name, *args)`` any other input. A
    variable may only read inputs it declared.
    """

    def __init__(self, sources, config, verbose=False):
        self.sources = sources
        self.config = config
        self.verbose = verbose
        # changed by the onProcess / onTestSubset metavariables
        self.sample = "test"
        self._cache = {}
        self._inputs = {}
        self._stack = []
        self._memo = {}
        self.inputs_used = set()
        self.timings = {}

    # -- variables -----------------------------------------------------
    def get(self, expr):
        """Value of another variable, ``run.get("roc(b, light)")``:
        computed once, then cached by its canonical name (per jet
        sample). A float for scalars, a dict of arrays for curves.
        """
        call = expr if isinstance(expr, Call) else vm.parse(expr)
        key = (self.sample, call.name)
        if key in self._cache:
            return self._cache[key]
        var = call.variable
        self._stack.append(var)
        start = time.time()
        try:
            value = var.func(self, *call.args)
        finally:
            self._stack.pop()
        value = _check_value(call, value)
        self.timings[call.name] = time.time() - start
        if self.verbose:
            print(
                f"  {call.name}: {_short(value)}"
                f" ({self.timings[call.name]:.1f} s)"
            )
        self._cache[key] = value
        return value

    # -- inputs --------------------------------------------------------
    def _require(self, name):
        if not self._stack:
            return
        var = self._stack[-1]
        if var is not None and name not in var.inputs:
            raise InputNotDeclared(
                f"variable {var.name} reads input {name!r} but declares only"
                f" {sorted(var.inputs)}"
            )

    def input(self, name, *args):
        """One input, ``run.input("minbias_events", n_entries)``: loaded
        once and shared. The calling variable must declare it in
        ``inputs=``, otherwise InputNotDeclared."""
        self._require(name)
        key = (name, args)
        if key not in self._inputs:
            loader = getattr(self.sources, name, None)
            if loader is None:
                raise RuntimeError(
                    f"input {name} is not available from"
                    f" {type(self.sources).__name__}"
                )
            if self.verbose:
                print(f"  loading {name}{args if args else ''}")
            self._inputs[key] = loader(*args)
            self.inputs_used.add(
                name if not args else f"{name}({', '.join(map(str, args))})"
            )
        return self._inputs[key]

    def jets(self):
        """The jet sample with model outputs: the test split unless a
        metavariable switched it."""
        self._require("test_jets")
        if self.sample == "test":
            return self.input("test_jets")
        if self.sample == "emulation":
            return self._unchecked_input("emulation_jets")
        kind, _, process = self.sample.partition(":")
        if kind == "full":
            return self._unchecked_input("process_jets", process)
        if kind == "subset":
            key = ("test_subset", (process,))
            if key not in self._inputs:
                test = self._unchecked_input("test_jets")
                full = self._unchecked_input("process_jets", process)
                self._inputs[key] = test.subset(test.indices_shared_with(full))
            return self._inputs[key]
        raise ValueError(f"unknown jet sample {self.sample!r}")

    def _unchecked_input(self, name, *args):
        self._stack.append(None)
        try:
            key = (name, args)
            if key not in self._inputs:
                self._inputs[key] = getattr(self.sources, name)(*args)
                self.inputs_used.add(
                    name
                    if not args
                    else f"{name}({', '.join(map(str, args))})"
                )
            return self._inputs[key]
        finally:
            self._stack.pop()

    def memo(self, key, compute, *args):
        """Cache an intermediate result shared by several variables (not
        a variable itself): ``compute(*args)``, once per ``key``.
        ``compute`` must only read inputs that every caller declares."""
        k = (self.sample, key)
        if k not in self._memo:
            self._memo[k] = compute(*args)
        return self._memo[k]

    def class_labels(self):
        """Model class names -> output index."""
        return self.sources.class_labels()

    # -- class groups --------------------------------------------------
    def classes(self, group):
        """Output indices of a class or class group (config CLASS_GROUPS)."""
        labels = self.class_labels()
        if group == "all":
            return sorted(labels.values())
        names = self.config["CLASS_GROUPS"].get(group, [group])
        missing = [n for n in names if n not in labels]
        if missing:
            raise KeyError(
                f"class {group!r}: model has no class {missing}, model classes"
                f" {list(labels)}"
            )
        return [labels[n] for n in names]

    def has_classes(self, group):
        """Whether the model has every class of a class or class group
        (collections skip the others)."""
        labels = self.class_labels()
        if group == "all":
            return True
        return all(
            n in labels
            for n in self.config["CLASS_GROUPS"].get(group, [group])
        )


def _check_value(call, value):
    kind = call.variable.kind
    if kind == "scalar" or (kind == "inherit" and not isinstance(value, dict)):
        try:
            return float(value)
        except (TypeError, ValueError):
            raise TypeError(
                f"{call.name} is a scalar variable but returned"
                f" {type(value).__name__}"
            )
    if not isinstance(value, dict):
        raise TypeError(
            f"{call.name} is a curve variable and must return a dict, got"
            f" {type(value).__name__}"
        )
    return value


def _short(value):
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(f"{k}: {np.shape(v)}" for k, v in value.items())
            + "}"
        )
    return f"{value:.6g}"
