"""Generates docs/variables.md from the VariableManager: every variable
by group, then the collections. Call `python -m tagger.eval --docs` to
update the file. The VariableManager imports the modules of
tagger/eval/variables (variables and built-in collections) to read their
docstrings. The variable modules are grouped by their section comments
(# -- Title ---) and by the GROUPS list below. The collections are
listed with their members (variables or other collections) and the first
line of their docstring. The output is Markdown, with a preview style in
docs/variables.preview.css.
"""

import importlib
import inspect
import math
import os
import re

from tagger.eval.utils.parse import choices
from tagger.eval.variable_manager import ModelInfo, vm

DOCS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "docs", "variables.md"
)


def _by_module():
    modules = {}
    for var in vm.get_variables():
        modules.setdefault(var.module, []).append(var)
    return modules


# Section titles of the variable modules, in this order (others follow
# alphabetically)
GROUPS = [
    "jet_classifier",
    "jet_pt",
    "train",
    "seeds_tau",
    "seeds_hh",
    "fastpuppi",
]

# Model description used to list the members of the collections (the
# baseline model's classes)
DOC_CLASSES = [
    "b",
    "charm",
    "light",
    "gluon",
    "taup",
    "taum",
    "muon",
    "electron",
]


def _first_clause(doc):
    """Text before the first colon or full stop: the short description."""
    text = " ".join(doc.split())
    return re.split(r":|\.(?:\s|$)", text, maxsplit=1)[0].strip()


def _class_arguments(config):
    """Table of what a class argument (cls, sig, bkg) can be: the model
    classes, the class groups, all."""
    lines = [
        (
            "Class arguments (`cls`, `sig`, `bkg`) take one model class, a"
            " class group or `all`:"
        ),
        "",
        "| Argument | Jets with truth in |",
        "| --- | --- |",
        "| "
        + ", ".join(f"`{c}`" for c in DOC_CLASSES)
        + " | that class (models trained with `pileup: true` also have"
        " `pileup`) |",
    ]
    lines += [
        f"| `{g}` | {', '.join(members)} |"
        for g, members in config["CLASS_GROUPS"].items()
    ]
    return lines + [
        "| `all` | every class of the model |",
        "",
        (
            "Collections skip a group if the model lacks one of its classes."
            " The groups are defined in config `CLASS_GROUPS`."
        ),
        "",
    ]


def _compact(doc):
    """A docstring as one Markdown block: its paragraphs on separate
    lines (hard line break, a trailing backslash) instead of separate
    paragraphs with a blank line between them.
    """
    paragraphs = [" ".join(p.split()) for p in doc.split("\n\n") if p.strip()]
    return "\\\n".join(paragraphs)


# Longest default written next to a config key; longer values are left
# out (the key name alone)
MAX_DEFAULT_LENGTH = 100
# Lists with more items than this are written as their first two items
# and the last, if evenly spaced
MAX_LIST_ITEMS = 13


def _format_number(x):
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    if isinstance(x, float):
        text = f"{x:.4g}"
        return text if any(c in text for c in ".e") else text + ".0"
    return repr(x)


def _is_evenly_spaced(values):
    steps = [b - a for a, b in zip(values, values[1:])]
    return all(math.isclose(s, steps[0], rel_tol=1e-6) for s in steps)


def _format_default(value):
    """A config value as short Python-like text, or None if it is too
    long to read in a line."""
    if isinstance(value, bool) or value is None:
        text = repr(value)
    elif isinstance(value, (int, float)):
        text = _format_number(value)
    elif isinstance(value, str):
        text = repr(value)
    elif (
        isinstance(value, (list, tuple))
        and value
        and all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for v in value
        )
    ):
        items = [_format_number(v) for v in value]
        if len(items) > MAX_LIST_ITEMS:
            if not _is_evenly_spaced(value):
                return None
            items = items[:2] + ["..."] + items[-1:]
        brackets = "()" if isinstance(value, tuple) else "[]"
        text = brackets[0] + ", ".join(items) + brackets[1]
    elif (
        isinstance(value, (list, tuple))
        and value
        and len(value) <= MAX_LIST_ITEMS
    ):
        # mixed lists such as the TAU_SCANS ranges: [[first, last,
        # step], overflow]
        items = [_format_default(v) for v in value]
        if any(v is None for v in items):
            return None
        brackets = "()" if isinstance(value, tuple) else "[]"
        text = brackets[0] + ", ".join(items) + brackets[1]
    elif isinstance(value, dict) and value:
        entries = [(k, _format_default(v)) for k, v in value.items()]
        if any(v is None for _, v in entries):
            return None
        text = "{" + ", ".join(f"{k!r}: {v}" for k, v in entries) + "}"
    else:
        return None
    return text if len(text) <= MAX_DEFAULT_LENGTH else None


def _config_reference(config, key, rest):
    """The entry of config ``key`` that the text after it names, as
    ``KEY["a"]["b"]`` (a dict entry is written either way, KEY["a"] or
    KEY a), its value and the length of the text it used up.
    """
    value, name, used = config[key], key, 0
    while isinstance(value, dict):
        m = re.match(r'\["(\w+)"\]| (\w+)', rest[used:])
        if not m or (m.group(1) or m.group(2)) not in value:
            break
        entry = m.group(1) or m.group(2)
        value, name, used = value[entry], f'{name}["{entry}"]', used + m.end()
    return name, value, used


def _with_defaults(text, config):
    """A description with the first mention of each config key (or dict
    entry) written as code with its default, e.g. `PT_HIST_BINS = 50`.
    Keys whose value is too long keep their plain name.
    """
    out, pos, seen = [], 0, set()
    for m in re.finditer(r"(?<![`\w])[A-Z][A-Z0-9_]{2,}\b", text):
        if m.start() < pos or m.group() not in config:
            continue
        name, value, used = _config_reference(
            config, m.group(), text[m.end() :]
        )
        default = _format_default(value)
        if name in seen or default is None:
            continue
        seen.add(name)
        out += [text[pos : m.start()], f"`{name} = {default}`"]
        pos = m.end() + used
    return "".join(out) + text[pos:]


def _choices_line(var):
    """The allowed values of the Literal arguments, as a last line of
    the description."""
    parts = [
        f"`{p}`: {', '.join(allowed)}"
        for p, t in var.params
        if (allowed := choices(t))
    ]
    return "\\\n" + "; ".join(parts) if parts else ""


def _section_of(mod, lineno):
    """Title of the last section comment (``# -- Title ---`` or a
    ``# ===`` banner) above a line."""
    title = None
    lines = inspect.getsource(mod).splitlines()
    for i, line in enumerate(lines[: lineno - 1]):
        m = re.match(r"# -- (.+?) -+$", line)
        if m:
            title = m.group(1)
        elif (
            re.match(r"# =+$", line)
            and i + 2 < len(lines)
            and re.match(r"# =+$", lines[i + 2])
        ):
            title = lines[i + 1].lstrip("# ").strip()
    return title[0].upper() + title[1:] if title else None


def _collection_members(name, config):
    """Member collections, or else the variable names, of a collection
    (in order, without repeats)."""
    colls = vm.get_collections(config)
    info = ModelInfo(DOC_CLASSES, ["pt"], ["loss"])
    nested = [m for m in colls[name].build(config, info) if m in colls]
    if nested:
        return nested
    return list(
        dict.fromkeys(
            vm.parse(e).variable.name
            for e in vm.get_collection(name, config, info)
        )
    )


def markdown(config):
    config = {
        **config,
        "COLLECTIONS": {},
    }  # document the built-in collections only
    modules = _by_module()
    order = [m for g in GROUPS for m in modules if m.rsplit(".", 1)[-1] == g]
    order += sorted(m for m in modules if m not in order)
    lines = [
        "# Evaluation variables",
        "",
        (
            "<!-- Generated by `python -m tagger.eval --docs`. Do not edit,"
            " edit the docstrings instead. -->"
        ),
        "",
        (
            "The evaluation calculates physics quantities associated with a"
            " given jet or event. Every such quantity is a **variable**,"
            " called by an expression such as `auc(taus, jets)`."
        ),
        (
            "All variables and collections are held by the VariableManager"
            " (`from tagger.eval import vm`,"
        ),
        "e.g. `vm.get_variable(\"auc\")`).",
        "`tagger.eval` computes collections of variables and saves them;",
        "`tagger.plot` draws the saved results.",
        "",
        "```",
        "python -m tagger.eval -m output/baseline -e \"auc(taus, jets)\"",
        "python -m tagger.eval -m output/baseline -c jetClassifier",
        "```",
        "",
        "## 1. Variables",
        "",
    ]
    for n, module in enumerate(order, 1):
        mod = importlib.import_module(module)
        title = _first_clause(mod.__doc__ or "") or module.rsplit(".", 1)[-1]
        lines += [f"### 1.{n} {title}", ""]
        if module.endswith(".jet_classifier"):
            lines += _class_arguments(config)
        # variables in source order, under the section comments of the
        # module (# -- Title ---)
        current = None
        for var in sorted(
            modules[module], key=lambda v: v.func.__code__.co_firstlineno
        ):
            section = _section_of(mod, var.func.__code__.co_firstlineno)
            if section and section != current:
                current = section
                lines += [f"#### {section}", ""]
            description = _with_defaults(_compact(var.doc), config)
            lines += [
                f"##### `{var.signature()}`",
                "",
                description + _choices_line(var),
                "",
            ]  # purple via docs/variables.preview.css
    lines += ["## 2. Collections", ""]
    for name, coll in sorted(vm.get_collections(config).items()):
        members = ", ".join(
            f"'{m}'" for m in _collection_members(name, config)
        )
        # name purple and bold via docs/variables.preview.css (h5
        # code:first-child)
        lines += [
            f"##### `collections.{name}` `= [{members}]`",
            "",
            _first_clause(coll.doc),
            "",
        ]
    lines += ADDING_A_VARIABLE
    return "\n".join(lines).rstrip() + "\n"


ADDING_A_VARIABLE = '''## 3. Adding a variable

Write a function in a module under `tagger/eval/variables/` (every
module there is loaded) and register it with `@vm.variable`:

```python
from tagger.eval import vm


@vm.variable(inputs={"test_jets"}, kind="scalar")
def meanScore(run, cls: str):
    """Mean ``cls`` score of the jets whose truth is ``cls``."""
    jets = run.jets()
    idx = run.classes(cls)
    is_cls = jets.y_true[:, idx].sum(axis=1) > 0
    return float(jets.y_pred[is_cls][:, idx].sum(axis=1).mean())
```

- `inputs`: what the variable reads (`test_jets`, `model`,
  `minbias_events`, ...); reading anything else fails.
- `kind`: `scalar` (one number) or `curve` (a dict of arrays).
- Arguments after `run` are typed: `str` (class names), `float`, `int`,
  `Literal["a", "b"]` (a fixed set of names) or `Expr` (another
  expression).
- The docstring is required: it is the description on this page.
- `run.get("roc(b, light)")` reuses another variable; `run.memo(key, f,
  *args)` caches shared intermediate results.

Use it right away, or add it to a collection (a `@vm.collection()` in
`variables/collections.py`, or config `COLLECTIONS`), then regenerate
this page:

```
python -m tagger.eval -m output/baseline -e "meanScore(b)"
python -m tagger.eval --docs
```
'''.splitlines()


def write(config, path=DOCS_PATH):
    path = os.path.normpath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(markdown(config))
    return path
