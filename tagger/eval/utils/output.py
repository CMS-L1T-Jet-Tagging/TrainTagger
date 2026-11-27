"""Result files: output/<model>/eval/<collection>.json (+ .npz for curve
arrays).

The json holds provenance, scalars and the non-array fields of curves.
Curve arrays go to the npz under the key ``<variable>/<field>``.
:func:`read` puts both back together.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
from typing import Any

import numpy as np

FORMAT_VERSION = 1


def eval_dir(model_dir: str) -> str:
    """Results folder of a model.

    Args:
        model_dir: Model folder.

    Returns:
        str: <model_dir>/eval.
    """
    return os.path.join(model_dir, "eval")


def paths(model_dir: str, collection: str) -> tuple[str, str]:
    """Result files of one collection.

    Args:
        model_dir: Model folder.
        collection: Collection name.

    Returns:
        tuple: Paths of the json and the npz file.
    """
    base = os.path.join(eval_dir(model_dir), collection)
    return base + ".json", base + ".npz"


def git_commit() -> str:
    """Commit of the tagger code, stored with every result.

    Returns:
        str: The commit hash, with -dirty if files under tagger/ are
            modified or untracked, or
        unknown outside a git repository.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.check_output(
            ["git", "-C", here, "rev-parse", "HEAD"], text=True
        ).strip()
        root = subprocess.check_output(
            ["git", "-C", here, "rev-parse", "--show-toplevel"], text=True
        ).strip()
        # modified or untracked code under tagger/ makes the result not
        # reproducible from the commit alone
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", root, "status", "--porcelain", "--", "tagger"],
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")


def _is_array(value):
    return isinstance(value, np.ndarray) or (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(isinstance(v, (int, float, np.number)) for v in value)
    )


def write(
    model_dir: str,
    collection: str,
    values: dict[str, Any],
    provenance: dict[str, Any],
) -> str:
    """Write the results of one collection.

    Args:
        model_dir: Model folder; the files go to <model_dir>/eval.
        collection: Collection name.
        values: Expression -> value (a float, or a dict of arrays and
            scalars for a curve).
        provenance: What the results were made from, stored in the json.

    Returns:
        str: Path of the json file.
    """
    json_path, npz_path = paths(model_dir, collection)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    scalars, curves, arrays = {}, {}, {}
    for name, value in values.items():
        if isinstance(value, dict):
            meta = {}
            for field, item in value.items():
                if _is_array(item):
                    arrays[f"{name}/{field}"] = np.asarray(item)
                else:
                    meta[field] = item
            curves[name] = meta
        else:
            scalars[name] = float(value)
    doc = {
        "format": FORMAT_VERSION,
        "collection": collection,
        "provenance": {
            **provenance,
            "written": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "scalars": scalars,
        "curves": curves,
    }
    with open(json_path, "w") as f:
        json.dump(doc, f, indent=1, default=_json_default)
    np.savez_compressed(npz_path, **arrays)
    return json_path


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


class Results:
    """Results of one collection for one model. ``res[name]`` gives a
    float or a dict of arrays."""

    def __init__(self, model_dir, collection, doc, arrays):
        self.model_dir = model_dir
        self.collection = collection
        self.provenance = doc["provenance"]
        self.scalars = doc["scalars"]
        self.curves = {}
        for name, meta in doc["curves"].items():
            self.curves[name] = dict(meta)
        for key in arrays.files:
            name, _, field = key.rpartition("/")
            self.curves.setdefault(name, {})[field] = arrays[key]

    def __contains__(self, name):
        return name in self.scalars or name in self.curves

    def __getitem__(self, name):
        if name in self.scalars:
            return self.scalars[name]
        if name in self.curves:
            return self.curves[name]
        raise KeyError(name)

    def names(self):
        return list(self.scalars) + list(self.curves)


class MissingResult(FileNotFoundError):
    pass


def read(model_dir: str, collection: str) -> Results:
    """Read the results of one collection.

    Args:
        model_dir: Model folder.
        collection: Collection name.

    Returns:
        Results: res[expression] gives a float or a dict of arrays.

    Raises:
        MissingResult: The collection has no results; the message gives
            the eval command.
    """
    json_path, npz_path = paths(model_dir, collection)
    if not os.path.exists(json_path):
        raise MissingResult(
            f"no {collection} results for {model_dir}. Run first:\n    python"
            f" -m tagger.eval -m {model_dir} -c {collection}"
        )
    with open(json_path) as f:
        doc = json.load(f)
    with np.load(npz_path, allow_pickle=False) as arrays:
        return Results(model_dir, collection, doc, arrays)
