#!/usr/bin/env python3

"""Combine FastPUPPI ROOT files and make their event identifiers unique.

By default both the reconstructed-jet and generator-jet trees are copied. A
single per-file event mapping is shared by the trees so rows belonging to the
same event retain the same identifier after merging.
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
from uproot.source.file import MemmapSource


DEFAULT_TREES = ("outnano/Jets", "outnano/GenJets")
_NUMBER = re.compile(r"(\d+)")


def _natural_sort_key(path):
    """Sort file2.root before file10.root."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in _NUMBER.split(path.name)
    ]


def _normalise_tree_names(tree_names):
    """Return a non-empty, de-duplicated tuple of tree names."""
    if isinstance(tree_names, str):
        tree_names = (tree_names,)
    else:
        tree_names = tuple(tree_names)

    tree_names = tuple(dict.fromkeys(tree_names))
    if not tree_names:
        raise ValueError("At least one tree must be selected")
    return tree_names


def _remap_events(event_array, event_map):
    """Apply a previously constructed event-ID mapping."""
    old_events = ak.to_numpy(event_array)

    missing_events = set(map(int, np.unique(old_events))).difference(event_map)
    if missing_events:
        raise KeyError(
            "Event mapping is incomplete; missing IDs: "
            + ", ".join(map(str, sorted(missing_events)))
        )

    new_events = np.fromiter(
        (event_map[int(old_event_id)] for old_event_id in old_events),
        dtype=old_events.dtype,
        count=len(old_events),
    )
    return ak.Array(new_events)


def _collect_event_ids(input_root, tree_names, step_size):
    """Collect the union of event IDs appearing in all selected trees."""
    event_ids = set()
    for tree_name in tree_names:
        for arrays in input_root[tree_name].iterate(
            expressions=["event"],
            step_size=step_size,
            library="np",
            how=dict,
        ):
            event_ids.update(map(int, arrays["event"]))
    return sorted(event_ids)


def combine_sample(
    sample_dir,
    output_file,
    tree_names=DEFAULT_TREES,
    step_size="100 MB",
):
    sample_dir = Path(sample_dir)
    output_file = Path(output_file)
    tree_names = _normalise_tree_names(tree_names)

    if not sample_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {sample_dir}")

    output_resolved = output_file.resolve()
    input_files = sorted(
        (
            path
            for path in sample_dir.glob("*.root")
            if path.resolve() != output_resolved
        ),
        key=_natural_sort_key,
    )
    if not input_files:
        raise FileNotFoundError(f"No ROOT files found in {sample_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Force the local-file source. The fsspec source can wait indefinitely on
    # some DICE installations even though these are ordinary local paths.
    branch_types = {}
    with uproot.open(input_files[0], handler=MemmapSource) as first_file:
        for tree_name in tree_names:
            if tree_name not in first_file:
                raise KeyError(
                    f"Tree {tree_name!r} not found in {input_files[0]}"
                )
            branch_types[tree_name] = first_file[tree_name].typenames()
            if "event" not in branch_types[tree_name]:
                raise KeyError(f"Tree {tree_name!r} has no 'event' branch")

    # Write beside the destination and replace it only after a complete merge.
    temp_handle = tempfile.NamedTemporaryFile(
        prefix=f".{output_file.name}.",
        suffix=".tmp.root",
        dir=output_file.parent,
        delete=False,
    )
    temp_path = Path(temp_handle.name)
    temp_handle.close()

    next_event_id = 1
    total_entries = dict.fromkeys(tree_names, 0)

    try:
        with uproot.recreate(temp_path) as output_root:
            output_trees = dict.fromkeys(tree_names)

            for file_index, input_file in enumerate(input_files, start=1):
                with uproot.open(input_file, handler=MemmapSource) as input_root:
                    for tree_name in tree_names:
                        if tree_name not in input_root:
                            raise KeyError(
                                f"Tree {tree_name!r} not found in {input_file}"
                            )
                        if input_root[tree_name].typenames() != branch_types[tree_name]:
                            raise TypeError(
                                f"Branch schema for {tree_name!r} in {input_file} "
                                f"does not match the schema in {input_files[0]}"
                            )

                    # Construct one mapping for both trees. The union preserves
                    # events that contain gen jets but no reconstructed jets.
                    old_event_ids = _collect_event_ids(
                        input_root, tree_names, step_size
                    )
                    event_map = {
                        old_event_id: next_event_id + offset
                        for offset, old_event_id in enumerate(old_event_ids)
                    }
                    next_event_id += len(event_map)

                    file_entries = dict.fromkeys(tree_names, 0)
                    for tree_name in tree_names:
                        input_tree = input_root[tree_name]
                        for arrays in input_tree.iterate(
                            step_size=step_size,
                            library="ak",
                            how=dict,
                        ):
                            arrays["event"] = _remap_events(
                                arrays["event"], event_map
                            )
                            if output_trees[tree_name] is None:
                                # Passing the first arrays lets uproot infer a
                                # writable schema for std::vector branches.
                                # mktree also writes this first chunk.
                                output_trees[tree_name] = output_root.mktree(
                                    tree_name, arrays
                                )
                            else:
                                output_trees[tree_name].extend(arrays)

                            chunk_entries = len(arrays["event"])
                            file_entries[tree_name] += chunk_entries
                            total_entries[tree_name] += chunk_entries

                print(
                    f"[{file_index}/{len(input_files)}] {input_file.name}: "
                    + ", ".join(
                        f"{tree_name}={file_entries[tree_name]} rows"
                        for tree_name in tree_names
                    )
                    + f", {len(event_map)} events"
                )

            empty_trees = [
                tree_name
                for tree_name, output_tree in output_trees.items()
                if output_tree is None
            ]
            if empty_trees:
                raise ValueError(
                    "All inputs are empty for tree(s): " + ", ".join(empty_trees)
                )

        # NamedTemporaryFile starts with mode 0600. Match the usual mode for a
        # newly created output instead so group-readable datasets stay usable.
        current_umask = os.umask(0)
        os.umask(current_umask)
        temp_path.chmod(0o666 & ~current_umask)
        os.replace(temp_path, output_file)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise

    total_events = next_event_id - 1
    print(
        "Wrote "
        + ", ".join(
            f"{tree_name}={total_entries[tree_name]} rows"
            for tree_name in tree_names
        )
        + f" containing {total_events} unique events to {output_file}"
    )
    return total_entries, total_events


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Combine FastPUPPI ROOT files and replace each input file's event "
            "IDs with globally unique IDs shared by all selected trees."
        )
    )
    parser.add_argument("sample_dir", help="Directory containing input ROOT files")
    parser.add_argument("output_file", help="Combined output ROOT file")
    parser.add_argument(
        "--tree",
        dest="trees",
        action="append",
        help=(
            "TTree to combine; repeat for multiple trees. The default is "
            + ", ".join(DEFAULT_TREES)
        ),
    )
    parser.add_argument(
        "--step-size",
        default="100 MB",
        help="Maximum uncompressed data read per chunk (default: 100 MB)",
    )
    args = parser.parse_args()

    combine_sample(
        args.sample_dir,
        args.output_file,
        args.trees or DEFAULT_TREES,
        args.step_size,
    )


if __name__ == "__main__":
    main()
