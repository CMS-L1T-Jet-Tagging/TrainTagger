#!/usr/bin/env python3
"""Merge several flat JSON objects into one, without duplicate keys.
 
Usage:
    python merge_json.py a.json b.json c.json d.json -o merged.json
    python merge_json.py results/*.json -o merged.json --on-conflict last
 
Rules:
  * The result contains the union of all keys.
  * A key present in several files with the same value appears once.
  * A key present in several files with DIFFERENT values is a conflict.
    --on-conflict error (default): print the conflicts and exit with code 1,
                                   so the CI job fails instead of silently
                                   picking a value.
    --on-conflict first / last:    keep the value from the first / last file.
  * Key order follows first appearance, so output is deterministic.
"""
import argparse
import json
import sys
 
 
def load(path):
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        sys.exit(f"{path}: expected a top-level JSON object, got {type(data).__name__}")
    return data
 
 
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="input JSON files, in priority order")
    p.add_argument("-o", "--output", required=True, help="merged output file")
    p.add_argument("--on-conflict", choices=["error", "first", "last"], default="error")
    args = p.parse_args()
 
    merged, source, conflicts = {}, {}, []
    for path in args.inputs:
        for key, value in load(path).items():
            if key not in merged:
                merged[key], source[key] = value, path
            elif merged[key] != value:
                conflicts.append((key, source[key], merged[key], path, value))
                if args.on_conflict == "last":
                    merged[key], source[key] = value, path
            # identical value: duplicate, dropped
 
    if conflicts:
        print(f"{len(conflicts)} conflicting key(s):", file=sys.stderr)
        for key, p1, v1, p2, v2 in conflicts:
            print(f"  {key}: {v1!r} ({p1}) vs {v2!r} ({p2})", file=sys.stderr)
        if args.on_conflict == "error":
            sys.exit(1)
 
    with open(args.output, "w") as f:
        json.dump(merged, f, indent=4)
        f.write("\n")
    print(f"Wrote {len(merged)} keys from {len(args.inputs)} files to {args.output}")
 
 
if __name__ == "__main__":
    main()
