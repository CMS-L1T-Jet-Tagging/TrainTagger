#!/usr/bin/env python3
"""Build (and optionally submit) a per-file FastPUPPI HTCondor DAG."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_CMSSW = Path("/dice/users/ek19824/CMSSW_17_0_0_pre2")


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def files_from_list(list_path: Path) -> Iterable[Path]:
    for line_number, raw_line in enumerate(list_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("file:"):
            line = line[5:]
        candidate = Path(line).expanduser()
        if not candidate.is_absolute():
            candidate = list_path.parent / candidate
        if candidate.suffix.lower() != ".root":
            raise ValueError(f"{list_path}:{line_number}: expected a .root file, got {raw_line!r}")
        yield candidate.resolve()


def discover_inputs(sources: list[str], recursive: bool) -> list[Path]:
    discovered: list[Path] = []
    for raw_source in sources:
        source = Path(raw_source).expanduser().resolve()
        if source.is_dir():
            iterator = source.rglob("*.root") if recursive else source.glob("*.root")
            discovered.extend(sorted(iterator, key=natural_key))
        elif source.is_file() and source.suffix.lower() == ".root":
            discovered.append(source)
        elif source.is_file():
            discovered.extend(files_from_list(source))
        else:
            raise ValueError(f"Input does not exist: {source}")

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in discovered:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not resolved.is_file():
            raise ValueError(f"ROOT input does not exist: {resolved}")
        if resolved.suffix.lower() != ".root":
            raise ValueError(f"Expected a .root input: {resolved}")
        if any(character.isspace() or character == '"' for character in str(resolved)):
            raise ValueError(f"Whitespace and double quotes are unsupported in input paths: {resolved}")
        seen.add(resolved)
        unique.append(resolved)

    if not unique:
        raise ValueError("No ROOT files were found")
    return unique


def quote_dag(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def vars_line(node: str, values: dict[str, object]) -> str:
    assignments = " ".join(f'{key}="{quote_dag(value)}"' for key, value in values.items())
    return f"VARS {node} {assignments}"


def original_sample_name(path: Path) -> str:
    stem = path.stem
    return stem.removeprefix("inputs170X_")


def stage_paths(source: Path, output_dir: Path, stage: str) -> tuple[Path | None, Path | None]:
    sample = original_sample_name(source)
    inputs_path = output_dir / "inputs170X" / f"inputs170X_{sample}.root"
    ntuple_path = output_dir / "jetNtuples170X" / f"jetNtuple170X_{sample}.root"
    if stage == "inputs":
        return inputs_path, None
    if stage == "ntuples":
        return None, ntuple_path
    return inputs_path, ntuple_path


def parse_args() -> argparse.Namespace:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Create one HTCondor job per ROOT file and, by default, chain the "
            "input-production and jet-ntuple jobs with DAGMan."
        )
    )
    parser.add_argument(
        "sources",
        nargs="+",
        help="ROOT files, directories containing ROOT files, or text file lists",
    )
    parser.add_argument(
        "--stage",
        choices=("both", "inputs", "ntuples"),
        default="both",
        help="Run the full DAG or generate only one stage (default: both)",
    )
    parser.add_argument("--recursive", action="store_true", help="Search input directories recursively")
    parser.add_argument("--max-events", type=int, default=-1, help="cmsRun maxEvents for every job")
    parser.add_argument("--cmssw-dir", type=Path, default=DEFAULT_CMSSW)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "output")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_DIR / "output" / "condor_runs" / timestamp,
        help="New directory for the DAG, manifest, and logs",
    )
    parser.add_argument("--max-input-jobs", type=int, default=75, help="Maximum number of input jobs to run concurrently")
    parser.add_argument("--max-ntuple-jobs", type=int, default=75, help="Maximum number of ntuple jobs to run concurrently")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--submit", action="store_true", help="Submit after generating the DAG")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_events < -1:
        raise ValueError("--max-events must be -1 or non-negative")
    if args.max_input_jobs < 1 or args.max_ntuple_jobs < 1:
        raise ValueError("MAXJOBS limits must be positive")
    if args.retries < 0:
        raise ValueError("--retries cannot be negative")

    cmssw_dir = args.cmssw_dir.expanduser().resolve()
    required_paths = [
        cmssw_dir / "src",
        cmssw_dir / "src/FastPUPPI/NtupleProducer/python/runInputs151X_htcondor.py",
        cmssw_dir / "src/FastPUPPI/NtupleProducer/python/runJetNtuple_htcondor.py",
    ]
    for required_path in required_paths:
        if not required_path.exists():
            raise ValueError(f"Required path is missing: {required_path}")

    inputs = discover_inputs(args.sources, args.recursive)
    output_dir = args.output_dir.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    for label, condor_path in (
        ("script directory", SCRIPT_DIR),
        ("CMSSW directory", cmssw_dir),
        ("output directory", output_dir),
        ("run directory", run_dir),
    ):
        if any(character.isspace() or character == '"' for character in str(condor_path)):
            raise ValueError(f"Whitespace and double quotes are unsupported in the {label}: {condor_path}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"Run directory already exists and is not empty: {run_dir}")

    stage1_dir = output_dir / "inputs170X"
    stage2_dir = output_dir / "jetNtuples170X"
    stage1_logs = run_dir / "logs" / "inputs170X"
    stage2_logs = run_dir / "logs" / "jetNtuples170X"
    for directory in (stage1_dir, stage2_dir, stage1_logs, stage2_logs):
        directory.mkdir(parents=True, exist_ok=True)

    dag_lines: list[str] = []
    manifest_lines = ["index\tsource\tinputs170X\tjetNtuple170X"]
    output_names: set[str] = set()

    for index, source in enumerate(inputs):
        stage1_path, stage2_path = stage_paths(source, output_dir, args.stage)
        planned_names = [path.name for path in (stage1_path, stage2_path) if path is not None]
        duplicates = output_names.intersection(planned_names)
        if duplicates:
            raise ValueError(
                "Duplicate input basenames would overwrite output files: " + ", ".join(sorted(duplicates))
            )
        output_names.update(planned_names)

        stage1_text = str(stage1_path) if stage1_path else ""
        stage2_text = str(stage2_path) if stage2_path else ""
        manifest_lines.append(f"{index}\t{source}\t{stage1_text}\t{stage2_text}")

        input_node = f"INPUT_{index:05d}"
        ntuple_node = f"NTUPLE_{index:05d}"
        common = {
            "script_dir": SCRIPT_DIR,
            "cmssw_dir": cmssw_dir,
            "max_events": args.max_events,
        }

        if args.stage in ("both", "inputs"):
            assert stage1_path is not None
            dag_lines.append(f"JOB {input_node} {SCRIPT_DIR / 'inputs.sub'}")
            dag_lines.append(
                vars_line(
                    input_node,
                    common
                    | {
                        "input_file": source,
                        "output_name": stage1_path.name,
                        "output_path": stage1_path,
                        "stdout": stage1_logs / f"{index:05d}.out",
                        "stderr": stage1_logs / f"{index:05d}.err",
                        "joblog": stage1_logs / f"{index:05d}.log",
                    },
                )
            )
            dag_lines.append(f"CATEGORY {input_node} INPUTS")
            dag_lines.append(f"RETRY {input_node} {args.retries}")

        if args.stage in ("both", "ntuples"):
            assert stage2_path is not None
            ntuple_input = stage1_path if args.stage == "both" else source
            assert ntuple_input is not None
            dag_lines.append(f"JOB {ntuple_node} {SCRIPT_DIR / 'ntuples.sub'}")
            dag_lines.append(
                vars_line(
                    ntuple_node,
                    common
                    | {
                        "input_file": ntuple_input,
                        "output_name": stage2_path.name,
                        "output_path": stage2_path,
                        "stdout": stage2_logs / f"{index:05d}.out",
                        "stderr": stage2_logs / f"{index:05d}.err",
                        "joblog": stage2_logs / f"{index:05d}.log",
                    },
                )
            )
            dag_lines.append(f"CATEGORY {ntuple_node} NTUPLES")
            dag_lines.append(f"RETRY {ntuple_node} {args.retries}")

        if args.stage == "both":
            dag_lines.append(f"PARENT {input_node} CHILD {ntuple_node}")
        dag_lines.append("")

    if args.stage in ("both", "inputs"):
        dag_lines.append(f"MAXJOBS INPUTS {args.max_input_jobs}")
    if args.stage in ("both", "ntuples"):
        dag_lines.append(f"MAXJOBS NTUPLES {args.max_ntuple_jobs}")

    dag_path = run_dir / "workflow.dag"
    manifest_path = run_dir / "manifest.tsv"
    dag_path.write_text("\n".join(dag_lines) + "\n")
    manifest_path.write_text("\n".join(manifest_lines) + "\n")

    jobs_per_file = 2 if args.stage == "both" else 1
    print(f"Discovered {len(inputs)} input ROOT files")
    print(f"Prepared {len(inputs) * jobs_per_file} jobs in {dag_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Stage-1 outputs: {stage1_dir}")
    print(f"Stage-2 outputs: {stage2_dir}")

    if args.submit:
        subprocess.run(["condor_submit_dag", str(dag_path)], cwd=run_dir, check=True)
    else:
        print(f"Not submitted. Inspect the manifest, then run:\n  condor_submit_dag {dag_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
