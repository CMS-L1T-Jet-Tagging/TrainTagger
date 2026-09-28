import os
from argparse import ArgumentParser

import yaml

from tagger.data.tools import make_data


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", help="Dataset YAML configuration")
    args = parser.parse_args()

    with open(args.config) as stream:
        config = yaml.safe_load(stream)

    make_data(
        infile=config["input"],
        outdir=config["output"],
        step_size=config.get("step_size", "100MB"),
        extras=config.get("extras", "extra_fields"),
        ratio=config.get("ratio", 1.0),
        tree=config.get("tree", "outnano/Jets"),
        num_workers=config.get("num_workers", 8),
    )

    for signal in config.get("signal_processes", []):
        if os.path.exists(signal["output"]):
            print(f"Signal output exists, skipping: {signal['output']}")
            continue

        make_data(
            infile=signal["input"],
            outdir=signal["output"],
            step_size=config.get("step_size", "100MB"),
            extras=config.get("extras", "extra_fields"),
            ratio=config.get("ratio", 1.0),
            tree=signal.get("tree", config.get("tree", "outnano/Jets")),
            num_workers=config.get("num_workers", 8),
        )
