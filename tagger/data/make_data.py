import os
from argparse import ArgumentParser
import yaml

# Import from other modules
from tagger.data.tools import make_data

if __name__ == "__main__":

    parser = ArgumentParser()
    # Making input arguments
    parser.add_argument(
        '-y',
        '--yaml',
        default='./tagger/data/samples.yml',
        help='Path to input training data configuration file',
    )
    parser.add_argument(
        '-i',
        '--input',
        default='./data',
        help='Path to input training data configuration file',
    )
    parser.add_argument('-r', '--ratio', default=1, type=float, help='Ratio (0-1) of the input data root file to process')
    parser.add_argument('-s', '--step', default='100MB', help='The maximum memory size to process input root file')
    parser.add_argument('-e', '--extras', default='extra_fields', help='Which extra fields to add to output tuples, in puppicand_fields.yml')
    parser.add_argument('-t', '--tree', default='outnano/Jets', help='Tree within the ntuple containing the jets')
    parser.add_argument('-sig', '--signal-processes', default=[], nargs='*', help='Specify all signal process for individual plotting')
    parser.add_argument('-nw', '--num_workers', default=8, type=int, help='How many threads to run the data splitting with')
    args = parser.parse_args()

    sample_config = yaml.safe_load(open(args.yaml, "r"))
    print('Processing samples:', list(sample_config['samples'].keys()))

    # Format all the signal processes used for plotting later
    for sample, info in sample_config['samples'].items():
        file = os.path.join(args.input, f"{sample}.root")
        outdir = os.path.join("training_data", sample)
        if not os.path.exists(outdir):
            make_data(
                infile=file,
                outdir=outdir,
                step_size=args.step,
                extras=args.extras,
                ratio=args.ratio,
                tree=args.tree,
                num_workers = args.num_workers,
                remove_unmatched=info.get('remove_unmatched', True),
                apply_cuts=info.get('apply_cuts', True),
            )

    # Format all the signal processes used for plotting later
    for signal_process in args.signal_processes:
        signal_input = os.path.join(os.path.dirname(args.input), f"{signal_process}.root")
        signal_output = os.path.join("signal_process_data", signal_process)
        if not os.path.exists(signal_output):
            make_data(
                infile=signal_input,
                outdir=signal_output,
                step_size=args.step,
                extras=args.extras,
                ratio=args.ratio,
                tree=args.tree,
                num_workers = args.num_workers,
            )
