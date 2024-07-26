from argparse import ArgumentParser

def get_common_parser():
    parser = ArgumentParser()
    parser.add_argument('-w', '--workdir', dest='workdir', default='.')
    parser.add_argument('-s', '--save', dest='save', action='store_true')
    parser.add_argument('-v', '--verbose', dest='verbose', default=False)
    return parser

def handle_common_args(args):
    if args.verbose:
        import sys
        import logging
        logging.basicConfig(stream=sys.stdout, level=logging.WARNING)






