from argparse import ArgumentParser

from tagger.model.common import fromFolder

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m', '--model_path', default='output/baseline', help='Input model path for conversion')
    parser.add_argument('-o', '--outpath', default='firmware/L1TSC4NGJetModel', help='Jet tagger synthesized output directory')

    args = parser.parse_args()

    # Load the model
    model = fromFolder(args.model_path)
    model.hls4ml_convert(args.outpath, build=False)
