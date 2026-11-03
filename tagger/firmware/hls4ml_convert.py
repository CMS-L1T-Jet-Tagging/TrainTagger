from argparse import ArgumentParser

from tensorflow import keras

from tagger.model.common import fromFolder

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m', '--model_path', default='output/baseline', help='Input model path for conversion')
    parser.add_argument(
        '-o', '--outpath', default='firmware/L1TSC4NGJetModel', help='Jet tagger synthesized output directory'
    )

    args = parser.parse_args()

    # Load the model
    model = fromFolder(args.model_path)

    # Replace this with our new code
    # model.firmware_convert(args.outpath, build=False)

    # Split the model into sub-models for embedding, jet ID, and pT
    embedding_model = keras.Model(model.jet_model.input, model.jet_model.get_layer('pool').output)
    jet_id_model = keras.Model(model.jet_model.get_layer('Dense_0_jetID').output, model.jet_model.get_layer('jet_id_output').output)
    pt_model = keras.Model(model.jet_model.get_layer('Dense_0_pT').output, model.jet_model.get_layer('pT_output').output)
    # autoencoder_model = ...

    # Now write code to convert each sub-model using hls4ml

