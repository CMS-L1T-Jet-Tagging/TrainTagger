from argparse import ArgumentParser
import os

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import hls4ml
# Import from other modules
from tagger.plot import style
from tagger.data.tools import make_data, load_data, to_ML
from tagger.plot.common import plot_2d
from tagger.model.common import fromFolder

style.set_style()


def getReports(indir):
    data_ = {}

    report_csynth = Path(
        '{}/L1TSC4NGJetModel_test_prj/solution1/syn/report/L1TSC4NGJetModel_test_csynth.rpt'.format(indir))

    if report_csynth.is_file():
        print('Found valid vsynth and synth in {}! Fetching numbers'.format(indir))

        with report_csynth.open() as report:
            lines = np.array(report.readlines())
            lat_line = lines[np.argwhere(
                np.array(['Latency (cycles)' in line for line in lines])).flatten()[0] + 3]
            data_['latency_clks'] = int(lat_line.split('|')[2])
            data_['latency_mus'] = float(lat_line.split('|')[2])*5.0/1000.
            data_['latency_ii'] = int(lat_line.split('|')[6])

            resource_line = lines[np.argwhere(
                np.array(['|Utilization (%)' in line for line in lines])).flatten()[0]]
            try:
                data_['bram_rel'] = int(resource_line.split('|')[2])
            except ValueError:
                data_['bram_rel'] = 0
            data_['dsp_rel'] = int(resource_line.split('|')[3])
            data_['ff_rel'] = int(resource_line.split('|')[4])
            data_['lut_rel'] = int(resource_line.split('|')[5])

            total_line = lines[np.argwhere(
                np.array(['|Total ' in line for line in lines])).flatten()[0]]
            data_['bram'] = int(total_line.split('|')[2])
            data_['dsp'] = int(total_line.split('|')[3])
            data_['ff'] = int(total_line.split('|')[4])
            data_['lut'] = int(total_line.split('|')[5])

    return data_


def doPlots(model, outputdir, inputdir):
    os.makedirs(outputdir, exist_ok=True)

    modelsAndNames = {"model": model}

    data, _, class_labels, input_vars, extra_vars = load_data(
        inputdir, percentage=100, test_ratio=0.0)
    X_test, Y_test, pt_target, truth_pt, _ = to_ML(data, class_labels)

    labels = list(class_labels.keys())

    model.hls4ml_convert("temp", build=False)
    y_hls, y_ptreg_hls = model.hls_jet_model.predict(np.ascontiguousarray(X_test))
    y_class, y_ptreg = model.jet_model.predict(np.ascontiguousarray(X_test))

    for i, label in enumerate(labels):
        plt.clf()
        min_x = min(np.amin(y_hls[:, i]), np.amin(y_class[:, i]))
        max_x = max(np.amax(y_hls[:, i]), np.amax(y_class[:, i]))
        figure = plot_2d(np.array(y_class[:, i]), np.array(y_hls[:, i]), (min_x, max_x), (
            min_x, max_x), "Tensorflow", "hls4ml", style.CLASS_LABEL_STYLE[label]+" score")
        plt.savefig("%s/%s_score_2D.png" %
                    (outputdir, label), bbox_inches='tight')
        plt.savefig("%s/%s_score_2D.pdf" %
                    (outputdir, label), bbox_inches='tight')

    plt.clf()
    figure = plot_2d(y_ptreg[:, 0], y_ptreg_hls[:, 0],
                     (min(np.amin(y_ptreg_hls), np.amin(y_ptreg)),
                      max(np.amax(y_ptreg_hls), np.amax(y_ptreg))),
                     (min(np.amin(y_ptreg_hls), np.amin(y_ptreg)),
                      max(np.amax(y_ptreg_hls), np.amax(y_ptreg))),
                     "Tensorflow", "hls4ml", "Regression score")
    plt.savefig("%s/%s_score_2D.png" %
                (outputdir, "Regression"), bbox_inches='tight')
    plt.savefig("%s/%s_score_2D.pdf" %
                (outputdir, "Regression"), bbox_inches='tight')
    plt.close()

    wp, wph, ap, aph = hls4ml.model.profiling.numerical(
        model=model.jet_model, hls_model=model.hls_jet_model, X=X_test)
    ap.savefig(outputdir+"/model_activations_profile.png")
    wp.savefig(outputdir+"/model_weights_profile.png")
    aph.savefig(outputdir+"/model_activations_profile_opt.png")
    wph.savefig(outputdir+"/model_weights_profile_opt.png")

    y_hls, hls4ml_trace = model.hls_jet_model.trace(np.ascontiguousarray(X_test))
    keras_trace = hls4ml.model.profiling.get_ymodel_keras(model.jet_model, X_test)

    for layer in hls4ml_trace.keys():
        print("Doing profiling 2d for layer", layer)
        min_x = min(np.amin(hls4ml_trace[layer]), np.amin(keras_trace[layer]))
        max_x = max(np.amax(hls4ml_trace[layer]), np.amax(keras_trace[layer]))
        plot_2d(hls4ml_trace[layer].flatten(), keras_trace[layer].flatten(),
                (min_x, max_x),
                (min_x, max_x),
                "hls4ml {}".format(layer), "Tensorflow  {}".format(layer), layer + " agreement")
        plt.plot([min_x, max_x], [min_x, max_x], c="gray")
        plt.savefig(f"{outputdir}/profile_2d_{layer}.png", bbox_inches='tight')
        plt.savefig(f"{outputdir}/profile_2d_{layer}.pdf", bbox_inches='tight')
        plt.close()

    return


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m', '--model_path', default='output/baseline',
                        help='Input model path for comparison')
    parser.add_argument('-o', '--outpath', default='output/baseline/plots/profile',
                        help='Jet tagger plotting directory')
    parser.add_argument('-of', '--outpath_firmware',
                        default='output/baseline/firmware', help='Jet tagger firmware directory')
    parser.add_argument('-i', '--input', default='data/jetTuple_extended_5.root',
                        help='Path to profiling data rootfile')
    parser.add_argument('-r', '--remake', default=False,
                        help='Remake profiling data? ')
    parser.add_argument('-y', '--yaml_config',
                        default='tagger/model/configs/baseline.yaml', help='YAML config for model')

    args = parser.parse_args()

    model = fromFolder(args.model_path)

    if args.remake:
        make_data(infile=args.input, outdir="profiling_data/",
                  extras='extra_emulation_fields', tree="outnano/Jets")

    doPlots(model, args.outpath, "profiling_data/")

    report = getReports(args.outpath_firmware + '/' +
                        model.hls4ml_config['project_name'])

    # if os.path.isfile("mlflow_run_id.txt"):

    #     f = open("mlflow_run_id.txt", "r")
    #     run_id = (f.read())
    #     mlflow.get_experiment_by_name(os.getenv('CI_COMMIT_REF_NAME'))
    #     with mlflow.start_run(experiment_id=1,
    #                         run_name=args.name,
    #                         run_id=run_id # pass None to start a new run
    #                         ):
    #         mlflow.log_metric('FF',report['ff_rel'])
    #         mlflow.log_metric('LUT',report['lut_rel'])
    #         mlflow.log_metric('BRAM',report['bram_rel'])
    #         mlflow.log_metric('DSP',report['dsp_rel'])
    #         mlflow.log_metric('Latency cc',report['latency_clks'])
    #         mlflow.log_metric('Latency us',report['latency_mus'])
    #         mlflow.log_metric('Initiation Interval ',report['latency_ii'])

    #         mlflow.log_param('Input Precision ',precisions[0])
    #         mlflow.log_param('Class Precision ',precisions[1])
    #         mlflow.log_param('Regression Precision ',precisions[2])

    print("===================")
    print('Input Precision : ',  model.hls4ml_config['input_precision'])
    print('Class Precision : ', model.hls4ml_config['class_precision'])
    print('Regression Precision : ', model.hls4ml_config['reg_precision'])
    print(" Resource Usage of a VU13P")
    print('Flip Flops : ', report['ff_rel'], ' %')
    print('Look Up Tables : ', report['lut_rel'], ' %')
    print('Block RAM : ', report['bram_rel'], ' %')
    print('Digital Signal Processors : ', report['dsp_rel'], ' %')
    print('Latency : ', report['latency_clks'], ' clock cycles')
    print('Latency : ', report['latency_mus'], ' mus')
    print('Initiation Interval : ', report['latency_mus'], ' clock cycles')
    print("===================")
