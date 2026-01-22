import os
from argparse import ArgumentParser
from pathlib import Path
import re

import hls4ml
import matplotlib.pyplot as plt
import numpy as np

from tagger.data.tools import load_data, make_data, to_ML
from tagger.model.common import fromFolder

# Import from other modules
from tagger.plot import style
from tagger.plot.common import plot_2d

import sklearn

import xml.etree.ElementTree as ET

style.set_style()

def read_hls_report(filename: str) -> dict:
  '''
  Extract estimated performance metrics from HLS C Synthesis such as:
    latency (min, max), interval (min, max), resources
  Parameters
  ----------
  filename : string
    Name of XML HLS report file
  Returns
  ----------
  dictionary of extracted report contents
  '''
  
  if os.path.exists(filename):
    report = {}
    xml = ET.parse(filename)
    PE = xml.find('PerformanceEstimates')
    if PE is not None:
      SoOL = PE.find('SummaryOfOverallLatency')
      if SoOL is not None:
        report['latency_best'] = SoOL.find('Best-caseLatency')
        report['latency_worst'] = SoOL.find('Worst-caseLatency')
        report['interval_best'] = SoOL.find('Interval-min')
        report['interval_worst'] = SoOL.find('Interval-max')
    AE = xml.find('AreaEstimates')
    if AE is not None:
      R = AE.find('Resources')
      if R is not None:
        report['lut'] = R.find('LUT')
        report['ff'] = R.find('FF')
        report['dsp'] = R.find('DSP')
        report['bram18'] = R.find('BRAM_18K')

    for key in report.keys():
      if key is not None:
        report[key] = int(report[key].text)
    return report
  else:
    return None

def read_vsynth_report(filename):
  section = 0
  if os.path.exists(filename):
    report = {}
    f = open(filename, 'r')
    for line in f.readlines():
      # track which report section the line is in for filtering
      if '1. CLB Logic' in line:
        section = 1
      elif '1.1 Summary of Registers by Type' in line:
        section = 1.1
      elif '2. BLOCKRAM' in line:
        section = 2
      elif '3. ARITHMETIC' in line:
        section = 3
      elif '4. I/O' in line:
        section = 4

      # extract the value from the tables in each section
      if section == 1 and 'CLB LUTs*' in line:
        report['lut'] = int(line.split('|')[2])
      elif section == 1 and 'CLB Registers' in line:
        report['ff'] = int(line.split('|')[2])
      elif section == 2 and 'RAMB18' in line and 'Note' not in line:
        report['bram18'] = int(line.split('|')[2])
      elif section == 3 and 'DSPs' in line:
        report['dsp'] = int(line.split('|')[2])
    return report
  else:
    return None


def read_hls_log(filename: str) -> dict:
  '''
  Extract build metrics from HLS C Synthesis log such as:
    synthesis time, synthesis memory usage
  Parameters
  ----------
  filename : string
    Name of HLS log file
  Returns
  ----------
  dictionary of extracted log contents
  '''
  if os.path.exists(filename):
    report = {}
    f = open(filename, 'r')
    for line in f.readlines():
      if 'HLS 200-112' in line: # build summary line
        search = 'Total elapsed time: ([0-9]+)\.*([0-9]*) seconds'
        m = re.search(search, line)
        if m is not None:
          report['time_seconds'] = float(m.group(1))
          if m.group(2) != '':
            report['time_seconds'] += float(m.group(2))/100
        else:
          report['time_seconds'] = None
        search = 'peak allocated memory: ([0-9]+)\.*([0-9]*) ([k,M,G])B'
        m = re.search(search, line)
        if m is not None:
          mem = float(m.group(1))
          if m.group(2) != '':
            mem += float(m.group(2))/1000
          div = 1
          if m.group(3) == 'G':
            div = 1
          elif m.group(3) == 'M':
            div=1e3
          elif m.group(3) == 'k':
            div=1e6
          report['memory_GB'] = mem / div
        else:
          report['memory_GB'] = None
          report['time_seconds'] = None
          
      if 'Estimated Fmax' in line:
        search = '([0-9]+)\.*([0-9]*) MHz'
        f = re.search(search, line)
        if f is not None:
          report['FMax'] = float(f.group(1))
    return report
  else:
    return None

def getReports(indir):
        
    data_ = {}

    report_csynth = Path('{}/L1TSC4NGJetModel_prj/solution1/syn/report/L1TSC4NGJetModel_csynth.xml'.format(indir))
    report_vsynth = Path('{}/vivado_synth.rpt'.format(indir))
    log = Path('{}/vitis_hls.log'.format(indir))
    
    data_['hls_report'] = read_hls_report(report_csynth)
    data_['vsynth_report'] = read_vsynth_report(report_vsynth)
    data_['hls_log'] = read_hls_log(log)
    
    return data_


def doPlots(model, outputdir, inputdir, trace=False):
    os.makedirs(outputdir, exist_ok=True)

    data, _, class_labels, input_vars, extra_vars = load_data(inputdir, percentage=1, test_ratio=0.0)
    X_test, Y_test, pt_target, truth_pt, _ = to_ML(data, class_labels)

    labels = list(class_labels.keys())

    model.firmware_convert("temp", build=False)
    y_hls, y_ptreg_hls = model.hls_jet_model.predict(np.ascontiguousarray(X_test))
    y_class, y_ptreg = model.jet_model.predict(np.ascontiguousarray(X_test))

    print(
        "MSE between keras and hls4ml for regression is",
        sklearn.metrics.mean_absolute_error(y_ptreg[:, 0],y_ptreg_hls[:, 0])
    )
    

    for i, label in enumerate(labels):
        plt.clf()
        min_x = min(np.amin(y_hls[:, i]), np.amin(y_class[:, i]))
        max_x = max(np.amax(y_hls[:, i]), np.amax(y_class[:, i]))
        figure = plot_2d(
            np.array(y_class[:, i]),
            np.array(y_hls[:, i]),
            (min_x, max_x),
            (min_x, max_x),
            "Keras",
            "hls4ml",
            style.CLASS_LABEL_STYLE[label] + " score",
        )
        figure.savefig("%s/%s_score_2D.png" % (outputdir, label), bbox_inches='tight')
        figure.savefig("%s/%s_score_2D.pdf" % (outputdir, label), bbox_inches='tight')
        
        print(
            "MSE between keras and hls4ml for " + label + " classification is",
            sklearn.metrics.mean_absolute_error(np.array(y_class[:, i]),np.array(y_hls[:, i]))
        )

    plt.clf()
    figure = plot_2d(
        y_ptreg[:, 0],
        y_ptreg_hls[:, 0],
        (min(np.amin(y_ptreg_hls), np.amin(y_ptreg)), max(np.amax(y_ptreg_hls), np.amax(y_ptreg))),
        (min(np.amin(y_ptreg_hls), np.amin(y_ptreg)), max(np.amax(y_ptreg_hls), np.amax(y_ptreg))),
        "Keras",
        "hls4ml",
        "Regression score",
    )
    figure.savefig("%s/%s_score_2D.png" % (outputdir, "Regression"), bbox_inches='tight')
    figure.savefig("%s/%s_score_2D.pdf" % (outputdir, "Regression"), bbox_inches='tight')
    plt.close()
    
    if trace==True:

      wp, wph, ap, aph = hls4ml.model.profiling.numerical(model=model.jet_model, hls_model=model.hls_jet_model, X=X_test)
      ap.savefig(outputdir + "/model_activations_profile.png")
      wp.savefig(outputdir + "/model_weights_profile.png")
      aph.savefig(outputdir + "/model_activations_profile_opt.png")
      wph.savefig(outputdir + "/model_weights_profile_opt.png")

      y_hls, hls4ml_trace = model.hls_jet_model.trace(np.ascontiguousarray(X_test))
      keras_trace = hls4ml.model.profiling.get_ymodel_keras(model.jet_model, X_test)

      for layer in hls4ml_trace.keys():
          print("Doing profiling 2d for layer", layer)
          min_x = min(np.amin(hls4ml_trace[layer]), np.amin(keras_trace[layer]))
          max_x = max(np.amax(hls4ml_trace[layer]), np.amax(keras_trace[layer]))
          plot_2d(
              hls4ml_trace[layer].flatten(),
              keras_trace[layer].flatten(),
              (min_x, max_x),
              (min_x, max_x),
              "hls4ml {}".format(layer),
              "Keras  {}".format(layer),
              layer + " agreement",
          )
          plt.plot([min_x, max_x], [min_x, max_x], c="gray")
          plt.savefig(f"{outputdir}/profile_2d_{layer}.png", bbox_inches='tight')
          plt.savefig(f"{outputdir}/profile_2d_{layer}.pdf", bbox_inches='tight')
          plt.close()

    return


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m', '--model_path', default='output/baseline', help='Input model path for comparison')
    parser.add_argument('-o', '--outpath', default='output/baseline/plots/profile', help='Jet tagger plotting directory')
    parser.add_argument(
        '-of', '--outpath_firmware', default='output/baseline/firmware', help='Jet tagger firmware directory'
    )
    parser.add_argument('-i', '--input', default='data/jetTuple_extended_5.root', help='Path to profiling data rootfile')
    parser.add_argument('-r', '--remake', default=False, help='Remake profiling data? ')
    parser.add_argument('-p', '--doplots', default=False, help='Run profiling plots ')
    parser.add_argument('-d', '--data', default='training_data_baseline', help='What data to use for profiling ')
    parser.add_argument('-y', '--yaml_config', default='tagger/model/configs/baseline.yaml', help='YAML config for model')

    args = parser.parse_args()

    model = fromFolder(args.model_path)
    data_dir = args.data
    if args.remake:
        make_data(infile=args.input, outdir="profiling_data/", extras='extra_emulation_fields', tree="outnano/Jets")
        data_dir = "profiling_data/"
    
    if args.doplots:
        doPlots(model, args.outpath, data_dir)

    report = getReports(args.outpath_firmware + '/' + model.firmware_config['project_name'])

    print("===================")
    print('Input Precision : ', model.firmware_config['input_precision'])
    print('Class Precision : ', model.firmware_config['class_precision'])
    print('Regression Precision : ', model.firmware_config['reg_precision'])
    print("===================")
    print("   ")
    print("Resource Usage of a VU13P from CSynth")
    print("   ")
    print('Flip Flops : {} -> {:.2f} %'.format(report['hls_report']['ff'],(report['hls_report']['ff']/3456000)*100))
    print('Look Up Tables : {} -> {:.2f} %'.format(report['hls_report']['lut'],(report['hls_report']['lut']/1728000)*100))
    print('Digital Signal Processors : {} -> {:.2f} %'.format(report['hls_report']['dsp'],(report['hls_report']['dsp']/12288)*100))
    print('Block RAM : {} -> {:.2f} %'.format(report['hls_report']['bram18'],(report['hls_report']['bram18']/5376)*100))
    print('Latency :  {} clock cycles -> {:.2f} mus'.format(report['hls_report']['latency_best'],(report['hls_report']['latency_best']*model.firmware_config['clock_period'] * 1e-3)))
    print('Initiation Interval :  {} clock cycles -> {:.2f} mus'.format(report['hls_report']['interval_best'],(report['hls_report']['interval_best']*model.firmware_config['clock_period'] * 1e-3)))
    print('Estimated FMax :',  report['hls_log']['FMax'], 'MHz' )
    print("   ")
    print("===================")
    if bool(report['vsynth_report']):
        print("   ")
        print("Resource Usage of a VU13P from VSynth")
        print("   ")
        print('Flip Flops : {} -> {:.2f} %'.format(report['vsynth_report']['ff'],(report['vsynth_report']['ff']/3456000)*100))
        print('Look Up Tables : {} -> {:.2f} %'.format(report['vsynth_report']['lut'],(report['vsynth_report']['lut']/1728000)*100))
        print('Digital Signal Processors : {} -> {:.2f} %'.format(report['vsynth_report']['dsp'],(report['vsynth_report']['dsp']/12288)*100))
        print('Block RAM : {} -> {:.2f} %'.format(report['vsynth_report']['bram18'],(report['vsynth_report']['bram18']/5376)*100))
        print("   ")
        print("===================")
    print('Memory Usage :',  report['hls_log']['memory_GB'], 'GB' )
    print('Synth time :',  report['hls_log']['time_seconds'], 's' )
    print("===================")
