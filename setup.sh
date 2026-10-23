# To run before using the codes
export PYTHONPATH=$PYTHONPATH:$PWD
export CI_COMMIT_REF_NAME=local

# Hide TensorFlow's C++ start-up messages (no GPU/CUDA/TensorRT found etc.); set to 0 to see them again
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3}
# Hide harmless warnings raised inside Keras/QKeras, not by this repo:
# - deprecation of set_learning_phase
# - serialisation of QKeras quantizers, which QKeras stores as Keras constraints
for _filter in \
  'ignore:`tf.keras.backend.set_learning_phase` is deprecated:UserWarning' \
  'ignore:The `keras.constraints.serialize()` API should only be used for objects of type:UserWarning'; do
  case ",${PYTHONWARNINGS}," in
    *",${_filter},"*) ;;
    *) export PYTHONWARNINGS="${PYTHONWARNINGS:+${PYTHONWARNINGS},}${_filter}" ;;
  esac
done
unset _filter

# Set default versions of command line variables for local running
export Name=new_samples_baseline_5param_extended_trk
export Inputs=baseline
export Model=baseline
export N_PARAMS=5
export TRACK_ALGO=extended
export TRAIN=All200.root
export MINBIAS=MinBias_PU200.root
export BBBB=GluGluHHTo4B_PU200.root
export BBBB_EOS_DIR='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_240926_170X_v0.1_full/'
export BBTT=GluGluHHTo2B2Tau_PU200.root
export BBTT_EOS_DIR='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_240926_170X_v0.1_full/'
export VBFHTAUTAU=VBFHToTauTau_PU200.root
export NTUPLE_TREE=outnano/Jets
export SIGNAL=TT_PU200
export EOS_DATA_DIR='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_240926_170X_v0.1_full/'
export CI_PROJECT_NAME='TrainTagger'
export EOS_STORAGE_DIR="/eos/project/c/cms-l1t-jet-tagger/www/${CI_PROJECT_NAME}"

# CMSSW emulation, same defaults as the CI
export CMSSW_VERSION=CMSSW_17_0_0_pre2
export CMSSW_L1CT=CMS-L1T-Jet-Tagging:CMSSW_17_0_0_pre2_NGJet_plus_correlator
export FASTPUPPI_VERSION=CMSSW_17_0_0_pre2_NGJet_plus_correlator
export CMSSW_EMULATOR_WRAPPER=https://github.com/cms-hls4ml/L1TSC4NGJetModel.git
# v1_0_2 for baseline, v2_0_0 for weightedAverageSimple; set before sourcing to override
export EMULATION_WRAPPER_VERSION=${EMULATION_WRAPPER_VERSION:-v1_0_2}
# Emulator output written by setup_cmssw.sh
export JETTUPLE=$PWD/$CMSSW_VERSION/src/FastPUPPI/NtupleProducer/python/jetTuple_${TRACK_ALGO}_${N_PARAMS}.root
# Local run of the emulator on several processes: symlinked inputs and merged outputs
export L1TPF_LOCAL_INPUT_DIR=${L1TPF_LOCAL_INPUT_DIR:-$PWD/localinputs}
export OUTPATH=${OUTPATH:-$PWD/FPOutput}


# Mount that holds this checkout, bound into the cmssw-el8 container (e.g. /shared on NGT).
# Defaults to the top-level directory of $PWD; set WORK_MOUNT before sourcing to override.
export WORK_MOUNT=${WORK_MOUNT:-/$(echo $PWD | cut -d/ -f2)}

# Paths of the input files: the local copy in $DATA_DIR if it exists, otherwise the file on EOS
export DATA_DIR=${DATA_DIR:-$PWD/data}
_input_file() { if [ -f "$DATA_DIR/$2" ]; then echo "$DATA_DIR/$2"; else echo "$1$2"; fi; }
export TRAIN_FILE=$(_input_file "$EOS_DATA_DIR" "$TRAIN")
export SIGNAL_FILE=$(_input_file "$EOS_DATA_DIR" "${SIGNAL}.root")
export MINBIAS_FILE=$(_input_file "$EOS_DATA_DIR" "$MINBIAS")
export BBBB_FILE=$(_input_file "$BBBB_EOS_DIR" "$BBBB")
export BBTT_FILE=$(_input_file "$BBTT_EOS_DIR" "$BBTT")
export VBFHTAUTAU_FILE=$(_input_file "$EOS_DATA_DIR" "$VBFHTAUTAU")
unset -f _input_file
