#!/usr/bin/env bash
# Source this file (do not execute it) before running jobs that need TensorFlow:
#
#   source ci/setup_tagger_env.sh
#
# What it does:
#   1. Activates the mamba env (default: "tagger")
#   2. Puts the env's lib directory first on LD_LIBRARY_PATH so TensorFlow
#      and numpy load the env's libstdc++ (GLIBCXX_3.4.29+) instead of the
#      old system one in /usr/lib64.
#
# Optional overrides (set before sourcing):
#   TAGGER_ENV_NAME    env name or path          (default: tagger)
#   MAMBA_ROOT_PREFIX  mamba root prefix    (default: /opt/conda)
#   TAGGER_VERIFY=1    import tensorflow after setup as a sanity check
#
# Note: no "set -e" here on purpose; it would leak into the calling shell.
 
_tagger_env="${TAGGER_ENV_NAME:-tagger}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/opt/conda}"
 
if ! command -v mamba >/dev/null 2>&1; then
    echo "setup_tagger_env: mamba not found in PATH" >&2
    return 1 2>/dev/null || exit 1
fi
 
# Non-interactive CI shells don't load the mamba hook, so do it here.
eval "$(mamba shell hook --shell bash)"
 
if ! mamba activate "$_tagger_env"; then
    echo "setup_tagger_env: failed to activate env '$_tagger_env'" >&2
    return 1 2>/dev/null || exit 1
fi
 
# Prepend the env's lib dir (idempotent: skip if it is already first).
case "${LD_LIBRARY_PATH:-}" in
    "$CONDA_PREFIX/lib"|"$CONDA_PREFIX/lib:"*) ;;
    *) export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
 
# Add pip-installed NVIDIA CUDA libs (cusolver, cudnn, cublas, ...) so
# TensorFlow can dlopen all of them, not just the ones its RPATH covers.
_tagger_nv=""
for _d in "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/*/lib; do
    [ -d "$_d" ] && _tagger_nv="${_tagger_nv:+$_tagger_nv:}$_d"
done
case "${LD_LIBRARY_PATH:-}" in
    *site-packages/nvidia/*) ;;
    *) [ -n "$_tagger_nv" ] && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$_tagger_nv" ;;
esac
unset _tagger_nv _d
 
echo "setup_tagger_env: using $(command -v python) ($(python -V 2>&1))"
 
if [ "${TAGGER_VERIFY:-0}" = "1" ]; then
    python -c "import tensorflow as tf; print('TensorFlow', tf.__version__, 'GPUs:', tf.config.list_physical_devices('GPU'))" \
        || { echo "setup_tagger_env: tensorflow import failed" >&2; return 1 2>/dev/null || exit 1; }
fi
 
unset _tagger_env
# To run before using the codes
export PYTHONPATH=$PYTHONPATH:$PWD
export CI_COMMIT_REF_NAME=local

# Set default versions of command line variables for local running
export Name=DeepSetHGQ2
export Config=HGQ2/DeepSets_HGQ2.yaml
export Inputs=baseline
export Model=DeepSetsHGQ2
export N_PARAMS=5
export TRACK_ALGO=extended
export TRAIN=All200.root
export MINBIAS=MinBias_PU200.root
export BBBB=GluGluHHTo4B_PU200.root
export BBBB_EOS_DIR='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_090125_addGenH/'
export BBTT=GluGluHHTo2B2Tau_PU200.root
export BBTT_EOS_DIR='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_090125_addGenH/'
export VBFHTAUTAU=VBFHToTauTau_PU200.root
export NTUPLE_TREE=outnano/Jets
export SIGNAL=TT_PU200
export EOS_DATA_DIR='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_090125/'
export CI_PROJECT_NAME='TrainTagger'
export EOS_STORAGE_DIR='/eos/project/c/cms-l1t-jet-tagger/www/${CI_PROJECT_NAME}'