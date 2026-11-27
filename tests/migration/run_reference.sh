#!/bin/bash
# Produce the reference plots of the old tagger/plot scripts, with the same commands and arguments as CI/ml_pipeline.yml.
# Run from the repo root. Needs output/$MODEL (trained model + testing_data), data/ with the CI ntuples
# and signal_process_data/$SIGNAL. The result is copied to output/reference/$MODEL.
#
# Not covered, because they need inputs CI makes in other stages:
#   loss history  (written during training only)
#   makeEmulationPlot.py, hls4ml_profile.py  (need data/jetTuple_*.root and a compiled hls4ml model)
#   emulation_regression/*  (needs the CMSSW FPOutput on EOS)
set -u
MODEL=${MODEL:-baseline}
SIGNAL=${SIGNAL:-TT_PU200}
TREE=outnano/Jets
MINBIAS=data/MinBias_PU200.root
BBBB=data/GluGluHHTo4B_PU200.root
BBTT=data/GluGluHHTo2B2Tau_PU200.root
VBFHTAUTAU=data/VBFHToTauTau_PU200.root
M=output/$MODEL
LOG=$M/reference_logs
mkdir -p "$LOG" $M/plots/training  # CI/run_training.sh creates plots/training before --plot-basic

run() {
    local name=$1; shift
    echo "$(date +%H:%M:%S) start $name: $*"
    if /usr/bin/time -v "$@" > "$LOG/$name.log" 2>&1; then
        echo "$(date +%H:%M:%S) ok    $name ($(grep 'Maximum resident' "$LOG/$name.log" | awk '{print $NF/1e6 " GB"}'))"
    else
        echo "$(date +%H:%M:%S) FAIL  $name, see $LOG/$name.log"
    fi
}

run basic           python tagger/train/train.py --plot-basic -sig $SIGNAL -o $M
run pt_weights      python tagger/plot/pt_weights.py -m $M
run bbtt_rate       python tagger/plot/bbtt.py --deriveRate --minbias $MINBIAS -n 650000 --tree $TREE -m $M
run bbtt_wps        python tagger/plot/bbtt.py --deriveWPs --minbias $MINBIAS --signal $BBTT -n 650000 --tree $TREE -m $M
run bbtt_eff        python tagger/plot/bbtt.py --eff -s $BBTT -n 500000 --tree $TREE -m $M
run ditau_wps       python tagger/plot/diTaus.py --deriveWPs --minbias $MINBIAS -n 650000 --tree $TREE -m $M
run ditau_rate      python tagger/plot/diTaus.py --BkgRate --minbias $MINBIAS -n 500000 --tree $TREE -m $M
run ditau_eff       python tagger/plot/diTaus.py --eff --vbf_sample $VBFHTAUTAU -n 500000 --tree $TREE -m $M
run ditautopo_wps   python tagger/plot/diTaus_topo.py --deriveWPs --minbias $MINBIAS -n 1000000 --tree $TREE -m $M
run ditautopo_rate  python tagger/plot/diTaus_topo.py --BkgRate --minbias $MINBIAS -n 1000000 --tree $TREE -m $M
run ditautopo_eff   python tagger/plot/diTaus_topo.py --eff --vbf_sample $VBFHTAUTAU -n 1000000 --tree $TREE -m $M
run bbbb_wps        python tagger/plot/bbbb.py --deriveWPs --minbias $MINBIAS -n 650000 --tree $TREE -m $M
run bbbb_eff        python tagger/plot/bbbb.py --eff -s $BBBB --minbias $MINBIAS -n 500000 --tree $TREE -m $M
run singletau_wps   python tagger/plot/singleTau.py --deriveWPs --minbias $MINBIAS -n 1000000 --tree $TREE -m $M
run singletau_eff   python tagger/plot/singleTau.py --eff --vbf_sample $VBFHTAUTAU -n 1000000 --tree $TREE -m $M

mkdir -p output/reference
rm -rf output/reference/$MODEL
cp -r $M/plots output/reference/$MODEL
cp -r "$LOG" output/reference/$MODEL/logs
git rev-parse HEAD > output/reference/$MODEL/COMMIT
echo "$(date +%H:%M:%S) done, reference in output/reference/$MODEL"
