#!/bin/bash

# DEBUG
echo "Hostname: $(hostname)"
echo "Arguments: $@"

# setup environment
eval "$(/nfs/dust/cms/user/sewuchte/GNN_weaver/miniconda/bin/conda shell.zsh hook)"
# conda activate hls4ml-l1jets
conda activate hls4ml-l1jets-38
cd /nfs/dust/cms/user/sewuchte/L1-TrainingStuff/TrainingCode/

export COMMENT=$1

echo comment is ${COMMENT}
echo commands training are "${@:1}"
echo commands eval are "${@:1}"

ulimit -s unlimited

# ./train_0L_ttZ.sh "${@:2}"
# python training.py -f All200 -c btgc -i baseline --train-epochs 200 --model deepset --classweights --regression --learning-rate 0.001 --nNodes 20 --optimizer adam --train-batch-size 2048 --pruning
# python training.py -f $1 -c btgc -i $2 --train-epochs 1 --model $3 --classweights --regression --learning-rate $4 --nNodes 20 --optimizer adam --train-batch-size 2048 --pruning --strstamp 2024_03_13_v0
python3 training.py -f $1 -c btgc -i $2 --train-epochs 200 --model $3 --classweights --regression --learning-rate $4 --nNodes $6 --optimizer adam --train-batch-size 2048 --pruning --strstamp $5 --nLayers $7

echo =====================================================================================================================
echo =====================================================================================================================
echo =====================================================================================================================

export COMMENT=$1

# python makeResultPlot.py -f All200 -c btgc -i baseline -m DeepSet -o baseline_deepset_regression --splitTau --splitGluon --splitCharm --regression --timestamp 2024_2_27-21_41 --pruning
python3 makeResultPlot.py -f $1 -c btgc -i $2 -m $3 -o with_regression --splitTau --splitGluon --splitCharm --regression --timestamp $5 --pruning

echo =====================================================================================================================
echo =====================================================================================================================
echo =====================================================================================================================

source setup.sh

python3 synthesis.py -f $1 -c btgc -i $2 -m $3 -o with_regression --splitTau --splitGluon --splitCharm --regression --timestamp $5 -B --pruning