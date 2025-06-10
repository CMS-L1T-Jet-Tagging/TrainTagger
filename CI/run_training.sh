#!/bin/bash
if [[ "$1" == "False" ]]; then
    python tagger/train/train.py -n $Name -p 50
    eos cp ${EOS_STORAGE_DIR}/${EOS_STORAGE_DATADIR}/signal_process_data.tgz .
    tar -xf signal_process_data.tgz
    python tagger/train/train.py --plot-basic -n $Name -sig $SIGNAL
    cd output/baseline
    eos mkdir -p ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/model
    eos cp model/saved_model.h5 ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/model/saved_model.h5 .
    export MODEL_LOCATION=${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}
else
    mkdir -p output/baseline/model
    eos cp ${MODEL_LOCATION}/model/saved_model.h5 output/baseline/model
    eos cp ${MODEL_LOCATION}/extras/* output/baseline/
    mkdir -p output/baseline/testing_data
    eos cp ${MODEL_LOCATION}/testing_data/* output/baseline/testing_data
    eos cp ${MODEL_LOCATION}/signal_process_data.tgz .
    tar -xf signal_process_data.tgz
    python tagger/train/train.py --plot-basic -n $Name -sig $SIGNAL
fi
