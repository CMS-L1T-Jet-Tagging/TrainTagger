#!/bin/bash
# $1 RERUN_ON_TAG; if false triggers full retraining of the model, if true just run basic plotting on the tagged model

if [[ "$1" == "False" ]]; then
    python tagger/train/train.py -p 50 -y tagger/model/configs/$Model.yaml -o output/$Model
    eos cp ${EOS_STORAGE_DIR}/${EOS_STORAGE_DATADIR}/signal_process_data.tgz .
    tar -xf signal_process_data.tgz
    python tagger/train/train.py --plot-basic -sig $SIGNAL -y tagger/model/configs/$Model.yaml -o output/$Model
    cd output/$Model/model
    eos mkdir -p ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/model
    eos cp saved_model.keras ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/model/
    export MODEL_LOCATION=${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}
else
    mkdir -p output/$Model/model
    eos cp ${MODEL_LOCATION}/model/saved_model.* output/$Model/model
    eos cp ${MODEL_LOCATION}/extras/* output/$Model/
    mkdir -p output/$Model/testing_data
    eos cp ${MODEL_LOCATION}/testing_data/* output/$Model/testing_data
    eos cp ${MODEL_LOCATION}/signal_process_data.tgz .
    tar -xf signal_process_data.tgz
    python tagger/train/train.py --plot-basic -sig $SIGNAL -y tagger/model/configs/$Model.yaml -o output/$Model
fi
