#!/bin/bash
# $1 RERUN_ON_TAG; if false triggers full retraining of the model, if true just run basic plotting on the tagged model
# $2 RERUN_ON_OLDTRAINING if true run basic plotting on previosuly trained model
# $3 RETRAIN; if "True" initialize the model weights from $RETRAIN_URL before training
# $4 RUN_RETRAIN; if "True" the training input was prepared from $TRAINING_DATA_DIR by the retrain_data job
if [ "$1" == "False" ] && [ "$2" == "False" ]; then
    RETRAIN_OPT=""
    if [ "$3" == "True" ]; then
        echo "Retraining model initialized from: $RETRAIN_URL"
        RETRAIN_OPT="--retrain $RETRAIN_URL"
    fi
    python tagger/train/train.py -p 40 -y tagger/model/configs/$Model.yaml -o output/$Model $RETRAIN_OPT
    # Signal-process inputs for the basic ROC plots: retrain pipelines (RUN_RETRAIN=True)
    # produce retrain_signal_process_data.tgz when the signal nano AOD file sits next to the
    # retrain input; otherwise fall back to the standard dataset, and skip the per-signal
    # plots if neither tarball exists.
    SIG_OPT=""
    SIG_TGZ=""
    if [ "$4" == "True" ] && eos cp "${EOS_STORAGE_DIR}/${EOS_STORAGE_DATADIR}/retrain_signal_process_data.tgz" . 2>/dev/null; then
        SIG_TGZ=retrain_signal_process_data.tgz
    elif eos cp "${EOS_STORAGE_DIR}/${EOS_STORAGE_DATADIR}/signal_process_data.tgz" .; then
        SIG_TGZ=signal_process_data.tgz
    else
        echo "No signal_process_data tarball found, plotting basics without a dedicated signal process"
    fi
    if [ -n "$SIG_TGZ" ]; then
        tar -xf "$SIG_TGZ"
        SIG_OPT="-sig $SIGNAL"
    fi
    python tagger/train/train.py --plot-basic $SIG_OPT -y tagger/model/configs/$Model.yaml -o output/$Model
    cd output/$Model/model
    eos mkdir -p ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/model
    eos cp saved_model.keras ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/model/
    cd ..
    eos cp *.json ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/
    eos cp *.yaml ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/
    eos cp -r testing_data ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/
    rm -rf /eos/${EOS_STORAGE_DIR}/${EOS_STORAGE_PROCSDIR}
    export MODEL_LOCATION=${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}
else
    export MODEL_LOCATION=${EOS_STORAGE_DIR}/tags/${TAG}/
    mkdir -p output/$Model/model
    eos cp ${MODEL_LOCATION}/model/saved_model.* output/$Model/model
    eos cp ${MODEL_LOCATION}/model/*.yaml output/$Model/
    eos cp ${MODEL_LOCATION}/extras/* output/$Model/
    mkdir -p output/$Model/testing_data
    eos cp ${MODEL_LOCATION}/testing_data/* output/$Model/testing_data
    eos cp ${MODEL_LOCATION}/signal_process_data.tgz .
    tar -xf signal_process_data.tgz
    mkdir -p output/$Model/plots/training
    python tagger/train/train.py --plot-basic -sig $SIGNAL -y tagger/model/configs/$Model.yaml -o output/$Model
fi
