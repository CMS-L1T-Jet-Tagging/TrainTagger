source activate tagger
echo "Name: " ${Name}
echo "Model type: " ${Model}
echo "EOS STORAGE DIR: " ${EOS_STORAGE_DIR}
echo "EOS STORAGE SUBDIR: " ${EOS_STORAGE_SUBDIR}
echo "EOS STORAGE save directory: " ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}
echo "EOS STORAGE save directory latest tag: " ${EOS_STORAGE_DIR}/branches/${CI_COMMIT_REF_SLUG}/${Name}/latest


if [[-n $MODEL_OUTPUT_PATH]]; then
    export UPLOAD_DIRECTORY=$MODEL_OUTPUT_PATH
else
    export UPLOAD_DIRECTORY=${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}
fi

# Verify that all the variables exist so paths aren't overwritten
if [ -z "${Name}" ] || [ -z "${Model}" ] || [ -z "${EOS_STORAGE_DIR}" ] || [ -z "${EOS_STORAGE_SUBDIR}" ]; then
    echo "Missing a saving directory parameter, check print statments above"
    exit 1
fi
mkdir $Name
mkdir $Name/plots
mkdir $Name/configs

mv output/$Model/*.json $Name/configs
mv output/$Model/*.yaml $Name/configs
mv output/$Model/configs/* $Name/configs

mv output/$Model/model $Name/model

mv output/$Model/plots/training/ $Name/plots
mv output/$Model/plots/physics/ $Name/plots


eos mkdir -p $UPLOAD_DIRECTORY
eos cp -r $Name/model $UPLOAD_DIRECTORY
eos cp -r $Name/plots $UPLOAD_DIRECTORY

mkdir $Name/configs
mv output/$Model/*.json $Name/configs
mv output/$Model/*.yaml $Name/configs

eos cp -r $Name/configs $UPLOAD_DIRECTORY
eos cp output/$Model/results_dict.json $UPLOAD_DIRECTORY

fi

if [[ "$RUN_SYNTHESIS" == "True" ]]; then
    cd output/$Model/firmware/
    tar -cvf L1TSC4NGJetModel_$HLS4ML_NAME.tgz L1TSC4NGJetModel_$HLS4ML_NAME
    eos mkdir -p $UPLOAD_DIRECTORY/firmware/
    eos cp -r L1TSC4NGJetModel.tgz $UPLOAD_DIRECTORY/firmware/
    cd ../../..
    mv output/$Model/plots/profile $Name/plots
fi

if [[ "$RUN_EMULATION" == "True" ]]; then
    mv output/$Model/plots/emulation $Name/plots
    eos mkdir -p $UPLOAD_DIRECTORY/emulator/
    eos cp -r ${CMSSW_VERSION}/src/L1TSC4NGJetModel $UPLOAD_DIRECTORY/emulator
fi

cd ..
rm -rf php-plots
git clone https://gitlab-ci-token:${CI_JOB_TOKEN}@gitlab.cern.ch/cebrown/php-plots -b feature/update_extension_grouping
export PATH="${PATH}:/builds/ml_l1/php-plots/bin"
pb_copy_index.py TrainTagger/${Name} --recursive
pb_copy_index.py $UPLOAD_DIRECTORY --recursive
cd TrainTagger/$Name
ls $UPLOAD_DIRECTORY

eos mkdir -p $UPLOAD_DIRECTORY/model
eos mkdir -p $UPLOAD_DIRECTORY/plots
eos mkdir -p $UPLOAD_DIRECTORY/configs

eos cp -r model/* $UPLOAD_DIRECTORY/model
eos cp -r plots/* $UPLOAD_DIRECTORY/plots
eos cp -r configs/* $UPLOAD_DIRECTORY/configs

pb_copy_index.py $UPLOAD_DIRECTORY/plots --recursive

if [[-n $MODEL_OUTPUT_PATH]]; then
    eos rm ${EOS_STORAGE_DIR}/branches/${CI_COMMIT_REF_SLUG}/${Name}/latest || true
    eos ln ${EOS_STORAGE_DIR}/branches/${CI_COMMIT_REF_SLUG}/${Name}/latest $UPLOAD_DIRECTORY
