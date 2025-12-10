source activate tagger
echo "Name: " ${Name}
echo "Model type: " ${Model}
echo "EOS STORAGE DIR: " ${EOS_STORAGE_DIR}
echo "EOS STORAGE SUBDIR: " ${EOS_STORAGE_SUBDIR}
echo "EOS STORAGE save directory: " ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}
echo "EOS STORAGE save directory latest tag: " ${EOS_STORAGE_DIR}/branches/${CI_COMMIT_REF_SLUG}/${Name}/latest

# Verify that all the variables exist so paths aren't overwritten
if [ -z "${Name}" ] || [ -z "${Model}" ] || [ -z "${EOS_STORAGE_DIR}" ] || [ -z "${EOS_STORAGE_SUBDIR}" ]; then
    echo "Missing a saving directory parameter, check print statments above"
    exit 1
fi
mkdir $Name
mkdir $Name/plots
mv output/$Model/model $Name/model
mv output/$Model/plots/training/ $Name/plots
mv output/$Model/plots/physics/ $Name/plots

if [[ "$RUN_SYNTHESIS" == "True" ]]; then
    cd output/$Model/firmware/
    tar -cvf L1TSC4NGJetModel.tgz L1TSC4NGJetModel
    eos mkdir -p ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/firmware/
    cp -r L1TSC4NGJetModel.tgz ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/firmware/
    cd ../../..
    mv output/$Model/plots/profile $Name/plots
fi

if [[ "$RUN_EMULATION" == "True" ]]; then
    mv output/$Model/plots/emulation $Name/plots
    eos mkdir -p ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/emulator/
    cp -r ${CMSSW_VERSION}/src/L1TSC4NGJetModel ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/emulator
fi

cd ..
rm -rf php-plots
git clone https://gitlab-ci-token:${CI_JOB_TOKEN}@gitlab.cern.ch/cebrown/php-plots -b feature/update_extension_grouping
export PATH="${PATH}:/builds/ml_l1/php-plots/bin"
pb_copy_index.py TrainTagger/${Name} --recursive
pb_copy_index.py ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR} --recursive
cd TrainTagger/$Name
pb_deploy_plots.py model ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR} --recursive --extensions h5
pb_deploy_plots.py plots ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR} --recursive --extensions png,pdf,json
eos rm ${EOS_STORAGE_DIR}/branches/${CI_COMMIT_REF_SLUG}/${Name}/latest || true
eos ln ${EOS_STORAGE_DIR}/branches/${CI_COMMIT_REF_SLUG}/${Name}/latest ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}
