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
mkdir $Name/configs

mv output/$Model/*.json $Name/configs
mv output/$Model/*.yaml $Name/configs

mv output/$Model/model $Name/model

# eval results (tagger.eval), in the layout of the model folder so tagger.plot can read them back:
# eval/, hls4ml/eval/ and emulator/eval/ of the model versions, cmssw/eval/ of the CMSSW L1 baseline
mv output/$Model/eval $Name/eval
for version in hls4ml emulator; do
    if [ -d output/$Model/$version/eval ]; then
        mkdir -p $Name/$version
        mv output/$Model/$version/eval $Name/$version/eval
    fi
done
if [ -d output/cmssw/eval ]; then
    mkdir -p $Name/cmssw
    mv output/cmssw/eval $Name/cmssw/eval
fi

mv output/$Model/plots/training/ $Name/plots
mv output/$Model/plots/physics/ $Name/plots
mv output/$Model/plots/emulation_regression/ $Name/plots

if [[ "$RUN_SYNTHESIS" == "True" ]]; then
    cd output/$Model/firmware/
    tar -cvf L1TSC4NGJetModel.tgz L1TSC4NGJetModel
    eos mkdir -p ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/firmware/
    eos cp -r L1TSC4NGJetModel.tgz ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/firmware/
    cd ../../..
    mv output/$Model/plots/profile $Name/plots
fi

if [[ "$RUN_EMULATION" == "True" ]]; then
    mv output/$Model/plots/emulation $Name/plots
    eos mkdir -p ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/emulator/
    eos cp -r ${CMSSW_VERSION}/src/L1TSC4NGJetModel ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/emulator
fi

cd ..
rm -rf php-plots
git clone https://gitlab-ci-token:${CI_JOB_TOKEN}@gitlab.cern.ch/cebrown/php-plots -b feature/update_extension_grouping
export PATH="${PATH}:/builds/ml_l1/php-plots/bin"
pb_copy_index.py TrainTagger/${Name} --recursive
pb_copy_index.py ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR} --recursive
cd TrainTagger/$Name

ls ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}

eos mkdir -p ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/model
eos mkdir -p ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/plots
eos mkdir -p ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/configs

eos cp -r model/* ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/model
eos cp -r plots/* ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/plots
eos cp -r configs/* ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/configs

eos mkdir -p ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/eval
eos cp -r eval/* ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/eval
for results in hls4ml emulator cmssw; do
    if [ -d $results/eval ]; then
        eos mkdir -p ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/$results/eval
        eos cp -r $results/eval/* ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/$results/eval
    fi
done

pb_copy_index.py ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/plots --recursive

eos rm ${EOS_STORAGE_DIR}/branches/${CI_COMMIT_REF_SLUG}/${Name}/latest || true
eos ln ${EOS_STORAGE_DIR}/branches/${CI_COMMIT_REF_SLUG}/${Name}/latest ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}
