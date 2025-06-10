source activate tagger
mkdir $Name
mkdir $Name/plots
mv output/$Model/model $Name/model
mv output/$Model/plots/training/ $Name/plots
mv output/$Model/plots/physics/ $Name/plots

if [[ "$RUN_SYNTHESIS" == "True" ]]; then
    tar -cvf L1TSC4NGJetModel.tgz output/$Model/firmware
    eos mkdir -p ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/firmware/
    cp -r L1TSC4NGJetModel.tgz ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/firmware/
    mv output/$Model/plots/profile $Name/plots
fi

if [[ "$RUN_EMULATION" == "True" ]]; then
    mv output/$Model/plots/emulation $Name/plots
fi

cd ..
rm -rf php-plots
git clone https://gitlab-ci-token:${CI_JOB_TOKEN}@gitlab.cern.ch/cms-analysis/general/php-plots.git -b feature/update_extension_grouping
export PATH="${PATH}:/builds/ml_l1/php-plots/bin"
pb_copy_index.py TrainTagger/${Name} --recursive
pb_copy_index.py ${EOS_STORAGE_DIR} --recursive
cd TrainTagger/$Name
pb_deploy_plots.py model ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR} --recursive --extensions h5
pb_deploy_plots.py plots ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR} --recursive --extensions png,pdf,json
cd ../..
mv CI/mlflow_logger.py .
python mlflow_logger.py -w https://cms-l1t-jet-tagger.web.cern.ch/${CI_PROJECT_NAME}/${EOS_STORAGE_SUBDIR} \
      -f ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/firmware \
      -m ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/model \
      -p ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/plots \
      -n ${Name}
eos rm ${EOS_STORAGE_DIR}/branches/${CI_COMMIT_REF_SLUG}/${Name}/latest || true
eos ln ${EOS_STORAGE_DIR}/branches/${CI_COMMIT_REF_SLUG}/${Name}/latest ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}
