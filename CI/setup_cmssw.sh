#!/bin/bash
if [[ "$2" == "" ]]; then
    echo "Usage $0 [ -checkout | -compile | -run ] CMSSW_VERSION GITHUB_MASTER GITHUB_TAG [ GITHUB_PR ]"
    exit 1;
fi;

COMPILE=true; RUN=false
if [[ "$1" == "-checkout" ]]; then COMPILE=false; RUN=false; shift; fi;
if [[ "$1" == "-compile" ]]; then RUN=false; shift; fi;
if [[ "$1" == "-run" ]]; then RUN=true; shift; fi;

CMSSW_VERSION=$1
CMSSW_L1CT=$2

scram p CMSSW ${CMSSW_VERSION}
cd ${CMSSW_VERSION}/src
[ "$?" != "0" ] && { echo "Unable to set up CMSSW" >&2 ; exit 1; }
eval $(scram runtime -sh)
git cms-init  --upstream-only -q -y
echo "git cms-checkout-topic -u ${CMSSW_L1CT}"
git cms-checkout-topic -u ${CMSSW_L1CT}
echo "git remote add l1ct https://github.com/${CMSSW_L1CT%%:*}/cmssw.git -t ${CMSSW_L1CT##*:} -f"
git remote add l1ct https://github.com/${CMSSW_L1CT%%:*}/cmssw.git -t ${CMSSW_L1CT##*:} -f 2>&1 | grep -v 'new tag.*CMSSW'

git cms-addpkg L1Trigger/Phase2L1ParticleFlow
git cms-addpkg L1Trigger/Configuration

# # local copy of the json external to avoid tracking the version in all tcl scripts
# eval `scram tool info json | grep INCLUDE`; cp -r ${INCLUDE}/nlohmann/ .
# eval `scram tool info conifer | grep INCLUDE`; cp  ${INCLUDE}/* .

# # local copy data files from externals area if not found (e.g. compositeID json)
# RELDATA=$CMSSW_RELEASE_BASE/external/$SCRAM_ARCH/data
# for DATAFILE in \
#       L1Trigger/Phase2L1ParticleFlow/data/compositeID.json \
#       L1Trigger/Phase2L1ParticleFlow/data/jecs/jecs_20220308.root \
# ; do
#     if [ $RELDATA/$DATAFILE -nt $DATAFILE ]; then # includes the case where $DATAFILE is only in the release
#         test -d $(dirname $DATAFILE) || mkdir -p $(dirname $DATAFILE) 
#         cp -v $RELDATA/$DATAFILE $DATAFILE
#     else
#         echo "$DATAFILE is newer than the one from the CMSSW release"
#     fi;
# done
# # remove unnecessary packages
# perl -ne 'm/Calibration|DQM|Ntuples|HLTrigger|EventFilter.L1TRawToDigi/ or print' -i .git/info/sparse-checkout
# git read-tree -mu HEAD


git clone --quiet https://github.com/cms-hls4ml/hls4mlEmulatorExtras.git && \
  cd hls4mlEmulatorExtras &&
  git checkout -b v1.1.3 tags/v1.1.3
make 
make install
cd ..
git clone --quiet https://github.com/Xilinx/HLS_arbitrary_Precision_Types.git hls

git config user.email chris.brown@fpsl.net
git config user.name "Chriisbrown"



git clone --quiet https://github.com/CMS-L1T-Jet-Tagging/hls4ml-jettagger.git && \
  cd hls4ml-jettagger
  git checkout -b hls4ml-v081


ls ../../..
cp -r ../../../outputSynthesis/regression/Training_2024_10_10_vTEST/firmware MultiJetBaseline/
./setup.sh
cd ..

make 
make install
cd ..

git clone https://github.com/CMS-L1T-Jet-Tagging/FastPUPPI.git -b addMultiJet


if [[ "$COMPILE" == "false" ]]; then exit 0; fi
scram b -j 8 -k  2>&1 | tee ../compilation.log | grep '^>>\|[Ee]rror\|out of memory'
if grep -q 'out of memory' ../compilation.log; then
    for retry in 1 2 3; do
        scram b -j 2 -k 2>&1 | tee -a ../compilation.log | grep '^>>\|[Ee]rror\|out of memory' | grep -v 'Compiling python3 modules\|Package\|Product Rules\|symlink'
    done;
fi;
scram b 2>&1 || exit 1

if [[ "$RUN" == "false" ]]; then exit 0; fi

cd FastPUPPI/NtupleProducer/python

cmsenv

echo "Temporary workaround to get the input files"
curl -s https://cerminar.web.cern.ch/cerminar/data/14_0_X/fpinputs_131X/v3/TTbar_PU200/inputs131X_1.root -o inputs131X_1.root
cmsRun runPerformanceNTuple.py --tm18 2>&1 | tee cmsRun.log
