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
CMSSW_L1CT_TAG=$3
CMSSW_PR=$4

scram p CMSSW ${CMSSW_VERSION}
cd ${CMSSW_VERSION}/src
[ "$?" != "0" ] && { echo "Unable to set up CMSSW" >&2 ; exit 1; }
eval $(scram runtime -sh)
git cms-init  --upstream-only -q -y
echo "git cms-checkout-topic -u ${CMSSW_L1CT}"
git cms-checkout-topic -u ${CMSSW_L1CT}
echo "git remote add l1ct https://github.com/${CMSSW_L1CT%%:*}/cmssw.git -t ${CMSSW_L1CT##*:} -f"
git remote add l1ct https://github.com/${CMSSW_L1CT%%:*}/cmssw.git -t ${CMSSW_L1CT##*:} -f 2>&1 | grep -v 'new tag.*CMSSW'
[[ "$CMSSW_L1CT_TAG" != "" ]] && echo git checkout -b latest_tag ${CMSSW_L1CT_TAG}
[[ "$CMSSW_L1CT_TAG" != "" ]] && git checkout -b latest_tag ${CMSSW_L1CT_TAG}
if [[ "${CI_JOB_ID}" != "" ]]; then
    git config user.email gpetrbot@cern.ch
    git config user.name "CI bot"
fi;
if [[ "$CMSSW_PR" != "" ]]; then 
    echo "Including CMSSW PR ${CMSSW_PR}"
    if echo ${CMSSW_PR} | grep -q ":"; then
        echo "git cms-merge-topic -u ${CMSSW_PR}"
        git cms-merge-topic -u ${CMSSW_PR}
    else
        echo "git cms-merge-topic -u cms-l1t-offline:${CMSSW_PR}"
        git cms-merge-topic -u cms-l1t-offline:${CMSSW_PR}
    fi;
fi;
git cms-addpkg DataFormats/L1TParticleFlow
git cms-addpkg L1Trigger/Phase2L1ParticleFlow
git cms-addpkg L1Trigger/DemonstratorTools
git cms-addpkg L1Trigger/TrackTrigger
git cms-addpkg SimTracker/TrackTriggerAssociation
# local copy of the json external to avoid tracking the version in all tcl scripts
eval `scram tool info json | grep INCLUDE`; cp -r ${INCLUDE}/nlohmann/ .
eval `scram tool info conifer | grep INCLUDE`; cp  ${INCLUDE}/* .

# local copy data files from externals area if not found (e.g. compositeID json)
RELDATA=$CMSSW_RELEASE_BASE/external/$SCRAM_ARCH/data
for DATAFILE in \
      L1Trigger/Phase2L1ParticleFlow/data/compositeID.json \
      L1Trigger/Phase2L1ParticleFlow/data/jecs/jecs_20220308.root \
; do
    if [ $RELDATA/$DATAFILE -nt $DATAFILE ]; then # includes the case where $DATAFILE is only in the release
        test -d $(dirname $DATAFILE) || mkdir -p $(dirname $DATAFILE) 
        cp -v $RELDATA/$DATAFILE $DATAFILE
    else
        echo "$DATAFILE is newer than the one from the CMSSW release"
    fi;
done
# remove unnecessary packages
perl -ne 'm/Calibration|DQM|Ntuples|HLTrigger|EventFilter.L1TRawToDigi/ or print' -i .git/info/sparse-checkout
git read-tree -mu HEAD

ls 
ls ..
ls ../..
pwd

ls ../../outputSynthesis/regression/
# Deal with Emulator
mv ../../emulation/CMSSW/L1MultiJetProducer_cff.py L1Trigger/Phase2L1ParticleFlow/python
mv ../../emulation/CMSSW/L1MultiJetProducer.cc L1Trigger/Phase2L1ParticleFlow/plugins
mv ../../emulation/CMSSW/MultiJetId.cc L1Trigger/Phase2L1ParticleFlow/src
mv ../../emulation/CMSSW/MultiJetId.h L1Trigger/Phase2L1ParticleFlow/interface
mv ../../emulation/CMSSW/SimL1Emulator_cff.py L1Trigger/Configuration/python
mv ../../emulation/MultiJetTaggerMinimal .
mv ../../outputSynthesis/regression/Training_2024_22_08_vTEST/firmware MultiJetTaggerMinimal/MultiJetMinimal_test
cd MultiJetTaggerMinimal
./setup.sh
cd ..

git clone --quiet https://github.com/cms-hls4ml/hls4mlEmulatorExtras.git && \
  cd hls4mlEmulatorExtras &&
  git checkout -b v1.1.1 tags/v1.1.1
make 
make install
cd ..
git clone --quiet https://github.com/Xilinx/HLS_arbitrary_Precision_Types.git hls
cd MultiJetTaggerMinimal 
make 
make install
cd ..

if [[ "$COMPILE" == "false" ]]; then exit 0; fi
scram b -j 8 -k  2>&1 | tee ../compilation.log | grep '^>>\|[Ee]rror\|out of memory'
if grep -q 'out of memory' ../compilation.log; then
    for retry in 1 2 3; do
        scram b -j 2 -k 2>&1 | tee -a ../compilation.log | grep '^>>\|[Ee]rror\|out of memory' | grep -v 'Compiling python3 modules\|Package\|Product Rules\|symlink'
    done;
fi;
scram b 2>&1 || exit 1

if [[ "$RUN" == "false" ]]; then exit 0; fi

git clone git@github.com:CMS-L1T-Jet-Tagging/FastPUPPI.git -b dev/14_0_X-leptons

cd FastPUPPI/NtupleProducer/

mv ../../../../emulation/CMSSW/JetNTuplizer.cc plugins
mv ../../../../emulation/CMSSW/runPerformanceNTuple.py python

cd python

cmsenv

## See if can read from EOS
if [[ "$(id -gn)" == "zh" ]] && test -d /eos/cms/store/cmst3 ; then
    echo "Will read input files from CMS T3 system"
else
    echo "Temporary workaround to get the input files"
    curl -s https://cerminar.web.cern.ch/cerminar/data/14_0_X/fpinputs_131X/v3/TTbar_PU200/inputs131X_1.root -o inputs131X_1.root
    echo 'process.source.fileNames = ["file:inputs131X_1.root"]' >> runPerformanceNTuple.py
fi
cmsRun runPerformanceNTuple.py --tm18 2>&1 | tee cmsRun.log
mv -v cmsRun.log  ../../../..
mv -v *.dump  ../../../../../dumpfiles/
