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

cd L1Trigger/Phase2L1ParticleFlow
mv data/hadcorr_HGCal3D_TC.root .
rm -r data
git clone https://github.com/cms-data/L1Trigger-Phase2L1ParticleFlow.git
mv L1Trigger-Phase2L1ParticleFlow data
mv hadcorr_HGCal3D_TC.root data
cd ../..

git clone --quiet https://github.com/cms-hls4ml/hls4mlEmulatorExtras.git && \
  cd hls4mlEmulatorExtras &&
  git checkout -b v1.1.3 tags/v1.1.3
make
make install
cd ..
git clone --quiet https://github.com/Xilinx/HLS_arbitrary_Precision_Types.git hls

git clone --quiet ${CMSSW_EMULATOR_WRAPPER}
cd L1TSC4NGJetModel
git checkout emulator_test

cp -r ../../../output/$Model/firmware/L1TSC4NGJetModel/firmware .
./setup.sh v1

make
make install
cd ..

git clone https://github.com/CMS-L1T-Jet-Tagging/FastPUPPI.git -b 15_1_X_NGJet


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
echo ${TRACK_ALGO}
echo  ${N_PARAMS}
sed -i -e 's/trktype = "extended"/trktype = "'${TRACK_ALGO}'"/g' runJetNtuple.py
sed -i -e 's/nparam = 5/nparam = '${N_PARAMS}'/g' runJetNtuple.py
echo "Temporary workaround to get the input files"
echo $'\nprocess.source.fileNames = ["file:/eos/cms/store/cmst3/group/l1tr/FastPUPPI/15_1_X/fpinputs_151X/v1/TT_PU200/inputs151X_10.root"]' >> runJetNtuple.py
<<<<<<< HEAD
echo $'\nprocess.l1tSC4NGJetProducer.l1tSC4NGJetModelPath = cms.string(os.environ["CMSSW_BASE"]+"/src/L1TSC4NGJetModel/L1TSC4NGJetModel")' >> runJetNtuple.py
=======
echo $'\nprocess.l1tSC4NGJetProducer.l1tSC4NGJetModelPath = cms.string(os.environ["CMSSW_BASE"]+"/src/L1TSC4NGJetModel/L1TSC4NGJetModel_v1")' >> runJetNtuple.py
>>>>>>> origin/main
cat runJetNtuple.py
cmsRun runJetNtuple.py --tm18 2>&1 | tee cmsRun.log
