#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 CONFIG INPUT_FILE OUTPUT_FILE MAX_EVENTS CMSSW_DIR" >&2
    exit 2
fi

CONFIG=$1
INPUT_FILE=$2
OUTPUT_FILE=$3
MAX_EVENTS=$4
CMSSW_DIR=$5
JOB_WORKDIR=${_CONDOR_SCRATCH_DIR:-$(pwd)}

echo "============================================================"
echo "Host:       $(hostname)"
echo "Started:    $(date --iso-8601=seconds)"
echo "Config:     ${CONFIG}"
echo "Input:      ${INPUT_FILE}"
echo "Output:     ${OUTPUT_FILE}"
echo "Max events: ${MAX_EVENTS}"
echo "CMSSW:      ${CMSSW_DIR}"
echo "Scratch:    ${JOB_WORKDIR}"
echo "============================================================"

echo "Checking CMS environment path"
if ! timeout 60 test -r /cvmfs/cms.cern.ch/cmsset_default.sh; then
    echo "Cannot read /cvmfs/cms.cern.ch/cmsset_default.sh" >&2
    exit 10
fi
echo "Checking CMSSW source directory"
if ! timeout 60 test -d "${CMSSW_DIR}/src"; then
    echo "CMSSW source directory does not exist: ${CMSSW_DIR}/src" >&2
    exit 11
fi

echo "Loading the CMS environment"
source /cvmfs/cms.cern.ch/cmsset_default.sh
echo "CMS environment loaded"
cd "${CMSSW_DIR}/src"
echo "Building the scram runtime environment"
if ! SCRAM_RUNTIME=$(timeout 300 scram runtime -sh); then
    echo "scram runtime failed or exceeded five minutes" >&2
    exit 14
fi
eval "${SCRAM_RUNTIME}"
unset SCRAM_RUNTIME
echo "scram runtime environment loaded"
cd "${JOB_WORKDIR}"

if [[ ! -r "${CONFIG}" ]]; then
    echo "Transferred cmsRun config is missing: ${CONFIG}" >&2
    exit 12
fi

# PoolSource needs the file: protocol for an absolute path on the shared
# filesystem. Leave explicitly qualified file:/ and root:// URIs unchanged.
case "${INPUT_FILE}" in
    file:*|root://*) INPUT_URI=${INPUT_FILE} ;;
    /*)              INPUT_URI="file:${INPUT_FILE}" ;;
    *)               INPUT_URI="file:$(pwd)/${INPUT_FILE}" ;;
esac

echo "Starting cmsRun"
stdbuf -oL -eL cmsRun "${CONFIG}" \
    inputFile="${INPUT_URI}" \
    outputFile="${OUTPUT_FILE}" \
    maxEvents="${MAX_EVENTS}"

if [[ ! -s "${OUTPUT_FILE}" ]]; then
    echo "cmsRun returned successfully but did not create ${OUTPUT_FILE}" >&2
    exit 13
fi

echo "Finished: $(date --iso-8601=seconds)"
ls -lh "${OUTPUT_FILE}"
