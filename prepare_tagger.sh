#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$PWD
export CI_COMMIT_REF_NAME=local

micromamba activate tagger2
