#!/bin/bash
mkdir outputs
eos cp -r {$1}* outputs

python tagger/plot/merge_ROCs.py outputs

