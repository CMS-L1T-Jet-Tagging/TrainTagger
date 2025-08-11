#!/bin/bash
mkdir outputs
for d in $1 ; do
    echo $d 
    echo $1/$d
    cp $1/$d/latest/plots/plotting_dict.pkl outputs/${d}_plotting_dict.pkl
done

python tagger/plot/merge_ROCs.py outputs

