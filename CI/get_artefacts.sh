#!/bin/bash
mkdir outputs
for d in $1 ; do
    cp $1/$d/latest/plots/training/basic_ROC.png outputs/basic_ROC_$d.png
done

