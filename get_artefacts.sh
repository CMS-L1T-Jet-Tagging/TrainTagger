#!/bin/bash
for dir in $1/* ; do
    cp $1/$dir/latest/plots/training/basic_ROC.png basic_ROC_$d.png
done

