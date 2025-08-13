#!/bin/bash
mkdir outputs

array=($(eos ls ${1}))

for i in "${array[@]}"
do
   echo "$i"
   echo "======="
   mkdir outputs/$i
   eos cp -r ${1}/$i/latest/plots/*.pkl outputs/$i

done

python tagger/plot/merge_ROCs.py outputs

