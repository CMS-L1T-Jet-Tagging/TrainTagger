#!/bin/bash

# Find all model names in the output directory
folder=$1
models=$(ls -1 output/${folder}/*_training_out.txt | sed 's/_training_out.txt//' | xargs -I {} basename {} | sort -u)

# Run the python program for each model
for model in $models; do
  echo "=== $model ==="
  python output/read_output.py $model $folder
  echo ""
done
