export PYTHONPATH=/builds/cebrown/TrainTagger:$PYTHONPATH
export ARTIFACTS_ACCESS_KEY=MINIROOT
export ARTIFACTS_ALIAS=MinIO-Runner
export ARTIFACTS_BUCKET=jetmc
export ARTIFACTS_HOST=cebrown-desktop-ngt:9000
export ARTIFACTS_SECRET_KEY="M1N110_U$£R"

source activate l1_training
mkdir data
cd data
mc cp $ARTIFACTS_ALIAS/$ARTIFACTS_BUCKET/baselineTRK_4param_021024/All200_part0.root ./
cd ..
python datatools/createDataset.py -i data/All200_part0.root -o data/All -t baseline
python train/training.py -t data/All -c btgc -i baseline --train-epochs 15 --model DeepSet --classweights --regression --learning-rate 0.001 --nNodes 16 --optimizer adam --train-batch-size 2048 --strstamp 2024_10_10_vTEST --nLayers 2 --pruning
python evaluation/makeResultPlot.py -t data/All -f trainings_regression_weighted -c btgc -i baseline --model DeepSet -o with_regression --regression --timestamp 2024_10_10_vTEST --pruning
python evaluation/makeInputPlots.py -t data/All -f trainings_regression_weighted -c btgc -i baseline


#get jetTuple.root
docker cp jetTuple.root 785ffefdcccd:/builds/cebrown/TrainTagger
python datatools/createDataset.py -i data/jetTuple.root -o data/Emu -t baseline