# Training Jet Taggers for CMS Phase 2 L1 Trigger

Documentation for training a L1T Jet Tagging model

# Produce NTuples

`python3 createDataset_chunks_new.py -f extendedAll200`

# Training the model
`python3 training.py -f extendedAll200 -c btgc -i minimal --train-epochs 15 --model DeepSet --classweights --regression --learning-rate 0.001 --nNodes 16 --optimizer adam --train-batch-size 2048 --strstamp 2024_07_22_vTEST --nLayers 2 --pruning --test`

# Producing validation plots
`python3 makeResultPlot.py -f extendedAll200 -c btgc -i minimal --model DeepSet -o NewBaselineDuc --regression --pruning --timestamp 2024_07_22_vTEST`

# Synthesis
`python3 synthesis.py -f extendedAll200 -c btgc -i minimal -m DeepSet -o regression --regression --timestamp 2024_07_22_vTEST --pruning -B`