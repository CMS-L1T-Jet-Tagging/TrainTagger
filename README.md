# Training Jet Taggers for CMS Phase 2 L1 Trigger

Documentation for training a L1T Jet Tagging model. 

## Produce Training Dataset

Creating the training datasets involve several steps: 

1. Taking the RAW samples and pruning them, this is currently done for all, and stored in here:

```
/eos/cms/store/cmst3/group/l1tr/FastPUPPI/14_0_X/fpinputs_131X/v9a/
```

2. This samples would then be run by the nTuplizer, which is part of the [FastPUPPI](https://github.com/CMS-L1T-Jet-Tagging/FastPUPPI/tree/dev/14_0_X-leptons) repo. In particular the `runPerformanceNTuple.py`, which calls `jetNTuplizer.cc`. Note that to submit jobs as part of this setup, you also need the [submission](https://github.com/CMS-L1T-Jet-Tagging/submission/tree/dev/14_0_X-leptons) repo as well. 
3. Finally, from the ntuple outputs of FastPUPPI, we could create the training parquet files using:
```
python3 createDataset_chunks_new.py -f extendedAll200
```

## Training the model

The model can be trained using this command:

```
python3 training.py -f extendedAll200 -c btgc -i minimal --train-epochs 15 --model DeepSet --classweights --regression --learning-rate 0.001 --nNodes 16 --optimizer adam --train-batch-size 2048 --strstamp 2024_07_22_vTEST --nLayers 2 --pruning --test
```


## Synthesize the model to HDL Codes
```
python3 synthesis.py -f extendedAll200 -c btgc -i minimal -m DeepSet -o regression --regression --timestamp 2024_07_22_vTEST --pruning -B
```
