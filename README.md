# Training Jet Taggers for CMS Phase 2 L1 Trigger

Documentation for training a L1T Jet Tagging model. 

## Conda environment

Create conda environment:

```
conda-env create -f environment.yml
```

Activate the environment:

```
conda activate l1_training
```

And then do whatever you want in this environment (edit files, open notebooks, etc.). To deactivate the environment:

```
conda deactivate
```

If you make any update for the environment, please edit the `environment.yml` file and run:

```
conda env update --file environment.yml  --prune
```

Reference on conda environment here: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

## Produce Training Dataset

Creating the training datasets involve several steps: 

1. Taking the RAW samples and pruning/sliming them. This can be done running the `runInputs_X_X.py` scripts in [FastPUPPI](https://github.com/CMS-L1T-Jet-Tagging/FastPUPPI/tree/dev/14_0_X-leptons), which also uses [submission](https://github.com/CMS-L1T-Jet-Tagging/submission) repo. This is currently done for all, and stored in here:

```
/eos/cms/store/cmst3/group/l1tr/FastPUPPI/14_0_X/fpinputs_131X/v9a/
```

2. This samples would then be run by the nTuplizer, which is part of the [FastPUPPI](https://github.com/CMS-L1T-Jet-Tagging/FastPUPPI/tree/dev/14_0_X-leptons) repo. In particular the `runPerformanceNTuple.py`, which calls `jetNTuplizer.cc`. Note that to submit jobs as part of this setup, you also need the [submission](https://github.com/CMS-L1T-Jet-Tagging/submission/tree/dev/14_0_X-leptons) repo as well. 

3. Finally, from the ntuple outputs of FastPUPPI, we could create the training parquet files using:

```
python utils/createDataset.py -i <input ntuple root file> -o <output directory>
```

The -i and -o values are optional, see the script for the default values.

## Training

### Training the model

The model can be trained using this command:

```
python train/training.py -f extendedAll200 -c btgc -i minimal --train-epochs 15 --model DeepSet --classweights --regression --learning-rate 0.001 --nNodes 16 --optimizer adam --train-batch-size 2048 --strstamp 2024_07_22_vTEST --nLayers 2 --pruning --test
```


## Synthesize the model to HDL Codes
```
python3 synthesis.py -f extendedAll200 -c btgc -i minimal -m DeepSet -o regression --regression --timestamp 2024_07_22_vTEST --pruning -B
```
