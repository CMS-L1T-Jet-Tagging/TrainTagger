# python3 training.py -f extendedAll200 -c btgc -i baselineHW --train-epochs 200 --model DeepSet --classweights --regression --learning-rate 0.001 --nNodes 16 --optimizer adam --train-batch-size 2048 --strstamp 2024_07_23_v1 --nLayers 2 --pruning
# python3 training.py -f extendedAll200 -c btgc -i baselineHWMe --train-epochs 200 --model DeepSet --classweights --regression --learning-rate 0.001 --nNodes 16 --optimizer adam --train-batch-size 2048 --strstamp 2024_07_23_v2 --nLayers 2 --pruning
# python3 training.py -f extendedAll200 -c btgc -i baselineEmulator --train-epochs 200 --model DeepSet --classweights --regression --learning-rate 0.001 --nNodes 16 --optimizer adam --train-batch-size 2048 --strstamp 2024_07_23_v3 --nLayers 2 --pruning
# python3 training.py -f extendedAll200 -c btgc -i baselineEmulatorMe --train-epochs 200 --model DeepSet --classweights --regression --learning-rate 0.001 --nNodes 16 --optimizer adam --train-batch-size 2048 --strstamp 2024_07_23_v4 --nLayers 2 --pruning
# python3 training.py -f extendedAll200 -c btgc -i baselineEmulatorAdd --train-epochs 200 --model DeepSet --classweights --regression --learning-rate 0.001 --nNodes 16 --optimizer adam --train-batch-size 2048 --strstamp 2024_07_23_v5 --nLayers 2 --pruning
# python3 training.py -f extendedAll200 -c btgc -i minimal --train-epochs 200 --model DeepSet --classweights --regression --learning-rate 0.001 --nNodes 16 --optimizer adam --train-batch-size 2048 --strstamp 2024_07_23_v6 --nLayers 2 --pruning
# python3 training.py -f extendedAll200 -c btgc -i minimalMe --train-epochs 200 --model DeepSet --classweights --regression --learning-rate 0.001 --nNodes 16 --optimizer adam --train-batch-size 2048 --strstamp 2024_07_23_v7 --nLayers 2 --pruning
# python3 training.py -f extendedAll200 -c btgc -i ext1 --train-epochs 200 --model DeepSet --classweights --regression --learning-rate 0.001 --nNodes 16 --optimizer adam --train-batch-size 2048 --strstamp 2024_07_23_v8 --nLayers 2 --pruning
# python3 training.py -f extendedAll200 -c btgc -i ext2 --train-epochs 200 --model DeepSet --classweights --regression --learning-rate 0.001 --nNodes 16 --optimizer adam --train-batch-size 2048 --strstamp 2024_07_23_v9 --nLayers 2 --pruning
# python3 training.py -f extendedAll200 -c btgc -i ext3 --train-epochs 200 --model DeepSet --classweights --regression --learning-rate 0.001 --nNodes 16 --optimizer adam --train-batch-size 2048 --strstamp 2024_07_23_v10 --nLayers 2 --pruning
# python3 training.py -f extendedAll200 -c btgc -i ext4 --train-epochs 200 --model DeepSet --classweights --regression --learning-rate 0.001 --nNodes 16 --optimizer adam --train-batch-size 2048 --strstamp 2024_07_23_v11 --nLayers 2 --pruning
# python3 training.py -f extendedAll200 -c btgc -i ext5 --train-epochs 200 --model DeepSet --classweights --regression --learning-rate 0.001 --nNodes 16 --optimizer adam --train-batch-size 2048 --strstamp 2024_07_23_v12 --nLayers 2 --pruning
# python3 training.py -f extendedAll200 -c btgc -i ext6 --train-epochs 200 --model DeepSet --classweights --regression --learning-rate 0.001 --nNodes 16 --optimizer adam --train-batch-size 2048 --strstamp 2024_07_23_v13 --nLayers 2 --pruning


# python3 makeResultPlot.py -f extendedAll200 -c btgc -i baselineHW --model DeepSet -o NewBaseline --regression --pruning --timestamp 2024_07_23_v1
# python3 makeResultPlot.py -f extendedAll200 -c btgc -i baselineHWMe --model DeepSet -o NewBaseline --regression --pruning --timestamp 2024_07_23_v2
# python3 makeResultPlot.py -f extendedAll200 -c btgc -i baselineEmulator --model DeepSet -o NewBaseline --regression --pruning --timestamp 2024_07_23_v3
# python3 makeResultPlot.py -f extendedAll200 -c btgc -i baselineEmulatorMe --model DeepSet -o NewBaseline --regression --pruning --timestamp 2024_07_23_v4
# python3 makeResultPlot.py -f extendedAll200 -c btgc -i baselineEmulatorAdd --model DeepSet -o NewBaseline --regression --pruning --timestamp 2024_07_23_v5
# python3 makeResultPlot.py -f extendedAll200 -c btgc -i minimal --model DeepSet -o NewBaseline --regression --pruning --timestamp 2024_07_23_v6
# python3 makeResultPlot.py -f extendedAll200 -c btgc -i minimalMe --model DeepSet -o NewBaseline --regression --pruning --timestamp 2024_07_23_v7
# python3 makeResultPlot.py -f extendedAll200 -c btgc -i ext1 --model DeepSet -o NewBaseline --regression --pruning --timestamp 2024_07_23_v8
# python3 makeResultPlot.py -f extendedAll200 -c btgc -i ext2 --model DeepSet -o NewBaseline --regression --pruning --timestamp 2024_07_23_v9
# python3 makeResultPlot.py -f extendedAll200 -c btgc -i ext3 --model DeepSet -o NewBaseline --regression --pruning --timestamp 2024_07_23_v10
# python3 makeResultPlot.py -f extendedAll200 -c btgc -i ext4 --model DeepSet -o NewBaseline --regression --pruning --timestamp 2024_07_23_v11
# python3 makeResultPlot.py -f extendedAll200 -c btgc -i ext5 --model DeepSet -o NewBaseline --regression --pruning --timestamp 2024_07_23_v12
# python3 makeResultPlot.py -f extendedAll200 -c btgc -i ext6 --model DeepSet -o NewBaseline --regression --pruning --timestamp 2024_07_23_v13


python3 training.py -f extendedAll200 -c btgc -i ext3 --train-epochs 100 --model DeepSet --classweights --regression --learning-rate 0.001 --nNodes 32 --optimizer adam --train-batch-size 2048 --strstamp 2024_07_25_v10 --nLayers 2 --pruning
