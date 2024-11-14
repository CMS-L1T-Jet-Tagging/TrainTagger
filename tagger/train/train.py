from argparse import ArgumentParser

#Import from other modules
from tagger.data.tools import make_data, load_data, to_ML
from models import baseline_model

# from qkeras import quantized_bits

#GLOBAL PARAMETERS TO BE DEFINED WHEN TRAINING
# INPUT_QUANTIZER = quantized_bits(bits=15, integer=12, symmetric=0, alpha=1) #To be consistent with firmware implementation
BATCH_SIZE = 1024
EPOCHS = 100
VALIDATION_SPLIT = 0.1 # 10% of training set will be used for validation set. 

# sparsity parameters
I_SPARSITY = 0.2 #Initial sparsity
F_SPARSITY = 0.7 #Final sparsity

def train(out_dir, percent):

    #Load the data
    data = load_data("training_data/", percentage=percent)

    #Make into ML-like data for training
    test = to_ML(data)

    #Get the model
    # model = baseline_model()

    #Train it with sparsity

    #Produce some basic plots with the training

    return

if __name__ == "__main__":

    parser = ArgumentParser()

    #Making input arguments
    parser.add_argument('--make-data', action='store_true', help='Prepare the data if set.')
    parser.add_argument('-i','--input', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_ntuples_v131Xv9/baselineTRK_4param_021024/All200.root' , help = 'Path to input training data')
    parser.add_argument('-s','--step', default='100MB' , help = 'The maximum memory size to process input root file')

    #Training argument
    parser.add_argument('-o','--output', default='output/baseline', help = 'Output model directory path, also save evaluation plots')
    parser.add_argument('-p','--percent', default=100, type=int, help = 'Percentage of how much processed data to train on')
    
    args = parser.parse_args()

    #Either make data or start the training
    if args.make_data:
        make_data(infile=args.input, step_size=args.step) #Write to training_data/, can be specified using outdir, but keeping it simple here for now
    else:
        train(args.output, args.percent)
