from tagger.data.tools import make_data
from qkeras import quantized_bits
from .models import baseline_model

#GLOBAL PARAMETERS TO BE DEFINED WHEN TRAINING
INPUT_QUANTIZER = quantized_bits(bits=15, integer=12, symmetric=0, alpha=1) #To be consistent with firmware implementation
BATCH_SIZE = 1024
EPOCHS = 100
VALIDATION_SPLIT = 0.1 # 10% of training set will be used for validation set. 

# sparsity parameters
I_SPARSITY = 0.2 #Initial sparsity
F_SPARSITY = 0.7 #Final sparsity

def train(out_dir, infile):

    #Create training data set from infile (see README on how to create it)
    make_data(infile=infile)

    #Get the model
    model = baseline_model()

    #Print summary
    print(model.summary())

    #Train it with sparsity

    #Produce some basic plots after training

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-o','--output', default='output/baseline', help = 'Output model directory path, also save evaluation plots')
    parser.add_argument('-i','--input', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_ntuples_v131Xv9/baselineTRK_4param_021024/All200.root' , help = 'Path to input training data')
    args = parser.parse_args()

    train(args.ouput, args.input)
