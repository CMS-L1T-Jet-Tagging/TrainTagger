import yaml

def make_data(infile='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_ntuples_v131Xv9/baselineTRK_4param_021024/All200.root', 
              outdir='../training_data/',
              tags="baseline_hardware_inputs", nconstit = 16, doLeptons = True):
    """
    Process the data set in chunks from the input ntuples file.

    Parameters:
        infile (str): The input file path.
        outdir (str): The output directory.
        tags (str): input tags to use from pfcands, defined in pfcand_fields.yml
        nconstit (int): Number of constituents.
    """

    # Load the YAML file as a dictionary
    with open("pfcand_fields.yml", "r") as file: pfcand_fields = yaml.safe_load(file)
