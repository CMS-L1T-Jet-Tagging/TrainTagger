# To run before using the codes
export PYTHONPATH=$PYTHONPATH:$PWD
export CI_COMMIT_REF_NAME=local

# Set default versions of command line variables for local running
export Name=new_samples_baseline_5param_extended_trk
export Inputs=baseline
export Model=baseline
export N_PARAMS=5
export TRACK_ALGO=extended
export TRAIN=All200.root
export MINBIAS=MinBias_PU200.root
export BBBB=GluGluHHTo4B_PU200.root
export BBBB_EOS_DIR='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_090125_addGenH/'
export BBTT=GluGluHHTo2B2Tau_PU200.root
export BBTT_EOS_DIR='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_090125_addGenH/'
export VBFHTAUTAU=VBFHToTauTau_PU200.root
export NTUPLE_TREE=outnano/Jets
export SIGNAL=TT_PU200
export EOS_DATA_DIR='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_090125/'
