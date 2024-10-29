import os,gc
import argparse
gc.set_threshold(0)

import uproot
import numpy as np
import awkward as ak
import tensorflow as tf
from datatools import dataset

def processPerFeatureSet(data_split_, name_, features_, chunk_, outFolder, nconstit, doLeptons = True):
    """
    Process and save the dataset per feature set.

    Parameters:
        data_split_ (dict): The data split by class.
        name_ (str): Name of the feature set.
        features_ (list): List of feature names.
        chunk_ (int): The chunk number.
        outFolder (str): The output directory.
        nconstit (int): Number of constituents.
    """

    print (f"Process chunk {chunk_} for {name_}")
    
    # Create and save training data
    classes_, var_names_, x_, y_, x_global_, y_target_ = dataset.createAndSaveTrainingData(data_split_, features_)
    X_train_, X_test_, Y_train_, Y_test_, x_global_train_, x_global_test_, y_target_train_, y_target_test_ = dataset.splitAndShuffle(x_, y_, x_global_, y_target_, len(features_), shuffleConst = False)

    # Reshape arrays for different classes
    x_b_ = np.reshape(classes_["b"]["x"],[-1, nconstit, len(features_)])
    x_b_global = classes_["b"]["x_global"]
    x_taup_ = np.reshape(classes_["taup"]["x"],[-1, nconstit, len(features_)])
    x_taup_global = classes_["taup"]["x_global"]
    x_taum_ = np.reshape(classes_["taum"]["x"],[-1, nconstit, len(features_)])
    x_taum_global = classes_["taum"]["x_global"]
    x_gluon_ = np.reshape(classes_["gluon"]["x"],[-1, nconstit, len(features_)])
    x_gluon_global = classes_["gluon"]["x_global"]
    x_charm_ = np.reshape(classes_["charm"]["x"],[-1, nconstit, len(features_)])
    x_charm_global = classes_["charm"]["x_global"]
    x_bkg_ = np.reshape(classes_["bkg"]["x"],[-1, nconstit, len(features_)])
    x_bkg_global = classes_["bkg"]["x_global"]
    if doLeptons:
        x_muon_ = np.reshape(classes_["muon"]["x"],[-1, nconstit, len(features_)])
        x_muon_global = classes_["muon"]["x_global"]
        x_electron_ = np.reshape(classes_["electron"]["x"],[-1, nconstit, len(features_)])
        x_electron_global = classes_["electron"]["x_global"]

    # save data to quarquet files
    ak.to_parquet(X_train_, f"{outFolder}/X_{name_}_train_{chunk_}.parquet")
    ak.to_parquet(Y_train_, f"{outFolder}/Y_{name_}_train_{chunk_}.parquet")
    ak.to_parquet(X_test_, f"{outFolder}/X_{name_}_test_{chunk_}.parquet")
    ak.to_parquet(Y_test_, f"{outFolder}/Y_{name_}_test_{chunk_}.parquet")
    ak.to_parquet(x_global_train_, f"{outFolder}/X_global_{name_}_train_{chunk_}.parquet")
    ak.to_parquet(x_global_test_, f"{outFolder}/X_global_{name_}_test_{chunk_}.parquet")
    ak.to_parquet(y_target_train_, f"{outFolder}/Y_target_{name_}_train_{chunk_}.parquet")
    ak.to_parquet(y_target_test_, f"{outFolder}/Y_target_{name_}_test_{chunk_}.parquet")
    ak.to_parquet(x_b_, f"{outFolder}/X_{name_}_b_{chunk_}.parquet")
    ak.to_parquet(x_b_global, f"{outFolder}/X_global_{name_}_b_{chunk_}.parquet")
    ak.to_parquet(x_taup_, f"{outFolder}/X_{name_}_taup_{chunk_}.parquet")
    ak.to_parquet(x_taup_global, f"{outFolder}/X_global_{name_}_taup_{chunk_}.parquet")
    ak.to_parquet(x_taum_, f"{outFolder}/X_{name_}_taum_{chunk_}.parquet")
    ak.to_parquet(x_taum_global, f"{outFolder}/X_global_{name_}_taum_{chunk_}.parquet")
    ak.to_parquet(x_gluon_, f"{outFolder}/X_{name_}_gluon_{chunk_}.parquet")
    ak.to_parquet(x_gluon_global, f"{outFolder}/X_global_{name_}_gluon_{chunk_}.parquet")
    ak.to_parquet(x_charm_, f"{outFolder}/X_{name_}_charm_{chunk_}.parquet")
    ak.to_parquet(x_charm_global, f"{outFolder}/X_global_{name_}_charm_{chunk_}.parquet")
    ak.to_parquet(x_bkg_, f"{outFolder}/X_{name_}_bkg_{chunk_}.parquet")
    ak.to_parquet(x_bkg_global, f"{outFolder}/X_global_{name_}_bkg_{chunk_}.parquet")
    if doLeptons:
        ak.to_parquet(x_muon_, f"{outFolder}/X_{name_}_muon_{chunk_}.parquet")
        ak.to_parquet(x_muon_global, f"{outFolder}/X_global_{name_}_muon_{chunk_}.parquet")
        ak.to_parquet(x_electron_, f"{outFolder}/X_{name_}_electron_{chunk_}.parquet")
        ak.to_parquet(x_electron_global, f"{outFolder}/X_global_{name_}_electron_{chunk_}.parquet")

    del classes_, var_names_, x_, y_, x_global_, y_target_
    del X_train_,Y_train_,X_test_,Y_test_,x_global_train_,x_global_test_,y_target_train_,y_target_test_
    del x_b_,x_b_global,x_taup_,x_taup_global,x_taum_,x_taum_global,x_gluon_,x_gluon_global,x_charm_,x_charm_global,x_bkg_,x_bkg_global
    if doLeptons:
        del x_muon_,x_muon_global,x_electron_,x_electron_global


def createDataset(infile, outdir, inputs, nconstit = 16, doLeptons = True):
    """
    Process the data set in chunks from the input ntuples file.

    Parameters:
        infile (str): The input file path.
        outdir (str): The output directory.
        nconstit (int): Number of constituents.
    """
    
    if not os.path.exists(outdir): os.makedirs(outdir)
    print ("Using the following outdir", outdir)

    # Transform into Awkward arrays and filter its contents
    filter = "/(jet)_(reject|eta|eta_phys|phi|phi_phys|pt|pt_phys|pt_raw|bjetscore|tauscore|taupt|pt_corr|tauflav|muflav|elflav|taudecaymode|lepflav|taucharge|genmatch_pt|genmatch_eta|genmatch_phi|genmatch_mass|genmatch_hflav|genmatch_lep_vis_pt|genmatch_lep_pt|genmatch_pflav|npfcand|pfcand_pt|pfcand_pt_rel|pfcand_pt_rel_log|pfcand_pt_log|pfcand_eta|pfcand_phi|pfcand_puppiweight|pfcand_emid|pfcand_pt_rel_phys|pfcand_pt_phys|pfcand_eta_phys|pfcand_phi_phys|pfcand_dphi_phys|pfcand_deta_phys|pfcand_quality|pfcand_tkquality|pfcand_z0|pfcand_dxy|pfcand_dxy_custom|pfcand_dxy_phys|pfcand_dxy_physSquared|pfcand_id|pfcand_charge|pfcand_pperp_ratio|pfcand_ppara_ratio|pfcand_deta|pfcand_dphi|pfcand_etarel|pfcand_track_valid|pfcand_track_rinv|pfcand_track_phizero|pfcand_track_tanl|pfcand_track_z0|pfcand_track_d0|pfcand_track_chi2rphi|pfcand_track_chi2rz|pfcand_track_bendchi2|pfcand_track_hitpattern|pfcand_track_mvaquality|pfcand_track_mvaother|pfcand_track_chi2|pfcand_track_chi2norm|pfcand_track_qual|pfcand_track_npar|pfcand_track_nstubs|pfcand_track_vx|pfcand_track_vy|pfcand_track_vz|pfcand_track_pterror|pfcand_cluster_hovere|pfcand_cluster_sigmarr|pfcand_cluster_abszbarycenter|pfcand_cluster_emet|pfcand_cluster_egvspion|pfcand_cluster_egvspu|pfcand_isPhoton|pfcand_isElectronPlus|pfcand_isElectronMinus|pfcand_isMuonPlus|pfcand_isMuonMinus|pfcand_isNeutralHadron|pfcand_isChargedHadronPlus|pfcand_isChargedHadronMinus|pfcand_isfilled|pfcand_energy|pfcand_mass)/"

    num_entries =f = uproot.open(infile)["jetntuple/Jets"].num_entries
    num_entries_done = 0
    chunk = 0
    for data in uproot.iterate(infile, filter_name = filter, how = "zip"):

        num_entries_done = num_entries_done + len(data)
        print ("Processed", num_entries_done, "out of", num_entries, "|", np.round(float(num_entries_done)/float(num_entries)*100.,1),"%")
        jet_cut = (data['jet_pt_phys'] > 15.) & (np.abs(data['jet_eta_phys']) < 2.4) & (data['jet_reject'] == 0 )
        data = data[jet_cut]
        dataset.addResponseVars(data)

        print("Data Fields:", data.fields)
        print("JET PFcand Fields:", data.jet_pfcand.fields)

        data_split = dataset.splitFlavors(data, doLeptons = doLeptons)

        # Process and save datasets for a given feature set

        processPerFeatureSet(data_split, inputs, dict_fields[inputs], chunk, outdir, nconstit, doLeptons = doLeptons)

        chunk = chunk + 1

        del data_split, data, jet_cut


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process input ntuples.')
    parser.add_argument('-i', '--infile', help='Input file name.', default='/eos/user/s/sewuchte/L1Trigger/ForDuc/nTuples/All200.root')
    parser.add_argument('-o', '--outdir', help='Ouput directory path.', default='data/')
    parser.add_argument('-t','--inputs', default='minimal', help = 'Which inputs to run, options are baseline, ext1, ext2, all.')
    args = parser.parse_args()

    doLeptons = True

    allowedInputs = dict_fields.keys()

    if args.inputs not in allowedInputs: raise ValueError("args.inputs not in allowed inputs! Options are", allowedInputs)


    #Print the arguments
    for arg in vars(args): print('%s: %s' %(arg, getattr(args, arg)))

    #Create the dataset parquet files
    createDataset(args.infile, args.outdir, args.inputs, doLeptons = doLeptons)
