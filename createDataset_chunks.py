from imports import *
from dataset import *
import argparse
gc.set_threshold(0)

# All PF candidate properties
pfcand_fields_all = [
    'puppiweight',
    'pt_rel','pt_rel_log',
    'pt_rel_phys',
    'pt',
    'pt_phys',
    'dxy',
    'dxy_custom',
    'id','charge','pperp_ratio','ppara_ratio',
    'deta','dphi',
    'deta_phys','dphi_phys',
    'etarel','track_chi2',
    'track_chi2norm','track_qual','track_npar','track_vx','track_vy','track_vz','track_pterror',
    'cluster_hovere','cluster_sigmarr','cluster_abszbarycenter','cluster_emet',
    'pt_log',
    'eta','phi',
    'eta_phys','phi_phys',

    'emid','quality','tkquality',
    'track_valid','track_rinv',
    'track_phizero','track_tanl','track_z0','z0',
    'track_d0','track_chi2rphi','track_chi2rz',
    'track_bendchi2','track_hitpattern','track_nstubs',
    # 'track_mvaquality',
    'track_mvaother',
    'mass',
    'isPhoton', 'isElectronPlus', 'isElectronMinus', 'isMuonPlus', 'isMuonMinus', 'isNeutralHadron', 'isChargedHadronPlus', 'isChargedHadronMinus',
    'isfilled',
    ]

pfcand_fields_baselineHW = [
    'pt_phys','eta_phys','phi_phys',
    'isPhoton', 'isElectronPlus', 'isElectronMinus', 'isMuonPlus', 'isMuonMinus', 'isNeutralHadron', 'isChargedHadronPlus', 'isChargedHadronMinus',
    'z0', 'dxy',
]

pfcand_fields_baselineHWMe = [
    'charge','id',
    'pt_phys','eta_phys','phi_phys',
    'z0', 'dxy',
]

pfcand_fields_baselineEmulator = [
    'pt_rel_phys','deta_phys','dphi_phys',
    "track_vx","track_vy","track_vz",
    'isPhoton', 'isElectronPlus', 'isElectronMinus', 'isMuonPlus', 'isMuonMinus', 'isNeutralHadron', 'isChargedHadronPlus', 'isChargedHadronMinus',
]

pfcand_fields_baselineEmulatorAdd = [
    'pt_rel_phys','deta_phys','dphi_phys',
    "track_vx","track_vy","track_vz",
    'isPhoton', 'isElectronPlus', 'isElectronMinus', 'isMuonPlus', 'isMuonMinus', 'isNeutralHadron', 'isChargedHadronPlus', 'isChargedHadronMinus',
    'isfilled',
]

pfcand_fields_baselineEmulatorMe = [
    'pt_rel_phys','deta_phys','dphi_phys',
    "track_vx","track_vy","track_vz",
    'charge','id',
]

pfcand_fields_minimal = [
    'pt_rel_phys','deta_phys','dphi_phys',
    'isPhoton', 'isElectronPlus', 'isElectronMinus', 'isMuonPlus', 'isMuonMinus', 'isNeutralHadron', 'isChargedHadronPlus', 'isChargedHadronMinus',
]

pfcand_fields_minimalMe = [
    'pt_rel_phys','deta_phys','dphi_phys',
    'charge','id',
]

pfcand_fields_ext1 = [
    'pt_rel_phys','deta_phys','dphi_phys',
    'pt_phys', 'eta_phys','phi',
    'isPhoton', 'isElectronPlus', 'isElectronMinus', 'isMuonPlus', 'isMuonMinus', 'isNeutralHadron', 'isChargedHadronPlus', 'isChargedHadronMinus',
    'z0', 'dxy',
    'isfilled',
    'puppiweight', 'emid', 'quality',
]

pfcand_fields_ext2 = [
    'pt_rel_phys','deta_phys','dphi_phys',
    'pt_phys','eta_phys','phi_phys', 'mass',
    'charge','id',
    'z0', 'dxy',
    'isfilled',
    'puppiweight', 'emid', 'quality',
]

pfcand_fields_ext3 = [
    'pt_rel_phys','deta_phys','dphi_phys',
    'pt_phys','eta_phys','phi_phys', 'mass',
    'isPhoton', 'isElectronPlus', 'isElectronMinus', 'isMuonPlus', 'isMuonMinus', 'isNeutralHadron', 'isChargedHadronPlus', 'isChargedHadronMinus',
    'z0', 'dxy',
    'isfilled',
    'puppiweight', 'emid', 'quality',
]

pfcand_fields_ext4 = [
    'pt_rel_phys','deta_phys','dphi_phys',
    'isPhoton', 'isElectronPlus', 'isElectronMinus', 'isMuonPlus', 'isMuonMinus', 'isNeutralHadron', 'isChargedHadronPlus', 'isChargedHadronMinus',
    'z0', 'dxy',
    'isfilled',
    'puppiweight', 'emid', 'quality',
]

pfcand_fields_ext5 = [
    'pt_rel_phys','deta_phys','dphi_phys',
    'charge','id',
    'z0', 'dxy',
    'isfilled',
    'puppiweight', 'emid', 'quality',
]

pfcand_fields_ext6 = [
    'pt_rel_phys','deta_phys','dphi_phys',
    'pt_phys','eta_phys','phi_phys', 'mass',
    'isPhoton', 'isElectronPlus', 'isElectronMinus', 'isMuonPlus', 'isMuonMinus', 'isNeutralHadron', 'isChargedHadronPlus', 'isChargedHadronMinus',
    'z0', 'dxy',
    'isfilled',
]


dict_fields = {
    "all" : pfcand_fields_all,
    "baselineHW" : pfcand_fields_baselineHW,
    "baselineHWMe" : pfcand_fields_baselineHWMe,
    "baselineEmulator" : pfcand_fields_baselineEmulator,
    "baselineEmulatorMe" : pfcand_fields_baselineEmulatorMe,
    "baselineEmulatorAdd" : pfcand_fields_baselineEmulatorAdd,
    "minimal" : pfcand_fields_minimal,
    "minimalMe" : pfcand_fields_minimalMe,
    "ext1" : pfcand_fields_ext1,
    "ext2" : pfcand_fields_ext2,
    "ext3" : pfcand_fields_ext3,
    "ext4" : pfcand_fields_ext4,
    "ext5" : pfcand_fields_ext5,
    "ext6" : pfcand_fields_ext6,
}


def processPerFeatureSet(data_split_, name_, features_, chunk_,outFolder, nconstit):
    print ("Process chunk",chunk_,"for",name_)
    # make the split
    classes_, var_names_, x_, y_, x_global_, y_target_ = createAndSaveTrainingData(data_split_, features_)
    X_train_, X_test_, Y_train_, Y_test_, x_global_train_, x_global_test_, y_target_train_, y_target_test_ = splitAndShuffle(x_, y_, x_global_, y_target_, len(features_), shuffleConst = False)

    # make all arrays
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
    x_muon_ = np.reshape(classes_["muon"]["x"],[-1, nconstit, len(features_)])
    x_muon_global = classes_["muon"]["x_global"]
    x_electron_ = np.reshape(classes_["electron"]["x"],[-1, nconstit, len(features_)])
    x_electron_global = classes_["electron"]["x_global"]

    # save data to parket files
    ak.to_parquet(X_train_, outFolder+"/X_"+name_+"_train_"+str(chunk_)+".parquet")
    ak.to_parquet(Y_train_, outFolder+"/Y_"+name_+"_train_"+str(chunk_)+".parquet")
    ak.to_parquet(X_test_, outFolder+"/X_"+name_+"_test_"+str(chunk_)+".parquet")
    ak.to_parquet(Y_test_, outFolder+"/Y_"+name_+"_test_"+str(chunk_)+".parquet")
    ak.to_parquet(x_global_train_, outFolder+"/X_global_"+name_+"_train_"+str(chunk_)+".parquet")
    ak.to_parquet(x_global_test_, outFolder+"/X_global_"+name_+"_test_"+str(chunk_)+".parquet")
    ak.to_parquet(y_target_train_, outFolder+"/Y_target_"+name_+"_train_"+str(chunk_)+".parquet")
    ak.to_parquet(y_target_test_, outFolder+"/Y_target_"+name_+"_test_"+str(chunk_)+".parquet")
    ak.to_parquet(x_b_, outFolder+"/X_"+name_+"_b_"+str(chunk_)+".parquet")
    ak.to_parquet(x_b_global, outFolder+"/X_global_"+name_+"_b_"+str(chunk_)+".parquet")
    ak.to_parquet(x_taup_, outFolder+"/X_"+name_+"_taup_"+str(chunk_)+".parquet")
    ak.to_parquet(x_taup_global, outFolder+"/X_global_"+name_+"_taup_"+str(chunk_)+".parquet")
    ak.to_parquet(x_taum_, outFolder+"/X_"+name_+"_taum_"+str(chunk_)+".parquet")
    ak.to_parquet(x_taum_global, outFolder+"/X_global_"+name_+"_taum_"+str(chunk_)+".parquet")
    ak.to_parquet(x_gluon_, outFolder+"/X_"+name_+"_gluon_"+str(chunk_)+".parquet")
    ak.to_parquet(x_gluon_global, outFolder+"/X_global_"+name_+"_gluon_"+str(chunk_)+".parquet")
    ak.to_parquet(x_charm_, outFolder+"/X_"+name_+"_charm_"+str(chunk_)+".parquet")
    ak.to_parquet(x_charm_global, outFolder+"/X_global_"+name_+"_charm_"+str(chunk_)+".parquet")
    ak.to_parquet(x_bkg_, outFolder+"/X_"+name_+"_bkg_"+str(chunk_)+".parquet")
    ak.to_parquet(x_bkg_global, outFolder+"/X_global_"+name_+"_bkg_"+str(chunk_)+".parquet")
    ak.to_parquet(x_muon_, outFolder+"/X_"+name_+"_muon_"+str(chunk_)+".parquet")
    ak.to_parquet(x_muon_global, outFolder+"/X_global_"+name_+"_muon_"+str(chunk_)+".parquet")
    ak.to_parquet(x_electron_, outFolder+"/X_"+name_+"_electron_"+str(chunk_)+".parquet")
    ak.to_parquet(x_electron_global, outFolder+"/X_global_"+name_+"_electron_"+str(chunk_)+".parquet")

    # del X_train_, Y_train_, X_test_, Y_test_, x_b_,x_taup_,x_taum_, x_gluon_, x_bkg_, x_electron_, x_muon_, x_b_global, x_taup_global, x_taum_global, x_gluon_global, x_charm_global, x_charm_, x_bkg_global, x_muon_global, x_electron_global


def createDataset(filetag):
    # Open the input ROOT files and check its contents
    fname = "../nTuples/"+filetag+".root"
    outFolder = "datasetsNewComplete2/"
    # outFolder = "datasetsNewComplete_plotting/"
    outFolder = outFolder + "/" + filetag + "/" + "/btgc/"
    nconstit = 16

    if not os.path.exists(outFolder):
        os.makedirs(outFolder)

    print ("Use the following outfolder", outFolder)

    # Transform into Awkward arrays and filter its contents
    filter = "/(jet)_(reject|eta|eta_phys|phi|phi_phys|pt|pt_phys|pt_raw|bjetscore|tauscore|taupt|pt_corr|tauflav|muflav|elflav|taudecaymode|lepflav|taucharge|genmatch_pt|genmatch_eta|genmatch_phi|genmatch_mass|genmatch_hflav|genmatch_lep_vis_pt|genmatch_lep_pt|genmatch_pflav|npfcand|pfcand_pt|pfcand_pt_rel|pfcand_pt_rel_log|pfcand_pt_log|pfcand_eta|pfcand_phi|pfcand_puppiweight|pfcand_emid|pfcand_pt_rel_phys|pfcand_pt_phys|pfcand_eta_phys|pfcand_phi_phys|pfcand_dphi_phys|pfcand_deta_phys|pfcand_quality|pfcand_tkquality|pfcand_z0|pfcand_dxy|pfcand_dxy_custom|pfcand_id|pfcand_charge|pfcand_pperp_ratio|pfcand_ppara_ratio|pfcand_deta|pfcand_dphi|pfcand_etarel|pfcand_track_valid|pfcand_track_rinv|pfcand_track_phizero|pfcand_track_tanl|pfcand_track_z0|pfcand_track_d0|pfcand_track_chi2rphi|pfcand_track_chi2rz|pfcand_track_bendchi2|pfcand_track_hitpattern|pfcand_track_mvaquality|pfcand_track_mvaother|pfcand_track_chi2|pfcand_track_chi2norm|pfcand_track_qual|pfcand_track_npar|pfcand_track_nstubs|pfcand_track_vx|pfcand_track_vy|pfcand_track_vz|pfcand_track_pterror|pfcand_cluster_hovere|pfcand_cluster_sigmarr|pfcand_cluster_abszbarycenter|pfcand_cluster_emet|pfcand_cluster_egvspion|pfcand_cluster_egvspu|pfcand_isPhoton|pfcand_isElectronPlus|pfcand_isElectronMinus|pfcand_isMuonPlus|pfcand_isMuonMinus|pfcand_isNeutralHadron|pfcand_isChargedHadronPlus|pfcand_isChargedHadronMinus|pfcand_isfilled|pfcand_energy|pfcand_mass)/"


    chunk = 0
    for data in uproot.iterate(fname, filter_name = filter, how = "zip"):

        # jet_cut = (data['jet_pt_phys'] > 15.) & (np.abs(data['jet_eta_phys']) < 2.4) & (data['jet_genmatch_pt'] > 5.)
        jet_cut = (data['jet_pt_phys'] > 15.) & (np.abs(data['jet_eta_phys']) < 2.4) & (data['jet_genmatch_pt'] > 5.) & (data['jet_reject'] == 0 )
        data = data[jet_cut]
        addResponseVars(data)

        print("Data Fields:", data.fields)
        print("JET PFcand Fields:", data.jet_pfcand.fields)

        data_split = splitFlavors(data)

        # Create datasets
        processPerFeatureSet(data_split, "all", pfcand_fields_all, chunk, outFolder, nconstit)
        processPerFeatureSet(data_split, "baselineHW", pfcand_fields_baselineHW, chunk, outFolder, nconstit)
        processPerFeatureSet(data_split, "baselineHWMe", pfcand_fields_baselineHWMe, chunk, outFolder, nconstit)
        processPerFeatureSet(data_split, "baselineEmulator", pfcand_fields_baselineEmulator, chunk, outFolder, nconstit)
        processPerFeatureSet(data_split, "baselineEmulatorMe", pfcand_fields_baselineEmulatorMe, chunk, outFolder, nconstit)
        processPerFeatureSet(data_split, "baselineEmulatorAdd", pfcand_fields_baselineEmulatorAdd, chunk, outFolder, nconstit)
        processPerFeatureSet(data_split, "minimal", pfcand_fields_minimal, chunk, outFolder, nconstit)
        processPerFeatureSet(data_split, "minimalMe", pfcand_fields_minimalMe, chunk, outFolder, nconstit)
        processPerFeatureSet(data_split, "ext1", pfcand_fields_ext1, chunk, outFolder, nconstit)
        processPerFeatureSet(data_split, "ext2", pfcand_fields_ext2, chunk, outFolder, nconstit)
        processPerFeatureSet(data_split, "ext3", pfcand_fields_ext3, chunk, outFolder, nconstit)
        processPerFeatureSet(data_split, "ext4", pfcand_fields_ext4, chunk, outFolder, nconstit)
        processPerFeatureSet(data_split, "ext5", pfcand_fields_ext5, chunk, outFolder, nconstit)
        processPerFeatureSet(data_split, "ext6", pfcand_fields_ext6, chunk, outFolder, nconstit)

        chunk = chunk + 1

        del data_split


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file', help='input file name part')
    args = parser.parse_args()

    print('#'*30)
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('#'*30)

    createDataset(args.file)
