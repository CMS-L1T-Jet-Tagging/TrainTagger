import numpy as np
# load dataset of one process folder
# if one wants to load the entire dataset, skip the data_dir or data_files argument
import pandas as pd
import awkward as ak

max_n_jets = 12
max_n_constituents = 16
min_constituent_pt = 0
min_matched_jet_pt = 5
min_reco_jet_pt = 15
max_jet_eta = 2.4

def PIDtoOneHot(PID):
    pdgid_table = {'photon' : 22, 'electron_plus':-11, 'electron_minus':11, 'muon_plus': -13, 'muon_minus':13,  'NeutralHadron':111,'ChargedHadronPlus': 211, 'ChargedHadronMinus':-211 }
    one_hot_PID = np.zeros(8)

    match PID:
        case 0:
            pass
        case 22:
            one_hot_PID[0] = 1
        case -11:
            one_hot_PID[1] = 1
        case 11:
            one_hot_PID[2] = 1
        case -13:
            one_hot_PID[3] = 1
        case 13:
            one_hot_PID[4] = 1
        case 111 | 311 | 2112 :
            one_hot_PID[5] = 1
        case 211  | 2212 | 321 | 3312 | 3222 | 3112:
            one_hot_PID[6] = 1
        case -211  | -2212 | -321 | -3312 | -3222 | -3112:
            one_hot_PID[7] = 1
        case _:
            print(PID)
    return one_hot_PID


def PIDtoMass(PID):
    mass = 0.13
    match PID:
        case 22:
            mass = 0
        case -11 | 11:
            mass = 0.005
        case -13 | 13:
            mass = 0.105
        case 111 | 311 | 2112 :
            mass = 0.5
        case _:
            mass = 0.13
    return mass

def phiWrap(phi):
    if phi < -np.pi:
        return phi + 2*np.pi
    
    elif phi > np.pi:
        return phi - 2*np.pi
    else:
        return phi
    
    
def matchGenJet(trPt, trEta, trPhi, genPt, genEta, genPhi):
    ##### PROBABLY NEEDS FIXING!!!! 
    matched_pt = []
    
    genPt_ = genPt.copy()
    genEta_ = genEta.copy()
    genPhi_ = genPhi.copy()

    if (len(genPt) == 0):
        for i in range(len(trPt)):
            matched_pt.append(-1) 
        return matched_pt
    
    if (len(trPt) > 0) and (len(genPt) > 0):
        for ijet in range(len(trPt)):
            diffs_eta = abs(genEta_ - trEta[ijet])
            diffs_phi = abs(genPhi_ - trPhi[ijet])
            diffs_sum = diffs_eta + diffs_phi
            if min(abs(diffs_sum)) < 0.1:
                argmin = np.argmin(abs(diffs_sum))
                matched_pt.append(genPt_[argmin])
                genPt_[argmin] = 9999
                genEta_[argmin] = 9999
                genPhi_[argmin] = 9999
            else:
                matched_pt.append(-1)  
    return matched_pt
    
def jet_label(class_label):
    new_class_label = 0
    match class_label:
        case 5 | -5:
            new_class_label = 0
        case 4 | -4:
            new_class_label = 1
        case 0 | 1 | -1 | 2 | -2 | 3 | -3 :
            new_class_label = 2
        case 21 | -21:
            new_class_label = 3
        case -15:
            new_class_label = 4
        case 15:
            new_class_label = 5
        case 13:
            new_class_label = 6
        case 11:
            new_class_label = 7
        case _:
            new_class_label = -1 
    return new_class_label 


def process_single_file(path):
    
    columns = ['L1T_PUPPIPart_PuppiW','L1T_PUPPIPart_PT','L1T_JetPuppiAK4_PT','L1T_PUPPIPart_D0','L1T_PUPPIPart_DZ','L1T_PUPPIPart_PID','L1T_PUPPIPart_Charge','L1T_PUPPIPart_Eta','L1T_PUPPIPart_Phi','L1T_JetPuppiAK4_Eta', 'L1T_JetPuppiAK4_Phi',
           'L1T_JetPuppiAK4_Constituents', 'L1T_JetPuppiAK4_ConstituentsIdx','L1T_PUPPIPart_Mass','L1T_JetPuppiAK4_Flavor','Gen_JetAK4_PT','FullReco_JetAK4_PT','Gen_JetAK4_Phi','FullReco_JetAK4_Phi','Gen_JetAK4_Eta','FullReco_JetAK4_Eta','Gen_Part_PID','Gen_Part_Phi','Gen_Part_PT','Gen_Part_Eta','FullReco_JetPuppiAK4_Flavor']


    dataset = pd.read_parquet(path,columns=columns)
    
    
    n_events = len(dataset['L1T_JetPuppiAK4_ConstituentsIdx'])
    
    feature_array = []
    pt_target_array = []
    class_label_array = []
    reco_pt_array = []

    for ievent,constituents in enumerate(dataset['L1T_JetPuppiAK4_ConstituentsIdx']):
        matched_gen_pt = matchGenJet(dataset['L1T_JetPuppiAK4_PT'][ievent],dataset['L1T_JetPuppiAK4_Eta'][ievent],dataset['L1T_JetPuppiAK4_Phi'][ievent],dataset['Gen_JetAK4_PT'][ievent],dataset['Gen_JetAK4_Eta'][ievent],dataset['Gen_JetAK4_Phi'][ievent])
        for ijet, jet_constituents in enumerate(dataset['L1T_JetPuppiAK4_ConstituentsIdx'][ievent]):
            if (dataset['L1T_JetPuppiAK4_PT'][ievent][ijet] > min_reco_jet_pt) and (abs(dataset['L1T_JetPuppiAK4_Eta'][ievent][ijet]) < max_jet_eta):
                feature_vector = np.zeros([max_n_constituents,20])

                pt_target = matched_gen_pt[ijet]
                
                if pt_target > min_matched_jet_pt:
                
                    reco_pt = dataset['L1T_JetPuppiAK4_PT'][ievent][ijet]

                    class_label = dataset['L1T_JetPuppiAK4_Flavor'][ievent][ijet]
                    #print(ievent, ijet)
                    #print( 'flavour: ', dataset['L1T_JetPuppiAK4_Flavor'][ievent][ijet])
                    # print( 'flavour 2: ', dataset['FullReco_JetPuppiAK4_Flavor'][ievent])
                    
                    
                    # print( ' gen part: ',dataset['Gen_Part_PID'][ievent][0:20])
                    # print( ' gen part: ',dataset['Gen_Part_PT'][ievent][0:20])
                    # print( ' gen part: ',dataset['Gen_Part_Eta'][ievent][0:20])
                    # print( ' gen part: ',dataset['Gen_Part_Phi'][ievent][0:20])
                    
                    # print( ' tr part: ',dataset['L1T_PUPPIPart_PID'][ievent][0:20])
                    # print( ' tr part: ',dataset['L1T_PUPPIPart_PT'][ievent][0:20])
                    # print( ' tr part: ',dataset['L1T_PUPPIPart_Eta'][ievent][0:20])
                    # print( ' tr part: ',dataset['L1T_PUPPIPart_Phi'][ievent][0:20])
                    
                    # print( ' gen Jet: ',dataset['Gen_JetAK4_PT'][ievent])
                    # print( ' gen Jet: ',dataset['Gen_JetAK4_Eta'][ievent])
                    # print( ' gen Jet: ',dataset['Gen_JetAK4_Phi'][ievent])
                    
                    # print( ' tr Jet: ',dataset['L1T_JetPuppiAK4_PT'][ievent])
                    # print( ' tr Jet: ',dataset['L1T_JetPuppiAK4_Eta'][ievent])
                    # print( ' tr Jet: ',dataset['L1T_JetPuppiAK4_Phi'][ievent])
                    
                    # print("==========")
                    
                    #print(dataset['L1T_PUPPIPart_PID'])
                    #print(dataset['L1T_PUPPIPart_PuppiW'].to_numpy())
                    if ijet >= max_n_jets:
                        break
                    for iconstituent in range(max_n_constituents): 
                            try:
                                index = jet_constituents[iconstituent]
                                if index == -1:
                                    break
                            except:
                                break
                            if (dataset['L1T_PUPPIPart_PT'][ievent][index] > min_constituent_pt) and (abs(dataset['L1T_PUPPIPart_Eta'][ievent][index]) < max_jet_eta):
                                # print('puppi W: ', dataset['L1T_PUPPIPart_PuppiW'].to_numpy()[0][ievent][index])
                                # print('PID: ', dataset['L1T_PUPPIPart_PID'][ievent][index])
                    
                                feature_vector[iconstituent][0] = dataset['L1T_PUPPIPart_PT'][ievent][index]
                                feature_vector[iconstituent][1] = dataset['L1T_PUPPIPart_PT'][ievent][index] / dataset['L1T_JetPuppiAK4_PT'][ievent][ijet]
                                feature_vector[iconstituent][2] = np.log(dataset['L1T_PUPPIPart_PT'][ievent][index])
                                
                                
                                dphi = dataset['L1T_JetPuppiAK4_Phi'][ievent][ijet] -  dataset['L1T_PUPPIPart_Phi'][ievent][index]
                                dphi0 = dphi if dphi < np.pi else dphi - 2*np.pi
                                dphi1 = dphi if dphi > -np.pi else dphi + 2*np.pi
                                
                                
                                if abs(dataset['L1T_JetPuppiAK4_Eta'][ievent][ijet] -  dataset['L1T_PUPPIPart_Eta'][ievent][index]) > 1:
                                    print('jet eta: ', dataset['L1T_JetPuppiAK4_Eta'][ievent][ijet])
                                    print('puppi eta: ',dataset['L1T_PUPPIPart_Eta'][ievent][index])
                                    
                                    print('all puppi eta: ',dataset['L1T_PUPPIPart_Eta'][ievent])
                                    print('consituents: ',dataset['L1T_JetPuppiAK4_ConstituentsIdx'][ievent])
                                
                                feature_vector[iconstituent][3] =  dataset['L1T_JetPuppiAK4_Eta'][ievent][ijet] -  dataset['L1T_PUPPIPart_Eta'][ievent][index]
                                feature_vector[iconstituent][4] =  dphi0 if dphi > 0 else dphi1
                                
                                feature_vector[iconstituent][5] =  PIDtoMass(dataset['L1T_PUPPIPart_PID'][ievent][index])
                                
                                
                                #print('mass: ', dataset['L1T_PUPPIPart_Mass'][ievent][index],'PID: ',dataset['L1T_PUPPIPart_PID'][ievent][index])
                                
                                feature_vector[iconstituent][6:14] = PIDtoOneHot(dataset['L1T_PUPPIPart_PID'][ievent][index])
                                
                                feature_vector[iconstituent][14] =  dataset['L1T_PUPPIPart_DZ'][ievent][index]
                                dxy = dataset['L1T_PUPPIPart_D0'][ievent][index] + np.random.normal(0,0.0)
                                feature_vector[iconstituent][15] =  np.sqrt(abs(dxy))
                                feature_vector[iconstituent][16] =  1
                                feature_vector[iconstituent][17] =  dataset['L1T_PUPPIPart_PuppiW'][ievent][index]
                                feature_vector[iconstituent][18] =  0
                                feature_vector[iconstituent][19] =  0
                                

                    new_label = jet_label(class_label)
                    
                    feature_array.append(feature_vector)
                    pt_target_array.append(pt_target)
                    class_label_array.append(new_label)
                    reco_pt_array.append(reco_pt)
                

    new_feature_array = np.stack( feature_array)
    pt_target_array = np.array(pt_target_array)
    class_label_array = np.array(class_label_array)
    reco_pt_array = np.array(reco_pt_array)
    minus_ones = np.where(class_label_array == -1)
    
    new_feature_array_after = np.delete(new_feature_array, minus_ones, axis=0)
    pt_target_array_after = np.delete(pt_target_array, minus_ones, axis=0)
    class_label_array_after = np.delete(class_label_array, minus_ones, axis=0)
    reco_pt_array_after = np.delete(reco_pt_array, minus_ones, axis=0)
    
    return new_feature_array_after, class_label_array_after, pt_target_array_after, reco_pt_array_after
