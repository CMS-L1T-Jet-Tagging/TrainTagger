from imports import *

def getPadNParr(events, obj, n_pad, fields, cuts = None, name = None, pad_val = 0):
    '''
    This function filter objects and pads them to a certain length with a given value
    '''
    
    objects = events[obj]
    
    if not name: name = obj
    
    # cuts are defined as a dictionary containing the relevant keys:
    # cuttype, field and value
    if cuts:
        for cut in cuts:
            if cut["cuttype"] == "equals": objects = objects[objects[cut["field"]] == cut["value"]]
            else: raise Exception("Cuttype {} is not implemented.".format(cut["cuttype"]))
    
    pad_arrs = []
    var_names = []
        
    # padding with nones
    pad_arr = ak.pad_none(objects, n_pad, clip=True)
    
    # combining to numpy
    for i in range(n_pad):

        for var in fields:
            pad_arrs += [ak.to_numpy( ak.fill_none(pad_arr[var][:,i], pad_val) )]
            var_names.append( "{}_{}_{}".format(name, i, var) )
            
    return np.stack(pad_arrs), var_names

def formatData(data, objects, verbosity = 0):
    '''
    This function concatenates the padded arrays for different objects.
    It is controlled via a dictionary as defined above
    '''
    
    # this will be filled by all required objects
    dataList = [] 
    varList = []
    
    for obj in objects: 
        dat, names = getPadNParr(data, obj["key"], obj["n_obj"], obj["fields"], obj["cuts"] if "cuts" in obj else None, obj["name"] )
        dataList.append(dat)
        varList += names
        
    if verbosity > 0:
        print("The input variables are the following:")
        print(varList)
                
    # combining and returning (and transforming back so events are along the first axis...)
    return np.concatenate(dataList, axis = 0).T, varList

def readDataFromFile(filename, filter = "jet_*", applyBaseCut = True):
    f = uproot.open(filename)
    data = f["jetntuple/Jets"].arrays(
        filter_name = filter, 
        how = "zip")
    if applyBaseCut:
        # jet_ptmin =   (data['jet_pt'] > 15.) & (np.abs(data['jet_eta']) < 2.4)
        # jet_ptmin =   (data['jet_pt_phys'] > 15.) & (np.abs(data['jet_eta_phys']) < 2.4) & (data['jet_genmatch_pt'] > 0)
        jet_ptmin =   (data['jet_pt_phys'] > 15.) & (np.abs(data['jet_eta_phys']) < 2.4) & (data['jet_genmatch_pt'] > 0) & (data['jet_reject'] < 1)
        data = data[jet_ptmin]
    return data

def splitFlavors(data, splitTau = True, splitGluon = True, splitCharm = True):
    condition_b = (
        (data['jet_genmatch_pt'] > 0) &
        (data['jet_muflav'] == 0) &
        (data['jet_tauflav'] == 0) &
        (data['jet_elflav'] == 0) &
        (data['jet_genmatch_hflav'] == 5)
    )
    # data['label_b'] = (data['jet_genmatch_pt'] > -1) + (data['jet_muflav'] == 0) + (data['jet_tauflav'] == 0) + (data['jet_elflav'] == 0) + (data['jet_genmatch_hflav'] == 5)
    data['label_b'] = condition_b
    if splitTau:
        # condition_tau = (
        #     (data['jet_genmatch_pt'] > 0) &
        #     (data['jet_muflav'] == 0) &
        #     (data['jet_tauflav'] == 1) &
        #     (data['jet_elflav'] == 0)
        # )
        # data['label_tau'] = (data['jet_genmatch_pt'] > -1) + (data['jet_muflav'] == 0) + (data['jet_tauflav'] == 1) + (data['jet_elflav'] == 0)
        # data['label_tau'] = condition_tau
        condition_taup = (
            (data['jet_genmatch_pt'] > 0) &
            (data['jet_muflav'] == 0) &
            (data['jet_tauflav'] == 1) &
            (data['jet_taucharge'] > 0) &
            (data['jet_elflav'] == 0)
        )
        data['label_taup'] = condition_taup
        condition_taum = (
            (data['jet_genmatch_pt'] > 0) &
            (data['jet_muflav'] == 0) &
            (data['jet_tauflav'] == 1) &
            (data['jet_taucharge'] < 0) &
            (data['jet_elflav'] == 0)
        )
        data['label_taum'] = condition_taum
    if splitGluon:
        condition_gluon = (
            (data['jet_genmatch_pt'] > 0) &
            (data['jet_muflav'] == 0) &
            (data['jet_tauflav'] == 0) &
            (data['jet_elflav'] == 0) &
            (data['jet_genmatch_hflav'] == 0) &
            (data['jet_genmatch_pflav'] == 21)
        )
        # data['label_gluon'] = (data['jet_genmatch_pt'] > -1) + (data['jet_muflav'] == 0) + (data['jet_tauflav'] == 0) + (data['jet_elflav'] == 0) + (data['jet_genmatch_hflav'] == 0 ) + (data['jet_genmatch_pflav'] == 21)
        data['label_g'] = condition_gluon

    if splitCharm:
        condition_charm = (
            (data['jet_genmatch_pt'] > 0) &
            (data['jet_muflav'] == 0) &
            (data['jet_tauflav'] == 0) &
            (data['jet_elflav'] == 0) &
            (data['jet_genmatch_hflav'] == 4)
        )
        # data['label_charm'] = (data['jet_genmatch_pt'] > -1) + (data['jet_muflav'] == 0) + (data['jet_tauflav'] == 0) + (data['jet_elflav'] == 0) + (data['jet_genmatch_hflav'] == 4)
        data['label_c'] = condition_charm

    condition_uds = (
            (data['jet_genmatch_pt'] > 0) &
            (data['jet_muflav'] == 0) &
            (data['jet_tauflav'] == 0) &
            (data['jet_elflav'] == 0) &
            (data['jet_genmatch_hflav'] == 0) &
            ((abs(data['jet_genmatch_pflav']) == 0) | (abs(data['jet_genmatch_pflav']) == 1) | (abs(data['jet_genmatch_pflav']) == 2) | (abs(data['jet_genmatch_pflav']) == 3))
        )
    data['label_uds'] = condition_uds

    condition_muon = (
            (data['jet_genmatch_pt'] > 0) &
            (data['jet_muflav'] == 1) &
            (data['jet_tauflav'] == 0) &
            (data['jet_elflav'] == 0)
        )
    data['label_muon'] = condition_muon

    condition_electron = (
            (data['jet_genmatch_pt'] > 0) &
            (data['jet_muflav'] == 0) &
            (data['jet_tauflav'] == 0) &
            (data['jet_elflav'] == 1)
        )
    data['label_electron'] = condition_electron

    # data['target_pt'] = np.clip(((data["label_b"]) | (data["label_c"]) | (data["label_uds"]) | (data["label_g"]))*ak.nan_to_num(data["jet_genmatch_pt"]/data["jet_pt_phys"],nan=0,posinf=0,neginf=0)+((data["label_tau"]) | (data["label_muon"]) | (data["label_electron"]))*ak.nan_to_num((data["jet_genmatch_lep_vis_pt"]/data["jet_pt_phys"]),nan=0,posinf=0,neginf=0),0.3,2)
    # data['target_pt'] = np.clip(((data["label_b"]) | (data["label_c"]) | (data["label_uds"]) | (data["label_g"]))*ak.nan_to_num(data["jet_genmatch_pt"]/data["jet_pt_phys"],nan=0,posinf=0,neginf=0)+((data["label_tau"]) | (data["label_taup"]) | (data["label_taum"]) | (data["label_muon"]) | (data["label_electron"]))*ak.nan_to_num((data["jet_genmatch_lep_vis_pt"]/data["jet_pt_phys"]),nan=0,posinf=0,neginf=0),0.3,2)
    data['target_pt'] = np.clip(((data["label_b"]) | (data["label_c"]) | (data["label_uds"]) | (data["label_g"]))*ak.nan_to_num(data["jet_genmatch_pt"]/data["jet_pt_phys"],nan=0,posinf=0,neginf=0)+((data["label_taup"]) | (data["label_taum"]) | (data["label_muon"]) | (data["label_electron"]))*ak.nan_to_num((data["jet_genmatch_lep_vis_pt"]/data["jet_pt_phys"]),nan=0,posinf=0,neginf=0),0.3,2)
    print(data['target_pt'])

    data_b = data[(condition_b)]
    # data['label_b'] = (data[(condition_b)] > 0)
    if splitTau:
        # data_tau = data[(condition_tau)]
        data_taup = data[(condition_taup)]
        data_taum = data[(condition_taum)]
    # else:
        # data_tau = None
    if splitGluon:
        data_gluon = data[(condition_gluon)]
    else:
        data_gluon = None
    if splitCharm:
        data_charm = data[(condition_charm)]
    else:
        data_charm = None
    
    data_muon = data[(condition_muon)]
    data_electron = data[(condition_electron)]

    # Definition of background (non-b jets)
    # if splitTau and splitGluon and splitCharm:
    #     condition_bkg = (~condition_b) & (~condition_tau) & (~condition_gluon) & (~condition_charm)
    # elif splitTau and splitGluon:
    #     condition_bkg = (~condition_b) & (~condition_tau) & (~condition_gluon)
    # elif splitTau:
    #     condition_bkg = (~condition_b) & (~condition_tau)
    # elif splitGluon:
    #     condition_bkg = (~condition_b) & (~condition_gluon)
    # else:
    #     condition_bkg = (~condition_b)

    # data_bkg = data[condition_bkg]
    data_bkg = data[condition_uds]

    # Sanity check
    print("Sanity check:")
    print("Length of data:", len(data))
    sum = 0
    print("Length of data_b:", len(data_b))
    sum = sum + len(data_b)
    if splitTau:
        # print("Length of data_tau:", len(data_tau))
        print("Length of data_taup:", len(data_taup))
        print("Length of data_taum:", len(data_taum))
        sum = sum + len(data_taup)
        sum = sum + len(data_taum)
    if splitGluon:
        print("Length of data_gluon:", len(data_gluon))
        sum = sum + len(data_gluon)
    if splitCharm:
        print("Length of data_charm:", len(data_charm))
        sum = sum + len(data_charm)
    print("Length of data_bkg:", len(data_bkg))
    sum = sum + len(data_bkg)
    print("Length of data_muon:", len(data_muon))
    sum = sum + len(data_muon)
    print("Length of data_electron:", len(data_electron))
    sum = sum + len(data_electron)
    print("Sum:", sum)

    if not len(data) == sum:
        print ("ERROR: Data splitting does not match!!")

    # return {"b": data_b, "tau": data_tau, "gluon": data_gluon, "charm": data_charm, "bkg": data_bkg, "muon": data_muon, "electron": data_electron, "taup": data_taup, "taum": data_taum}
    return {"b": data_b, "taup": data_taup, "taum": data_taum, "gluon": data_gluon, "charm": data_charm, "bkg": data_bkg, "muon": data_muon, "electron": data_electron}


def reduceDatasetToMin(dataJson):
    # lenghts = [len(dataJson[key]) for key in dataJson]
    lenghts = []
    for key in dataJson:
        if dataJson[key] is not None:
            lenghts.append(len(dataJson[key]))
    minL = min(lenghts)
    print ("Reduce to", minL, "per class!")

    for key in dataJson:
        if dataJson[key] is not None:
            random_indices = np.random.choice(len(dataJson[key]), size=minL, replace=False)
            dataJson[key] = dataJson[key][random_indices]


def addResponseVars(data):
    data['jet_ptUncorr_div_ptGen'] = ak.nan_to_num(data['jet_pt_phys']/data['jet_genmatch_pt'], copy=True, nan=0.0, posinf=0., neginf=0.)
    data['jet_ptCorr_div_ptGen'] = ak.nan_to_num(data['jet_pt_corr']/data['jet_genmatch_pt'], copy=True, nan=0.0, posinf=0., neginf=0.)
    data['jet_ptRaw_div_ptGen'] = ak.nan_to_num(data['jet_pt_raw']/data['jet_genmatch_pt'], copy=True, nan=0.0, posinf=0., neginf=0.)


def createAndSaveTrainingData(data_json, keys, nconstit=16):
    print ("createAndSaveTrainingData with keys:", keys)
    objects = [{"name" : "pfcand", "key" : "jet_pfcand", "fields" : keys, "n_obj" : nconstit},]
    classes = {}
    n_classes = 1
    for key in data_json:
        if data_json[key] is not None:
            classes[key] = {}
            x, var_names = formatData(data_json[key], objects, verbosity = 0)
            if key == "bkg":
                classIdx = 0
            else:
                classIdx = n_classes
                n_classes = n_classes + 1
            classes[key]["idx"] = classIdx
            classes[key]["x"] = x
            classes[key]["x_global"] = data_json[key]
            classes[key]["target_pt"] = data_json[key]["target_pt"]

    # creating labels
    for key in classes:
        if key == "bkg":
            y = np.zeros(len(classes[key]["x"]))
        else:
            y = np.ones(len(classes[key]["x"])) * classes[key]["idx"]
        classes[key]["y"] = y

    # creating pt target
    # np.clip(((label_b) | (label_c) | (label_uds) | (label_g))*np.nan_to_num(jet_hlt_genmatch_pt/jet_hlt_pt,nan=0,posinf=0,neginf=0)+((label_taup) | (label_taum))*np.nan_to_num((jet_hlt_genmatch_lep_vis_pt/jet_hlt_pt),nan=0,posinf=0,neginf=0),0.3,2)
    # for key in classes:
    #     y_target = target_pt

    # combining signal & bkg
    x_all = np.concatenate([classes[key]["x"] for key in classes])
    y_all = np.concatenate([classes[key]["y"] for key in classes])
    y_all = tf.keras.utils.to_categorical(y_all, num_classes=n_classes)

    x_global = ak.concatenate([classes[key]["x_global"] for key in classes])

    y_target = ak.concatenate([classes[key]["target_pt"] for key in classes])

    print (y_target)

    return classes, var_names, x_all, y_all, x_global, y_target

def splitAndShuffle(x, y, x_global, y_target, nvars, nconstit = 16, testfraction = 0.2, shuffleConst = True):
    # Reshape Data and Split into Train and Test
    from sklearn.model_selection import train_test_split

    # Reshape inputs and targets 
    x = np.reshape(x,[-1, nconstit, nvars])

    if shuffleConst:
        # Shuffle jet constituents
        print("splitAndShuffle: before --->> x[0,0:4,0] = ",x[0, 0:nconstit, 0])
        for i in range(x.shape[0]):
            x[i] = x[i, np.random.permutation(nconstit), :]
        print("splitAndShuffle: after --->> x[0,0:4,0] = ",x[0, 0:nconstit, 0])

    # Split data into train and test shuffling events
    X_train, X_test, Y_train, Y_test, x_global_train, x_global_test, y_target_train, y_target_test = train_test_split(x, y, x_global, y_target, test_size = testfraction, shuffle = True)

    return X_train, X_test, Y_train, Y_test, x_global_train, x_global_test, y_target_train, y_target_test