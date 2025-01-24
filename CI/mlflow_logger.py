import os, sys
import json
from argparse import ArgumentParser

import mlflow


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-w','--website', default='https://cebrown.web.cern.ch/' , help = 'Plotting Website')    
    parser.add_argument('-f','--firmware', default='/eos/cms/store/cmst3/group/l1tr/MultiJetTagger/main/firmware' , help = 'Firmware archive')    
    parser.add_argument('-m','--model', default=='/eos/cms/store/cmst3/group/l1tr/MultiJetTagger/main/model', help = 'Model archive')
    parser.add_argument('-p','--plots', default=='/eos/cms/store/cmst3/group/l1tr/MultiJetTagger/main/plots', help = 'Plots archive')

    args = parser.parse_args()

    f = open("mlflow_run_id.txt", "r")
    run_id = (f.read())

    mlflow.get_experiment_by_name(os.getenv('CI_COMMIT_REF_NAME'))
    with mlflow.start_run(experiment_id=1,
                                run_name=args.name,
                                run_id=run_id # pass None to start a new run
                                ):

        mlflow.log_param("Plots Website: ",args.website)
        mlflow.log_param("Firmware Archive: ",args.firmware)
        mlflow.log_param("Model Archive: ",args.model)
        mlflow.log_param("Plots Archive: ",args.plots)
        mlflow.log_artifact("output/baseline/model/saved_model.h5")
