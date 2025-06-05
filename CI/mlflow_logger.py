import os
from argparse import ArgumentParser

import mlflow

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-w', '--website', default='https://cms-l1t-jet-tagger.web.cern.ch/', help='Plotting Website')
    parser.add_argument('-f', '--firmware', default='/eos/project/c/cms-l1t-jet-tagger/CI/main/firmware', help='Firmware archive')
    parser.add_argument('-m', '--model', default='/eos/project/c/cms-l1t-jet-tagger/CI/main/model', help='Model archive')
    parser.add_argument('-p', '--plots', default='/eos/project/c/cms-l1t-jet-tagger/CI/main/plots', help='Plots archive')
    parser.add_argument('-n', '--name', default='baseline', help='Model experiment name')

    args = parser.parse_args()

    f = open("mlflow_run_id.txt", "r")
    run_id = f.read()

    mlflow.get_experiment_by_name(os.getenv('CI_COMMIT_REF_NAME'))
    with mlflow.start_run(experiment_id=1, run_name=args.name, run_id=run_id):  # pass None to start a new run

        mlflow.log_param("Plots Website: ", args.website)
        mlflow.log_param("Firmware Archive: ", args.firmware)
        mlflow.log_param("Model Archive: ", args.model)
        mlflow.log_param("Plots Archive: ", args.plots)
        mlflow.log_artifact("output/baseline/model/saved_model.keras")
