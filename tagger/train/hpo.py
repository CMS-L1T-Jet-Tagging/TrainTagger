from argparse import ArgumentParser
import os, shutil, json

#Import from other modules
from tagger.data.tools import make_data, load_data, to_ML
from tagger.plot.basic import loss_history, basic
import models

#Third parties
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from hgq.utils.sugar import FreeEBOPs
from sklearn.utils.class_weight import compute_class_weight
import mlflow
from datetime import datetime
from train import train_weights,save_test_data


from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.air.integrations.keras import ReportCheckpointCallback

def train_model_wrapper(config):

    VALIDATION_SPLIT = 0.1
    BATCH_SIZE = 1024
    EPOCHS = 100
    os.system('pwd')
    os.system('cp -r /builds/ml_l1/TrainTagger/training_data/ .')
    ID = np.random.randint(0,1000)
    out_dir = '/builds/ml_l1/TrainTagger/training_data/output/baseline/trials/'+str(ID)
    os.makedirs(out_dir, exist_ok=True)
    #Load the data, class_labels and input variables name, not really using input variable names to be honest
    data_train, data_test, class_labels, input_vars, extra_vars = load_data("training_data/", percentage=50)

    #Save input variables and extra variables metadata
    with open(os.path.join(out_dir, "input_vars.json"), "w") as f: json.dump(input_vars, f, indent=4) #Dump output variables
    with open(os.path.join(out_dir, "extra_vars.json"), "w") as f: json.dump(extra_vars, f, indent=4) #Dump output variables

    #Make into ML-like data for training
    X_train, y_train, pt_target_train, truth_pt_train, reco_pt_train = to_ML(data_train, class_labels)

    #Save X_test, y_test, and truth_pt_test for plotting later
    X_test, y_test, _, truth_pt_test, reco_pt_test = to_ML(data_test, class_labels)
    save_test_data(out_dir, X_test, y_test, truth_pt_test, reco_pt_test, class_labels)

    #Calculate the sample weights for training
    sample_weight = train_weights(y_train, truth_pt_train, class_labels)

    #Get input shape
    input_shape = X_train.shape[1:] #First dimension is batch size
    output_shape = y_train.shape[1:]

    #Dynamically get the model
    try:
        model_func = getattr(models, 'baseline')
        model = model_func(input_shape, output_shape)  # Assuming the model function doesn't require additional arguments
    except AttributeError:
        raise ValueError(f"Model '{'baseline'}' is not defined in the 'models' module.")

    #Train it with a pruned model
    num_samples = X_train.shape[0] * (1 - VALIDATION_SPLIT)
    opt = keras.optimizers.Adam(learning_rate=config["learning_rate"])
    model.compile(optimizer=opt,
                            loss={'jet_id_output': 'categorical_crossentropy', 'pT_output': keras.losses.Huber( delta=config["delta"])},
                            metrics = {'jet_id_output': 'categorical_accuracy', 'pT_output': ['mae', 'mean_squared_error']},
                            weighted_metrics = {'jet_id_output': 'categorical_accuracy', 'pT_output': ['mae', 'mean_squared_error']})

    print(model.summary())

    #Now fit to the data
    callbacks = [FreeEBOPs(),
                 EarlyStopping(monitor='val_loss', patience=10),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
                 ReportCheckpointCallback(metrics={"val_loss": "val_loss"})]

    history = model.fit({'model_input': X_train},
                            {'jet_id_output': y_train, 'pT_output': pt_target_train},
                            sample_weight=sample_weight,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            verbose=2,
                            validation_split=VALIDATION_SPLIT,
                            callbacks = [callbacks],
                            shuffle=True)

    #Export the model
    export_path = os.path.join(out_dir, "model/")
    os.makedirs(export_path, exist_ok=True)
    model.save(export_path+'saved_model.keras')
    print(f"Model saved to {export_path}")

    plot_path = os.path.join(out_dir, "plots/training")
    os.makedirs(plot_path, exist_ok=True)

    #Plot history
    loss_history(plot_path, history)
    os.system('rm -r training_data')


def tune_deepset():
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=400, grace_period=20
    )

    tuner = tune.Tuner(
        tune.with_resources(train_model_wrapper, resources={"cpu":2 , "gpu": 0.1}),
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=sched,
            num_samples=10,
        ),
        run_config=tune.RunConfig(
            name="exp",
            stop={"val_loss": 1.0},
        ),
        param_space={
            "threads": 1,
            "learning_rate": tune.uniform(0.001, 0.1),
            "delta": tune.uniform(0.1, 2.0),
        },
    )
    results = tuner.fit()
    return results

    

results = tune_deepset()
print(f"Best hyperparameters found were: {results.get_best_result().config} | val_loss: {results.get_best_result().metrics['val_loss']}")

