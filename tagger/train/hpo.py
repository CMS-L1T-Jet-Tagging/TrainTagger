from argparse import ArgumentParser
from tagger.model.common import fromFolder, fromYaml
from tagger.train.train import save_test_data, train_weights
from tagger.data.tools import load_data, to_ML


def tune(model, out_dir, percent):

    # Load the data, class_labels and input variables name, not really using input variable names to be honest
    data_train, data_test, class_labels, input_vars, extra_vars = load_data("training_data/", percentage=percent)
    model.set_labels(
        input_vars,
        extra_vars,
        class_labels,
    )

    # Make into ML-like data for training
    X_train, y_train, pt_target_train, truth_pt_train, reco_pt_train = to_ML(data_train, class_labels)

    # Save X_test, y_test, and truth_pt_test for plotting later
    X_test, y_test, _, truth_pt_test, reco_pt_test = to_ML(data_test, class_labels)
    save_test_data(out_dir, X_test, y_test, truth_pt_test, reco_pt_test)

    # Calculate the sample weights for training
    sample_weight = train_weights(
        y_train,
        reco_pt_train,
        class_labels,
        weightingMethod=model.training_config['weight_method'],
        debug=model.run_config['debug'],
    )
    if model.run_config['debug']:
        print("DEBUG - Checking sample_weight:")
        print(sample_weight)

    # Get input shape
    input_shape = X_train.shape[1:]  # First dimension is batch size
    output_shape = y_train.shape[1:]
    
    model.init_tuner()
    model.build_model(input_shape, output_shape, tuner=model.tuner)
    # Train it with a pruned model
    num_samples = X_train.shape[0] * (1 - model.training_config['validation_split'])
    model.compile_model(num_samples)

    model.tune(X_train, y_train, pt_target_train, sample_weight)
    
    model.save()

    return model.jet_model.hyperparameters

if __name__ == "__main__":

    parser = ArgumentParser()
    # Training argument
    parser.add_argument(
        '-o', '--output', default='output/baseline', help='Output model directory path, also save evaluation plots'
    )
    parser.add_argument('-p', '--percent', default=100, type=int, help='Percentage of how much processed data to train on')
    parser.add_argument(
        '-y', '--yaml_config', default='tagger/model/configs/baseline_larger.yaml', help='YAML config for model'
    )

    args = parser.parse_args()
    
    model = fromYaml(args.yaml_config, args.output)
    
    results = tune(model, args.output, args.percent )
    
    yaml_config_str = args.yaml_config.split(".")
    new_config = yaml_config_str[0] + '_optimised' + yaml_config_str[1]
    model.toYaml(new_config)
        
    print(f"Best hyperparameters found were: {results}")
