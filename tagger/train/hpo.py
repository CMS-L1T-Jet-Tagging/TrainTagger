import kubeflow.katib as katib 
import time

def objective(parameters):
    batch_size = parameters["batch_size"]

    # Simulate a training
    import time
    time.sleep(100)
    loss = 0.1

    # Katib parses metrics in this format: <metric-name>=<metric-value>.
    print(f"loss={loss}")

katib_client = katib.KatibClient()
parameters = {
    "batch_size": katib.search.int(min=4, max=64),
}
name = "tune-experiment"
katib_client.tune(
    name=name,
    objective=objective,
    parameters=parameters,
    objective_metric_name="loss",
    max_trial_count=5,
    resources_per_trial={"gpu": "2"},
)

# Wait for the trials to finish
katib_client.wait_for_experiment_condition(name=name)
print(katib_client.get_optimal_hyperparameters(name))

