"""Training diagnostics: the loss history saved by train.py
(training_history.json), and the SHAP feature importance of the model
outputs."""

import numpy as np

from tagger.eval.variable_manager import vm

# -- Loss history ------------------------------------------------------


@vm.variable(inputs={"training_history"}, kind="curve")
def lossHistory(run, loss: str):
    """Loss per epoch of ``loss`` (a Keras history key such as
    jet_id_output_loss), on the training and the validation split.
    Fields ``epoch`` (from 1), ``train``, ``val``.
    """
    history = run.input("training_history")
    if loss not in history:
        raise KeyError(
            f"no loss {loss!r} in the training history, it has"
            f" {sorted(history)}"
        )
    train = np.asarray(history[loss])
    return {
        "epoch": np.arange(1, len(train) + 1),
        "train": train,
        "val": np.asarray(history["val_" + loss]),
    }


# -- Explainability ----------------------------------------------------
BRANCHES = {"classification": "jet_id_output", "regression": "pT_output"}


@vm.variable(inputs={"model", "test_jets"}, kind="curve")
def shapImportance(run, branch: str):
    """Mean |SHAP value| per output and input feature, for ``branch``
    classification or regression.

    shap.GradientExplainer on the first SHAP_N_JETS test jets (also the
    background), with seed SHAP_SEED, so the same model always gives the
    same values. Per jet the values are summed over candidates, then |.|
    is averaged over jets. Fields ``values`` (outputs x features),
    ``features``, ``outputs`` (class names, or ``Regression``).
    """
    # imported here: a slow import, needed only by this variable
    import shap

    if branch not in BRANCHES:
        raise ValueError(
            f"shapImportance: branch must be one of {list(BRANCHES)}, got"
            f" {branch!r}"
        )
    model = run.input("model")
    model_inputs = run.jets().model_inputs
    sub = model.get_branch_model(BRANCHES[branch])
    n = run.config["SHAP_N_JETS"]
    x = [model_inputs[name][:n] for name in sub.input_names]
    explainer = shap.GradientExplainer(sub, x)
    values = explainer.shap_values(
        x[0] if len(x) == 1 else x, rseed=run.config["SHAP_SEED"]
    )
    if len(x) > 1:
        per_input = values
        values = per_input[sub.input_names.index("basic_input")]
    values = np.sum(values, axis=1)  # (jets, features, outputs)
    features = list(model.particle_input_vars)
    if "jet_features" in sub.input_names:
        jet_values = per_input[sub.input_names.index("jet_features")]
        values = np.concatenate((values, jet_values), axis=1)
        # the jet feature names differ per branch, as in the old
        # plot_shaply
        features += list(
            model.jet_input_vars
            if branch == "classification"
            else model.inputs["jet_features"]
        )
    importance = np.mean(
        np.abs(np.transpose(values, (2, 0, 1))), axis=1
    )  # (outputs, features)
    outputs = (
        list(run.class_labels())
        if branch == "classification"
        else ["Regression"]
    )
    return {"values": importance, "features": features, "outputs": outputs}
