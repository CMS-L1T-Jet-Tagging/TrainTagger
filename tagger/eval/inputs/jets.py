"""Jet samples: jets with truth and model outputs (the test split,
signal processes)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class JetSample:
    """Jets with truth and model outputs.

    y_true: one-hot truth (n_jets, n_classes); y_pred: class scores
    (n_jets, n_classes); pt_ratio: regressed pT correction (n_jets,) or
    None; truth_pt, reco_pt: (n_jets,) or None; model_inputs: model
    input arrays by name (basic_input is (n_jets, n_candidates,
    n_features)); reco_eta: jet eta (n_jets,), None for models trained
    before it was saved.
    """

    y_true: np.ndarray
    y_pred: np.ndarray
    pt_ratio: np.ndarray = None
    truth_pt: np.ndarray = None
    reco_pt: np.ndarray = None
    model_inputs: dict = None
    reco_eta: np.ndarray = None

    def __len__(self):
        return len(self.y_true)

    def subset(self, idx):
        pick = lambda a: None if a is None else a[idx]
        return JetSample(
            self.y_true[idx],
            self.y_pred[idx],
            pick(self.pt_ratio),
            pick(self.truth_pt),
            pick(self.reco_pt),
            (
                None
                if self.model_inputs is None
                else {k: v[idx] for k, v in self.model_inputs.items()}
            ),
            pick(self.reco_eta),
        )

    def indices_shared_with(self, other):
        """Indices of jets whose basic_input also occurs in ``other``
        (old basic.py filter_process)."""
        process_data = np.unique(other.model_inputs["basic_input"], axis=0)
        unique_test, idx_unique_test = np.unique(
            self.model_inputs["basic_input"], axis=0, return_index=True
        )
        comparison = np.concatenate((unique_test, process_data), axis=0)
        _, index, counts = np.unique(
            comparison, axis=0, return_index=True, return_counts=True
        )
        return idx_unique_test[index[counts == 2]]
