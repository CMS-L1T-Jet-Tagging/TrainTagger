"""Toy jets, events and model for the eval and plot tests: no files,
the model a small model class built in memory."""

import awkward as ak
import numpy as np
import tensorflow as tf

from tagger.data.tools import group_id_values
from tagger.eval import config as config_mod
from tagger.eval.variable_manager import EvalRun
from tagger.eval.inputs.jets import JetSample
from tagger.model.QKerasModel import QKerasModel

CLASSES = {
    "b": 0,
    "charm": 1,
    "light": 2,
    "gluon": 3,
    "taup": 4,
    "taum": 5,
    "muon": 6,
    "electron": 7,
}
FEATURES = [
    "pt",
    "pt_rel",
    "pt_log",
    "deta",
    "dphi",
    "mass",
    "isPhoton",
    "isElectronPlus",
    "isElectronMinus",
    "isMuonPlus",
    "isMuonMinus",
    "isNeutralHadron",
    "isChargedHadronPlus",
    "isChargedHadronMinus",
    "z0",
    "dxy",
    "isfilled",
    "puppiweight",
]


def jets(n=2000, seed=0, skill=2.0):
    """Jets whose true class gets a score boost of ``skill`` before the
    softmax."""
    rng = np.random.default_rng(seed)
    k = len(CLASSES)
    labels = rng.integers(0, k, n)
    y_true = np.eye(k, dtype=np.float32)[labels]
    logits = rng.normal(size=(n, k)) + skill * y_true
    y_pred = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    truth_pt = rng.uniform(15, 500, n)
    reco_pt = truth_pt * rng.normal(0.9, 0.1, n)
    pt_ratio = rng.normal(1.1, 0.05, n)
    x = rng.uniform(0, 1, size=(n, 16, len(FEATURES)))
    n_filled = rng.integers(
        1, 17, n
    )  # >= 1: all-zero jets are duplicates, which filter_process drops
    filled = np.arange(16)[None, :] < n_filled[:, None]
    x[:, :, 16] = filled
    x[~filled] = 0
    reco_eta = rng.uniform(-2.4, 2.4, n)
    return JetSample(
        y_true,
        y_pred.astype(np.float32),
        pt_ratio,
        truth_pt,
        reco_pt,
        {"basic_input": x},
        reco_eta,
    )


class ToyModel(QKerasModel):
    """A small two-branch model class (class scores and pT correction)
    on 16 candidates of the toy features, built on creation. predict and
    get_branch_model are those of the model classes."""

    def __init__(self):
        super().__init__(output_dir="")
        self.particle_input_vars = list(FEATURES)
        self.inputs = {"basic_features": ["basic_input"], "jet_features": []}
        self.build_model((16, len(FEATURES)), (len(CLASSES),))

    def build_model(self, inputs_shape, outputs_shape):
        """Dense layer, average over the candidates, then one dense
        layer per output; seeded, so every toy model has the same
        weights."""
        tf.keras.utils.set_random_seed(0)
        inp = tf.keras.Input(inputs_shape, name="basic_input")
        h = tf.keras.layers.GlobalAveragePooling1D()(
            tf.keras.layers.Dense(4, activation="relu")(inp)
        )
        out_id = tf.keras.layers.Dense(
            outputs_shape[0], activation="softmax", name=self.output_id_name
        )(h)
        out_pt = tf.keras.layers.Dense(1, name=self.output_pt_name)(h)
        self.jet_model = tf.keras.Model(inp, [out_id, out_pt])


class ToySources:
    def __init__(self, sample=None, process=None):
        self.sample = sample if sample is not None else jets()
        # the process sample shares its first 300 jets with the test
        # sample
        self.process = (
            process
            if process is not None
            else _with_shared(self.sample, jets(500, seed=1), 300)
        )

    def class_labels(self):
        return dict(CLASSES)

    def particle_input_vars(self):
        return list(FEATURES)

    def describe(self):
        return {"model_dir": "toy"}

    def model(self):
        if not hasattr(self, "_model"):
            self._model = ToyModel()
        return self._model

    def training_history(self):
        return {
            "loss": [3.0, 2.0, 1.5],
            "jet_id_output_loss": [2.0, 1.5, 1.2],
            "val_jet_id_output_loss": [2.1, 1.6, 1.4],
            "val_loss": [3.1, 2.2, 1.7],
        }

    def has_eta(self):
        return self.sample.reco_eta is not None

    def loss_names(self):
        return ["jet_id_output_loss"]

    def test_jets(self):
        return self.sample

    def process_jets(self, process):
        return self.process


def _with_shared(test, other, n_shared):
    idx = np.arange(n_shared)
    shared = test.subset(idx)
    return JetSample(
        np.concatenate([shared.y_true, other.y_true]),
        np.concatenate([shared.y_pred, other.y_pred]),
        np.concatenate([shared.pt_ratio, other.pt_ratio]),
        np.concatenate([shared.truth_pt, other.truth_pt]),
        np.concatenate([shared.reco_pt, other.reco_pt]),
        {
            "basic_input": np.concatenate(
                [
                    shared.model_inputs["basic_input"],
                    other.model_inputs["basic_input"],
                ]
            )
        },
        np.concatenate([shared.reco_eta, other.reco_eta]),
    )


def run(sources=None, overrides=()):
    return EvalRun(
        sources or ToySources(), config_mod.load(overrides=overrides)
    )


class ToyEvents:
    """Stand-in for tagger.eval.inputs.events.EventSample: jets of
    n_events events (4 to 8 jets each) with the ntuple branches the seed
    variables read and random model outputs."""

    def __init__(self, n_events=3000, seed=0, signal=False, pseudo=""):
        self.pseudo = (
            pseudo  # "cmssw": the CMSSW pseudo-model, no network outputs
        )
        rng = np.random.default_rng(seed)
        n_jets = rng.integers(4, 9, n_events)
        n = int(n_jets.sum())
        event = np.repeat(np.arange(n_events), n_jets)
        pt = rng.uniform(31, 150, n) if signal else rng.uniform(5, 150, n)
        eta = rng.uniform(-2.3, 2.3, n) if signal else rng.uniform(-3, 3, n)
        logits = rng.normal(size=(n, len(CLASSES)))
        self._outputs = (
            (
                np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
            ).astype(np.float32),
            rng.normal(1.0, 0.1, n).astype(np.float32),
        )
        b = rng.uniform(0, 1.5, n)
        b[rng.uniform(size=n) < 0.1] = -1
        self._branches = {
            k: ak.Array(v)
            for k, v in {
                "event": event,
                "jet_pt": pt,
                "jet_eta_phys": eta,
                "jet_phi_phys": rng.uniform(-np.pi, np.pi, n),
                "jet_taupt": pt * rng.uniform(0.7, 1.3, n),
                "jet_tauscore": rng.uniform(0, 1, n),
                "jet_bjetscore": b,
                "jet_tauflav": (rng.uniform(size=n) < 0.5).astype(np.int32),
                "jet_genmatch_pt": (
                    pt if signal else pt * rng.normal(1, 0.1, n)
                ),
                "jet_genmatch_eta": (
                    eta if signal else eta + rng.normal(0, 0.05, n)
                ),
                "jet_genmatch_dR": rng.uniform(0, 0.5, n),
                "genHH_mass": np.repeat(
                    rng.uniform(250, 1000, n_events), n_jets
                ),
            }.items()
        }
        self.n_entries = n

    def __len__(self):
        return self.n_entries

    def describe(self):
        return "toy events"

    def has_branch(self, name):
        return name in self._branches

    def branch(self, name):
        return self._branches[name]

    @property
    def n_events(self):
        return len(np.unique(self._branches["event"]))

    def outputs(self):
        if self.pseudo:
            raise RuntimeError("pseudo-model without network outputs")
        return self._outputs

    def grouped(
        self, *arrays, num_elements=2, ordering_var=None, with_event_id=False
    ):

        event_id, grouped = group_id_values(
            self._branches["event"],
            *arrays,
            num_elements=num_elements,
            ordering_var=ordering_var,
        )
        return (event_id, grouped) if with_event_id else grouped


class ToySeedSources(ToySources):
    """Toy seed inputs; pseudo="cmssw" gives the same events as the
    CMSSW pseudo-model."""

    def __init__(self, pseudo=""):
        super().__init__()
        self.pseudo = pseudo

    def class_labels(self):
        if self.pseudo:
            raise RuntimeError("pseudo-model without class scores")
        return super().class_labels()

    def minbias_events(self, n_entries):
        if not hasattr(self, "_minbias"):
            self._minbias = ToyEvents(seed=10, pseudo=self.pseudo)
        return self._minbias

    def signal_events(self, sample, n_entries):
        if not hasattr(self, "_signal"):
            self._signal = {}
        if sample not in self._signal:
            self._signal[sample] = ToyEvents(
                2000,
                seed=20 + len(self._signal),
                signal=True,
                pseudo=self.pseudo,
            )
        return self._signal[sample]


# Toy MinBias rate scale: rates are 100 x the passing fraction, so the
# seed target rates (14-31 kHz) lie on the toy grids
SEED_OVERRIDES = ["MINBIAS_RATE=100"]
SEED_COLLECTIONS = [
    "seedSingleTau",
    "seedDiTau",
    "seedDiTauTopo",
    "seedBbbb",
    "seedBbtt",
]
