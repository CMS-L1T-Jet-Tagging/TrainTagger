"""Sources: reads the inputs of the variables (see INPUTS in this
package) for one model or pseudo-model."""

import json
import os
import shutil
import tempfile

import numpy as np

from tagger.data.tools import load_data, make_data, to_ML
from tagger.eval.inputs import MODEL_VERSIONS, PSEUDO_MODELS
from tagger.eval.inputs.events import CmsswModel, EventSample
from tagger.eval.inputs.fastpuppi import load_process
from tagger.eval.inputs.jets import JetSample


class Sources:
    """Inputs for one model folder (``-m output/<model>``), the
    ``-m cmssw`` pseudo-model, or a version of a model (``pseudo``
    hls4ml or emulator, ``model_dir`` the model folder).
    """

    def __init__(
        self,
        model_dir,
        process_dir="signal_process_data",
        config=None,
        pseudo="",
    ):
        self.model_dir = model_dir
        self.process_dir = process_dir
        self.config = config or {}
        self.pseudo = pseudo
        self._class_labels = None
        self._model = None

    def describe(self):
        out = {"model_dir": os.path.abspath(self.model_dir)}
        if self.pseudo:
            out["pseudo_model"] = self.pseudo
        return out

    def _no_network(self, what):
        if self.pseudo in PSEUDO_MODELS:
            raise RuntimeError(
                f"the {self.pseudo} pseudo-model has no {what}; it supports"
                " the seed collections only"
            )

    def _emulation_only(self, what):
        self._no_network(what)
        if self.pseudo in MODEL_VERSIONS:
            raise RuntimeError(
                f"-m {self.pseudo}:<model> has the emulation jets only"
                f" (collection emulation), not the {what}"
            )

    def class_labels(self):
        self._no_network("class scores")
        if self._class_labels is None:
            with open(os.path.join(self.model_dir, "class_labels.json")) as f:
                self._class_labels = json.load(f)
        return self._class_labels

    def particle_input_vars(self):
        if self.pseudo in PSEUDO_MODELS:
            return []
        with open(
            os.path.join(self.model_dir, "particle_input_vars.json")
        ) as f:
            return json.load(f)

    def training_history(self):
        path = os.path.join(self.model_dir, "training_history.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} missing: it is written by train.py since the eval"
                " refactor"
            )
        with open(path) as f:
            return json.load(f)

    def loss_names(self):
        """Per-output training losses in the history (the removed
        plot_loss plotted these), [] without history."""
        if self.pseudo:
            return []
        try:
            history = self.training_history()
        except FileNotFoundError:
            return []
        return [
            k
            for k in history
            if k.endswith("_loss") and not k.startswith("val_")
        ]

    def model(self):
        if self.pseudo in PSEUDO_MODELS:
            return CmsswModel()
        if self._model is None:
            # imported here: tagger.model loads TensorFlow
            from tagger.model.common import fromFolder

            self._model = fromFolder(self.model_dir)
        return self._model

    def _predict(self, model_inputs):
        return self.model().predict(model_inputs, verbose=0)

    def test_jets(self):
        self._emulation_only("test split")
        d = os.path.join(self.model_dir, "testing_data")
        with np.load(
            os.path.join(d, "test_dict.npz"), allow_pickle=False
        ) as f:
            model_inputs = {k: f[k] for k in f.files}
        y_true = np.load(os.path.join(d, "y_test.npz"))["label"]
        truth_pt = np.load(os.path.join(d, "truth_pt_test.npz"))["truth_pt"]
        reco_pt = np.load(os.path.join(d, "reco_pt_test.npz"))["reco_pt"]
        eta_path = os.path.join(d, "reco_eta_test.npz")
        reco_eta = (
            np.load(eta_path)["reco_eta"] if os.path.exists(eta_path) else None
        )
        y_pred, pt_ratio = self._predict(model_inputs)
        return JetSample(
            y_true, y_pred, pt_ratio, truth_pt, reco_pt, model_inputs, reco_eta
        )

    def has_pt_weights(self):
        """Whether the model has per-constituent pT weights
        (output_pt_weights_name set, WeightedAverageSimpleModel)."""
        if self.pseudo:
            return False
        return self.model().output_pt_weights_name is not None

    def has_eta(self):
        """Whether the test split has the jet eta
        (testing_data/reco_eta_test.npz, saved by train.py)."""
        return os.path.exists(
            os.path.join(self.model_dir, "testing_data", "reco_eta_test.npz")
        )

    def process_jets(self, process):
        self._emulation_only("process jets")
        model = self.model()
        data, _, class_labels = load_data(
            os.path.join(self.process_dir, process), 100, model, test_ratio=0
        )[:3]
        X, jet_features, y, _, truth_pt, reco_pt = to_ML(data, class_labels)
        raw = {"basic_input": X, "jet_features": jet_features}
        y_pred, pt_ratio = self._predict(model.prepare_inputs(raw)[0])
        return JetSample(
            y,
            y_pred,
            pt_ratio,
            truth_pt,
            reco_pt,
            raw,
            np.asarray(data["jet_eta_phys"]),
        )

    # -- ntuples for the trigger seeds ---------------------------------
    def _events(self, path, n_entries):
        if self.pseudo in MODEL_VERSIONS:
            self._emulation_only("seed ntuples")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} missing: set the sample path in the config"
                " (MINBIAS_SAMPLE, SIGNAL_SAMPLES)"
            )
        print(f"  reading {path} ({n_entries} jets)")
        return EventSample(path, self.config["TREE"], n_entries, self.model())

    def minbias_events(self, n_entries):
        """MinBias events, config MINBIAS_SAMPLE, first n_entries jets."""
        return self._events(self.config["MINBIAS_SAMPLE"], n_entries)

    def signal_events(self, sample, n_entries):
        """Signal events of config SIGNAL_SAMPLES[sample], first
        n_entries jets."""
        samples = self.config["SIGNAL_SAMPLES"]
        if sample not in samples:
            raise KeyError(
                f"unknown signal sample {sample!r}, config SIGNAL_SAMPLES has"
                f" {sorted(samples)}"
            )
        return self._events(samples[sample], n_entries)

    # -- The emulation ntuple ------------------------------------------
    # the network, its hls4ml build or the CMSSW emulator
    def emulation_jets(self):
        """Jets of config EMULATION_SAMPLE in file order, with the
        outputs of this model version."""
        self._no_network("emulation jets")
        model = self.model()
        data, class_labels = self._emulation_data(model)
        X, jet_features, y, _, _, reco_pt = to_ML(data, class_labels)
        if self.pseudo == "emulator":  # the CMSSW emulator has no pileup score
            zeros = np.zeros(len(data))
            y_pred = np.stack(
                [
                    (
                        zeros
                        if c == "pileup"
                        else np.asarray(data[f"jet_SC4NGJet_score_{c}"])
                    )
                    for c in class_labels
                ],
                axis=1,
            )
            pt_ratio = np.asarray(data["jet_SC4NGJet_score_regression"])
        else:
            model_inputs, _ = model.prepare_inputs(
                {"basic_input": X, "jet_features": jet_features}
            )
            model_inputs = {
                k: np.ascontiguousarray(v, dtype=np.float64)
                for k, v in model_inputs.items()
            }
            if self.pseudo == "hls4ml":
                y_pred, pt_ratio = self._hls4ml_predict(model, model_inputs)
            else:
                y_pred, pt_ratio = model.predict(model_inputs, verbose=0)
        truth_pt = np.asarray(data["jet_genmatch_pt"])
        return JetSample(
            y, np.asarray(y_pred), pt_ratio, truth_pt, reco_pt, None
        )

    def _hls4ml_predict(self, model, model_inputs):
        """Class scores (quantised to config HLS4ML_SCORE_BITS) and pT
        correction of the hls4ml build, converted into a temporary
        folder."""
        folder = tempfile.mkdtemp(prefix="hls4ml_")
        try:
            model.firmware_convert(folder, build=False)
            hls = model.hls_jet_model
            hls_inputs = [
                model_inputs[var.name] for var in hls.get_input_variables()
            ]
            y, pt_ratio = hls.predict(
                hls_inputs[0] if len(hls_inputs) == 1 else hls_inputs
            )
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        scale = 2 ** (self.config["HLS4ML_SCORE_BITS"] - 1)
        return (
            np.round(np.asarray(y) * scale) / scale,
            np.asarray(pt_ratio)[:, 0],
        )

    def _emulation_data(self, model):
        """make_data output of config EMULATION_SAMPLE in
        EMULATION_DATA_DIR (made once, remade when the sample file
        changes), loaded without shuffling."""
        sample, folder = (
            self.config["EMULATION_SAMPLE"],
            self.config["EMULATION_DATA_DIR"],
        )
        if not os.path.exists(sample):
            raise FileNotFoundError(
                f"{sample} missing: set config EMULATION_SAMPLE (in CI the"
                " ntuple of the CMSSW build)"
            )
        stamp = {
            "sample": os.path.abspath(sample),
            "size": os.path.getsize(sample),
            "mtime": os.path.getmtime(sample),
        }
        stamp_path = os.path.join(folder, "sample.json")
        print(f"  reading the emulation jets of {sample}")
        if not os.path.exists(stamp_path) or _read_json(stamp_path) != stamp:
            shutil.rmtree(folder, ignore_errors=True)
            make_data(
                infile=sample,
                outdir=folder,
                extras="extra_emulation_fields",
                tree=self.config["EMULATION_TREE"],
            )
            with open(stamp_path, "w") as f:
                json.dump(stamp, f)
        data, _, class_labels = load_data(
            folder, 100, model, test_ratio=0, shuffle=False
        )[:3]
        return data, class_labels

    # -- FastPUPPI performance ntuples ---------------------------------
    # of the model's CMSSW emulation
    def fastpuppi_events(self, process):
        """The gen jets, gen particles and L1 jet collections of one
        process (config FASTPUPPI_DIR)."""
        self._emulation_only("FastPUPPI ntuples")
        print(f"  reading {process} of {self.config['FASTPUPPI_DIR']}")
        return load_process(self.config["FASTPUPPI_DIR"], process, self.config)


def _read_json(path):
    with open(path) as f:
        return json.load(f)
