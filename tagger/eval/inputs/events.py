"""Event samples from the jet ntuples (MinBias and signal) for the
trigger seeds, and the CMSSW pseudo-model.

An :class:`EventSample` reads the first ``n_entries`` jets of a tree,
reads branches lazily, runs the model once on every jet and groups by
event id with tagger.data.tools.group_id_values. For the ``-m cmssw``
pseudo-model (the CMSSW L1 baseline, see :class:`CmsswModel`) there is
no network: the seed variables read its branches instead.
"""

import numpy as np
import uproot

from tagger.data.tools import extract_array, extract_nn_inputs, group_id_values


class EventSample:
    """The first ``n_entries`` jets of one ntuple tree, with the model
    outputs on every jet."""

    def __init__(self, path, tree, n_entries, model):
        self.path = path
        self.tree_name = tree
        self.n_entries = n_entries
        self.model = model
        self.pseudo = getattr(
            model, "pseudo", ""
        )  # "cmssw" for the CMSSW pseudo-model
        self._tree = None
        self._branches = {}
        self._outputs = None
        self._nn_inputs = None

    def __len__(self):
        return len(self.branch("event"))

    def describe(self):
        return f"{self.path}:{self.tree_name}[:{self.n_entries}]"

    @property
    def tree(self):
        if self._tree is None:
            self._tree = uproot.open(self.path)[self.tree_name]
        return self._tree

    def has_branch(self, name):
        return name in self.tree.keys()

    def branch(self, name):
        """One branch as an awkward array (tagger.data.tools.extract_array)."""
        if name not in self._branches:
            self._branches[name] = extract_array(
                self.tree, name, self.n_entries
            )
        return self._branches[name]

    @property
    def n_events(self):
        """Number of distinct events among the loaded jets (the rate
        denominator of the old scripts)."""
        return len(np.unique(self.branch("event")))

    def nn_inputs(self):
        """(particle inputs, jet inputs) as awkward arrays
        (tagger.data.tools.extract_nn_inputs)."""
        if self._nn_inputs is None:
            self._nn_inputs = extract_nn_inputs(
                self.tree,
                self.model.particle_input_vars,
                self.model.jet_input_vars,
                n_entries=self.n_entries,
            )
        return self._nn_inputs

    def outputs(self):
        """(class scores (n_jets, n_classes), pT correction (n_jets,))
        of the model on every jet."""
        if self.pseudo:
            raise RuntimeError(
                f"the {self.pseudo} pseudo-model has no network outputs"
            )
        if self._outputs is None:
            inputs, jet_inputs = self.nn_inputs()
            raw = {
                "basic_input": np.asarray(inputs),
                "jet_features": (
                    np.asarray(jet_inputs)
                    if self.model.jet_input_vars
                    else None
                ),
            }
            self._outputs = self.model.predict(
                self.model.prepare_inputs(raw)[0], verbose=0
            )
        return self._outputs

    def grouped(
        self, *arrays, num_elements=2, ordering_var=None, with_event_id=False
    ):
        """Group per-jet arrays by event, keeping events with at least
        ``num_elements`` jets, optionally sorted within the event by
        ``ordering_var`` (descending)."""
        event_id, grouped = group_id_values(
            self.branch("event"),
            *arrays,
            num_elements=num_elements,
            ordering_var=ordering_var,
        )
        return (event_id, grouped) if with_event_id else grouped


class CmsswModel:
    """The CMSSW L1 baseline as a pseudo-model (``-m cmssw``): the
    Phase-2 L1 algorithms already in CMSSW, whose outputs the jet
    ntuples store per jet, the NN PUPPI tau (score jet_tauscore, pT
    jet_taupt) and the L1 b-tag (jet_bjetscore). It is not the CMSSW
    emulation of our own model (the jet_SC4NGJet_score_* branches). No
    network: the seed variables read these branches. As in the old
    scripts, its singleTau working point is derived at the network's
    target rate; diTau, diTauTopo and bbbb use the fixed working points
    of config CMSSW_WPS."""

    pseudo = "cmssw"
