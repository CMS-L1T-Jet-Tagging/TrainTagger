"""Config loading: the defaults < --config file < --set KEY=VALUE.

The defaults are the YAML files in config/defaults/: one per collection
(as in variables/), the inputs, and the keys shared by several
collections; every key is defined once. Override them with
``--config my.yaml`` (a YAML file that sets any of these keys) and then
with ``--set KEY=VALUE`` (VALUE in YAML, e.g.
``--set MISTAG_POINTS=[0.1, 0.01]``).
"""

from tagger.eval.config.loader import config_hash, load

__all__ = ["config_hash", "load"]
