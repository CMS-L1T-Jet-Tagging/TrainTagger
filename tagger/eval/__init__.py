"""Evaluation: compute named variables from a model and its samples, see
docs/variables.md.

    from tagger.eval import vm

The VariableManager is imported on first use, so ``tagger.eval.config``
stays light for scripts that only read the config.
"""


def __getattr__(name):
    if name == "vm":
        from tagger.eval.variable_manager import vm

        return vm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
