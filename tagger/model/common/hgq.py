from functools import wraps
import numpy as np

from keras import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from hgq.config import LayerConfigScope, QuantizerConfigScope, QuantizerConfig
from hgq.regularizers import MonoL1
from hgq.constraints import MinMax




def quantization_decorator(build_func):
    """Decorator used to include additional
    saving functionality for child classes
    """

    @wraps(build_func)
    def wrapper(self, input_shape: tuple, output_shape: tuple):
        
        scope0 = QuantizerConfigScope(
            default_q_type='kbi',
            b0=7,
            overflow_mode='wrap',
            i0=0,
            fr=MonoL1(1.e-8),
            ir=MonoL1(1.e-8),
        )

        scope1 = QuantizerConfigScope(
            default_q_type='kif',
            place='datalane',
            overflow_mode='wrap',
            f0=7,
            fr=MonoL1(1.e-8),
            ic=MinMax(0, 12),
        )

        scope2 = LayerConfigScope(
            enable_ebops=True, 
            heterogeneous_axis=None, 
            beta0=self.training_config['beta']
        )
        
        with scope0, scope1, scope2:
        
            # self.iq_conf = QuantizerConfig(k0=1, i0=11, f0=12, trainable=False,round_mode='RND',overflow_mode='SAT')
            # self.oq_conf_jet_id = QuantizerConfig(k0=0, i0=12, f0=12, trainable=False,round_mode='RND',overflow_mode='SAT')
            # self.oq_conf_pt = QuantizerConfig(k0=1, i0=9, f0=6, trainable=False,round_mode='RND',overflow_mode='SAT')

            self.iq_conf = None
            self.oq_conf_jet_id = None
            self.oq_conf_pt = None

            with (
                QuantizerConfigScope(place=('weight', 'bias'), overflow_mode='SAT_SYM'),
                QuantizerConfigScope(place='datalane', heterogeneous_axis=None)
            ):

                return build_func(self, input_shape, output_shape)

    return wrapper


def get_ebops(model: Model, print_layers: bool = False,) -> float:
    """Calculate total EBOPs recursively."""
    ebops = 0.0
    for layer in model._flatten_layers(include_self=False, recursive=True,):
        if hasattr(layer, "ebops"):
            ebops_layer = float(layer.ebops)
            if print_layers:
                print(f"Layer {layer.name}: {ebops_layer} EBOPs")
            ebops += ebops_layer

    return ebops


def log_beta_schedule(epoch, max_epochs=100):
    log_beta_start = np.log10(1e-7)
    log_beta_end = np.log10(1e-4)
    log_beta = log_beta_start + (log_beta_end - log_beta_start) * (epoch / max_epochs)
    return 10 ** log_beta

class EarlyStoppingWithEbopsThres(EarlyStopping):
    """Early stopping enabled only after reaching an EBOPs threshold.

    If ``start_from_epoch`` is None, monitoring begins on the first epoch
    for which EBOPs is at or below ``ebops_threshold``.

    If ``start_from_epoch`` is an integer, monitoring begins only once both:
      - the requested epoch has been reached, and
      - EBOPs is at or below ``ebops_threshold``.
    """

    def __init__(
        self,
        ebops_threshold: float,
        monitor: str = "val_loss",
        min_delta: float = 0,
        patience: int = 0,
        verbose: int = 0,
        mode: str = "auto",
        baseline: float | None = None,
        restore_best_weights: bool = False,
        start_from_epoch: int | None = None,
        model_attr: str | None = None,
    ):
        self.ebops_threshold = ebops_threshold
        self.apply_early_stop = False
        self.model_attr = model_attr

        # Keras requires an integer here.
        # If None, we handle the start condition ourselves.
        self.keras_start_epoch = (
            0 if start_from_epoch is None else start_from_epoch
        )

        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            baseline=baseline,
            restore_best_weights=restore_best_weights,
            start_from_epoch=self.keras_start_epoch,
        )

    def on_epoch_end(self, epoch, logs=None):
        assert isinstance(self.model, Model)

        inner_model = getattr(self.model, self.model_attr) if self.model_attr else self.model

        ebops = get_ebops(inner_model, print_layers=False)

        # Do not update EarlyStopping's internal state until the
        # requested EBOP region has been reached.
        if ebops <= self.ebops_threshold and not self.apply_early_stop:
            self.apply_early_stop = True
            print(f"EBOPs threshold reached: {ebops} < {self.ebops_threshold}. Early stopping monitoring enabled.")

        if not self.apply_early_stop:
            # Not yet in the requested EBOP region, so do not update EarlyStopping's internal state.
            return

        # Once here:
        # - if start_from_epoch=None, Keras start_from_epoch is 0, so
        #   monitoring begins immediately.
        # - if start_from_epoch was specified, the parent callback will
        #   enforce that epoch as usual.
        super().on_epoch_end(epoch, logs)

class ReduceLROnPlateauWithEbopsThres(ReduceLROnPlateau):
    """Reduce learning rate on plateau only after an EBOPs threshold is met.

    This callback reduces the learning rate when:

    - EBOPs is at or below ``ebops_threshold``, and
    - the monitored metric has stopped improving.

    Until the EBOPs threshold is reached, the internal state of
    ``ReduceLROnPlateau`` is not updated. Therefore, the patience counter
    effectively starts once the model reaches the requested EBOPs region.

    Parameters
    ----------
    ebops_threshold : float
        The EBOPs threshold. Learning-rate reduction is disabled while
        the model's EBOPs is above this value.
    monitor : str, default "val_loss"
        Quantity to be monitored.
    factor : float, default 0.1
        Factor by which the learning rate will be reduced:
        ``new_lr = lr * factor``.
    patience : int, default 10
        Number of epochs with no improvement after which the learning
        rate will be reduced.
    verbose : int, default 0
        Verbosity mode. 0 is silent and 1 prints messages when the
        callback changes the learning rate.
    mode : {"auto", "min", "max"}, default "auto"
        In ``"min"`` mode, the learning rate is reduced when the monitored
        quantity has stopped decreasing. In ``"max"`` mode, it is reduced
        when the quantity has stopped increasing. In ``"auto"`` mode,
        the direction is inferred from the monitored quantity.
    min_delta : float, default 1e-4
        Minimum change in the monitored quantity to qualify as an
        improvement.
    cooldown : int, default 0
        Number of epochs to wait after reducing the learning rate before
        resuming normal monitoring.
    min_lr : float, default 0.0
        Lower bound on the learning rate.
    """

    def __init__(
        self,
        ebops_threshold: float,
        monitor: str = "val_loss",
        factor: float = 0.1,
        patience: int = 10,
        verbose: int = 0,
        mode: str = "auto",
        min_delta: float = 1e-4,
        cooldown: int = 0,
        min_lr: float = 0.0,
        model_attr: str | None = None,
    ):
        self.ebops_threshold = ebops_threshold
        self.apply_lr_reduction = False
        self.model_attr = model_attr

        super().__init__(
            monitor=monitor,
            factor=factor,
            patience=patience,
            verbose=verbose,
            mode=mode,
            min_delta=min_delta,
            cooldown=cooldown,
            min_lr=min_lr,
        )

    def on_epoch_end(self, epoch, logs=None):
        assert isinstance(self.model, Model)

        inner_model = getattr(self.model, self.model_attr) if self.model_attr else self.model

        ebops = get_ebops(inner_model, print_layers=False)

        if ebops <= self.ebops_threshold and not self.apply_lr_reduction:
            # Once the threshold has been passed once, the learning rate reduction is enabled for the rest of training.
            self.apply_lr_reduction = True
            print(f"EBOPs threshold reached: {ebops} < {self.ebops_threshold}. Learning rate reduction enabled.")

        if not self.apply_lr_reduction:
            return

        # Once the EBOPs threshold is reached, allow the parent callback to
        # update its internal state and potentially reduce the learning rate.
        super().on_epoch_end(epoch, logs)