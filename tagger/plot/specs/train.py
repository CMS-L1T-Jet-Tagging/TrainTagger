"""Plots of the train results. File names are those of the removed
JetTagModel.plot_loss.

Each plot is a function that returns its PlotSpec; the collection at the
end lists the plots.
"""

from tagger.plot.registry import PlotSpec, Series, plot_collection


def loss_history(loss):
    """training/loss_<loss>_history: training and validation loss per
    epoch of one model output."""
    return PlotSpec(
        name=f"training/loss_{loss}_history",
        plotter="loss_history",
        series=[
            Series(f"lossHistory({loss})"),
        ],
        options=dict(
            xlabel="Epoch",
            ylabel="Loss " + loss,
            legend=dict(loc="upper right"),
        ),
    )


# -- Collection --------------------------------------------------------
@plot_collection(reads=["train"])
def train(config, info):
    """Training and validation loss per epoch, one plot per model output."""
    return [loss_history(loss) for loss in info.losses]
