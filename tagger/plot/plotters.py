"""Generic plotters. Each takes a PlotSpec and a list of models (label +
result lookup) and returns a figure.

With several models the plot becomes an overlay: a single-series plot
gets one colour per model, a multi-series plot one line style per model,
and every legend entry is prefixed with the model label.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402
from scipy.interpolate import make_interp_spline  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402

from tagger.plot import style  # noqa: E402
from tagger.plot.registry import (
    format_label,
    needs_missing_reference,
)  # noqa: E402

PLOTTERS = {}


def close_all():
    """Close every open figure, e.g. after a plotter stopped half way."""
    plt.close("all")


def plotter(func):
    PLOTTERS[func.__name__] = func
    return func


def _figure(opts, ncols=1):
    width = opts.get("figwidth", style.FIGURE_SIZE[0])
    fig, axes = plt.subplots(
        1, ncols, figsize=(width * ncols, style.FIGURE_SIZE[1]), squeeze=False
    )
    if opts.get("cms", True):
        for ax in axes[0]:
            _cms(ax, opts)
    return fig, axes[0]


def _cms(ax, opts):
    hep.cms.label(
        llabel=style.CMSHEADER_LEFT,
        rlabel=style.CMSHEADER_RIGHT,
        ax=ax,
        fontsize=opts.get("cms_fontsize", style.CMSHEADER_SIZE),
    )


def _label(template, model, models):
    text = format_label(template, model.scalar)
    return (
        f"{model.label}: {text}"
        if len(models) > 1 and text
        else (model.label if len(models) > 1 else text)
    )


def _overlay_style(base, i_model, i_series, n_models, n_series):
    s = dict(base)
    if n_models > 1:
        if n_series == 1:
            s["color"] = style.color_cycle[i_model % len(style.color_cycle)]
        else:
            s.setdefault(
                "color", style.color_cycle[i_series % len(style.color_cycle)]
            )  # same series, same colour
            s["linestyle"] = style.LINESTYLES[i_model % len(style.LINESTYLES)]
    return s


def _finish(ax, opts, sort_keys=None):
    grid = opts.get("grid", True)
    if grid:
        ax.grid(True, **(grid if isinstance(grid, dict) else {}))
    if "xlabel" in opts:
        ax.set_xlabel(opts["xlabel"], **opts.get("xlabel_kw", {}))
    if "ylabel" in opts:
        ax.set_ylabel(opts["ylabel"], **opts.get("ylabel_kw", {}))
    if opts.get("logy"):
        ax.set_yscale("log")
    if opts.get("logx"):
        ax.set_xscale("log")
    if "xlim" in opts:
        ax.set_xlim(*opts["xlim"])
    if "ylim" in opts:
        ax.set_ylim(*opts["ylim"])
    legend = opts.get("legend", {})
    if legend is not None:
        handles, labels = ax.get_legend_handles_labels()
        if sort_keys is not None and len(sort_keys) == len(handles):
            order = np.argsort(sort_keys, kind="stable")
            handles, labels = [handles[i] for i in order], [
                labels[i] for i in order
            ]
        if handles:
            leg = ax.legend(handles, labels, **legend)
            if legend.get("title"):
                leg._legend_box.align = "left"


@plotter
def curve_overlay(spec, models):
    """Lines (or error bars when the series has ``yerr``) of curve
    fields ``x`` vs ``y``.

    Options: xlabel, ylabel, logx, logy, xlim, ylim, grid (True, False
    or kwargs of ax.grid), legend (kwargs of ax.legend, None for none),
    legend_sort (one scalar expression per series: legend ordered by its
    value), cms.
    """
    opts = spec.options
    fig, (ax,) = _figure(opts)
    sort_keys = [] if "legend_sort" in opts else None
    for im, model in enumerate(models):
        for i, s in enumerate(spec.series):
            c = model.get(s.expr)
            kw = _overlay_style(s.style, im, i, len(models), len(spec.series))
            label = _label(s.label, model, models)
            if s.yerr:
                ax.errorbar(c[s.x], c[s.y], yerr=c[s.yerr], label=label, **kw)
            else:
                ax.plot(c[s.x], c[s.y], label=label, **kw)
            if sort_keys is not None:
                sort_keys.append(model.scalar(opts["legend_sort"][i]))
    _finish(ax, opts, sort_keys)
    return fig


@plotter
def histogram(spec, models):
    """Step histograms from curve fields ``edges`` and ``y`` (default
    ``counts``).

    Options as curve_overlay, plus density (default True: each histogram
    normalised to unit area); xlim "edges" sets the x range to the bin
    edges of the first series.
    """
    opts = dict(spec.options)
    fig, (ax,) = _figure(opts)
    if opts.get("xlim") == "edges":
        edges = np.asarray(
            models[0].get(spec.series[0].expr)["edges"], dtype=float
        )
        opts["xlim"] = (edges[0], edges[-1])
    for im, model in enumerate(models):
        for i, s in enumerate(spec.series):
            c = model.get(s.expr)
            edges = np.asarray(c["edges"], dtype=float)
            heights = np.asarray(
                c[s.y if s.y != "y" else "counts"], dtype=float
            )
            if opts.get("density", True):
                total = heights.sum()
                heights = (
                    heights / (total * np.diff(edges))
                    if total > 0
                    else heights
                )
            kw = _overlay_style(
                {"histtype": "step", **s.style},
                im,
                i,
                len(models),
                len(spec.series),
            )
            ax.hist(
                edges[:-1],
                bins=edges,
                weights=heights,
                label=_label(s.label, model, models),
                **kw,
            )
    _finish(ax, opts)
    return fig


@plotter
def grid_2d(spec, models):
    """A matrix (curve field ``y``, default ``matrix``) with its cells
    annotated, one panel per model.

    The axis tick labels come from the curve field ``x`` (default
    ``classes``), mapped through style.CLASS_LABEL_STYLE. Options:
    xlabel, ylabel, clim, round (decimals of the annotations).
    """
    opts = spec.options
    (s,) = spec.series
    fig, axes = _figure(opts, ncols=len(models))
    for ax, model in zip(axes, models):
        c = model.get(s.expr)
        matrix = np.round(
            c[s.y if s.y != "y" else "matrix"],
            opts.get("round", style.MATRIX_DECIMALS),
        )
        names = c[s.x if s.x != "x" else "classes"]
        display = ConfusionMatrixDisplay(
            matrix,
            display_labels=[style.CLASS_LABEL_STYLE.get(n, n) for n in names],
        )
        display.plot(ax=ax)
        display.im_.set_clim(*opts.get("clim", (0, 1)))
        display.im_.colorbar.remove()
        cax = make_axes_locatable(ax).append_axes(
            "right", size="5%", pad=style.COLORBAR_PAD
        )
        plt.colorbar(display.im_, cax=cax)
        if len(models) > 1:
            ax.set_title(model.label)
    return fig


@plotter
def stacked_bars(spec, models):
    """Horizontal stacked bars of curve field ``y`` (default ``values``,
    outputs x categories), one panel per model. Category names from
    field ``x`` (default ``features``, through
    style.INPUT_FEATURE_STYLE), stack names from field ``outputs``
    (through style.CLASS_LABEL_STYLE). Categories are sorted by their
    total. Options: xlabel, fontsize, legend."""
    opts = spec.options
    (s,) = spec.series
    fig, axes = _figure(opts, ncols=len(models))
    fontsize = opts.get("fontsize", style.STACKED_BARS_FONTSIZE)
    for ax, model in zip(axes, models):
        c = model.get(s.expr)
        values = np.asarray(c[s.y if s.y != "y" else "values"])
        names = c[s.x if s.x != "x" else "features"]
        order = np.argsort(values.sum(axis=0))
        pos, left = np.arange(len(order)), np.zeros(len(order))
        for i, out in enumerate(c["outputs"]):
            ax.barh(
                pos,
                values[i][order],
                style.STACKED_BARS_HEIGHT,
                left=left,
                align="center",
                label=style.CLASS_LABEL_STYLE.get(out, out),
                color=style.SHAP_COLORS[i % len(style.SHAP_COLORS)],
            )
            left += values[i][order]
        ax.legend(
            **opts.get("legend", dict(loc="lower right", fontsize=fontsize))
        )
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")
        for side in ("right", "top", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(color="#333333", labelcolor="#333333")
        ax.set_yticks(
            pos,
            [style.INPUT_FEATURE_STYLE.get(names[i], names[i]) for i in order],
            fontsize=fontsize,
        )
        ax.set_xlabel(opts.get("xlabel", ""), fontsize=fontsize)
        if len(models) > 1:
            ax.set_title(model.label)
    fig.tight_layout()
    return fig


@plotter
def loss_history(spec, models):
    """Training and validation loss per epoch from a lossHistory curve
    (fields ``train``, ``val``). Options as curve_overlay."""
    opts = spec.options
    fig, (ax,) = _figure(opts)
    (s,) = spec.series
    for im, model in enumerate(models):
        c = model.get(s.expr)
        for i, (field, label) in enumerate(
            (("train", "Train Loss"), ("val", "Validation Loss"))
        ):
            kw = _overlay_style(
                {"linewidth": style.LINEWIDTH, **s.style},
                im,
                i,
                len(models),
                2,
            )
            ax.plot(c[field], label=_label(label, model, models), **kw)
    _finish(ax, opts)
    return fig


@plotter
def scalar_vs_x(spec, models):
    """Scalars against a number in the expression: the series ``expr``
    contains ``{x}``, which is replaced by each value of options
    ``x_values``. Option swap_axes draws the values on the y axis and
    the scalars on the x axis; other options as curve_overlay.
    """
    opts = spec.options
    fig, (ax,) = _figure(opts)
    xs = opts["x_values"]
    for im, model in enumerate(models):
        for i, s in enumerate(spec.series):
            ys = [
                model.scalar(s.expr.replace("{x}", repr(float(x)))) for x in xs
            ]
            kw = _overlay_style(
                {"marker": "o", **s.style},
                im,
                i,
                len(models),
                len(spec.series),
            )
            xy = (ys, xs) if opts.get("swap_axes") else (xs, ys)
            ax.plot(*xy, label=_label(s.label, model, models), **kw)
    _finish(ax, opts)
    return fig


# -- trigger seed plots ------------------------------------------------


def _header(ax, opts):
    """A text-only legend entry (option ``header``), e.g. the eta range,
    as in the old seed plots."""
    if opts.get("header"):
        ax.plot([], [], "none", linestyle="none", label=opts["header"])


def _style_kw(s, im, i, models, n_series):
    base = dict(s.style)
    if "c" in base:  # specs may give c=; the overlay logic sets color
        base["color"] = base.pop("c")
    kw = _overlay_style(base, im, i, len(models), n_series)
    if len(models) > 1 and n_series > 1:
        kw.pop("linestyle", None)
        kw["alpha"] = 1.0 if im == 0 else style.OVERLAY_ALPHA
    return kw


@plotter
def rate_grid(spec, models):
    """Rate of a seed on a threshold grid as coloured squares, with the
    working point line on top.

    The first series is the grid (curve fields ``x``, ``y`` and ``rate``
    or option ``z``, drawn for the first model only), every further
    series a line (fields ``x``, ``y``) per model. Options: zlabel
    (colour bar label), xlabel, ylabel, xlim, ylim, legend,
    cms_fontsize.
    """
    opts = spec.options
    fig, (ax,) = _figure(opts)
    grid, lines = spec.series[0], spec.series[1:]
    g = models[0].get(grid.expr)
    im = ax.scatter(
        g[grid.x],
        g[grid.y],
        c=g[opts.get("z", "rate")],
        s=style.RATE_GRID_MARKERSIZE,
        marker="s",
        cmap="Spectral_r",
        linewidths=0,
        norm=matplotlib.colors.LogNorm(),
    )
    plt.colorbar(im, ax=ax).set_label(opts.get("zlabel", "Rate [kHz]"))
    if "xlabel" in opts:
        ax.set_ylabel(opts["ylabel"])
        ax.set_xlabel(opts["xlabel"])
    ax.set_xlim(opts["xlim"])
    ax.set_ylim(opts["ylim"])
    for i_model, model in enumerate(models):
        for s in lines:
            c = model.get(s.expr)
            kw = {
                "linewidth": style.LINEWIDTH,
                "color": "firebrick",
                **s.style,
            }
            if len(models) > 1:
                kw["color"] = style.color_cycle[
                    i_model % len(style.color_cycle)
                ]
            ax.plot(c[s.x], c[s.y], label=_label(s.label, model, models), **kw)
    ax.legend(**opts.get("legend", dict(loc="upper right")))
    return fig


@plotter
def eff_points(spec, models):
    """Binned efficiencies (curve fields ``edges``, ``eff``,
    ``err_low``, ``err_up``) as points with errors.

    Points at the bin centres, errors asymmetric Clopper-Pearson. Empty
    bins (no denominator) are not drawn. Options: header, zero_nan_err
    (NaN errors drawn as 0), hline (y of a dashed line), hlines ((y,
    xmin, xmax) of a dashed line), background (dict expr, label: a
    normalised histogram, fields edges and counts, in grey), text ((x,
    y, label template) in a box), errorbar_kw, xlabel, ylabel, xlim,
    ylim, legend, grid (default False), cms_fontsize.
    """
    opts = spec.options
    fig, (ax,) = _figure(opts)
    _header(ax, opts)
    if (
        "background" in opts
    ):  # normalised distribution in grey behind the points
        bkg = models[0].get(opts["background"]["expr"])
        hep.histplot(
            (bkg["counts"], bkg["edges"]),
            ax=ax,
            histtype="step",
            color="grey",
            label=opts["background"]["label"],
        )
    for im, model in enumerate(models):
        for i, s in enumerate(spec.series):
            c = model.get(s.expr)
            edges = np.asarray(c["edges"], dtype=float)
            x, y = (edges[1:] + edges[:-1]) / 2, c["eff"]
            yerr = np.vstack([c["err_low"], c["err_up"]])
            if opts.get("zero_nan_err"):
                yerr = np.nan_to_num(yerr, nan=0.0)
            kw = {
                "fmt": "o",
                **opts.get(
                    "errorbar_kw",
                    {
                        "markersize": style.LINEWIDTH,
                        "linewidth": style.ERRORBAR_LINEWIDTH,
                    },
                ),
                **_style_kw(s, im, i, models, len(spec.series)),
            }
            ax.errorbar(
                x, y, yerr=yerr, label=_label(s.label, model, models), **kw
            )
    if "hline" in opts:
        ax.axhline(
            opts["hline"],
            linestyle="dashed",
            color="black",
            linewidth=style.REFERENCE_LINEWIDTH,
        )
    if "hlines" in opts:
        ax.hlines(
            *opts["hlines"],
            linestyles="dashed",
            color="black",
            linewidth=style.HH_REFERENCE_LINEWIDTH,
        )
    if "text" in opts:
        x, y, template = opts["text"]
        ax.text(
            x,
            y,
            format_label(template, models[0].scalar),
            fontsize=style.SMALL_SIZE,
            bbox=dict(facecolor="white", alpha=style.TEXT_BOX_ALPHA),
        )
    _finish(ax, {"grid": False, **opts})
    return fig


@plotter
def band_overlay(spec, models):
    """Lines of curve fields ``x`` vs ``y`` with a band y +- ``yerr``
    (series field, default ``err``).

    Options: header, xlabel, ylabel, logy, xlim, ylim, legend, grid
    (default False), cms_fontsize.
    """
    opts = spec.options
    fig, (ax,) = _figure(opts)
    _header(ax, opts)
    for im, model in enumerate(models):
        for i, s in enumerate(spec.series):
            c = model.get(s.expr)
            x, y, err = (
                np.asarray(c[s.x]),
                np.asarray(c[s.y]),
                np.asarray(c[s.yerr or "err"]),
            )
            kw = {
                "linewidth": style.LINEWIDTH,
                **_style_kw(s, im, i, models, len(spec.series)),
            }
            ax.plot(x, y, label=_label(s.label, model, models), **kw)
            ax.fill_between(
                x,
                y - err,
                y + err,
                color=kw.get("color"),
                alpha=style.BAND_ALPHA,
            )
    _finish(ax, {"grid": False, **opts})
    return fig


@plotter
def mesh_panels(spec, models):
    """One pcolormesh panel per series (curve fields ``edges`` and
    ``y``), first model only, with a common colour bar. Options: titles
    (one per series), clim, xlabel, ylabel, figwidth_factor,
    cms_fontsize.
    """
    opts = spec.options
    model = models[0]
    n = len(spec.series)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(
            opts.get("figwidth_factor", style.MESH_PANELS_WIDTH_FACTOR)
            * style.FIGURE_SIZE[0],
            style.FIGURE_SIZE[1],
        ),
    )
    axes = np.atleast_1d(axes)
    for ax, s, title in zip(axes, spec.series, opts.get("titles", [""] * n)):
        c = model.get(s.expr)
        edges = c["edges"]
        im = ax.pcolormesh(
            edges,
            edges,
            np.asarray(c[s.y]).T,
            vmin=opts.get("clim", (0, 1))[0],
            vmax=opts.get("clim", (0, 1))[1],
        )
        ax.set_xlabel(opts.get("xlabel", ""))
        ax.set_ylabel(opts.get("ylabel", ""))
        ax.set_title(
            format_label(title, model.scalar), pad=style.MESH_PANELS_TITLE_PAD
        )
        _cms(ax, opts)
    fig.colorbar(im, ax=axes.ravel().tolist())
    return fig


@plotter
def ratio_map(spec, models):
    """The ratio of the ``num`` fields of two series (curves with
    ``edges``, ``num``, ``genpt1_frac``, ``genpt2_frac``; NaN where
    undefined) on a diverging colour scale around 1, with the marginal
    distributions of the first series on top and to the right. First
    model only. Options: title (vertical text on the left), xlabel,
    ylabel, vmax, cms_fontsize.
    """
    opts = spec.options
    s_num, s_den = spec.series
    c = models[0].get(s_num.expr)
    num, den = np.asarray(c["num"], dtype=float), np.asarray(
        models[0].get(s_den.expr)["num"], dtype=float
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(den != 0, num / np.where(den != 0, den, 1), np.nan)
    edges = np.asarray(c["edges"])
    width, height = style.RATIO_MAP_SIZE_FACTORS
    fig = plt.figure(
        figsize=(style.FIGURE_SIZE[0] * width, style.FIGURE_SIZE[1] * height)
    )
    grid = style.RATIO_MAP_GRID
    gs = GridSpec(
        len(grid["height_ratios"]), len(grid["width_ratios"]), **grid
    )
    ax_title = fig.add_subplot(gs[1, 0])
    ax_title.axis("off")
    ax_title.text(
        0,
        style.RATIO_MAP_TITLE_Y,
        opts.get("title", ""),
        fontsize=style.MEDIUM_SIZE,
        rotation=90,
    )
    ax = fig.add_subplot(gs[1, 3])
    norm = matplotlib.colors.TwoSlopeNorm(
        vmin=0.0, vcenter=1.0, vmax=opts.get("vmax", style.RATIO_MAP_VMAX)
    )
    im = ax.pcolormesh(edges, edges, ratio.T, norm=norm, cmap="coolwarm")
    ax.set_xlabel(opts.get("xlabel", ""))
    ax.set_ylabel(opts.get("ylabel", ""))
    ax.set_xticks(ax.get_xticks()[:-1])
    fmt = matplotlib.ticker.FuncFormatter(
        lambda v, _: (
            "{:.0f}".format(v) if float(v).is_integer() else "{:.2f}".format(v)
        )
    )
    ax_top = fig.add_subplot(gs[0, 3], sharex=ax)
    ax_top.bar(
        edges[:-1],
        c["genpt1_frac"],
        width=np.diff(edges),
        align="edge",
        color="gray",
        alpha=style.BAR_ALPHA,
    )
    ax_top.set_yticks(style.RATIO_MAP_FRACTION_TICKS)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.yaxis.set_major_formatter(fmt)
    _cms(ax_top, opts)
    ax_right = fig.add_subplot(gs[1, 4], sharey=ax)
    ax_right.barh(
        edges[:-1],
        c["genpt2_frac"],
        height=np.diff(edges),
        align="edge",
        color="gray",
        alpha=style.BAR_ALPHA,
    )
    ax_right.set_xticks(style.RATIO_MAP_FRACTION_TICKS)
    ax_right.xaxis.set_major_formatter(fmt)
    ax_right.tick_params(axis="y", labelleft=False)
    fig.colorbar(
        im,
        cax=fig.add_subplot(gs[1, 1]),
        aspect=style.RATIO_MAP_COLORBAR_ASPECT,
    )
    return fig


@plotter
def text_summary(spec, models):
    """A text file instead of a figure: options ``lines``, each a label
    template with [[expr:fmt]] references, for the first model. Lines
    that need reference results (cmssw/, hls4ml/, emulator/) the model
    does not have are left out."""
    lines = [
        line
        for line in spec.options["lines"]
        if not needs_missing_reference(line, models[0])
    ]
    return "".join(
        format_label(line, models[0].scalar) + "\n" for line in lines
    )


# -- comparisons of two model versions per jet (emulation) -------------
@plotter
def pairwise_2d(spec, models):
    """Per-jet 2D histogram of two series on the same jets (curve field
    ``y`` of each, e.g. one model version against another), 50 x 50
    bins, log colour scale, for the first model. Options: xrange,
    yrange, xlabel, ylabel, title."""
    opts = spec.options
    sx, sy = spec.series
    x = np.asarray(models[0].get(sx.expr)[sx.y], dtype=float)
    y = np.asarray(models[0].get(sy.expr)[sy.y], dtype=float)
    return plot_2d(
        x,
        y,
        opts["xrange"],
        opts["yrange"],
        opts.get("xlabel", ""),
        opts.get("ylabel", ""),
        opts.get("title", ""),
    )


@plotter
def agreement(spec, models):
    """A text file: per field in option ``fields``, the percentage of
    jets whose values in the two series are exactly equal (bit
    accuracy), for the first model. Option ``text``: the line template
    with {field} and {percent}."""
    sa, sb = spec.series
    a, b = models[0].get(sa.expr), models[0].get(sb.expr)
    lines = []
    for f in spec.options["fields"]:
        percent = 100 * np.mean(np.asarray(a[f]) == np.asarray(b[f]))
        lines.append(
            spec.options["text"].format(field=f, percent=percent) + "\n"
        )
    return "".join(lines)


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(fig, str):  # text_summary
        with open(f"{path}.txt", "w") as f:
            f.write(fig)
        return
    fig.savefig(f"{path}.pdf", bbox_inches="tight")
    fig.savefig(f"{path}.png", bbox_inches="tight")
    plt.close(fig)


# -- FastPUPPI jets (emulationRegression) ------------------------------
@plotter
def pt_map(spec, models):
    """A (gen pT x L1 pT) map (curve fields ``edges``, ``counts``) of
    the first model, log colour scale; with two series their ratio (0
    where the second is empty) on a diverging scale around 1. Options:
    xlabel, ylabel, zlabel. As in the old plot_basic.py, gen pT is drawn
    on the y axis and L1 pT on the x axis."""
    opts = spec.options
    maps = [models[0].get(s.expr) for s in spec.series]
    edges = np.asarray(maps[0]["edges"], dtype=float)
    fig, ax = plt.subplots(figsize=style.FIGURE_SIZE)
    hep.cms.label(
        llabel=style.CMSHEADER_LEFT,
        rlabel=style.CMSHEADER_RIGHT,
        ax=ax,
        fontsize=style.MEDIUM_SIZE,
    )
    if len(maps) == 2:
        num, den = (np.asarray(m["counts"], dtype=float) for m in maps)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(den != 0, num / den, 0)
        max_dev = max(abs(z.min() - 1), abs(z.max() - 1))
        norm, cmap = (
            TwoSlopeNorm(vmin=1 - max_dev, vcenter=1, vmax=1 + max_dev),
            plt.cm.coolwarm.copy(),
        )
    else:
        z = np.asarray(maps[0]["counts"], dtype=float)
        norm, cmap = LogNorm(), plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")
    im = ax.imshow(
        z,
        origin="lower",
        cmap=cmap,
        norm=norm,
        extent=[edges[0], edges[-1], edges[0], edges[-1]],
    )
    ax.set_aspect("equal")
    ax.set_xlabel(opts.get("xlabel", ""))
    ax.set_ylabel(opts.get("ylabel", ""))
    fig.colorbar(im, ax=ax, shrink=style.COLORBAR_SHRINK).set_label(
        opts.get("zlabel", ""), fontsize=style.LARGE_SIZE
    )
    return fig


@plotter
def mass_peak(spec, models):
    """Mass distributions (curve fields ``edges``, ``x``, ``fraction``,
    ``err``, ``mean``, ``sigma_eff``) of the first model: the first
    series as a step line, the others with error bars, each labelled
    with its mean and sigma_eff, and a vertical line at option ``mass``
    (label ``mass_label``). Options: xlabel, legend."""
    opts = spec.options
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    ax.axvline(
        opts["mass"],
        color="black",
        linestyle="--",
        linewidth=style.MASS_LINEWIDTH,
        label=opts.get("mass_label", ""),
    )
    for i, s in enumerate(spec.series):
        c = models[0].get(s.expr)
        label = (
            s.label
            + "\n"
            + f"($\\mu$ = {float(c['mean']):.1f}, $\\sigma_{{eff}}$ ="
            f" {float(c['sigma_eff']):.1f})"
        )
        if i == 0:
            ax.step(
                np.asarray(c["edges"])[:-1],
                c["fraction"],
                where="post",
                label=label,
                **{"linewidth": style.MASS_PEAK_LINEWIDTH, **s.style},
            )
        else:
            ax.errorbar(
                c["x"],
                c["fraction"],
                yerr=c["err"],
                label=label,
                **{
                    "fmt": "o-",
                    "markersize": style.MASS_PEAK_MARKERSIZE,
                    "linewidth": style.MASS_PEAK_LINEWIDTH,
                    "capsize": style.MASS_PEAK_CAPSIZE,
                    **s.style,
                },
            )
    ax.set_xlabel(opts.get("xlabel", ""))
    ax.set_ylabel("Fraction")
    ax.legend(**opts.get("legend", {}))
    _cms(ax, opts)
    return fig


@plotter
def turn_on(spec, models):
    """Turn-on curves (curve fields ``x``, ``xerr``, ``eff``, ``err``,
    ``xmax``) of the first model with error bars and a cubic spline
    through the points. Options: xlabel, ylabel, legend.
    """
    opts = spec.options
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    _cms(ax, opts)
    xmax = None
    for s in spec.series:
        c = models[0].get(s.expr)
        x, eff = np.asarray(c["x"], dtype=float), np.asarray(
            c["eff"], dtype=float
        )
        kw = {k: v for k, v in s.style.items() if k != "linestyle"}
        ax.errorbar(
            x,
            eff,
            xerr=c["xerr"],
            yerr=c["err"],
            fmt="o",
            capsize=style.EFF_POINTS_CAPSIZE,
            label=format_label(s.label, models[0].scalar),
            **kw,
        )
        x_smooth = np.linspace(x.min(), x.max(), style.TURN_ON_SPLINE_POINTS)
        ax.plot(
            x_smooth,
            make_interp_spline(x, eff, k=3)(x_smooth),
            color=kw.get("color"),
            linestyle=s.style.get("linestyle", "-"),
        )
        xmax = float(c["xmax"])
    ax.legend(**opts.get("legend", {}))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, style.TURN_ON_YMAX)
    ax.set_xlabel(opts.get("xlabel", ""), fontsize=style.LARGEST_SIZE)
    ax.set_ylabel(opts.get("ylabel", ""), fontsize=style.LARGEST_SIZE)
    return fig


@plotter
def hist_2d(spec, models):
    """A 2D histogram (curve fields ``xedges``, ``yedges``, ``counts``:
    x bins x y bins) of the first model as a colour map. Options:
    xlabel, ylabel, zlabel, cmap, cms_fontsize."""
    opts = spec.options
    (s,) = spec.series
    c = models[0].get(s.expr)
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    x, y = np.meshgrid(
        np.asarray(c["xedges"], dtype=float),
        np.asarray(c["yedges"], dtype=float),
    )
    im = ax.pcolormesh(
        x,
        y,
        np.asarray(c["counts"], dtype=float).T,
        cmap=opts.get("cmap", "viridis"),
        shading="auto",
    )
    _cms(ax, opts)
    ax.set_xlabel(opts.get("xlabel", ""))
    ax.set_ylabel(opts.get("ylabel", ""))
    plt.colorbar(im, ax=ax).set_label(opts.get("zlabel", ""))
    return fig


# -- Firmware profiling ------------------------------------------------
# figure helper of tagger/firmware/hls4ml_profile.py
def plot_2d(
    variable_one, variable_two, range_one, range_two, name_one, name_two, title
):
    """2D histogram of two arrays, 50 x 50 bins in the given ranges, log
    colour scale."""
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(
            style.FIGURE_SIZE[0] + style.PROFILE_EXTRA_WIDTH,
            style.FIGURE_SIZE[1],
        ),
    )
    hep.cms.label(
        llabel=style.CMSHEADER_LEFT,
        rlabel=style.CMSHEADER_RIGHT,
        ax=ax,
        fontsize=style.CMSHEADER_SIZE,
    )
    hist2d = ax.hist2d(
        variable_one,
        variable_two,
        range=(range_one, range_two),
        bins=style.PROFILE_HIST_BINS,
        norm=matplotlib.colors.LogNorm(),
        cmap="jet",
    )
    ax.set_xlabel(name_one)
    ax.set_ylabel(name_two)
    cbar = plt.colorbar(hist2d[3], ax=ax)
    cbar.set_label("a.u.")
    plt.suptitle(title)
    return fig
