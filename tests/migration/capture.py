"""Record the data matplotlib draws, keyed by the saved file. Temporary,
for the old-vs-new comparison.

Patches Axes.plot / errorbar / hist / imshow and Figure.savefig. Each
figure collects its drawn arrays; on savefig (png only) they are stored
under the file path relative to the plot folder, without extension.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.axes  # noqa: E402
import matplotlib.figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CAPTURED = {}
_root = [None]


def _arr(a):
    try:
        return np.asarray(a, dtype=float)
    except (TypeError, ValueError):
        return np.asarray([np.nan])


def _record(ax, kind, arrays, label=None):
    fig = ax.figure
    if not hasattr(fig, "_captured"):
        fig._captured = []
    fig._captured.append((kind, [_arr(a) for a in arrays], label))


_orig = {}


def install(root):
    _root[0] = os.path.abspath(root)
    if _orig:
        return  # already patched, only the root changes
    A, F = matplotlib.axes.Axes, matplotlib.figure.Figure
    _orig.update(
        plot=A.plot,
        errorbar=A.errorbar,
        hist=A.hist,
        imshow=A.imshow,
        barh=A.barh,
        savefig=F.savefig,
        scatter=A.scatter,
        fill_between=A.fill_between,
        pcolormesh=A.pcolormesh,
        bar=A.bar,
        stairs=A.stairs,
    )

    def stairs(self, values, edges=None, *args, **kw):
        _record(
            self,
            "stairs",
            [values, edges if edges is not None else []],
            kw.get("label"),
        )
        return _orig["stairs"](self, values, edges, *args, **kw)

    def scatter(self, x, y, *args, c=None, **kw):
        _record(
            self,
            "scatter",
            [x, y, c if c is not None else []],
            kw.get("label"),
        )
        return _orig["scatter"](self, x, y, *args, c=c, **kw)

    def fill_between(self, x, y1, y2=0, *args, **kw):
        lo, hi = np.fmin(_arr(y1), _arr(y2)), np.fmax(
            _arr(y1), _arr(y2)
        )  # order of y1, y2 does not matter
        _record(self, "band", [x, lo, hi], kw.get("label"))
        return _orig["fill_between"](self, x, y1, y2, *args, **kw)

    def pcolormesh(self, *args, **kw):
        _record(
            self,
            "mesh",
            [a for a in args if not isinstance(a, str)],
            kw.get("label"),
        )
        return _orig["pcolormesh"](self, *args, **kw)

    def bar(self, x, height, *args, **kw):
        _record(self, "bar", [x, height], kw.get("label"))
        return _orig["bar"](self, x, height, *args, **kw)

    def plot(self, *args, **kw):
        xy = [a for a in args if not isinstance(a, str)]
        _record(self, "line", xy, kw.get("label"))
        return _orig["plot"](self, *args, **kw)

    def errorbar(self, x, y, yerr=None, *args, **kw):
        if kw.get("label") is not None:
            _record(
                self,
                "errorbar",
                [x, y, yerr if yerr is not None else []],
                kw.get("label"),
            )
        A.plot, saved = _orig["plot"], A.plot  # errorbar calls plot internally
        try:
            return _orig["errorbar"](self, x, y, yerr, *args, **kw)
        finally:
            A.plot = saved

    def hist(self, *args, **kw):
        n, bins, patches = _orig["hist"](self, *args, **kw)
        ns = n if isinstance(n, list) or (np.ndim(n) == 2) else [n]
        labels = kw.get("label")
        labels = labels if isinstance(labels, list) else [labels] * len(ns)
        for h, lab in zip(ns, labels):
            _record(self, "hist", [h, bins], lab)
        return n, bins, patches

    def imshow(self, X, *args, **kw):
        _record(self, "image", [X])
        return _orig["imshow"](self, X, *args, **kw)

    def barh(self, y, width, *args, **kw):
        left = kw.get("left", 0)
        _record(
            self,
            "barh",
            [y, width, np.broadcast_to(left, np.shape(width)).copy()],
            kw.get("label"),
        )
        return _orig["barh"](self, y, width, *args, **kw)

    def savefig(self, fname, *args, **kw):
        path = os.path.abspath(str(fname))
        if path.endswith(".png"):
            key = os.path.relpath(path[:-4], _root[0])
            CAPTURED[key] = list(getattr(self, "_captured", []))
        return _orig["savefig"](self, fname, *args, **kw)

    A.plot, A.errorbar, A.hist, A.imshow, A.barh, F.savefig = (
        plot,
        errorbar,
        hist,
        imshow,
        barh,
        savefig,
    )
    A.scatter, A.fill_between, A.pcolormesh, A.bar, A.stairs = (
        scatter,
        fill_between,
        pcolormesh,
        bar,
        stairs,
    )

    def pyplot_savefig(*args, **kw):
        return plt.gcf().savefig(*args, **kw)

    plt.savefig = pyplot_savefig


def compare(old, new, rtol=1e-6, atol=1e-9):
    """Per file: 'same', or a reason. Only arrays are compared (not
    labels, colours, axes)."""
    report = {}
    for key in sorted(set(old) | set(new)):
        if key not in new:
            report[key] = "only old"
            continue
        if key not in old:
            report[key] = "only new"
            continue
        a, b = _canon(old[key]), _canon(new[key])
        if len(a) != len(b):
            report[key] = f"{len(a)} drawn items old, {len(b)} new"
            continue
        bad = []
        for i, (x, y) in enumerate(zip(a, b)):
            if len(x) != len(y) or not all(
                _close(u, v, rtol, atol) for u, v in zip(x, y)
            ):
                bad.append(i)
        report[key] = "same" if not bad else f"differs in items {bad}"
    return report


def _canon(items):
    """Arrays of each item, sorted so that drawing order does not matter."""
    out = []
    for kind, arrays, _ in items:
        arrays = [np.ravel(a) for a in arrays if np.size(a)]
        if kind == "errorbar" and arrays:
            # old pandas efficiency plots carry the under/overflow and
            # NaN bins with x = -inf, inf, nan
            keep = np.isfinite(arrays[0])
            arrays = [a[keep] if a.shape == keep.shape else a for a in arrays]
        out.append(arrays)
    return sorted(
        out,
        key=lambda arrs: tuple(round(float(np.nansum(a)), 6) for a in arrs),
    )


def _close(u, v, rtol, atol):
    return u.shape == v.shape and np.allclose(
        u, v, rtol=rtol, atol=atol, equal_nan=True
    )
