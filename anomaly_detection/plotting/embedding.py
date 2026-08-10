import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.covariance import EmpiricalCovariance
from sklearn.manifold import TSNE

def mean_embedding_norm(z):
    """Mean L2 norm of embeddings; larger values indicate points lie farther from the origin on average."""
    return np.linalg.norm(z, axis=1).mean()

def mean_cosine_similarity(z):
    """Average pairwise cosine similarity; high values suggest directional collapse, low values suggest angular diversity."""
    z = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-12, None)
    sim = z @ z.T
    n = len(z)
    return (sim.sum() - n) / (n*(n-1))

def covariance_trace(z):
    """Trace of embedding covariance; measures total variance/spread across latent dimensions."""
    z = z - z.mean(axis=0)
    cov = np.cov(z, rowvar=False)
    return np.trace(cov)

def effective_rank(z, eps=1e-12):
    """Participation-ratio rank of covariance; higher values mean variance is distributed across more latent directions."""
    z = z - np.mean(z, axis=0, keepdims=True)
    cov = np.cov(z, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, eps, None)

    return (eigvals.sum() ** 2) / np.sum(eigvals ** 2)

def uniformity(z, t=2):
    """Log expected exp(-t||zi-zj||^2); less negative values indicate tighter clustering, more negative indicate better global spread."""
    z = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-12, None)
    sq = pdist(z, metric="sqeuclidean")
    return np.log(np.mean(np.exp(-t * sq)))


def silhouette(z_bkg, y_bkg):
    """Silhouette coefficient from labels; higher values mean samples are closer to their own class than to others."""
    return silhouette_score(z_bkg, y_bkg)

def davies_bouldin(z_bkg, y_bkg):
    """Davies-Bouldin index from labels; lower values indicate compact, well-separated class clusters."""
    return davies_bouldin_score(z_bkg, y_bkg)

def calinski_harabasz(z_bkg, y_bkg):
    """Calinski-Harabasz score from labels; higher values indicate strong between-class separation vs within-class spread."""
    return calinski_harabasz_score(z_bkg, y_bkg)

def linear_probe_accuracy(z_bkg, y_bkg, cv=5):
    """Cross-validated logistic-regression accuracy; estimates how linearly decodable labels are in latent space."""
    clf = LogisticRegression(max_iter=5000)
    scores = cross_val_score(clf, z_bkg, y_bkg, cv=cv)
    return scores.mean()

def knn_accuracy(z_bkg, y_bkg, k=20, cv=5):
    """Cross-validated kNN accuracy; measures local neighborhood label consistency in latent space."""
    clf = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(clf, z_bkg, y_bkg, cv=cv)
    return scores.mean()

def neighbour_purity(z_bkg, y_bkg, k=10):
    """Fraction of same-label points among each sample's k nearest neighbors; higher values imply cleaner local class structure."""
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(z_bkg)
    indices = nn.kneighbors(z_bkg, return_distance=False)[:,1:]
    purity = np.mean([
        np.mean(y_bkg[indices[i]] == y_bkg[i])
        for i in range(len(z_bkg))
    ])
    return purity

def mahalanobis_auroc(z_bkg, z_sig):
    z_bkg = np.asarray(z_bkg)
    z_sig = np.asarray(z_sig)

    if z_bkg.ndim != 2 or z_sig.ndim != 2:
        raise ValueError("Expected 2D arrays for z_bkg and z_sig")
    if z_bkg.shape[0] == 0 or z_sig.shape[0] == 0:
        return float("nan")

    cov = EmpiricalCovariance().fit(z_bkg)
    bkg_scores = cov.mahalanobis(z_bkg)
    sig_scores = cov.mahalanobis(z_sig)
    scores = np.concatenate([bkg_scores, sig_scores])
    labels = np.concatenate([
        np.zeros(len(bkg_scores)),
        np.ones(len(sig_scores))
    ])

    return roc_auc_score(labels, scores)

def knn_auroc(z_bkg, z_sig, k=10):
    """AUROC from mean distance to background kNN; higher values indicate stronger background-vs-signal separation."""
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(z_bkg)

    d_bkg, _ = nn.kneighbors(z_bkg)
    d_bkg = d_bkg[:, 1:]  # Exclude self-distance
    d_sig, _ = nn.kneighbors(z_sig)
    score_bkg = d_bkg.mean(axis=1)
    score_sig = d_sig.mean(axis=1)
    scores = np.concatenate([score_bkg, score_sig])
    labels = np.concatenate([
        np.zeros(len(score_bkg)),
        np.ones(len(score_sig))
    ])

    return roc_auc_score(labels, scores)

def plot_eigenvalue_spectrum(z, output_file):
    """Plot sorted covariance eigenvalues on a log scale; visualizes variance distribution across latent dimensions."""
    z = z - np.mean(z, axis=0, keepdims=True)
    cov = np.cov(z, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals_sorted = np.sort(eigvals)[::-1]
    cumvar = np.cumsum(eigvals_sorted) / np.sum(eigvals_sorted)

    plt.figure(figsize=(5.5, 5))
    ax1 = plt.gca()
    ax1.bar(range(len(eigvals_sorted)), np.log(eigvals_sorted), alpha=0.7, width=0.8, label="Eigenvalues")
    ax1.set_xlabel("Eigenvalue Index (sorted)")
    ax1.set_ylabel("Log Eigenvalue")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(cumvar)), cumvar, color="red", marker="o", linewidth=2, label="Cumulative Variance")
    ax2.set_ylabel("Cumulative Variance Explained")
    ax2.set_ylim(0, 1.02)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()

def plot_pca_explained_variance(z_pca, output_file):
    """
    Plot the PCA explained variance ratio (scree plot) together with the
    cumulative explained variance.
    """

    explained = np.var(z_pca, axis=0) / np.sum(np.var(z_pca, axis=0))
    cumulative = np.cumsum(explained)
    components = np.arange(1, len(explained) + 1)

    fig, ax1 = plt.subplots(figsize=(5.5, 5))

    # Individual explained variance
    ax1.bar(components, explained, alpha=0.7, width=0.8, label="Explained variance")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_xlim(0.5, len(components) + 0.5)
    ax1.grid(True, axis="y", alpha=0.3)

    # Cumulative explained variance
    ax2 = ax1.twinx()
    ax2.plot(components, cumulative, color="red", marker="o", linewidth=2, label="Cumulative")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_ylim(0, 1.02)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()

def plot_pairwise_distance_hist(z, y, output_file):
    """Plot histogram of pairwise distances between embeddings; visualizes overall spread and clustering."""
    
    inter_class_dists = []
    for i, label_i in enumerate(np.unique(y)):
        for j, label_j in enumerate(np.unique(y)):
            if i < j:
                mask_i = y == label_i
                mask_j = y == label_j
                dists = cdist(z[mask_i], z[mask_j], metric='euclidean')
                inter_class_dists.append(dists.flatten())
    inter_class_dists = np.concatenate(inter_class_dists)

    intra_class_dists = []
    for label in np.unique(y):
        mask = y == label
        if np.sum(mask) > 1:
            intra_class_dists.append(pdist(z[mask], metric='euclidean'))
    if len(intra_class_dists) > 0:
        intra_class_dists = np.concatenate(intra_class_dists)

    plt.figure(figsize=(5.5, 5))
    plt.hist(inter_class_dists, bins=50, density=True, alpha=0.7, label='Inter-class')
    if len(intra_class_dists) > 0:
        plt.hist(intra_class_dists, bins=50, density=True, alpha=0.7, label='Intra-class')
    plt.legend()
    plt.xlabel("Euclidean Pairwise Distance")
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()

def plot_pca_embedding_2d(
    z_pca,
    output_file,
    class_info=None,
    class_array=None,
    variable_name=None,
    variable_array=None,
    percentile=99,
    log_variable=False,
):
    """Project embeddings to 2D using PCA and plot (optionally colored by class or variable)."""

    pct_min, pct_max = (100 - percentile) / 2, 100 - (100 - percentile) / 2
    xlim = np.percentile(z_pca[:, 0], [pct_min, pct_max])
    ylim = np.percentile(z_pca[:, 1], [pct_min, pct_max])

    fig, ax = plt.subplots(figsize=(5.5, 5))

    if class_array is not None:
        class_entries = _resolve_class_entries(class_array, class_info)
        _plot_class_scatter(ax, z_pca, i=1, j=0, class_array=class_array, class_entries=class_entries)
    elif variable_array is not None:
        _plot_variable_scatter(
            ax, z_pca, i=1, j=0, variable_array=variable_array,
            variable_name=variable_name, log_variable=log_variable,
        )
    else:
        ax.scatter(z_pca[:, 0], z_pca[:, 1], alpha=0.7)

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()

def plot_pca_embedding_2d_matrix(
    z_pca,
    output_file,
    n_components=3,
    class_info=None,
    class_array=None,
    variable_name=None,
    variable_array=None,
    percentile=99,
    log_variable=False,
):
    """Project embeddings to 2D using PCA and plot a matrix of scatter plots for the first n_components.

    Layout:
        - Diagonal: histogram of each PCA component. Stacked by class if `class_array`
          is given (colors/labels from `class_info` when available, else a tab10 fallback).
        - Upper triangle: scatter of component j vs i, colored by class if `class_array`
          is given.
        - Lower triangle: scatter of component j vs i, colored by `variable_array` if given
          (optionally log10-transformed via `log_variable`).

    `class_info` format (optional): a mapping of arbitrary keys to dicts like
        {"class": <value in class_array>, "label": "Display name", "color": "#rrggbb"}
    Any class present in `class_array` but missing from `class_info` gets a default
    label/color; any color unset in `class_info` also gets a default fallback color.
    """
    pct_min, pct_max = (100 - percentile) / 2, 100 - (100 - percentile) / 2
    component_limits = [np.percentile(z_pca[:, k], [pct_min, pct_max]) for k in range(n_components)]

    class_entries = _resolve_class_entries(class_array, class_info) if class_array is not None else None

    fig, axes = plt.subplots(
        n_components, n_components, figsize=(5.5 * n_components, 5 * n_components)
    )

    for i in range(n_components):
        ilim = component_limits[i]
        for j in range(n_components):
            jlim = component_limits[j]
            ax = axes[i, j]

            if i == j:
                _plot_diagonal_hist(ax, z_pca, i, ilim, class_array, class_entries)
                ax.set_xlabel(f"PCA Component {i + 1}")
                ax.set_ylabel("Count")
                ax.set_xlim(jlim)

            elif i < j:
                _plot_class_scatter(ax, z_pca, i, j, class_array, class_entries)
                ax.set_xlabel(f"PCA Component {j + 1}")
                ax.set_ylabel(f"PCA Component {i + 1}")
                ax.set_xlim(jlim)
                ax.set_ylim(ilim)

            else:  # i > j
                _plot_variable_scatter(
                    ax, z_pca, i, j, variable_array, variable_name, log_variable
                )
                ax.set_xlabel(f"PCA Component {j + 1}")
                ax.set_ylabel(f"PCA Component {i + 1}")
                ax.set_xlim(jlim)
                ax.set_ylim(ilim)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()

def plot_tsne_embedding_2d(
    z_tsne,
    output_file,
    class_info=None,
    class_array=None,
    variable_name=None,
    variable_array=None,
    percentile=100,
    log_variable=False,
):
    """Plot a precomputed 2D t-SNE embedding (optionally colored by class or variable).

    Expects `z_tsne` to already be the fitted 2D t-SNE output (see `fit_tsne_embedding`).
    Mirrors `plot_pca_embedding_2d` so the two can share the same class/variable coloring logic.
    """
    pct_min, pct_max = (100 - percentile) / 2, 100 - (100 - percentile) / 2
    xlim = np.percentile(z_tsne[:, 0], [pct_min, pct_max])
    ylim = np.percentile(z_tsne[:, 1], [pct_min, pct_max])

    fig, ax = plt.subplots(figsize=(5.5, 5))

    if class_array is not None:
        class_entries = _resolve_class_entries(class_array, class_info)
        _plot_class_scatter(ax, z_tsne, i=1, j=0, class_array=class_array, class_entries=class_entries)
    elif variable_array is not None:
        _plot_variable_scatter(
            ax, z_tsne, i=1, j=0, variable_array=variable_array,
            variable_name=variable_name, log_variable=log_variable,
        )
    else:
        ax.scatter(z_tsne[:, 0], z_tsne[:, 1], alpha=0.7)

    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def fit_tsne_embedding(z, **tsne_kwargs):
    """Fit t-SNE once on `z`, returning the 2D embedding to reuse across multiple plots.

    Fit this once and pass the result to plot_tsne_embedding_2d for both your
    class-colored and process-colored plots, so both share identical axes/geometry.
    """
    defaults = dict(n_components=2, init="pca", learning_rate="auto", random_state=0)
    defaults.update(tsne_kwargs)
    tsne = TSNE(**defaults)
    return tsne.fit_transform(z)


def _resolve_class_entries(class_array, class_info):
    """Return a list of (class_value, label, color) for every class present in class_array.

    Labels/colors come from `class_info` where available; anything missing (an
    unmapped class, or a mapped class with no color) is filled in from a tab10
    fallback, so the result always has a complete color/label for every class.
    """
    classes_present = np.unique(class_array)
    cmap = plt.get_cmap("tab10")

    labels_by_class, colors_by_class = {}, {}
    if class_info is not None:
        for info in class_info.values():
            cls = info.get("class")
            if cls in classes_present:
                labels_by_class[cls] = info.get("label", str(cls))
                colors_by_class[cls] = info.get("color")

    entries = []
    for idx, cls in enumerate(classes_present):
        label = labels_by_class.get(cls, str(cls))
        color = colors_by_class.get(cls) or cmap(idx % 10)
        entries.append((cls, label, color))
    return entries


def _plot_diagonal_hist(ax, z_pca, component_idx, xlim, class_array, class_entries):
    """Plot a histogram of one PCA component, stacked by class if class info is given."""
    if class_array is None:
        ax.hist(z_pca[:, component_idx], bins=50, range=xlim, color="gray", alpha=0.7)
        return

    data = [z_pca[class_array == cls, component_idx] for cls, _, _ in class_entries]
    colors = [color for _, _, color in class_entries]
    labels = [label for _, label, _ in class_entries]
    ax.hist(data, bins=50, range=xlim, stacked=True, alpha=0.85, color=colors, label=labels)
    ax.legend(title="Classes", fontsize=7)


def _plot_class_scatter(ax, z_pca, i, j, class_array, class_entries):
    """Scatter component j (x) vs component i (y), colored by class if given."""
    if class_array is None:
        ax.scatter(z_pca[:, j], z_pca[:, i], s=5, alpha=0.7)
        return

    for cls, label, color in class_entries:
        mask = class_array == cls
        ax.scatter(z_pca[mask, j], z_pca[mask, i], s=3, alpha=0.7, label=label, color=color)
    ax.legend(title="Classes", fontsize=8)


def _plot_variable_scatter(ax, z_pca, i, j, variable_array, variable_name, log_variable):
    """Scatter component j (x) vs component i (y), colored by a continuous variable if given."""
    if variable_array is None:
        ax.scatter(z_pca[:, j], z_pca[:, i], alpha=0.7)
        return

    var_plot = np.log10(variable_array + 1) if log_variable else variable_array
    scatter = ax.scatter(z_pca[:, j], z_pca[:, i], c=var_plot, cmap="viridis", s=3, alpha=0.7)


def plot_pairwise_distance_vs_delta_pt(z, jet_pt, output_file, bins=20):
    """Plot pairwise distances between embeddings as a function of delta PT between jets."""
    dists = pdist(z, metric='euclidean')
    jet_pt_pairs = pdist(jet_pt.reshape(-1, 1), metric='euclidean')

    delta_pt_bins = np.logspace(0, 4, bins)
    median_dists = np.zeros(len(delta_pt_bins) - 1)
    centers = 0.5 * (delta_pt_bins[:-1] + delta_pt_bins[1:])
    for i in range(len(delta_pt_bins) - 1):
        mask = (jet_pt_pairs >= delta_pt_bins[i]) & (jet_pt_pairs < delta_pt_bins[i+1])
        if np.any(mask):
            median_dists[i] = np.median(dists[mask])
        else:
            median_dists[i] = np.nan

    plt.figure(figsize=(5.5, 5))
    plt.plot(centers, median_dists, marker='o', linestyle='-')
    plt.xscale('log')
    plt.xlabel(r"$\Delta p_T$ (GeV)")
    plt.ylabel("Median Pairwise Distance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()

def plot_class_distance_matrix(class_info, z, y, output_file):
    """Plot heatmap of average pairwise distances between classes."""
    classes = sorted({info["class"] for info in class_info.values()})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    avg_dists = np.full((n_classes, n_classes), np.nan, dtype=float)

    for info_i in class_info.values():
        for info_j in class_info.values():
            i = class_to_idx[info_i["class"]]
            j = class_to_idx[info_j["class"]]

            # existing distance computation
            dists = cdist(z[info_i["mask"]], z[info_j["mask"]], metric="euclidean")
            avg_dists[i, j] = dists.mean()

    plt.figure(figsize=(5.5, 5))
    plt.imshow(avg_dists, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Centroid Distance')
    plt.xticks(range(n_classes), classes, rotation=60, ha='right')
    plt.yticks(range(n_classes), classes)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()