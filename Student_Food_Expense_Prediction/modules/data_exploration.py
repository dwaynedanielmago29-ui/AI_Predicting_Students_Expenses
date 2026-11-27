"""
data_exploration.py
Data Exploration module

Supports the manuscript's dataset/matrix diagnostics and eigen-analysis
(Sections 3.2, 3.4, 3.5, 3.6 and 5.1 to 5.2). It only looks at the data and
the training-set matrices; it does not fit the regression models.

    - descriptive statistics, missing values, value counts, outliers
    - correlation analysis
    - matrix diagnostics: rank, Gram matrix, determinant, condition number
    - eigen-analysis of the covariance matrix of standardized training data,
      explained variance, residual checks, reconstruction error, choice of k
    - figures: histograms, correlation heatmap, scree and cumulative plots
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:  # works as "python -m modules.data_exploration" and as a plain script
    from modules import dataset_manager as dm
except ImportError:  # pragma: no cover
    import dataset_manager as dm

FIGURE_DIR = dm.BASE_DIR / "outputs" / "figures"
VARIANCE_TARGET = 0.95   # Section 3.6: smallest k with >= 95% training variance


# ---------------------------------------------------------------------------
# Descriptive exploration
# ---------------------------------------------------------------------------
def summary_statistics(df):
    """Count, mean, SD, min, median and max of each numeric column."""
    return df[dm.NUMERIC_COLUMNS].agg(["count", "mean", "std", "min", "median", "max"]).T.round(2)


def missing_report(df):
    """Missing count and percentage per column."""
    return pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percent": (df.isnull().mean() * 100).round(2),
    })


def value_counts_report(df, max_unique=12):
    """Frequency table of columns with few distinct (coded or discrete) values."""
    return {c: df[c].value_counts().sort_index()
            for c in dm.NUMERIC_COLUMNS if df[c].nunique() <= max_unique}


def correlation_matrix(df):
    """Pearson correlation matrix."""
    return df[dm.NUMERIC_COLUMNS].corr().round(3)


def target_correlations(df):
    """Correlation of each feature with the target, strongest first."""
    corr = df[dm.NUMERIC_COLUMNS].corr()[dm.TARGET].drop(dm.TARGET)
    return corr.reindex(corr.abs().sort_values(ascending=False).index).round(3)


def iqr_outliers(df):
    """Count of IQR-rule outliers per column (unusual values check, Section 2.1)."""
    rows = {}
    for col in dm.NUMERIC_COLUMNS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((df[col] < low) | (df[col] > high)).sum())
        rows[col] = {"lower_bound": round(low, 2), "upper_bound": round(high, 2), "outliers": n}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Matrix diagnostics (Sections 3.2 and 3.4)
# ---------------------------------------------------------------------------
def matrix_diagnostics(X_train, X_train_std):
    """
    Rank, linear independence, Gram matrix, determinant and conditioning.

    G = X_std^T X_std is used for the determinant and condition number.
    The determinant is reported as sign and log|det|, because the raw value
    depends on scale and a large determinant alone does not prove good
    conditioning.
    """
    n, p = X_train.shape
    s = np.linalg.svd(X_train, compute_uv=False)
    tol = s.max() * max(X_train.shape) * np.finfo(float).eps   # same rule as NumPy
    rank = int(np.linalg.matrix_rank(X_train, tol=tol))

    G = dm.gram_matrix(X_train_std)
    sign, logdet = np.linalg.slogdet(G)
    return {
        "n": n,
        "p": p,
        "rank_X_train": rank,
        "rank_tolerance": float(tol),
        "columns_linearly_independent": rank == p,
        "gram_shape": G.shape,
        "gram_matrix": G,
        "det_sign": int(sign),
        "log_abs_det": float(logdet),
        "det_scientific": float(sign * np.exp(logdet)),
        "cond_X_train_std": float(np.linalg.cond(X_train_std)),
        "cond_gram": float(np.linalg.cond(G)),
        "singular_values_X_std": np.linalg.svd(X_train_std, compute_uv=False),
    }


# ---------------------------------------------------------------------------
# Eigen-analysis and PCA preparation (Sections 3.5 and 3.6)
# ---------------------------------------------------------------------------
def eigen_analysis(X_train_std, variance_target=VARIANCE_TARGET):
    """
    Eigen-decomposition of C = cov(X_train_std), a symmetric p x p matrix.

    Returns eigenvalues (descending), unit eigenvectors (columns of V),
    explained-variance ratios, residual norms ||Cv - lambda v||_2,
    reconstruction error ||C - V Lambda V^T||_F and the smallest k that
    reaches the variance target.
    """
    C = np.cov(X_train_std, rowvar=False)               # ddof = 1
    eigvals, eigvecs = np.linalg.eigh(C)                # ascending order
    order = np.argsort(eigvals)[::-1]
    eigvals, V = eigvals[order], eigvecs[:, order]

    ratio = eigvals / eigvals.sum()
    cumulative = np.cumsum(ratio)
    k = int(np.searchsorted(cumulative, variance_target - 1e-12) + 1)

    residuals = np.array([np.linalg.norm(C @ V[:, i] - eigvals[i] * V[:, i])
                          for i in range(len(eigvals))])
    recon_error = float(np.linalg.norm(C - V @ np.diag(eigvals) @ V.T, ord="fro"))

    return {
        "covariance": C,
        "eigenvalues": eigvals,
        "eigenvectors": V,
        "explained_ratio": ratio,
        "cumulative_ratio": cumulative,
        "residual_norms": residuals,
        "reconstruction_error": recon_error,
        "orthonormal_error": float(np.linalg.norm(V.T @ V - np.eye(V.shape[1]), ord="fro")),
        "k_selected": k,
        "variance_target": variance_target,
    }


def project_pca(X_train_std, X_test_std, V, k):
    """Z_train = X_train_std V_k and Z_test = X_test_std V_k (same eigenvectors)."""
    Vk = V[:, :k]
    return X_train_std @ Vk, X_test_std @ Vk


# ---------------------------------------------------------------------------
# Figures (black and white; matplotlib optional)
# ---------------------------------------------------------------------------
def save_figures(df, eig, out_dir=FIGURE_DIR):
    """Save histograms, correlation heatmap, scree plot and cumulative variance plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping figures.")
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    def _save(fig, name):
        fig.tight_layout()
        path = out_dir / name
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    # Histograms
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.ravel(), dm.NUMERIC_COLUMNS):
        ax.hist(df[col], bins=10, color="0.6", edgecolor="black")
        ax.set_title(col)
        ax.set_ylabel("Frequency")
    axes.ravel()[-1].axis("off")
    _save(fig, "histograms.png")

    # Correlation heatmap
    corr = df[dm.NUMERIC_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="gray_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.6 else "black", fontsize=8)
    fig.colorbar(im)
    ax.set_title("Correlation Matrix")
    _save(fig, "correlation_heatmap.png")

    # Scree plot
    lam = eig["eigenvalues"]
    pcs = np.arange(1, len(lam) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pcs, lam, "ko-")
    ax.set_xticks(pcs)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Scree Plot")
    ax.grid(True, linestyle=":", color="0.5")
    _save(fig, "scree_plot.png")

    # Cumulative explained variance
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pcs, eig["cumulative_ratio"] * 100, "ks-")
    ax.axhline(eig["variance_target"] * 100, color="black", linestyle="--",
               label=f"{eig['variance_target']:.0%} target")
    ax.set_xticks(pcs)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("Cumulative Explained Variance")
    ax.legend()
    ax.grid(True, linestyle=":", color="0.5")
    _save(fig, "cumulative_variance.png")

    return saved


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------
def run_exploration(df=None, make_figures=True):
    """Print the full exploration report and optionally save the figures."""
    df = dm.load_model_ready() if df is None else df
    data = dm.prepare_data(df)

    print("\nSUMMARY STATISTICS")
    print(summary_statistics(df).to_string())

    print("\nMISSING VALUES")
    print(missing_report(df).to_string())

    print("\nVALUE COUNTS")
    for col, counts in value_counts_report(df).items():
        print(f"\n{col}")
        print(counts.to_string())

    print("\nCORRELATION MATRIX")
    print(correlation_matrix(df).to_string())
    print("\nCORRELATION WITH TARGET (strongest first)")
    print(target_correlations(df).to_string())

    print("\nIQR OUTLIER CHECK")
    print(iqr_outliers(df).to_string())

    diag = matrix_diagnostics(data["X_train"], data["X_train_std"])
    print("\nMATRIX DIAGNOSTICS (training set)")
    print(f"    n = {diag['n']}, p = {diag['p']}")
    print(f"    rank(X_train) = {diag['rank_X_train']}  "
          f"(tolerance {diag['rank_tolerance']:.2e})  "
          f"independent columns: {diag['columns_linearly_independent']}")
    print(f"    Gram matrix G = Xstd^T Xstd, shape {diag['gram_shape']}")
    print(f"    det(G): sign {diag['det_sign']:+d}, log|det| = {diag['log_abs_det']:.4f}, "
          f"det = {diag['det_scientific']:.4e}")
    print(f"    cond(Xstd) = {diag['cond_X_train_std']:.4f}   cond(G) = {diag['cond_gram']:.4f}")
    print("    Gram matrix:")
    print(pd.DataFrame(diag["gram_matrix"], index=dm.FEATURES, columns=dm.FEATURES)
          .round(3).to_string())

    eig = eigen_analysis(data["X_train_std"])
    print("\nEIGEN-ANALYSIS of C = cov(Xstd_train)")
    table = pd.DataFrame({
        "eigenvalue": eig["eigenvalues"],
        "explained_%": eig["explained_ratio"] * 100,
        "cumulative_%": eig["cumulative_ratio"] * 100,
        "residual ||Cv - lv||": eig["residual_norms"],
    }, index=[f"PC{i+1}" for i in range(len(eig["eigenvalues"]))])
    print(table.round(6).to_string())
    print("\nEigenvectors V (columns = PCs, rows = original features):")
    print(pd.DataFrame(eig["eigenvectors"], index=dm.FEATURES, columns=table.index)
          .round(4).to_string())
    print(f"\nReconstruction error ||C - V L V^T||_F = {eig['reconstruction_error']:.2e}")
    print(f"Orthonormality error ||V^T V - I||_F  = {eig['orthonormal_error']:.2e}")
    k = eig["k_selected"]
    print(f"Smallest k with >= {eig['variance_target']:.0%} training variance: k = {k} "
          f"(cumulative {eig['cumulative_ratio'][k-1]*100:.2f}%)")

    Ztr, Zte = project_pca(data["X_train_std"], data["X_test_std"], eig["eigenvectors"], k)
    print(f"Z_train shape {Ztr.shape}, Z_test shape {Zte.shape}")

    if make_figures:
        files = save_figures(df, eig)
        if files:
            print("\nFigures saved:")
            for f in files:
                print("    ", f)


if __name__ == "__main__":
    run_exploration()
