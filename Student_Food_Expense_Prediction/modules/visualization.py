import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    """
    Generates and saves linear algebra, PCA, and regression visualizations
    for the Student Daily Food Expenses project.
    """

    def __init__(self, output_dir="outputs/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")

    def _save_or_show(self, fig, filename, show=False):
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return filepath

    def plot_scree(self, explained_variance_ratio, cumulative_variance_ratio, show=False):
        """
        Plots individual and cumulative explained variance across principal components.
        """
        n_pcs = len(explained_variance_ratio)
        pcs = np.arange(1, n_pcs + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(pcs, explained_variance_ratio * 100, alpha=0.75, label="Individual Variance (%)", color="#4C72B0")
        ax.plot(pcs, cumulative_variance_ratio * 100, marker="o", color="#DD8452", linewidth=2, label="Cumulative Variance (%)")

        ax.axhline(80, color="gray", linestyle="--", linewidth=1, label="80% Threshold")
        ax.set_title("PCA Scree Plot: Explained Variance by Component", fontsize=13, fontweight="bold")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance (%)")
        ax.set_xticks(pcs)
        ax.set_ylim(0, 105)
        ax.legend(loc="best")

        return self._save_or_show(fig, "pca_scree_plot.png", show)

    def plot_pca_2d(self, X_pca, target=None, target_name="Daily Food Expense", show=False):
        """
        Plots the 2D projection of students onto PC1 and PC2.
        """
        if X_pca.shape[1] < 2:
            raise ValueError("X_pca must have at least 2 components for a 2D plot.")

        fig, ax = plt.subplots(figsize=(8, 6))
        if target is not None:
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=target, cmap="viridis", alpha=0.8, edgecolors="k", s=55)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(target_name)
        else:
            ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.8, color="#4C72B0", edgecolors="k", s=55)

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title("2D PCA Projection of Student Food Expenses", fontsize=13, fontweight="bold")
        ax.set_xlabel("Principal Component 1 (PC1)")
        ax.set_ylabel("Principal Component 2 (PC2)")

        return self._save_or_show(fig, "pca_2d_projection.png", show)

    def plot_biplot(self, X_pca, eigenvectors, feature_names, show=False):
        """
        Plots PC1 vs PC2 scores along with feature loading vectors.
        """
        fig, ax = plt.subplots(figsize=(9, 7))
        # Normalize scores for clean overlay
        xs = X_pca[:, 0] / np.max(np.abs(X_pca[:, 0]))
        ys = X_pca[:, 1] / np.max(np.abs(X_pca[:, 1]))

        ax.scatter(xs, ys, alpha=0.45, color="#4C72B0", label="Observations")

        for i, feature in enumerate(feature_names):
            ax.arrow(0, 0, eigenvectors[i, 0], eigenvectors[i, 1],
                     color="#C44E52", alpha=0.85, head_width=0.03, linewidth=1.5)
            ax.text(eigenvectors[i, 0] * 1.12, eigenvectors[i, 1] * 1.12,
                    feature, color="#8C1D18", fontsize=9, fontweight="bold", ha="center")

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_title("PCA Biplot (Scores & Feature Loadings)", fontsize=13, fontweight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        return self._save_or_show(fig, "pca_biplot.png", show)

    def plot_correlation_heatmap(self, df, show=False):
        """
        Plots a correlation matrix heatmap for numerical features.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    square=True, linewidths=0.5, ax=ax)
        ax.set_title("Feature Correlation Matrix Heatmap", fontsize=13, fontweight="bold")

        return self._save_or_show(fig, "correlation_heatmap.png", show)

    def plot_actual_vs_predicted(self, y_true, y_pred, model_name="Linear Regression", show=False):
        """
        Plots actual vs. predicted daily food expenses to support regression evaluation.
        """
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_true, y_pred, alpha=0.75, color="#55A868", edgecolors="k", s=50)

        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal Fit (y = x)")

        ax.set_title(f"Actual vs. Predicted Expenses ({model_name})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Actual Daily Food Expense")
        ax.set_ylabel("Predicted Daily Food Expense")
        ax.legend()

        safe_name = model_name.lower().replace(" ", "_")
        return self._save_or_show(fig, f"actual_vs_pred_{safe_name}.png", show)