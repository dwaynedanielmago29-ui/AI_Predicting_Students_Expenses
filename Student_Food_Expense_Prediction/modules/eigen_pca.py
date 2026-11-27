import numpy as np
import pandas as pd


class EigenPCA:
    """
    Computes covariance matrices, eigenvalues, eigenvectors, and performs
    Principal Component Analysis (PCA) from scratch using NumPy.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.eigenvalues = None
        self.eigenvectors = None
        self.explained_variance_ratio = None
        self.cumulative_variance_ratio = None
        self.components = None
        self.mean = None

    def fit(self, X):
        """
        Fits PCA on the standardized/centered feature matrix X (n_samples, n_features).
        """
        X_arr = np.asarray(X, dtype=float)
        self.mean = np.mean(X_arr, axis=0)
        X_centered = X_arr - self.mean

        # 1. Compute the Covariance Matrix: C = (1 / (n - 1)) * X^T * X
        n_samples = X_centered.shape[0]
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 2. Compute Eigenvalues and Eigenvectors
        # eigh is used because covariance matrices are symmetric (guaranteed real eigenvalues)
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

        # 3. Sort eigenvalues and corresponding eigenvectors in descending order
        sorted_indices = np.argsort(eig_vals)[::-1]
        self.eigenvalues = eig_vals[sorted_indices]
        self.eigenvectors = eig_vecs[:, sorted_indices]

        # 4. Calculate Explained Variance Ratio
        total_variance = np.sum(self.eigenvalues)
        self.explained_variance_ratio = self.eigenvalues / total_variance
        self.cumulative_variance_ratio = np.cumsum(self.explained_variance_ratio)

        # 5. Select top k components
        if self.n_components is None:
            self.n_components = X_arr.shape[1]
        elif isinstance(self.n_components, float) and 0 < self.n_components < 1:
            # Automatically pick components needed to reach target variance (e.g., 0.95)
            self.n_components = int(np.argmax(self.cumulative_variance_ratio >= self.n_components) + 1)

        self.components = self.eigenvectors[:, :self.n_components]
        return self

    def transform(self, X):
        """
        Projects dataset X onto the selected principal components.
        """
        if self.components is None:
            raise ValueError("PCA must be fitted before calling transform.")
        X_arr = np.asarray(X, dtype=float)
        X_centered = X_arr - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        """
        Fits PCA and returns the transformed dataset.
        """
        self.fit(X)
        return self.transform(X)

    def get_summary_table(self, feature_names=None):
        """
        Returns a DataFrame summarizing eigenvalues, variance ratios, and loadings.
        """
        if self.eigenvalues is None:
            raise ValueError("PCA must be fitted first.")

        pc_labels = [f"PC{i+1}" for i in range(len(self.eigenvalues))]
        summary = pd.DataFrame({
            "Principal Component": pc_labels,
            "Eigenvalue": np.round(self.eigenvalues, 4),
            "Explained Variance (%)": np.round(self.explained_variance_ratio * 100, 2),
            "Cumulative Variance (%)": np.round(self.cumulative_variance_ratio * 100, 2)
        })

        if feature_names is not None and len(feature_names) == self.eigenvectors.shape[0]:
            loadings = pd.DataFrame(
                np.round(self.eigenvectors, 4),
                index=feature_names,
                columns=pc_labels
            )
            return summary, loadings

        return summary