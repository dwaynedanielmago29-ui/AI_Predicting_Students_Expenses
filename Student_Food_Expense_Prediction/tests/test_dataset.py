"""
tests/test_dataset.py
Unit tests for the Dataset Management and Data Exploration modules.

Run from the project folder:
    python -m pytest tests -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the project root importable when pytest is started from anywhere
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sources import dataset_manager as dm          # noqa: E402
from sources import data_exploration as de         # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def df():
    return dm.load_model_ready()


@pytest.fixture(scope="module")
def prepared(df):
    return dm.prepare_data(df)


def make_small_df(n=10):
    """Small valid DataFrame for tests that need controlled data."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        dm.ID_COLUMN: [f"S{i:03d}" for i in range(1, n + 1)],
        "DailyAllowance_PHP": rng.choice([75, 125.5, 175.5], n),
        "MealsPurchased": rng.choice([0, 1, 2], n),
        "SnacksPurchased": rng.choice([0, 1, 2, 4], n),
        "CampusHours": rng.choice([5.5, 7.5, 11], n),
        dm.TARGET: rng.choice([25, 75.5, 125.5], n),
    })


# ---------------------------------------------------------------------------
# Model-ready dataset (manuscript Sections 1.4 and 2)
# ---------------------------------------------------------------------------
class TestModelReadyData:
    def test_file_exists(self):
        assert dm.MODEL_READY_PATH.exists()

    def test_sample_size_is_98(self, df):
        assert len(df) == dm.EXPECTED_CLEAN_N == 98

    def test_expected_columns(self, df):
        assert list(df.columns) == [dm.ID_COLUMN] + dm.FEATURES + [dm.TARGET]

    def test_four_predictors_one_target(self):
        assert len(dm.FEATURES) == 4
        assert dm.TARGET not in dm.FEATURES      # no data leakage

    def test_no_missing_values(self, df):
        assert df.isnull().sum().sum() == 0

    def test_no_duplicate_ids_or_rows(self, df):
        assert not df[dm.ID_COLUMN].duplicated().any()
        assert not df.duplicated().any()

    def test_all_numeric_columns_are_numeric(self, df):
        for col in dm.NUMERIC_COLUMNS:
            assert pd.api.types.is_numeric_dtype(df[col])

    def test_no_negative_values(self, df):
        assert (df[dm.NUMERIC_COLUMNS] >= 0).all().all()

    def test_documented_value_ranges(self, df):
        assert df["DailyAllowance_PHP"].min() == 75
        assert df["DailyAllowance_PHP"].max() == 325
        assert df["DailyAllowance_PHP"].nunique() == 6
        assert set(df["MealsPurchased"].unique()) == {0, 1, 2}
        assert set(df["SnacksPurchased"].unique()) == {0, 1, 2, 4}
        assert set(df["CampusHours"].unique()) == {5.5, 7.5, 11}

    def test_validate_reports_valid(self, df):
        report = dm.validate_dataset(df)
        assert report["is_valid"] is True
        assert report["matches_expected_n"] is True


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------
class TestValidation:
    def test_detects_missing_column(self):
        bad = make_small_df().drop(columns=["CampusHours"])
        report = dm.validate_dataset(bad)
        assert "CampusHours" in report["missing_columns"]
        assert report["is_valid"] is False

    def test_detects_missing_value(self):
        bad = make_small_df()
        bad.loc[0, "MealsPurchased"] = np.nan
        assert dm.validate_dataset(bad)["is_valid"] is False

    def test_detects_negative_value(self):
        bad = make_small_df()
        bad.loc[0, dm.TARGET] = -5
        report = dm.validate_dataset(bad)
        assert dm.TARGET in report["negative_values"]
        assert report["is_valid"] is False

    def test_detects_duplicate_ids(self):
        bad = make_small_df()
        bad.loc[1, dm.ID_COLUMN] = bad.loc[0, dm.ID_COLUMN]
        report = dm.validate_dataset(bad)
        assert report["duplicate_ids"] == 1
        assert report["is_valid"] is False

    def test_detects_zero_variance(self):
        bad = make_small_df()
        bad["CampusHours"] = 7.5
        report = dm.validate_dataset(bad)
        assert "CampusHours" in report["zero_variance_columns"]
        assert report["is_valid"] is False

    def test_detects_non_numeric(self):
        bad = make_small_df()
        bad["DailyAllowance_PHP"] = bad["DailyAllowance_PHP"].astype(str)
        assert "DailyAllowance_PHP" in dm.validate_dataset(bad)["non_numeric_columns"]


# ---------------------------------------------------------------------------
# Range conversion and cleaning (Section 2.1)
# ---------------------------------------------------------------------------
class TestCleaning:
    @pytest.mark.parametrize("value, expected", [
        ("100-150", 125.0),
        ("101 - 150", 125.5),
        ("PHP 51 - 100", 75.5),
        ("1,000", 1000.0),
        ("175.5", 175.5),
        (3, 3.0),
        (7.5, 7.5),
    ])
    def test_range_to_midpoint(self, value, expected):
        assert dm.range_to_midpoint(value) == pytest.approx(expected)

    @pytest.mark.parametrize("value", [None, np.nan, "none", ""])
    def test_range_to_midpoint_returns_nan(self, value):
        assert np.isnan(dm.range_to_midpoint(value))

    def _raw(self):
        return pd.DataFrame({
            "DailyAllowance_PHP": ["101-150", "PHP 51 - 100", None, "175.5", "175.5"],
            "MealsPurchased": [1, 2, 1, 1, 1],
            "SnacksPurchased": [0, 1, 1, 2, 2],
            "CampusHours": ["5-8", 7.5, 6, 6, 6],
            dm.TARGET: [50, 75, 60, "100-150", "100-150"],
        })

    def test_cleaning_excludes_incomplete_and_duplicates(self):
        clean, log = dm.clean_original_responses(self._raw())
        assert log["raw_rows"] == 5
        assert log["duplicates_removed"] == 1
        assert log["incomplete_removed"] == 1
        assert log["clean_rows"] == len(clean) == 3

    def test_cleaning_converts_ranges_to_midpoints(self):
        clean, _ = dm.clean_original_responses(self._raw())
        assert clean.loc[0, "DailyAllowance_PHP"] == pytest.approx(125.5)
        assert clean.loc[0, "CampusHours"] == pytest.approx(6.5)
        assert clean.loc[2, dm.TARGET] == pytest.approx(125.0)

    def test_cleaning_output_columns_and_ids(self):
        clean, _ = dm.clean_original_responses(self._raw())
        assert list(clean.columns) == [dm.ID_COLUMN] + dm.NUMERIC_COLUMNS
        assert clean[dm.ID_COLUMN].tolist() == ["S001", "S002", "S003"]

    def test_cleaning_drops_personal_information(self):
        raw = self._raw()
        raw["Name"] = list("abcde")
        raw["Email"] = list("vwxyz")
        clean, _ = dm.clean_original_responses(raw)
        assert "Name" not in clean.columns and "Email" not in clean.columns

    def test_cleaning_uses_column_map(self):
        raw = self._raw().rename(columns={"MealsPurchased": "Meals bought"})
        clean, _ = dm.clean_original_responses(raw, column_map={"Meals bought": "MealsPurchased"})
        assert "MealsPurchased" in clean.columns

    def test_cleaning_raises_when_required_column_missing(self):
        with pytest.raises(KeyError):
            dm.clean_original_responses(self._raw().drop(columns=["CampusHours"]))

    def test_simulated_100_to_98(self):
        """100 raw responses with 2 incomplete leave 98 complete ones."""
        raw = make_small_df(100).drop(columns=[dm.ID_COLUMN])
        raw.loc[3, "SnacksPurchased"] = np.nan
        raw.loc[50, dm.TARGET] = np.nan
        # make every row unique so none is removed as a duplicate
        raw["DailyAllowance_PHP"] = raw["DailyAllowance_PHP"] + np.arange(100) * 0.01
        clean, log = dm.clean_original_responses(raw)
        assert log["incomplete_removed"] == 2
        assert len(clean) == 98


# ---------------------------------------------------------------------------
# Split protocol (Section 2.2)
# ---------------------------------------------------------------------------
class TestSplit:
    def test_80_20_sizes(self, df):
        train, test = dm.train_test_split(df)
        assert len(train) == 78
        assert len(test) == 20

    def test_default_seed_and_ratio(self):
        assert dm.RANDOM_SEED == 42
        assert dm.TEST_SIZE == pytest.approx(0.20)

    def test_split_is_reproducible(self, df):
        a_train, a_test = dm.train_test_split(df, seed=42)
        b_train, b_test = dm.train_test_split(df, seed=42)
        pd.testing.assert_frame_equal(a_train, b_train)
        pd.testing.assert_frame_equal(a_test, b_test)

    def test_different_seed_gives_different_split(self, df):
        a_train, _ = dm.train_test_split(df, seed=42)
        b_train, _ = dm.train_test_split(df, seed=7)
        assert not a_train[dm.ID_COLUMN].equals(b_train[dm.ID_COLUMN])

    def test_train_and_test_are_disjoint_and_complete(self, df):
        train, test = dm.train_test_split(df)
        train_ids, test_ids = set(train[dm.ID_COLUMN]), set(test[dm.ID_COLUMN])
        assert train_ids.isdisjoint(test_ids)
        assert train_ids | test_ids == set(df[dm.ID_COLUMN])

    @pytest.mark.parametrize("bad", [0, 1, -0.2, 1.5])
    def test_invalid_test_size_raises(self, df, bad):
        with pytest.raises(ValueError):
            dm.train_test_split(df, test_size=bad)


# ---------------------------------------------------------------------------
# Matrix representation (Section 3.1)
# ---------------------------------------------------------------------------
class TestMatrices:
    def test_shapes(self, df):
        X, y = dm.get_X_y(df)
        assert X.shape == (98, 4)
        assert y.shape == (98,)

    def test_intercept_column(self, df):
        X, _ = dm.get_X_y(df, add_intercept=True)
        assert X.shape == (98, 5)
        assert np.all(X[:, 0] == 1)

    def test_target_not_in_X(self, df):
        X, y = dm.get_X_y(df)
        for j in range(X.shape[1]):
            assert not np.array_equal(X[:, j], y)

    def test_gram_matrix_is_symmetric_p_by_p(self, prepared):
        G = dm.gram_matrix(prepared["X_train_std"])
        assert G.shape == (4, 4)
        assert np.allclose(G, G.T)

    def test_gram_diagonal_is_n_minus_1_for_standardized_data(self, prepared):
        # with ddof=1 standardization each column has sum of squares n - 1
        G = dm.gram_matrix(prepared["X_train_std"])
        assert np.allclose(np.diag(G), len(prepared["X_train_std"]) - 1)

    def test_rank_one_update_identity(self, prepared):
        diff = dm.rank_one_update_check(prepared["X_train_std"], prepared["X_test_std"][0])
        assert diff < 1e-10


# ---------------------------------------------------------------------------
# Standardization (Section 3.3)
# ---------------------------------------------------------------------------
class TestStandardizer:
    def test_train_has_zero_mean_unit_std(self, prepared):
        Z = prepared["X_train_std"]
        assert np.allclose(Z.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(Z.std(axis=0, ddof=1), 1, atol=1e-10)

    def test_M_is_diagonal_inverse_sigma(self, prepared):
        s = prepared["scaler"]
        assert np.allclose(s.M, np.diag(1 / s.sigma))
        assert np.allclose(s.M - np.diag(np.diag(s.M)), 0)

    def test_test_set_uses_train_parameters(self, prepared):
        s = prepared["scaler"]
        expected = (prepared["X_test"] - s.mu) / s.sigma
        assert np.allclose(prepared["X_test_std"], expected)

    def test_parameters_come_from_training_only(self, prepared):
        assert np.allclose(prepared["scaler"].mu, prepared["X_train"].mean(axis=0))

    def test_inverse_transform_round_trip(self, prepared):
        s = prepared["scaler"]
        back = s.inverse_transform(prepared["X_train_std"])
        assert np.allclose(back, prepared["X_train"])

    def test_zero_variance_feature_raises(self):
        X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        with pytest.raises(ValueError):
            dm.Standardizer().fit(X)

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            dm.Standardizer().transform(np.ones((2, 2)))


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------
class TestPrepareData:
    def test_shapes(self, prepared):
        assert prepared["X_train"].shape == (78, 4)
        assert prepared["X_test"].shape == (20, 4)
        assert prepared["y_train"].shape == (78,)
        assert prepared["y_test"].shape == (20,)
        assert prepared["X_train_std"].shape == (78, 4)
        assert prepared["X_test_std"].shape == (20, 4)

    def test_no_nan_after_preparation(self, prepared):
        for key in ["X_train_std", "X_test_std", "y_train", "y_test"]:
            assert not np.isnan(prepared[key]).any()


# ---------------------------------------------------------------------------
# Data exploration module
# ---------------------------------------------------------------------------
class TestExploration:
    def test_summary_statistics_values(self, df):
        stats = de.summary_statistics(df)
        assert stats.loc["DailyAllowance_PHP", "count"] == 98
        assert stats.loc[dm.TARGET, "mean"] == pytest.approx(48.66, abs=0.01)

    def test_missing_report_all_zero(self, df):
        assert (de.missing_report(df)["missing_percent"] == 0).all()

    def test_target_correlations_sorted_by_strength(self, df):
        corr = de.target_correlations(df)
        assert list(corr.abs()) == sorted(corr.abs(), reverse=True)
        assert corr.index[0] == "MealsPurchased"

    def test_matrix_diagnostics_full_rank(self, prepared):
        diag = de.matrix_diagnostics(prepared["X_train"], prepared["X_train_std"])
        assert diag["rank_X_train"] == diag["p"] == 4
        assert diag["columns_linearly_independent"] is True
        assert diag["det_sign"] == 1
        assert diag["cond_gram"] < 100          # well-conditioned

    def test_linearly_dependent_columns_are_detected(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=(30, 1)), rng.normal(size=(30, 1))
        X = np.hstack([a, b, a + b])            # exact sum -> rank 2
        diag = de.matrix_diagnostics(X, X)
        assert diag["rank_X_train"] == 2
        assert diag["columns_linearly_independent"] is False

    def test_eigenvalues_sorted_descending(self, prepared):
        eig = de.eigen_analysis(prepared["X_train_std"])
        assert np.all(np.diff(eig["eigenvalues"]) <= 1e-12)

    def test_eigenvalues_sum_to_p_for_standardized_data(self, prepared):
        # trace of the correlation matrix equals the number of features
        eig = de.eigen_analysis(prepared["X_train_std"])
        assert eig["eigenvalues"].sum() == pytest.approx(4.0)

    def test_explained_variance_adds_to_one(self, prepared):
        eig = de.eigen_analysis(prepared["X_train_std"])
        assert eig["explained_ratio"].sum() == pytest.approx(1.0)
        assert eig["cumulative_ratio"][-1] == pytest.approx(1.0)

    def test_eigenpair_residuals_are_tiny(self, prepared):
        eig = de.eigen_analysis(prepared["X_train_std"])
        assert np.all(eig["residual_norms"] < 1e-10)

    def test_eigenvectors_are_orthonormal(self, prepared):
        eig = de.eigen_analysis(prepared["X_train_std"])
        V = eig["eigenvectors"]
        assert np.allclose(np.linalg.norm(V, axis=0), 1)
        assert eig["orthonormal_error"] < 1e-10

    def test_spectral_reconstruction_error_is_tiny(self, prepared):
        eig = de.eigen_analysis(prepared["X_train_std"])
        assert eig["reconstruction_error"] < 1e-10

    def test_selected_k_is_valid_and_reaches_target(self, prepared):
        eig = de.eigen_analysis(prepared["X_train_std"])
        k = eig["k_selected"]
        assert 1 <= k <= 4
        assert eig["cumulative_ratio"][k - 1] >= eig["variance_target"] - 1e-9
        if k > 1:
            assert eig["cumulative_ratio"][k - 2] < eig["variance_target"]

    def test_pca_projection_shapes_and_shared_eigenvectors(self, prepared):
        eig = de.eigen_analysis(prepared["X_train_std"])
        Ztr, Zte = de.project_pca(prepared["X_train_std"], prepared["X_test_std"],
                                  eig["eigenvectors"], k=3)
        assert Ztr.shape == (78, 3)
        assert Zte.shape == (20, 3)
        assert np.allclose(Zte, prepared["X_test_std"] @ eig["eigenvectors"][:, :3])

    def test_projected_training_components_are_uncorrelated(self, prepared):
        eig = de.eigen_analysis(prepared["X_train_std"])
        Ztr, _ = de.project_pca(prepared["X_train_std"], prepared["X_test_std"],
                                eig["eigenvectors"], k=4)
        cov = np.cov(Ztr, rowvar=False)
        assert np.allclose(cov - np.diag(np.diag(cov)), 0, atol=1e-10)
