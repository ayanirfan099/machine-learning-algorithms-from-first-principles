import subprocess
import sys

import numpy as np
import pytest

from first_principles.models import (
    PCA,
    KMeans,
    KNearestNeighbors,
    LinearRegression,
    LogisticRegression,
    sigmoid,
)


@pytest.mark.parametrize("model", [LinearRegression, LogisticRegression])
def test_gradients(model):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(12, 3))
    w = rng.normal(size=3)
    y = rng.integers(0, 2, 12) if model is LogisticRegression else rng.normal(size=12)
    _, analytic = model.loss_gradient(x, y, w)
    numeric = []
    for i in range(3):
        step = np.zeros(3)
        step[i] = 1e-6
        numeric.append(
            (model.loss_gradient(x, y, w + step)[0] - model.loss_gradient(x, y, w - step)[0]) / 2e-6
        )
    np.testing.assert_allclose(analytic, numeric, atol=1e-7)


def test_linear_and_binary():
    x = np.arange(20).reshape(-1, 1) / 20
    np.testing.assert_allclose(
        LinearRegression(iterations=5000).fit(x, 2 * x[:, 0] + 1).predict(x),
        2 * x[:, 0] + 1,
        atol=1e-5,
    )
    np.testing.assert_array_equal(sigmoid([-1000, 1000]), [0, 1])
    assert np.isfinite(
        LogisticRegression.loss_gradient(np.array([[1000.0]]), np.array([0]), np.array([1.0]))[0]
    )
    with pytest.raises(ValueError):
        LogisticRegression().fit(x, np.ones(20) * 2)


@pytest.mark.parametrize("model", [LinearRegression(), KNearestNeighbors()])
def test_validation(model):
    with pytest.raises(ValueError):
        model.fit([[np.nan]], [1])
    with pytest.raises(ValueError):
        model.fit([[1], [2]], [1])


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_knn(metric):
    assert KNearestNeighbors(1, metric).fit([[0], [1]], [0, 1]).predict([[0.9]])[0] == 1
    with pytest.raises(ValueError):
        KNearestNeighbors(3).fit([[0]], [0])


def test_kmeans_reproducibility_and_empty_cluster():
    x = np.array([[0.0, 0.0], [0, 0], [10, 10], [10, 10]])
    a = KMeans(3).fit(x)
    b = KMeans(3).fit(x)
    assert a.inertia_ == b.inertia_ == 0
    np.testing.assert_array_equal(a.predict(x), a.labels_)
    with pytest.raises(ValueError):
        KMeans(5).fit(x)


def test_pca():
    x = np.random.default_rng(2).normal(size=(20, 3))
    p = PCA(3).fit(x)
    np.testing.assert_allclose(p.inverse_transform(p.transform(x)), x, atol=1e-12)
    assert np.isfinite(PCA(1).fit(np.ones((4, 2))).explained_variance_ratio_).all()
    with pytest.raises(ValueError):
        PCA(4).fit(x)


def test_cli_help():
    assert (
        subprocess.run(
            [sys.executable, "-m", "first_principles.cli", "--help"],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def test_end_to_end(tmp_path):
    from first_principles.cli import run

    result = run(tmp_path)
    assert result["linear_prediction_max_difference"] < 1e-5
    assert result["knn"]["euclidean"]["agreement"] == 1
    assert result["pca"]["sklearn_variance_max_difference"] < 1e-10
    assert (tmp_path / "boundary.png").exists()
