import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, metrics
from sklearn.cluster import KMeans as SKMeans
from sklearn.decomposition import PCA as SPCA
from sklearn.linear_model import LinearRegression as SLinear
from sklearn.linear_model import LogisticRegression as SLogistic
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from .models import PCA, KMeans, KNearestNeighbors, LinearRegression, LogisticRegression


def run(output, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(400, 3))
    y = x @ np.array([1.0, -2.0, 3.0]) + rng.normal(0, 0.1, 400)
    a, b, c, d = train_test_split(x, y, random_state=seed)
    linear = LinearRegression().fit(a, c)
    reference = SLinear().fit(a, c)
    results = {
        "seed": seed,
        "data": "synthetic generated regression, classification and blobs",
        "python": platform.python_version(),
        "linear_mse": metrics.mean_squared_error(d, linear.predict(b)),
        "linear_prediction_max_difference": float(
            np.max(np.abs(linear.predict(b) - reference.predict(b)))
        ),
    }
    fig, ax = plt.subplots()
    ax.scatter(d, linear.predict(b), s=8)
    ax.set(xlabel="True target", ylabel="Prediction")
    fig.savefig(output / "regression.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    for rate in [0.005, 0.05, 0.2]:
        model = LinearRegression(rate).fit(a, c)
        ax.plot(model.loss_history_, label=str(rate))
    ax.set(xlabel="Iteration", ylabel="Training MSE", yscale="log")
    ax.legend(title="Learning rate")
    fig.savefig(output / "loss.png")
    plt.close(fig)
    x, y = datasets.make_classification(
        n_samples=500, n_features=2, n_redundant=0, n_informative=2, random_state=seed
    )
    a, b, c, d = train_test_split(x, y, stratify=y, random_state=seed)
    scale = StandardScaler().fit(a)
    a, b = scale.transform(a), scale.transform(b)
    logistic = LogisticRegression(iterations=4000).fit(a, c)
    baseline = SLogistic(C=1e8, max_iter=4000).fit(a, c)
    pred = logistic.predict(b)
    prob = logistic.predict_proba(b)[:, 1]
    results["logistic"] = {
        "accuracy": metrics.accuracy_score(d, pred),
        "precision": metrics.precision_score(d, pred),
        "recall": metrics.recall_score(d, pred),
        "f1": metrics.f1_score(d, pred),
        "roc_auc": metrics.roc_auc_score(d, prob),
        "sklearn_accuracy": metrics.accuracy_score(d, baseline.predict(b)),
        "probability_max_difference": float(np.max(np.abs(prob - baseline.predict_proba(b)[:, 1]))),
    }
    xx, yy = np.meshgrid(
        np.linspace(a[:, 0].min() - 1, a[:, 0].max() + 1, 120),
        np.linspace(a[:, 1].min() - 1, a[:, 1].max() + 1, 120),
    )
    fig, ax = plt.subplots()
    ax.contourf(
        xx, yy, logistic.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape), alpha=0.2
    )
    ax.scatter(b[:, 0], b[:, 1], c=d, s=9)
    ax.set_title("Held-out synthetic points")
    fig.savefig(output / "boundary.png")
    plt.close(fig)
    results["knn"] = {}
    for metric in ["euclidean", "manhattan"]:
        pred = KNearestNeighbors(metric=metric).fit(a, c).predict(b)
        ref = KNeighborsClassifier(metric=metric).fit(a, c).predict(b)
        results["knn"][metric] = {
            "agreement": float(np.mean(pred == ref)),
            "accuracy": metrics.accuracy_score(d, pred),
        }
    x, _ = datasets.make_blobs(n_samples=300, centers=3, random_state=seed)
    km = KMeans(seed=seed).fit(x)
    ref = SKMeans(n_clusters=3, n_init=10, random_state=seed).fit(x)
    results["kmeans"] = {
        "inertia": km.inertia_,
        "sklearn_inertia": ref.inertia_,
        "adjusted_rand_agreement": metrics.adjusted_rand_score(ref.labels_, km.labels_),
    }
    fig, ax = plt.subplots()
    ax.scatter(*x.T, c=km.labels_, s=10)
    ax.scatter(*km.cluster_centers_.T, marker="x", c="red")
    fig.savefig(output / "clusters.png")
    plt.close(fig)
    x = rng.normal(size=(200, 4))
    x[:, 3] = x[:, 0] + 0.1 * rng.normal(size=200)
    pca = PCA(2).fit(x)
    ref = SPCA(2).fit(x)
    z = pca.transform(x)
    results["pca"] = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "sklearn_variance_max_difference": float(
            np.max(np.abs(pca.explained_variance_ - ref.explained_variance_))
        ),
        "reconstruction_mse": float(np.mean((x - pca.inverse_transform(z)) ** 2)),
    }
    fig, ax = plt.subplots()
    ax.scatter(*z.T, s=9)
    ax.set(xlabel="PC1", ylabel="PC2")
    fig.savefig(output / "pca.png")
    plt.close(fig)
    results["runtime_seconds"] = perf_counter() - start
    (output / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description="Execute synthetic NumPy/sklearn comparisons")
    parser.add_argument("--output", default="results/synthetic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
