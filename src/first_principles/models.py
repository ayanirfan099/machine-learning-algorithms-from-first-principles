"""NumPy estimators with deterministic initialization and explicit numerical losses."""

import numpy as np


def matrix(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite 2D matrix")
    return x


def supervised(x, y):
    x = matrix(x)
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != len(x) or not np.isfinite(y).all():
        raise ValueError("Expected finite 1D labels matching rows")
    return x, y


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return np.exp(-np.logaddexp(0, -z))


class LinearRegression:
    def __init__(self, learning_rate=0.05, iterations=2000):
        if learning_rate <= 0 or iterations < 1:
            raise ValueError("learning_rate and iterations must be positive")
        self.learning_rate, self.iterations = learning_rate, iterations

    @staticmethod
    def loss_gradient(x, y, weights):
        error = x @ weights - y
        return np.mean(error**2), 2 * x.T @ error / len(x)

    def fit(self, x, y):
        x, y = supervised(x, y)
        self.n_features_in_ = x.shape[1]
        x = np.c_[np.ones(len(x)), x]
        self.weights_ = np.zeros(x.shape[1])
        self.loss_history_ = []
        for _ in range(self.iterations):
            loss, grad = self.loss_gradient(x, y, self.weights_)
            if not np.isfinite(loss):
                raise ValueError("Optimization diverged; scale inputs or reduce learning rate")
            self.loss_history_.append(float(loss))
            self.weights_ -= self.learning_rate * grad
        return self

    def predict(self, x):
        x = matrix(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("Feature count mismatch")
        return np.c_[np.ones(len(x)), x] @ self.weights_


class LogisticRegression(LinearRegression):
    @staticmethod
    def loss_gradient(x, y, weights):
        z = x @ weights
        return np.mean(np.logaddexp(0, z) - y * z), x.T @ (sigmoid(z) - y) / len(x)

    def fit(self, x, y):
        if not np.isin(np.asarray(y), [0, 1]).all():
            raise ValueError("Binary labels must be 0 or 1")
        return super().fit(x, y)

    def predict_proba(self, x):
        positive = sigmoid(super().predict(x))
        return np.c_[1 - positive, positive]

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


class KNearestNeighbors:
    def __init__(self, neighbors=5, metric="euclidean"):
        if neighbors < 1 or metric not in {"euclidean", "manhattan"}:
            raise ValueError("Invalid neighbor count or metric")
        self.neighbors, self.metric = neighbors, metric

    def fit(self, x, y):
        self.x_, self.y_ = supervised(x, y)
        self.x_, self.y_ = self.x_.copy(), self.y_.copy()
        if self.neighbors > len(self.x_):
            raise ValueError("More neighbors than samples")
        return self

    def predict(self, x):
        x = matrix(x)
        if x.shape[1] != self.x_.shape[1]:
            raise ValueError("Feature count mismatch")
        predictions = []
        for row in x:
            delta = np.abs(self.x_ - row)
            distances = delta.sum(axis=1) if self.metric == "manhattan" else (delta**2).sum(axis=1)
            nearest = np.argsort(distances, kind="stable")[: self.neighbors]
            labels, counts = np.unique(self.y_[nearest], return_counts=True)
            predictions.append(labels[np.argmax(counts)])
        return np.asarray(predictions)


class KMeans:
    def __init__(self, clusters=3, n_init=10, iterations=200, tolerance=1e-6, seed=42):
        if min(clusters, n_init, iterations) < 1 or tolerance < 0:
            raise ValueError("Invalid clustering parameters")
        self.clusters, self.n_init, self.iterations = clusters, n_init, iterations
        self.tolerance, self.seed = tolerance, seed

    def fit(self, x):
        x = matrix(x)
        if self.clusters > len(x):
            raise ValueError("More clusters than samples")
        rng = np.random.default_rng(self.seed)
        self.inertia_ = np.inf
        for _ in range(self.n_init):
            centers = x[rng.choice(len(x), self.clusters, replace=False)].copy()
            for iteration in range(self.iterations):
                labels = ((x[:, None] - centers) ** 2).sum(axis=2).argmin(axis=1)
                updated = np.array(
                    [
                        x[labels == k].mean(axis=0)
                        if np.any(labels == k)
                        else x[rng.integers(len(x))]
                        for k in range(self.clusters)
                    ]
                )
                shift = np.linalg.norm(updated - centers)
                centers = updated
                if shift <= self.tolerance:
                    break
            distances = ((x[:, None] - centers) ** 2).sum(axis=2)
            inertia = distances.min(axis=1).sum()
            if inertia < self.inertia_:
                self.cluster_centers_ = centers.copy()
                self.labels_ = distances.argmin(axis=1)
                self.inertia_ = float(inertia)
                self.n_iter_ = iteration + 1
        return self

    def predict(self, x):
        x = matrix(x)
        if x.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError("Feature count mismatch")
        return ((x[:, None] - self.cluster_centers_) ** 2).sum(axis=2).argmin(axis=1)


class PCA:
    def __init__(self, components=2):
        self.components = components

    def fit(self, x):
        x = matrix(x)
        if len(x) < 2 or not 1 <= self.components <= min(x.shape):
            raise ValueError("Invalid component count or sample count")
        self.mean_ = x.mean(axis=0)
        _, singular, vectors = np.linalg.svd(x - self.mean_, full_matrices=False)
        self.components_ = vectors[: self.components]
        variance = singular**2 / (len(x) - 1)
        self.explained_variance_ = variance[: self.components]
        self.explained_variance_ratio_ = (
            variance[: self.components] / variance.sum()
            if variance.sum()
            else np.zeros(self.components)
        )
        return self

    def transform(self, x):
        x = matrix(x)
        if x.shape[1] != len(self.mean_):
            raise ValueError("Feature count mismatch")
        return (x - self.mean_) @ self.components_.T

    def inverse_transform(self, z):
        z = matrix(z)
        if z.shape[1] != self.components:
            raise ValueError("Component count mismatch")
        return z @ self.components_ + self.mean_
