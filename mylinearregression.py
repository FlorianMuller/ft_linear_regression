import numpy as np


class MyLinearRegression():
    """ My personnal linear regression class to fit like a boss """

    def __init__(self, thetas, alpha=0.001, max_iter=1000, mean=None, std=None):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = np.array(
            thetas, copy=True).reshape(-1, 1).astype("float64")

        # Mean and standar deviation of the dataset to use zscore
        self.mean = mean
        self.std = std

    def get_params_(self):
        return self.__dict__

    def setup_zscore(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)

    def zscore(self, x):
        return (x - self.mean) / self.std

    def fit_(self, x, y):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if (x.size == 0 or y.size == 0 or self.thetas.size == 0
            or x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]
                or x.shape[1] + 1 != self.thetas.shape[0] or y.shape[1] != 1):
            return None

        # Use standardisation (or normalisation)
        if self.mean is not None and self.std is not None:
            x = self.zscore(x)

        x_prime = np.c_[np.ones(x.shape[0]), x]
        for _ in range(self.max_iter):
            nabla = (x_prime.T @ ((x_prime @ self.thetas) - y)) / y.shape[0]
            self.thetas = self.thetas - self.alpha * nabla

        return self.thetas

    def predict_(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if (x.size == 0 or self.thetas.size == 0
                or x.ndim != 2 or x.shape[1] + 1 != self.thetas.shape[0]):
            return None

        # Use standardisation (or normalisation)
        if self.mean is not None and self.std is not None:
            x = self.zscore(x)

        x_prime = np.c_[np.ones(x.shape[0]), x]
        return x_prime @ self.thetas

    def cost_(self, x, y):
        # Using one dimensional array to use dot product with `np.dot`
        # (np.dot use matmul with two dimensional array)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.flatten()

        y_hat = self.predict_(x).flatten()

        if (y.size == 0 or y.ndim != 1
                or y_hat is None or y.shape != y_hat.shape):
            return None

        y_diff = y_hat - y
        return np.dot(y_diff, y_diff) / (2 * y.shape[0])
