import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MyLR


def plot_all(thetas, x, y):
    # Line
    line_x = np.linspace(x.min(), x.max(), 100)
    line_y = predict(thetas, line_x)
    plt.plot(line_x, line_y, "c-", label="Model")

    # Prediction
    y_hat = predict(thetas, x)
    plt.plot(x, y_hat, "bo", label="Prediction")

    # Data
    plt.plot(x, y, "go", label="Data")

    plt.legend(loc="best")
    plt.xlabel("Km")
    plt.ylabel("Price")
    plt.show()


def predict(thetas, x):
    return thetas[0][0] + (thetas[1][0] * x)


def cost(thetas, x, y):
    y_hat = predict(thetas, x)
    return np.sum(((y_hat - y) ** 2) / (2 * y.shape[0]))


def train(thetas, x, y, alpha=1e-3, max_iter=1000):
    for i in range(max_iter):
        y_hat = predict(thetas, x)
        nabla0 = (1 / y.shape[0]) * np.sum(y_hat - y)
        nabla1 = (1 / y.shape[0]) * np.sum((y_hat - y) * x)

        thetas[0][0] = thetas[0][0] - alpha * nabla0
        thetas[1][0] = thetas[1][0] - alpha * nabla1

    return thetas


def main():
    data = pd.read_csv("./data/data.csv")

    x = np.array(data["km"]).reshape(-1, 1)
    y = np.array(data["price"]).reshape(-1, 1)
    thetas = np.array([[0.], [0.]])

    print("before", cost(thetas, x, y))
    # plot_all(thetas, x, y)

    train(thetas, x, y, alpha=1e-10, max_iter=10000)

    print("after", cost(thetas, x, y))
    print(thetas)
    plot_all(thetas, x, y)


if __name__ == "__main__":
    main()
