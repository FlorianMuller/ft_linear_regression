import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MyLR


def plot_all(x, y, lr):
    # Line
    line_x = np.linspace(0, 300000, 1000)
    line_y = lr.predict_(line_x)
    plt.plot(line_x, line_y, "c-", label="Model")

    # Prediction
    y_hat = lr.predict_(x)
    plt.plot(x, y_hat, "bo", label="Prediction")

    # Data
    plt.plot(x, y, "go", label="Data")

    plt.legend(loc="best")
    plt.xlabel("Km")
    plt.ylabel("Price")
    plt.show()


def main():
    data = pd.read_csv("./data/data.csv")

    x = np.array(data["km"]).reshape(-1, 1)
    y = np.array(data["price"]).reshape(-1, 1)

    lr = MyLR(np.zeros(x.shape[1] + 1), alpha=1e-3, max_iter=50000)

    # Use standardisation/normalisation
    lr.setup_zscore(x)

    # Before training
    print("Starting cost:", lr.cost_(x, y))
    plot_all(x, y, lr)

    # Training model
    lr.fit_(x, y)

    # After training
    print("Ending cost:", lr.cost_(x, y))
    print("Thetas: ", lr.thetas)
    plot_all(x, y, lr)


if __name__ == "__main__":
    main()
