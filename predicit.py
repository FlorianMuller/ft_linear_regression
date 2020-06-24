import numpy as np
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MyLR


def plot_result(x, y_hat, lr):
    # Line
    line_x = np.linspace(0, max(300000, x[0][0]), 1000)
    line_y = lr.predict_(line_x)
    plt.plot(line_x, line_y, "c-", label="Model")

    # Prediction
    plt.plot(x, y_hat, "bo", label="Prediction")

    plt.legend(loc="best")
    plt.xlabel("Km")
    plt.ylabel("Price")
    plt.show()


def create_lr():
    try:
        with open("config.mlr", "r") as f:
            config_str = f.read()
        config = eval(config_str, {"array": np.array})

        lr = MyLR(**config)
    except FileNotFoundError:
        lr = MyLR(np.zeros(2))

    return lr


def get_input():
    nbr = None
    while nbr is None:
        print("What's your car mileage ?")
        s = input("-> ")
        try:
            nbr = float(s)
            if nbr < 0:
                print("You mileage must be positif\n")
                nbr = None
        except ValueError:
            print("This is not a valid number\n")

    return nbr


def main():
    # Creting a MyLinearRegression instance
    lr = create_lr()

    while True:
        # Getting input from user
        x = np.array([[get_input()]])

        # Prediction
        y_hat = lr.predict_(x)
        print("Prediction: ", y_hat[0][0], end="\n\n")
        plot_result(x, y_hat, lr)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye bye 👋")
