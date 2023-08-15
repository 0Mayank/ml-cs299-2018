import numpy as np
import matplotlib.pyplot as plt


def load_dataset_from_csv(path):
    with open(path, "r") as f:
        headers = f.readline().strip().split(",")

    x_cols = [i for i in range(len(headers)) if headers[i].startswith("x")]
    y_cols = [i for i in range(len(headers)) if headers[i].startswith("y")]

    inputs = np.loadtxt(path, delimiter=",", skiprows=1, usecols=x_cols).astype(
        np.longdouble
    )
    labels = np.loadtxt(path, delimiter=",", skiprows=1, usecols=y_cols).astype(
        np.longdouble
    )

    return inputs, labels


def normalize(x: np.ndarray, mean=None, std=None):
    if mean == None:
        mean = x.mean()
    if std == None:
        std = x.std()

    x = (x - mean) / std
    return x


def prepend_x0(x):
    m, _ = x.shape
    x0 = np.ones((m, 1))
    X = np.append(x0, x, 1)
    return X


def plot_logistic(x: np.ndarray, y: np.ndarray, theta: np.ndarray):
    plt.figure()
    plt.plot(x[y == 1, 0], x[y == 1, 1], "bx", linewidth=1)
    plt.plot(x[y == 0, 0], x[y == 0, 1], "go", linewidth=1)

    x1 = np.array([min(x[:, 0]), max(x[:, 0])])
    x2 = np.empty(2)

    for i in range(x1.size):
        x2[i] = -(theta[1] * x1[i] + theta[0]) / theta[2]

    plt.plot(x1, x2, c="red", linewidth=1)
