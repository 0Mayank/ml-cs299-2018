from libpkg.linear_model import GLM
import libpkg.utils as utils
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression(GLM):

    def h(self, theta: np.ndarray, xi: np.ndarray):
        tx = (-theta.T @ xi).astype(np.longdouble)
        return 1 / (1 + np.exp(tx[0]))

    def predict(self, x):
        y = super().predict(x)

        for i in range(y.size):
            if y[i] < 0.5:
                y[i] = 0
            else:
                y[i] = 1

        return y


ds1_train_path = "data\\ds1_train.csv"
ds1_valid_path = "data\\ds1_valid.csv"

x_train, y_train = utils.load_dataset_from_csv(ds1_train_path)
mean = x_train.mean()
std = x_train.std()
x_train = utils.normalize(x_train)
x_valid, y_valid = utils.load_dataset_from_csv(ds1_valid_path)
x_valid = utils.normalize(x_valid, mean, std)

reg = LogisticRegression(max_iter=100)
reg.fit(x_train, y_train)

print("Accuracy on training set is: ", np.mean(reg.predict(x_train) == y_train))
utils.plot_logistic(x_train, y_train, reg.theta)

print("Accuracy on valid set is: ", np.mean(reg.predict(x_valid) == y_valid))
utils.plot_logistic(x_valid, y_valid, reg.theta)

plt.show()
