import matplotlib.pyplot as plt
import numpy as np
import util as util
from sklearn.preprocessing import Normalizer, StandardScaler
import cv2


def add_intercept(x):
    """Add intercept to matrix x.

    Args:
        x: 2D NumPy array.

    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x
    return new_x


def load_dataset(csv_path, label_col='y', add_intercept=False):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels (should be 'y' or 't').
         add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """
    allowed_label_cols = ('y', 't')
    if label_col not in allowed_label_cols:
        raise ValueError(f'Invalid label_col: {label_col} (expected {allowed_label_cols})')

    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        inputs = add_intercept(inputs)

    return inputs, labels


def plot(x, y, theta, save_path=None, correction=1.0):
    """Plot dataset and fitted logistic regression parameters.
    Args:
        x: Matrix of training examples, one per row.
        y: Vector of labels in {0, 1}.
        theta: Vector of parameters for logistic regression model.
        save_path: Path to save the plot.
        correction: Correction factor to apply (Problem 2(e) only).
    """
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    margin1 = (max(x[:, -2]) - min(x[:, -2])) * 0.2
    margin2 = (max(x[:, -1]) - min(x[:, -1])) * 0.2
    x1 = np.arange(min(x[:, -2]) - margin1, max(x[:, -2]) + margin1, 0.01)
    x2 = -(theta[0] / theta[2] * correction + theta[1] / theta[2] * x1)
    plt.plot(x1, x2, c='red', linewidth=2)
    plt.xlim(x[:, -2].min() - margin1, x[:, -2].max() + margin1)
    plt.ylim(x[:, -1].min() - margin2, x[:, -1].max() + margin2)
    plt.xlabel('x1')
    plt.ylabel('x2')
    if save_path:
        plt.savefig(save_path)


class GeneralizedLinearModel:
    """Generalized Linear Model for distribution following the exponential family.

    Example usage:
        > clf = GeneralizedLinearModel()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    GLM_BERNOULI, GLM_GAUSSIAN = range(2)

    def __init__(self, step_size=0.1, max_iter=1000, theta_0=None, verbose=True,
                 glm_type=GLM_BERNOULI):
        self.acc_hist = []
        self.loss_hist = []
        self.max_iter = max_iter
        self.theta = theta_0
        self.verbose = verbose
        self.step_size = step_size
        self.type = glm_type

    def h_theta(self, x):
        if self.type == self.GLM_BERNOULI:
            return 1 / (1 + np.exp(-np.dot(x, self.theta)) + 1e-6)
        if self.type == self.GLM_GAUSSIAN:
            return np.dot(x, self.theta)

    def fit(self, x, y):
        """Train the model using input features x and labels y.

        :param x: Inputs of shape (m, n).
        :param y: Labels of shape (m,)
        """
        for i in range(self.max_iter):
            m, n = x.shape

            if self.theta is None:
                self.theta = np.random.random(n)

            self.theta = self.theta + (1 / m) * self.step_size * (y - self.h_theta(x)) @ x
            loss = np.mean(np.abs(y - self.h_theta(x)))

            if self.type == self.GLM_BERNOULI:
                acc = np.sum(y.astype("int32") == (self.h_theta(x) > 0.5).astype("int32")) / m
                self.acc_hist.append(acc)

            self.loss_hist.append(loss)

            if self.verbose:
                if self.type == self.GLM_BERNOULI:
                    print(f'Iter {i:10d} Accuracy {acc:10.6f} Loss {loss:10.6f}')
                if self.type == self.GLM_GAUSSIAN:
                    print(f'Iter {i:10d} Loss {loss:10.6f}')

    def predict(self, x):
        """Make a prediction given new inputs x.

        :param x: Inputs of shape (m, n).
        :return: Outputs of shape (m,).
        """
        if self.type == self.GLM_BERNOULI:
            return self.h_theta(x) > 0.5
        return self.h_theta(x)


if __name__ == "__main__":
    # Existing main execution logic remains unchanged.
    pass
