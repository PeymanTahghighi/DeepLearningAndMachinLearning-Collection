import numpy as np
from util import load_dataset, add_intercept

class GDA():
    """
        Gaussian Discriminant Analysis

        Example usage:
            > gda = GDA()
            > gda.fit(x_train, y_train)
            > pred = gda.predict(x_valid)
    """
    def __init__(self) -> None:
        pass

    def fit(self, x, y):
        """
            Fit a GDA model to training set given x and y

            :param x: Training set of size (m,n)
            :param y: Labels of size (m,)
        """
        m, n = x.shape;
        phi = np.sum(y) / m;
        mu_0 = np.dot(x.T, (1-y)) / np.sum(1-y);
        mu_1 = np.dot(x.T, y) / np.sum(y);

        # Reshape y to compute pairwise product with mu
        y_reshaped = np.reshape(y, (m, -1))

        # Matrix comprises mu_0 and mu_1 based on the value of y. Shape(m, n)
        mu_x = y_reshaped * mu_1 + (1 - y_reshaped) * mu_0

        x_centered = x - mu_x

        sigma = np.dot(x_centered.T, x_centered) / m
        sigma_inv = np.linalg.inv(sigma)

        theta = np.dot(sigma_inv, (mu_1 - mu_0));
        theta_0 = 1 / 2 * mu_0 @ sigma_inv @ mu_0 - 1 / 2 * mu_1 @ sigma_inv @ mu_1 - np.log((1 - phi) / phi)

        self.theta = np.insert(theta, 0, theta_0);

    def predict(self, x):
        """Predict given data using fitted model

        :param x: Given datset of size (m,n)
        """
        return np.dot(add_intercept(x), self.theta) >= 0
        


if __name__ == "__main__":
    ds_training_set_path = 'data/ds1_train.csv'
    ds_valid_set_path = 'data/ds1_valid.csv'

    x_train, y_train = load_dataset(ds_training_set_path)
    x_valid, y_valid = load_dataset(ds_valid_set_path)
    
    gda = GDA();
    gda.fit(x_train, y_train);
    pred = gda.predict(x_valid);
    acc = np.sum(np.int32(pred) == np.int32(y_valid))/y_valid.shape[0];
    print(f'accuracy: {acc}')


