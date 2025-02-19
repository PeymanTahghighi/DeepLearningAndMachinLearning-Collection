from typing import Any
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import Normalizer
from sklearn.utils import shuffle
import util as util

# Hyperparameters
LEARNING_RATE = 0.0001
L2_REGULARIZATION = 0.00001

class Layer:
    """Densely connected layer for a DNN."""
    def __init__(self, in_dim, out_dim, use_act=True) -> None:
        self.w = np.random.randn(out_dim, in_dim) * 0.1
        self.use_act = use_act

    def __calc(self, x):
        """Forward propagation, storing gradients for backpropagation."""
        z = np.dot(self.w, x)
        self.x = x
        if self.use_act:
            h = self.act(z)
            self.t = (h > 0).astype("float64")
        else:
            h = z
        self.dw = x.T
        self.dx = self.w.T
        return h

    def act(self, inp):
        """ReLU activation function."""
        return np.maximum(inp, 0)

    def backprop(self, gradient):
        """Backpropagation through this layer."""
        if not self.use_act:
            self.w -= LEARNING_RATE * np.dot(gradient, self.dw)
            return np.dot(self.dx, gradient)
        self.w -= LEARNING_RATE * (np.dot(gradient * self.t, self.dw) - self.w * L2_REGULARIZATION)
        return np.dot(self.dx, gradient * self.t)

    def __call__(self, x) -> Any:
        """Forward propagation call."""
        return self.__calc(x)

class NN:
    """Neural Network (NN) using densely connected layers."""
    def __init__(self, layers: list, steps=100, verbose=False, batch_size=2) -> None:
        self.layers = layers
        self.steps = steps
        self.verbose = verbose
        self.batch_size = batch_size
    
    def __forward(self, x):
        """Forward propagation through all layers."""
        out = deepcopy(x)
        for l in self.layers:
            out = l(out)
        return out

    def __backprop(self, g):
        """Backpropagation through all layers."""
        for i in range(len(self.layers) - 1, -1, -1):
            g = self.layers[i].backprop(g)

    def act(self, inp):
        """Sigmoid activation function for final layer."""
        return 1 / (1 + np.exp(-inp) + 1e-6)

    def read_batch(self, x, y, b):
        """Retrieve a batch of data."""
        return x[b * self.batch_size: (b + 1) * self.batch_size], y[b * self.batch_size: (b + 1) * self.batch_size]

    def fit(self, x, y):
        """Train the network with forward and backward propagation."""
        m, _ = x.shape
        batch_steps = m // self.batch_size
        for i in range(self.steps):
            epoch_loss, epoch_acc = [], []
            for b in range(batch_steps):
                batch_x, batch_y = self.read_batch(x, y, b)
                out = self.__forward(batch_x.T)
                out = self.act(out)
                loss = -(np.log(out.T + 1e-4) * batch_y.reshape(-1, 1) + (1 - batch_y.reshape(-1, 1)) * np.log(1 - out.T + 1e-4))
                g = -(batch_y.reshape(out.T.shape[0], 1) - out.T)
                self.__backprop(g.T)
                acc = np.mean(batch_y.reshape(-1, 1) == (out.T > 0.5))
                epoch_loss.append(np.mean(loss))
                epoch_acc.append(acc)
            if self.verbose:
                print(f"Epoch {i}: Loss {np.mean(epoch_loss):.4f}, Acc {np.mean(epoch_acc):.4f}")

    def predict(self, x):
        """Predict using the trained model."""
        return self.act(self.__forward(x))

if __name__ == "__main__":
    ds_training_set_path = 'data/ds2_train.csv'
    ds_valid_set_path = 'data/ds2_valid.csv'

    x_train, y_train = util.load_dataset(ds_training_set_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(ds_valid_set_path, add_intercept=True)

    n_x = Normalizer()
    x_train = n_x.fit_transform(x_train)
    x_train, y_train = shuffle(x_train, y_train, random_state=42)

    layers = [Layer(3, 100), Layer(100, 5), Layer(5, 1, use_act=False)]
    nn = NN(layers, steps=1000, verbose=True, batch_size=1)
    nn.fit(x_train, y_train)
    y_pred = nn.predict(x_valid)
    acc = np.mean(y_valid.reshape(-1, 1) == (y_pred > 0.5))
    print(f'Accuracy on validation set: {acc:.6f}')
