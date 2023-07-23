from typing import Any
import numpy as np
from copy import deepcopy
import util as util
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.utils import shuffle

#hyperparameters
LEARNING_RATE = 0.1;
L2_REGULARIZATION = 0.0001;

class Layer(object):
    """Densly connected layer for DNN
    
    Example usage:
        > layer = Layer(in_dim = 10, out_dim = 20, use_act = False);
    """
    def __init__(self, in_dim, out_dim, use_act = True) -> None:
        self.__w = np.random.randn(in_dim, out_dim)*0.1;
        self.use_act = use_act;
        pass

    def __calc(self, x):
        """Forward propagation, it also stores gradients for backpropagation
        
        :param x: Input of shape (m,n)
        """
        z = np.dot(x,self.__w);
        if self.use_act is True:
            h = self.act(z);
        else:
            h = z;
        if self.use_act is False:
            self.__dw = x.T;
            self.__dx = self.__w.T;
        else:
            t = h > 0;
            self.__t = t.astype("float64");
            self.__dw = x.T;
            self.__dx = self.__w.T;
        return h;

    def act(self, inp):
        """Activation function which is ReLU
        
        :param inp: Input of size (m,n)
        """
        return np.maximum(inp, np.zeros_like(inp));

    def backprop(self, gradient):
        """Backpropagation through this layer
        
        :param gradient: gradient from previous layer, it could of various sizes!
        """
        if self.use_act is False:
            self.__w -= LEARNING_RATE * np.dot(self.__dw, gradient);
            return np.dot(gradient, self.__dx);
        self.__w -= LEARNING_RATE * np.dot(self.__dw, np.multiply(gradient, self.__t)) - self.__w * L2_REGULARIZATION;
        return np.dot(np.multiply(gradient, self.__t), self.__dx);

    def __call__(self, x) -> Any:
        """Forward propagation call
        
        :param x: Input of size (m,n)
        """
        return self.__calc(x);
        pass

class NN():
    """Class Neural Network (NN) which could be used to utilize densly connected layers for training
    
    Example usage:
        > nn = NN(layers_list, steps = 100, verbose = True, batch_size = 10)
        > nn. fit(x_train, y_train)
        > nn.predict(x_eval)
    """
    def __init__(self, layers : list, steps = 100, verbose = False, batch_size = 2) -> None:
        self.layers = layers;
        self.steps = steps;
        self.verbose = verbose;
        self.batch_size = batch_size;
        pass
    
    def __forward(self, x):
        """Forward propagation through all the layers
        
        :param x: Input of size (m,n)
        """
        out = deepcopy(x);
        for l in self.layers:
            out = l(out);
        return out;

    def __backprop(self, g):
        """Backpropagate through all the layers
        
        :param x: initial gradients
        """
        for i in range(len(self.layers)-1, -1, -1):
            g = self.layers[i].backprop(g);

    def act(self, inp):
        """Activation function for last layer which is Sigmoid
        
        :param inp: Input of size (m,1)
        """
        return 1 /(1+np.exp(-inp) + 1e-6);

    def read_batch(self, x, y, b):
        """Read and return a batch of data based on given batch size
        
        :param x: input of size (m,n)
        :param y: label targets of size (m,1)
        :param b: current batch iteration
        """
        if (b+1)*self.batch_size < x.shape[0]:
            return x[b*self.batch_size:(b+1)*self.batch_size], y[b*self.batch_size:(b+1)*self.batch_size]
        return x[b*self.batch_size:], y[b*self.batch_size:]
    
    def fit(self, x, y):
        """Learn hypothesis h(x) through iteration between forward and backward propagation
        
        :param x: input of size (m,n)
        :param y: label targets of size (m,1)
        """
        m,n = x.shape;
        batch_steps = int(m/self.batch_size);
        for i in range(self.steps):
            epoch_loss = [];
            epoch_acc = [];
            for b in range(batch_steps):
                batch_x, batch_y = self.read_batch(x,y,b);
                self.__bs = batch_x.shape[0];
                out = self.__forward(batch_x);
                out = self.act(out);
                loss = -(1/self.batch_size)*(np.log(out+1e-4)*batch_y.reshape(-1, 1) + (1-batch_y.reshape(-1, 1))*(np.log(1-out+1e-4)));
                g = -(1/self.batch_size)*((np.array(batch_y).reshape(out.shape[0], 1) - out));
                self.__backprop(g);
                acc = np.sum((batch_y.reshape(-1, 1).astype("int32")==(out>0.5).astype("int32"))) / self.batch_size;
                epoch_loss.append(np.mean(loss));
                epoch_acc.append(acc);
            if self.verbose is True:
                print(('\n' +  '%10s'*3)%('Epoch', 'Loss', 'Acc'));
                print(('\n' +  '%10g'*3)%(i, np.mean(epoch_loss),  np.mean(epoch_acc)));

        pass

    def predict(self, x):
        """Predict using learned parameters on the given input
        
        :param x: input of size (m,n)
        """

        out = self.__forward(x);
        return self.act(out);



if __name__=="__main__":

    ds_training_set_path = 'data/ds2_train.csv'
    ds_valid_set_path = 'data/ds2_valid.csv'

    x_train, y_train = util.load_dataset(ds_training_set_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(ds_valid_set_path, add_intercept=True)

    n_x = Normalizer();
    x_train = n_x.fit_transform(x_train);
    x_train, y_train = shuffle(x_train, y_train, random_state=42);

    layers = [Layer(3,100), Layer(100,5), Layer(5,1, use_act=False)];
    nn = NN(layers, steps=10, verbose=True, batch_size=8);
    nn.fit(x_train, y_train);
    y_pred = nn.predict(x_valid);
    acc = np.sum((y_valid.reshape(-1, 1).astype("int32")==(y_pred>0.5).astype("int32"))) / x_valid.shape[0];
    print('Acc on valid: %10f' %(acc));
