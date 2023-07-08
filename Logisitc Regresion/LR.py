import matplotlib.pyplot as plt
import numpy as np
import util as util

class LinearModel(object):
    def __init__(self, 
                 step_size, 
                 max_iter, 
                 theta_0=None, 
                 verbose = True) -> None:
        self.theta = theta_0;
        self.step_size = step_size;
        self.max_iter = max_iter;
        self.verbose = verbose;
        pass

    def fit(self, x, y):
        raise NotImplementedError("Subclass of Linearmodel should implement fit function");
        pass

    def predict(self, x):
        raise NotImplementedError("Subclass of Linearmodel should implement predict function");
        pass


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.1, max_iter=1000,theta_0=None, verbose=True):
        super().__init__(step_size, max_iter, theta_0, verbose);
        
    
    def h_theta(self, x):
        return 1/(1+np.exp(-np.dot(x, self.theta)) + 1e-6);

    def fit(self, x, y):
        """Make a prediction given new inputs x.

        :param x: Inputs of shape (m, n).
        :param y: Labels of shape (m,)
        """
        for i in range(self.max_iter):
            m,n = x.shape;
            
            if self.theta is None:
                self.theta = np.ones(n);

            loss = (y-self.h_theta(x));
            self.theta = self.theta + self.step_size * np.dot((loss.T),x);
            acc = np.sum(y.astype("int32")==(self.h_theta(x)>0.5).astype("int32")) / m;
            if self.verbose is True:
                print(('\n' + '%10s' + '%10g'*3) %('Iter', i, acc, np.mean(loss)));

    def predict(self, x):
        """Make a prediction given new inputs x.

        :param x: Inputs of shape (m, n).
        :return:  Outputs of shape (m,).
        """

        return x @ self.theta >= 0

ds1_training_set_path = 'data/ds1_train.csv'
ds1_valid_set_path = 'data/ds1_valid.csv'
ds2_training_set_path = 'data/ds2_train.csv'
ds2_valid_set_path = 'data/ds2_valid.csv'

if __name__=="__main__":
    x_train, y_train = util.load_dataset(ds1_training_set_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(ds1_valid_set_path, add_intercept=True)
    x_train = (x_train - np.min(x_train)) / np.max(x_train);
    LR = LogisticRegression();
    LR.fit(x_train, y_train);