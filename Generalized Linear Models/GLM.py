import matplotlib.pyplot as plt
import numpy as np
import util as util
from sklearn.preprocessing import Normalizer, StandardScaler
import cv2


class GeneralizedLinearModel():
    """Generalized Linear Model for distribution following the exponential family.

    Example usage:
        > clf = GeneralizedLinearModel()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    GLM_BERNOULI, GLM_GUASSIAN = range(2);
    def __init__(self, step_size=0.1, max_iter=1000,theta_0=None, verbose=True,
                 glm_type = GLM_BERNOULI):
        self.acc_hist = [];
        self.loss_hist = [];
        self.max_iter = max_iter;
        self.theta = theta_0;
        self.verbose = verbose;
        self.step_size = step_size;
        self.type = glm_type;
        
    def h_theta(self, x):
        if self.type == self.GLM_BERNOULI:
            return 1 / (1+np.exp(-np.dot(x, self.theta)) + 1e-6);
        if self.type == self.GLM_GUASSIAN:
            return np.dot(x, self.theta);

    def fit(self, x, y):
        """Make a prediction given new inputs x.

        :param x: Inputs of shape (m, n).
        :param y: Labels of shape (m,)
        """
        for i in range(self.max_iter):
            m,n = x.shape;

            if self.theta is None:
                self.theta = np.random.random(n);

            self.theta = self.theta + 1/m * self.step_size * (y-self.h_theta(x)) @ x;
            loss = np.mean(np.abs((y-self.h_theta(x))));

            if self.type == self.GLM_BERNOULI:
                acc = np.sum(y.astype("int32")==(self.h_theta(x)>0.5).astype("int32")) / m;
                self.acc_hist.append(acc);
            
            self.loss_hist.append(loss);

            if self.verbose is True:
                if self.type == self.GLM_BERNOULI:
                    print(('\n' + '%10s' + '%10g'*3) %('Iter', i, acc, loss));
                if self.type == self.GLM_GUASSIAN:
                    print(('\n' + '%10s' + '%10g'*2) %('Iter', i, loss));

    def predict(self, x):
        """Make a prediction given new inputs x.

        :param x: Inputs of shape (m, n).
        :return:  Outputs of shape (m,).
        """
        if self.type == self.GLM_BERNOULI:
            return self.h_theta(x) > 0.5;
        else:
            return self.h_theta(x);



if __name__=="__main__":

    ds_training_set_path = 'data/ds2_train.csv'
    ds_valid_set_path = 'data/ds2_valid.csv'

    x_train, y_train = util.load_dataset(ds_training_set_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(ds_valid_set_path, add_intercept=True)
    n_x = Normalizer();
    n_y = StandardScaler();
    x_train = n_x.fit_transform(x_train);

    #Standardize labels for gaussian regression
    #y_train = n_y.fit_transform(np.expand_dims(y_train, axis = 1)).squeeze();

    x_valid = n_x.transform(x_valid);

    LR = GeneralizedLinearModel(
        max_iter=500, 
        step_size=2.0, 
        glm_type=GeneralizedLinearModel.GLM_BERNOULI, 
        verbose=True);
    
    LR.fit(x_train, y_train);

    out_valid = LR.predict(x_valid);
    acc_valid = np.sum(y_valid.astype("int32")==(out_valid).astype("int32")) / x_valid.shape[0];
    print("Accuracy on validation set: ", f'{acc_valid}');

    fig, ax = plt.subplots(1,2);
    ax[0].plot(LR.loss_hist);
    ax[0].set_title("Loss");
    ax[1].plot(LR.acc_hist);
    ax[1].set_title("Accuracy");
    plt.show();