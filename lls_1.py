import numpy as np
from numpy.linalg import inv 

class LinearLeastSquare:

    def __init__(self):
        self.W = None 

    def fit(self, X_train , Y_train):
        self.W = np.matmul(np.matmul(np.linalg.inv( np.matmul(X_train.T  , X_train) ) , X_train.T ) , Y_train )

    def predict(self , X_test):
        y_pred = X_test @ self.W
        return y_pred

    def evaluate(self , x_test , y_test , metric):
        y_pred = self.predict(x_test)
        if metric == "mae" :
            loss = np.sum(np.abs(y_test - y_pred)) / len(y_test)
        elif metric == "mse":
            loss = np.sum((y_test - y_pred) ** 2) / len(y_test)
        elif metric == "rmse" :
            loss =  np.sqrt(np.sum((y_test - y_pred) ** 2) / len(y_test))

        return loss 