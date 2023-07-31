import numpy as np
import numpy.linalg as inv

class LinearLeastSquare :
    def __init__(self):
        self.w = None

    def fit(self , X_train , Y_train):
        # train 
        self.w = inv(X_train.T @ X_train) @ X_train.T @ Y_train

    def predict(self , X_test) :
        Y_pred = X_test @ self.w
        return Y_pred
    
    def evaluate(self , X_test , Y_test):
        pass
