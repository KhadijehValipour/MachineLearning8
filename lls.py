import numpy as np

class LinearLeastSquare :

    def __init__(self):

        self.W = None

    def fit(self , X_train , Y_train):
        # train 

        self.W = np.matmul(np.matmul(np.linalg.inv( X_train.T @ X_train) , X_train.T) , Y_train )

    def predict(self , X_test) :

        Y_pred = np.matmul(X_test,self.W)
        return Y_pred
    
    
    def evaluate(self , X_test , Y_test , metric ):

        Y_pred = self.predict(X_test)
        error = Y_test - Y_pred

        if metric == "mae" :
            loss = np.sum(np.abs(error)) / len(Y_test)

        elif  metric == "mse" :
            loss = np.sum((error) ** 2) / len(Y_test)


        return loss
