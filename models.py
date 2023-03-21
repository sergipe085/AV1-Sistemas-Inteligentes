
import numpy as np
from utils import get_accuracy

class mmq:
    def __init__(self):
        pass

    def estimate(self, x_treino, y_treino):
        X = x_treino
        Y = y_treino
        self.B = np.linalg.inv(X.T@X)@X.T@Y

    def execute(self, x):
        y_prev = x@self.B
        return y_prev

class mmq_regularizado:
    def __init__(self):
        pass

    def estimate_with_lambda(self, x_treino, y_treino, _lbda):
        X = x_treino
        Y = y_treino

        self.W = np.linalg.inv(X.T@X + _lbda*np.identity(3))@X.T@Y
    
    def estimate(self, x_treino, y_treino):
        self.estimate_with_lambda(x_treino, y_treino, self.lbda)

    def execute(self, x):
        y_prev = x@self.W
        return y_prev
    
    def find_lambda(self, x_treino, y_treino, x_teste, y_teste):
        i = 0.0
        max_accuracy = 0.0
        max_lambda = 0.0
        while i <= 1.0:
            
            self.estimate_with_lambda(x_treino, y_treino, i)
            y_prev = self.execute(x_teste)

            accuracy = get_accuracy(y_prev, y_teste)

            if accuracy > max_accuracy:
                max_accuracy = accuracy
                max_lambda = i

            i+=0.1

        self.lbda = max_lambda
	
