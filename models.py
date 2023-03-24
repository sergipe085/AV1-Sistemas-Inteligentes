
import numpy as np
from utils import get_accuracy, qualificate_index, qualificate, qualificate_minor, get_class_label

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

class naive_bayes:
    def estimate(self, x_treino, y_treino):
        x_treino_separado = [[], [], [], [], []]
        matrizes_de_covariancia = []
        mis = []

        # separar os dados para cada classe
        for i in range(len(x_treino)):
            x = x_treino[i]
            y = y_treino[i]
            y_index = qualificate_index(y)
            x_treino_separado[y_index].append(x)

        for xi in x_treino_separado:

            # estimar as matrizes de covariancia 
            matriz_de_covariancia = np.cov(np.array(xi).T)
            matrizes_de_covariancia.append(matriz_de_covariancia)

            # estimar os mis
            total = [0, 0]
            for i in xi:
                total[0] += i[0]
                total[1] += i[1]

            mis.append(np.array([total[0] / len(xi), total[1] / len(xi)]))
            bp = 1

        self.matrizes_de_covariancia = matrizes_de_covariancia
        self.mis = mis

    def execute(self, x, Regularizado):
        valores = []
        for i in range(0, 5):
            if (Regularizado == False):
                r = self.discriminante_normal(x, self.mis[i])
            else:
                r = self.discriminante_regularized(x, self.mis[i], self.matrizes_de_covariancia[i])
            valores.append(r)

        return valores

    def discriminante_normal(self, x, mi):        
        r = (x - mi).T @ (x - mi)
        return r
    
    def discriminante_regularized(self, x, mi, matriz_de_covariancia):
        reg_cov = np.eye(matriz_de_covariancia.shape[0]) * 1e-6
        mcv = matriz_de_covariancia + reg_cov
        mcv = np.diag(np.diag(mcv))
        r = (x - mi).T @ np.linalg.inv(mcv) @ (x - mi)
        return r
    
