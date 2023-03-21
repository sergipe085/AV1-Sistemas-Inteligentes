import numpy as np 
import matplotlib.pyplot as plt
from models import mmq, mmq_regularizado
from utils import get_accuracy, qualificate

Data = np . loadtxt ("EMG.csv", delimiter =",") 
Rotulos = np . loadtxt ("Rotulos.csv", delimiter =",")
RODADAS = 100

mapper = ["Neutro", "Sorriso", "Aberto", "Surpreso", "Grumpy"]

def check_results(y_prev_array, treino, Yteste):
	accuracy = get_accuracy(y_prev_array, Yteste)
	print(f"TREINO {treino}; ACCURACY: {accuracy:.10f}%")

def get_data():
	seed = np . random . permutation ( Data . shape [0]) 
	X = Data [ seed ,:] 
	Y = Rotulos [ seed ,:] 

	X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

	# treino / teste : 14 
	Xtreino = X [0: int( X. shape [0]*.8) ,:] 
	Ytreino = Y [0: int( X. shape [0]*.8) ,:] 
	
	Xteste = X[ int (X. shape [0]*.8) : ,:] 
	Yteste = Y[ int (X. shape [0]*.8) : ,:] 

	return Xtreino, Ytreino, Xteste, Yteste

## DUVIDA: mudar o lambda nao muda a acuracia, pq?
mmq_regularizado_model = mmq_regularizado()
Xtreino, Ytreino, Xteste, Yteste = get_data()
mmq_regularizado_model.find_lambda(Xtreino, Ytreino, Xteste, Yteste)

for i in range ( RODADAS ):
	Xtreino, Ytreino, Xteste, Yteste = get_data()

	## MMQ ##

	# estimacao do modelo (treino)
	mmq_model = mmq()
	mmq_model.estimate(Xtreino, Ytreino)

	# teste do modelo
	y_prev_array = mmq_model.execute(Xteste)
	check_results(y_prev_array, i, Yteste)

	## MMQ REGULARIZADO ##
	mmq_regularizado_model.estimate(Xtreino, Ytreino)
	y_prev_array = mmq_regularizado_model.execute(Xteste)
	check_results(y_prev_array, i, Yteste)

	## Naive Bayes ##

	## KNN ##

	## DMC ##	
