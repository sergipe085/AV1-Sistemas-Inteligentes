import numpy as np 
import matplotlib.pyplot as plt
from models import mmq, mmq_regularizado
from utils import get_accuracy

Data = np . loadtxt ("EMG.csv", delimiter =",") 
Rotulos = np . loadtxt ("Rotulos.csv", delimiter =",")
RODADAS = 100

def check_results(y_prev_array, rodada, Yteste, modelo):
	accuracy = get_accuracy(y_prev_array, Yteste)
	print(f"{modelo} - RODADA: {rodada}; ACCURACY: {accuracy:.10f}%")


def print_result(modelo_nome, acuracias):
	acuracia_total = 0.0
	for ac in acuracias:
		acuracia_total += ac

	media = acuracia_total / len(acuracias)
	menor = min(acuracias)
	maior = max(acuracias)
	print(f"{modelo_nome} - Media: {media:.2f}%, Menor: {menor:.2f}%, Maior: {maior:.2f}%")

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

def execute_mmq():
	print("Iniciando MMQ Ordinario...")

	acuracias = []

	for i in range(RODADAS):
		Xtreino, Ytreino, Xteste, Yteste = get_data()

		# estimacao do modelo (treino)
		mmq_model = mmq()
		mmq_model.estimate(Xtreino, Ytreino)

		# teste do modelo
		y_prev_array = mmq_model.execute(Xteste)
		acuracia = get_accuracy(y_prev_array, Yteste)
		acuracias.append(acuracia)

	print_result("MMQ Ordinario", acuracias)

def execute_mmq_regularizado():
	print("Iniciando MMQ Regularizado...")

	mmq_regularizado_model = mmq_regularizado()

	Xtreino, Ytreino, Xteste, Yteste = get_data()
	## DUVIDA: mudar o lambda nao muda a acuracia, pq?
	mmq_regularizado_model.find_lambda(Xtreino, Ytreino, Xteste, Yteste)

	acuracias = []

	for i in range(RODADAS):
		Xtreino, Ytreino, Xteste, Yteste = get_data()
		mmq_regularizado_model.estimate(Xtreino, Ytreino)
		y_prev_array = mmq_regularizado_model.execute(Xteste)

		acuracia = get_accuracy(y_prev_array, Yteste)
		acuracias.append(acuracia)
		
	print_result("MMQ Regularizado", acuracias)

## MMQ ##

execute_mmq()

## MMQ REGULARIZADO ##

execute_mmq_regularizado()

## Naive Bayes ##

## KNN ##

## DMC ##	
