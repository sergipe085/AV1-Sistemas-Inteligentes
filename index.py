import numpy as np 
import matplotlib.pyplot as plt
from models import mmq, mmq_regularizado, naive_bayes
from utils import get_accuracy, execution_time, qualificate_minor, qualificate, qualificate_index

Data = np . loadtxt ("EMG.csv", delimiter =",") 
Rotulos = np . loadtxt ("Rotulos.csv", delimiter =",")
RODADAS = 100

def print_result(modelo_nome, acuracias, tempo_previsao, tempo_execucao):
	acuracia_total = 0.0
	for ac in acuracias:
		acuracia_total += ac

	media = acuracia_total / len(acuracias)
	menor = min(acuracias)
	maior = max(acuracias)
	print(f"{modelo_nome} - Media: {media:.2f}%, Menor: {menor:.2f}%, Maior: {maior:.2f}%, Tempo Treino: {tempo_previsao:.3f} ms, Tempo Execucao: {tempo_execucao:.3f} ms")

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

def get_data_withouth_perceptron():
	seed = np . random . permutation ( Data . shape [0]) 
	X = Data [ seed ,:] 
	Y = Rotulos [ seed ,:] 

	# treino / teste : 14 
	Xtreino = X [0: int( X. shape [0]*.8) ,:] 
	Ytreino = Y [0: int( X. shape [0]*.8) ,:] 
	
	Xteste = X[ int (X. shape [0]*.8) : ,:] 
	Yteste = Y[ int (X. shape [0]*.8) : ,:] 

	return Xtreino, Ytreino, Xteste, Yteste

def execute_mmq():
	print("Iniciando MMQ Ordinario...")

	acuracias = []
	tempo_estimacao_total = 0.0
	tempo_execucao_total = 0.0

	for i in range(RODADAS):
		Xtreino, Ytreino, Xteste, Yteste = get_data()

		# estimacao do modelo (treino)
		mmq_model = mmq()
		a, _time_previsao = execution_time(lambda:mmq_model.estimate(Xtreino, Ytreino))

		# teste do modelo
		y_prev_array, _time_execucao = execution_time(lambda:mmq_model.execute(Xteste))
		acuracia = get_accuracy(y_prev_array, Yteste)
		acuracias.append(acuracia)

		tempo_execucao_total += _time_execucao
		tempo_estimacao_total += _time_previsao

	tempo_ms_execucao = (tempo_execucao_total / 100) * 1000
	tempo_ms_estimacao = (tempo_estimacao_total / 100) * 1000

	print_result("MMQ Ordinario", acuracias, tempo_ms_estimacao, tempo_ms_execucao)

def execute_mmq_regularizado():
	print("Iniciando MMQ Regularizado...")

	mmq_regularizado_model = mmq_regularizado()

	Xtreino, Ytreino, Xteste, Yteste = get_data()
	## DUVIDA: mudar o lambda nao muda a acuracia, pq?
	mmq_regularizado_model.find_lambda(Xtreino, Ytreino, Xteste, Yteste)

	acuracias = []
	tempo_estimacao_total = 0.0
	tempo_execucao_total = 0.0

	for i in range(RODADAS):
		Xtreino, Ytreino, Xteste, Yteste = get_data()

		a, _time_previsao = execution_time(lambda: mmq_regularizado_model.estimate(Xtreino, Ytreino))

		y_prev_array, _time_execucao = execution_time(lambda: mmq_regularizado_model.execute(Xteste))

		acuracia = get_accuracy(y_prev_array, Yteste)
		acuracias.append(acuracia)

		tempo_execucao_total += _time_execucao
		tempo_estimacao_total += _time_previsao

	tempo_ms_execucao = (tempo_execucao_total / 100) * 1000
	tempo_ms_estimacao = (tempo_estimacao_total / 100) * 1000
		
	print_result(f"MMQ Regularizado", acuracias, tempo_ms_estimacao, tempo_ms_execucao)

def execute_naive_bayes():
	for i in range(1):

		Xtreino, Ytreino, Xteste, Yteste = get_data_withouth_perceptron()

		nb = naive_bayes()
		nb.estimate(Xtreino, Ytreino)

		count = 0
		for i in range(len(Xteste)):
			y_prev_array = nb.execute(Xteste[i])

			getted = qualificate_minor(y_prev_array)
			expected = qualificate(Yteste[i])

			if getted == expected:
				count+=1

		acuracia = (count/len(Xteste)) * 100

		print_result("Naive Bayes", [acuracia], 0, 0)
            
#execute_mmq()
#execute_mmq_regularizado()
#execute_naive_bayes()

def plot_dados():
	fig, ax = plt.subplots()

	color_mapper = ["blue", "orange", "yellow", "red", "green"]

	i = 0
	while i <= len(Data):
		label_index = int((i % 5000)/1000)

		color = color_mapper[label_index]

		ax.scatter(Data[i : i + 1000, 0], Data[i : i + 1000, 1], c=color)

		i += 1000

	ax.set_xlabel('Sensor 1')
	ax.set_ylabel('Sensor 2')

	plt.show()

plot_dados()
