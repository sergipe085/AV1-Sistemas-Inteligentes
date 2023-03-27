import numpy as np 
import matplotlib.pyplot as plt
from models import mmq, mmq_regularizado, naive_bayes, naive_bayes_pooled, naive_bayes_friedman
from utils import get_accuracy, execution_time, qualificate_minor, qualificate, qualificate_index

Data = np . loadtxt ("EMG.csv", delimiter =",") 
Rotulos = np . loadtxt ("Rotulos.csv", delimiter =",")
RODADAS = 10

def print_result(modelo_nome, acuracias, tempo_previsao, tempo_execucao):
	a1, a2, a3, a4 = get_results_for_values(acuracias)
	tp1, tp2, tp3, tp4 = get_results_for_values(tempo_previsao)
	te1, te2, te3, te4 = get_results_for_values(tempo_execucao)

	print()
	print()
	print(f"Resultado - {modelo_nome}")
	print(f"Acuracia - Media: {a1:.2f}%, Menor: {a2:.2f}%, Maior: {a3:.2f}%, Desvio Padrao: {a4:.2f}")
	print(f"Tempo previsao - Media: {tp1:.2f} ms, Menor: {tp2:.2f} ms, Maior: {tp3:.2f} ms, Desvio Padrao: {tp4:.2f}")
	print(f"Tempo execucao - Media: {te1:.2f} ms, Menor: {te2:.2f} ms, Maior: {te3:.2f} ms, Desvio Padrao: {te4:.2f}")
	print()

def print_progress(progress, maxv):
	print("\r", end="")
	p = progress*100/maxv
	print(f"Progresso: {p:.2f} %", end = "")

def get_results_for_values(values):
	value_total = 0.0
	for v in values:
		value_total += v

	media = value_total / len(values)
	menor = min(values)
	maior = max(values)
	desvio_padrao = np.std(values)

	return media, menor, maior, desvio_padrao

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
	tempos_estimacao = []
	tempos_execucao = []
	print_progress(0, RODADAS)

	for i in range(RODADAS):
		Xtreino, Ytreino, Xteste, Yteste = get_data()

		# estimacao do modelo (treino)
		mmq_model = mmq()
		a, _time_previsao = execution_time(lambda:mmq_model.estimate(Xtreino, Ytreino))
		tempos_estimacao.append(_time_previsao * 1000)

		# teste do modelo
		y_prev_array, _time_execucao = execution_time(lambda:mmq_model.execute(Xteste))
		tempos_execucao.append(_time_execucao * 1000)

		acuracia = get_accuracy(y_prev_array, Yteste)
		acuracias.append(acuracia)

		print_progress(i, RODADAS)

	print_result("MMQ Ordinario", acuracias, tempos_estimacao, tempos_execucao)

def execute_mmq_regularizado():
	print("Iniciando MMQ Regularizado...")

	mmq_regularizado_model = mmq_regularizado()

	Xtreino, Ytreino, Xteste, Yteste = get_data()
	## DUVIDA: mudar o lambda nao muda a acuracia, pq?
	mmq_regularizado_model.find_lambda(Xtreino, Ytreino, Xteste, Yteste)

	acuracias = []
	tempos_estimacao = []
	tempos_execucao = []

	for i in range(RODADAS):
		Xtreino, Ytreino, Xteste, Yteste = get_data()

		a, _time_previsao = execution_time(lambda: mmq_regularizado_model.estimate(Xtreino, Ytreino))

		y_prev_array, _time_execucao = execution_time(lambda: mmq_regularizado_model.execute(Xteste))

		acuracia = get_accuracy(y_prev_array, Yteste)
		acuracias.append(acuracia)

		tempos_estimacao.append(_time_previsao * 1000)
		tempos_execucao.append(_time_execucao * 1000)
		
	print_result(f"MMQ Regularizado", acuracias, tempos_estimacao, tempos_execucao)

def execute_naive_bayes(Regularizado = False):
	nome_modelo = "Naive Bayes"
	if (Regularizado): nome_modelo = nome_modelo + " Regularizado"
	print(f"Iniciando {nome_modelo}...")
	acuracias = []
	tempos_estimacao = []
	tempos_execucao = []
	rodada_atual = 0
	for i in range(RODADAS):
		print_progress(rodada_atual, RODADAS)

		Xtreino, Ytreino, Xteste, Yteste = get_data_withouth_perceptron()

		nb = naive_bayes()
		_, tempo_estimacao = execution_time(lambda: nb.estimate(Xtreino, Ytreino))

		count = 0
		for i in range(len(Xteste)):
			y_prev_array, tempo_execucao = execution_time(lambda: nb.execute(Xteste[i], Regularizado))

			getted = qualificate_minor(y_prev_array)
			expected = qualificate(Yteste[i])

			if getted == expected:
				count+=1

		acuracia = (count/len(Xteste)) * 100

		acuracias.append(acuracia)
		tempos_estimacao.append(tempo_estimacao * 1000)
		tempos_execucao.append(tempo_execucao * 1000)

		rodada_atual += 1

	print_result(nome_modelo, acuracias, tempos_estimacao, tempos_execucao)
            
def execute_naive_bayes_pooled():
	nome_modelo = "Naive Bayes Pooled"
	print(f"Iniciando {nome_modelo}...")
	acuracias = []
	tempos_estimacao = []
	tempos_execucao = []
	rodada_atual = 0
	for i in range(RODADAS):
		print_progress(rodada_atual, RODADAS)

		Xtreino, Ytreino, Xteste, Yteste = get_data_withouth_perceptron()

		nb_pooled = naive_bayes_pooled()
		_, tempo_estimacao = execution_time(lambda: nb_pooled.estimate(Xtreino, Ytreino))

		count = 0
		for i in range(len(Xteste)):
			y_prev_array, tempo_execucao = execution_time(lambda: nb_pooled.execute_pooled(Xteste[i]))

			getted = qualificate_minor(y_prev_array)
			expected = qualificate(Yteste[i])

			if getted == expected:
				count+=1

		acuracia = (count/len(Xteste)) * 100

		acuracias.append(acuracia)
		tempos_estimacao.append(tempo_estimacao * 1000)
		tempos_execucao.append(tempo_execucao * 1000)

		rodada_atual += 1

	print_result(nome_modelo, acuracias, tempos_estimacao, tempos_execucao)

def execute_naive_bayes_fridman():
	nome_modelo = "Naive Bayes Friedman"
	print(f"Iniciando {nome_modelo}...")
	acuracias = []
	tempos_estimacao = []
	tempos_execucao = []
	rodada_atual = 0
	for i in range(RODADAS):
		print_progress(rodada_atual, RODADAS)

		Xtreino, Ytreino, Xteste, Yteste = get_data_withouth_perceptron()

		nb_fridman = naive_bayes_friedman(0.2)
		_, tempo_estimacao = execution_time(lambda: nb_fridman.estimate(Xtreino, Ytreino))

		count = 0
		for i in range(len(Xteste)):
			y_prev_array, tempo_execucao = execution_time(lambda: nb_fridman.execute_friedman(Xteste[i]))

			getted = qualificate_minor(y_prev_array)
			expected = qualificate(Yteste[i])

			if getted == expected:
				count+=1

		acuracia = (count/len(Xteste)) * 100

		acuracias.append(acuracia)
		tempos_estimacao.append(tempo_estimacao * 1000)
		tempos_execucao.append(tempo_execucao * 1000)

		rodada_atual += 1

	print_result(nome_modelo, acuracias, tempos_estimacao, tempos_execucao)

execute_mmq()
execute_mmq_regularizado()
# execute_naive_bayes(Regularizado=True)
execute_naive_bayes()
execute_naive_bayes_pooled()
execute_naive_bayes_fridman()

# plot_dados()
