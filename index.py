import numpy as np 
import matplotlib.pyplot as plt

Data = np . loadtxt ("EMG.csv", delimiter =",") 
Rotulos = np . loadtxt ("Rotulos.csv", delimiter =",")
RODADAS = 100

mapper = ["Neutro", "Sorriso", "Aberto", "Surpreso", "Grumpy"]

def qualificate(y_prev):
	minor_distance = 1000000.0
	minor_index = -1
	for i in range(len(y_prev)):
		distance = 1 - y_prev[i]
		if (distance < minor_distance):
			minor_distance = distance
			minor_index = i

	return mapper[minor_index]

def check_results(y_prev_array, treino):
	y_prev_array = Xteste@B;	
	correct_counter = 0
	total = 10000
	for i in range(len(y_prev_array)):
		expected = qualificate(Yteste[i])
		result = qualificate(y_prev_array[i])
		#print(f"EXPECTED: {expected} GET: {result}")
		if (expected == result):
			correct_counter += 1

	accuracy = (correct_counter/total) * 100
	print(f"TREINO {treino} = ACCURACY: {accuracy:.2f}%")

for i in range ( RODADAS ):
	seed = np . random . permutation ( Data . shape [0]) 
	X = Data [ seed ,:] 
	Y = Rotulos [ seed ,:] 

	X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

	# treino / teste : 14 
	Xtreino = X [0: int( X. shape [0]*.8) ,:] 
	Ytreino = Y [0: int( X. shape [0]*.8) ,:] 
	
	Xteste = X[ int (X. shape [0]*.8) : ,:] 
	Yteste = Y[ int (X. shape [0]*.8) : ,:] 

	## MMQ ##

	# estimacao do modelo (treino)
	B = np.linalg.inv(Xtreino.T@Xtreino)@Xtreino.T@Ytreino

	y_prev_array = Xteste@B;	
	check_results(y_prev_array, i)

	## MMQ regularizado ##

	## Naive Bayes ##

	## KNN ##

	## DMC ##
