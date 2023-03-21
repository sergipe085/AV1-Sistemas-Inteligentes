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

def get_accuracy(y_prev_array, Yteste):
	correct_counter = 0
	total = 10000
	for i in range(len(y_prev_array)):
		expected = qualificate(Yteste[i])
		result = qualificate(y_prev_array[i])
		if (expected == result):
			correct_counter += 1

	accuracy = (correct_counter/total) * 100
	return accuracy