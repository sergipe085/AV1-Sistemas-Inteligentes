import time

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

def qualificate_major(y_prev):
	maximo = 0
	maximo_index = -1
	for i in range(len(y_prev)):
		if (y_prev[i] > maximo):
			maximo = y_prev[i]
			maximo_index = i

	return mapper[maximo_index]

def qualificate_minor(y_prev):
	minimo = 100000000.0
	minimo_index = -1
	for i in range(len(y_prev)):
		if (y_prev[i] < minimo):
			minimo = y_prev[i]
			minimo_index = i

	return mapper[minimo_index]

def qualificate_index(y_prev):
	minor_distance = 1000000.0
	minor_index = -1
	for i in range(len(y_prev)):
		distance = 1 - y_prev[i]
		if (distance < minor_distance):
			minor_distance = distance
			minor_index = i

	return minor_index


def get_accuracy(y_prev_array, Yteste):
	correct_counter = 0
	total = 10000
	accuracy_accumulator = 0.0
	for i in range(len(y_prev_array)):
		expected = qualificate(Yteste[i])
		result = qualificate(y_prev_array[i])
		if (expected == result):
			correct_counter += 1

	accuracy = (correct_counter/total) * 100
	return accuracy

def execution_time(_function):
	start_time = time.time()

	response = _function()

	finish_time = time.time()

	total_time = finish_time - start_time
	return response, total_time

def get_class_label(y_prev, return_label=True):
    max_value = 0.0
    min_index = -1
    for i in range(len(y_prev)):
        value = y_prev[i]
        if value > max_value:
            max_value = value
            min_index = i

    if return_label:
        return mapper[min_index]
    else:
        return min_index