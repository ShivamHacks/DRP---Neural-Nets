import numpy as np

class Layer(object):

	def __init__(self, num_inputs, num_neurons):
		self.weights = np.random.randint(1, 2, [num_neurons, num_inputs]) # make randn

	def pass_through(self, input):
		return np.dot(input, self.weights.T)

class Network(object):

	def __init__(self, num_inputs, structure):
		self.numLayers = len(structure)
		self.layers = [ Layer(num_inputs, structure[0]) ]
		for i in range(1, len(structure)):
			self.layers.append(Layer(structure[i - 1], structure[i]))

	def evaluate(self, input):
		input = activation(self.layers[0].pass_through(input))
		for i in range(1, len(self.layers)):
			input = activation(self.layers[i].pass_through(input))
		return input

	def train_one(self, trainX, trainY, learningRate):

		# get activations
		layer_outputs = []
		layer_outputs.append(self.layers[0].pass_through(trainX))
		for i in range(1, len(self.layers)):
			layer_outputs.append(self.layers[i].pass_through(activation(layer_outputs[i-1])))
		
		# get errors
		errors = [None] * self.numLayers
		errors[self.numLayers - 1] = (self.evaluate(trainX) - trainY) * activation_derivative(layer_outputs[self.numLayers - 1])
		for i in reversed(range(0, self.numLayers - 1)):
			errors[i] = np.dot(self.layers[i + 1].weights.T, errors[i + 1]) * activation_derivative(layer_outputs[i])



def activation(input):
	return input # todo

def activation_derivative(input):
	return input # todo

myNN = Network(2, [3, 3, 2])
print [layer.weights for layer in myNN.layers]
print myNN.evaluate([0, 1])
print myNN.train_one([0, 1], [1, 0], 0.1)