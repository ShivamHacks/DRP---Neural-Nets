import numpy as np

class Layer(object):

	def __init__(self, num_inputs, num_neurons):
		# self.weights = np.random.randint(1, 2, [num_neurons, num_inputs]) # make randn
		self.weights = np.random.randn(num_neurons, num_inputs)


	def pass_through(self, input):
		return np.dot(input, self.weights.T)

class Network(object):

	def __init__(self, num_inputs, structure):
		self.structure = structure
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
		errors[self.numLayers - 1] = (self.evaluate(trainX) - trainY) \
		* activation_derivative(layer_outputs[self.numLayers - 1])
		for i in reversed(range(0, self.numLayers - 1)):
			errors[i] = np.dot(self.layers[i + 1].weights.T, errors[i + 1]) \
			* activation_derivative(layer_outputs[i])

		# propogate error backwards
		for i in range(1, self.numLayers):
			self.layers[i].weights -= learningRate * np.outer(errors[i], activation(layer_outputs[i-1]))
			#print learningRate * np.dot(errors[i], activation(layer_outputs[i-1]).T)
		self.layers[0].weights -= learningRate * np.outer(errors[0], trainX)

	def train(self):
		# train on quadratic
		x = []
		y = []
		for i in range(100):
			x.append([i, i])
			y.append([i**2, i**2])
		x = np.array(x, dtype=float)
		y = np.array(y, dtype=float)
		for i in range(len(x)):
			self.train_one(x[i], y[i], 0.01)



def activation(input):
	#return input # todo
	return 1.0/(1.0+np.exp(-input))

def activation_derivative(input):
	#return input # todo
	return activation(input)*(1-activation(input))

#myNN = Network(2, [3, 3, 2])
myNN = Network(2, [3, 3, 2])
print myNN.evaluate([50, 50])
myNN.train()
print myNN.evaluate([20, 20])
#print [layer.weights for layer in myNN.layers]
#print myNN.evaluate([0, 1])
#myNN.train_one([0, 1], [1, 0], 0.1)