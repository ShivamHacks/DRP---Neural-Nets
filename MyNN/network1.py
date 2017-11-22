import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle

# possibly change talk topic to:
# Neural Networks as Polynomial Regressors
# Using Neural Networks as a Universal Approximator

class Layer(object):

	def __init__(self, num_inputs, num_neurons):
		self.weights = np.zeros((num_neurons, num_inputs))
		self.biases = np.zeros((num_neurons, ))
		#print self.weights
		#self.biases = np.zeros((num_neurons, ))
		#self.weights = np.random.randint(10, 15, [num_neurons, num_inputs]) # make randn
		#self.weights = np.random.randn(num_neurons, num_inputs)
		#self.biases = np.random.randn(num_neurons, )

	def pass_through(self, input):
		#return np.dot(self.weights, input) + self.biases
		return np.dot(input, self.weights.T) + self.biases

	def pass_through_no_activation(self, input):
		return np.dot(input, self.weights.T) + self.biases
	def pass_through_activation(self, input):
		return activation(np.dot(input, self.weights.T) + self.biases)
	def pass_through_activation_derivative(self, input):
		return activation_derivative(np.dot(input, self.weights.T) + self.biases)

class Network(object):

	def __init__(self, structure):
		"""
		structure[0] = number of inputs
		structure[len(structure) - 1] = number of outputs
		"""
		self.structure = structure
		self.numLayers = len(structure) - 1
		print "Number of layers: " + str(self.numLayers)
		self.layers = []
		print "Network Mapping Layer By Layer: "
		for i in range(1, len(structure)):
			print str(structure[i -1]) + " to " + str(structure[i])
			self.layers.append(Layer(structure[i - 1], structure[i]))

	def evaluate(self, input):
		#input = activation(self.layers[0].pass_through(input))
		for i in range(0, self.numLayers - 1):
			input = activation(self.layers[i].pass_through(input))
		return self.layers[self.numLayers - 1].pass_through(input) # final output is not sigmoidal

	def train_one(self, trainX, trainY, learningRate):

		print "Evaluation for: X" + str(trainX)
		print "\tPredicted: " + str(self.evaluate(trainX))
		print "\tActual: " + str(trainY)
		print "\tDifference: " + str(self.evaluate(trainX) - trainY)
		#print "Prediction for: X" + str(trainX) + " and Y" + str(trainY) + " --> " + str(self.evaluate(trainX))
		#print "Error for: X" + str(trainX) + " and Y" + str(trainY) + " --> " + str(self.evaluate(trainX) - trainY)
		#print "Difference: " + str(self.evaluate(trainX) - trainY)

		# issue is structuring all the matrices

		# get activations
		"""layer_outputs = []
		#activations = []
		input = trainX
		for i in range(0, self.numLayers):
			layer_outputs.append(input)
			input = self.layers[i + 1].pass_through(input)
			#if i == (self.numLayers - 1): activations.append(input)
			#else: activations.append(activation(input))
		"""

		#print layer_outputs
		#print activations

		#layer_outputs.append(self.layers[self.numLayers - 1].pass_through(activations[self.numLayers - 2]))
		#activations.append(layer_outputs[len(layer_outputs) - 1])
		
		# get errors
		#errors = [None] * self.numLayers
		#errors[self.numLayers - 1] = (self.evaluate(trainX) - trainY) * trainX

		# layer inputs
		layer_inputs = []
		layer_outputs = []
		input = trainX
		for i in range(0, self.numLayers): # don't need last output
			layer_inputs.append(input)
			input = activation(self.layers[i].pass_through(input))
			layer_outputs.append(input)

		
	
		error = self.evaluate(trainX) - trainY # output error if last layer not sigmoidal, if not could do layer_outputs[-1]
		for i in reversed(range(0, self.numLayers)):
			#print error.shape
			#print np.array(activations[i - 1]).shape
			#print np.outer(error, activations[i - 1]).shape
			#print self.layers[i].weights.shape

			#print error.shape
			#print np.array(activations[i - 1]).reshape(3, 1)
			#print np.array(error).shape
			#print self.structure[i + 1]
			print np.array(error).reshape(1, self.structure[i+1]).shape
			print np.array(layer_outputs[i - 1]).reshape(self.structure[i], 1).T.shape
			print "Layer: " + str(i+1)
			print "Current Weight Shape: " + str(self.layers[i].weights.shape)
			print "Current Layer Error Shape: " + str(error.shape)
			print "Backpropogating Error Shape: " + str(np.outer(error, layer_outputs[i-1]).shape)

			#print "Next Layer Error Shape: " + str(np.outer(error, layer_inputs[i - 1]).shape)
			#print "Prev Layer Input Shape: " + str(np.array(layer_inputs[i - 1]).shape)

			self.layers[i].weights -= learningRate * np.outer(error, layer_outputs[i - 1])
			self.layers[i].biases -= learningRate * error
			#print np.multiply(error, activation_derivative(layer_outputs[i - 1])).shape

			error = np.multiply(np.dot(self.layers[i].weights.T , error), activation_derivative(layer_inputs[i])) # error should be same shape as weights of prev layer
			# ^ this is right according to all sources
			#print "SHAPE: " + str(error.shape)

			#error = np.dot(np.multiply(error, activation_derivative(layer_outputs[i - 1])), self.layers[i].weights)


		

			#np.dot(self.layers[i].weights.T, error) * activation_derivative(layer_outputs[i])
		#self.layers[0].weights -= learningRate * np.outer(error, trainX)
		#self.layers[0].biases -= learningRate * error

		# propogate error backwards
		"""
		for i in range(1, self.numLayers):
			self.layers[i].weights -= learningRate * np.outer(errors[i], activations[i-1])
			self.layers[i].biases -= learningRate * errors[i]
			#print learningRate * np.dot(errors[i], activation(layer_outputs[i-1]).T)
		self.layers[0].weights -= learningRate * np.outer(errors[0], trainX)
		self.layers[0].biases -= learningRate * errors[0]
		"""

	def train(self):
		# train on quadratic

		# num data points

		dataRange = 2
		step = 0.1
		numDataPoints = int(dataRange / step)
		numEpochs = 7

		"""x = []
		y = []
		for j in range( -int(dataRange/2) , int(dataRange/2), 1 ):
			#i = np.random.uniform(-1, 1)
			i = j
			x.append([i])
			y.append([i**2])"""
		
		x = np.arange(-1.0, 1.0, 0.1).reshape(numDataPoints, 1)
		y = x ** 2
		x, y = shuffle(x, y, random_state=6)

		#x = np.array(x, dtype=float)
		#y = np.array(y, dtype=float)
		#min_max_scaler = preprocessing.MinMaxScaler()
		#x = min_max_scaler.fit_transform(x)
		#y = min_max_scaler.fit_transform(y)
		for j in range(numEpochs): # numEpochs
			for k in range(numDataPoints):
				self.train_one(x[k], y[k], 0.01)



def activation(x):
	return x * (x > 0) #RelU
	#return input # todo
	#return 1.0/(1.0+np.exp(-input)) # last layer can't be sigmoidal because then values lie only between -1 and 1

def activation_derivative(x):
	#return input # todo
	return (x > 0) * 1 #RelU
	#return activation(input)*(1-activation(input)) # last layer can't be sigmoidal because then values lie only between -1 and 1


#myNN = Network([1, 3, 3, 1])
#myNN.train_one([2], [4], 0.1)


#myNN = Network(2, [3, 3, 2])
#print myNN.evaluate([1, 1])







myNN = Network([1, 3, 3, 1])
pred1 = myNN.evaluate(0.5)
myNN.train()
pred2 = myNN.evaluate(0.5)
print "before: " + str(pred1)
print "after : " + str(pred2)
print "actual: " + str(0.5**2)

#print activation(np.array([1, -2, -3, 4]))
#print activation_derivative(np.array([1, -2, -3, 4]))





#print [layer.weights for layer in myNN.layers]
#print myNN.evaluate([0, 1])
#myNN.train_one([0, 1], [1, 0], 0.1)