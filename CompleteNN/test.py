import numpy as np

np.random.seed(8)
training_X = np.random.rand(100, 5)
training_Y = np.random.rand()


Network nn = new Network([2, 3, 3, 2])
nn.SGD(training_data=training_data, epochs=100, eta=0.01)
