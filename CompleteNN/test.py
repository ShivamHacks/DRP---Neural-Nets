import numpy as np
from basic_network import Network

np.random.seed(6)

nn = Network([1, 3, 3, 1])

def func_to_predict(x): return x ** 3

dataRange = [-1.0, 1]
step = 0.05
numDataPoints = int( (dataRange[1] - dataRange[0]) / step) + 1
learning_rate = 0.1
numEpochs = 100

x = np.arange(dataRange[0], dataRange[1] + step, step)
y = func_to_predict(x)

train_x = np.array(x).reshape(numDataPoints, 1)
train_y = np.array(y).reshape(numDataPoints, 1)

nn.SGD(training_data=zip(x, y), epochs=100, eta=0.1, mini_batch_size=numDataPoints/5, test_data=zip(x, y))

y_test = []
for i in range(len(train_x)):
	y_test.append(nn.feedforward(train_x[i]))

print len(y_test)
y_test = np.array(y_test)
print y_test
#print y_test
#y_test = np.array(myNN.evaluate_multiple(x)).reshape(1, numDataPoints)

# PLOT

import matplotlib.pyplot as plt

plt.scatter(train_x, train_y, color='b')
plt.scatter(train_x, y_test, color='r')

x_graph = np.linspace(dataRange[0], dataRange[1], 1000)
plt.plot(x_graph, func_to_predict(x_graph), color='y')

plt.show()