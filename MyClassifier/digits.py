#from sklearn import datasets

#Load the digits dataset
#digits = datasets.load_digits()

#print digits


# Creates a random image 100*100 pixels
#mat = np.random.random((1000,10))

'''
Can make a linear classifier using position and pixel or a manual one (custom)
'''


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model

def get_data(csv, sample_size, test_size, random_state):
	df = pd.read_csv(csv)

	num_samples = int(len(df) * sample_size)
	num_train = int(num_samples * (1 - test_size))
	num_test = int(num_samples * test_size)

	df = df.sample(frac=sample_size, random_state=random_state)

	df_train, df_test = train_test_split(df, test_size=test_size)

	X_train = df_train[df_train.columns[1:]]
	y_train = df_train[df_train.columns[0]]

	X_test = df_test[df_test.columns[1:]]
	y_test = df_test[df_test.columns[0]]

	return X_train, X_test, y_train, y_test, num_samples, num_train, num_test

def score_model(model, X_train, X_test, y_train, y_test):
	model.fit(X_train, y_train)
	y_predicted = model.predict(X_test)
	score = accuracy_score(y_predicted, y_test)
	return score

X_train, X_test, y_train, y_test, num_samples, num_train, num_test = get_data('code/train.csv', 0.5, 0.2, 12345)

linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)
y_predicted = np.round(linear_regressor.predict(X_test))
linear_regressor_score = accuracy_score(y_predicted, y_test)

print 'Linear Score: ' + str(linear_regressor_score)
print 'Number of Samples: ' + str(num_samples)
print 'Train Size: ' + str(num_train)
print 'Test Size: ' + str(num_test)

'''

logistic_regressor = linear_model.LogisticRegression()
logistic_score = score_model(logistic_regressor, X_train, X_test, y_train, y_test)

print 'Logistic Score: ' + str(logistic_score)
print 'Number of Samples: ' + str(num_samples)
print 'Train Size: ' + str(num_train)
print 'Test Size: ' + str(num_test)

'''


'''

logistic_regressor = linear_model.LogisticRegression()
#linear_regressor = linear_model.LinearRegression()

logistic_score = score_model(logistic_regressor, X_train, X_test, y_train, y_test)
#linear_score = score_model(linear_regressor, X_train, X_test, y_train, y_test)

print 'Logistic Score: ' + str(logistic_score)
#print 'Linear Score: ' + linear_score
'''
'''

df = pd.read_csv('code/train.csv')[0:1000]
print len(df)
df_train, df_test = train_test_split(df, test_size=0.2)

X_train = df_train[df_train.columns[1:]]
y_train = df_train[df_train.columns[0]]

X_test = df_test[df_test.columns[1:]]
y_test = df_test[df_test.columns[0]]

logistic_regressor = linear_model.LogisticRegression()
linear_regressor = linear_model.LinearRegression()

logistic_regressor.fit(X_train, y_train)
linear_regressor.fit(X_train, y_train)

logistic_y_predicted = logistic_regressor.predict(X_test)

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, y_predicted)

'''

#y_classifed = np.round(y_predicted)

'''print y_classifed
print y_test

num_wrong = np.count_nonzero(y_classifed - y_test)
print num_wrong
'''

# Calculate Error


'''
X_test = pd.read_csv('code/test.csv')
y_test = regressor.predict(X_test)
y_classifed = np.round(y_test)
'''

# Calculate Error




'''
mat = np.random.random((28, 28))
print mat.shape
'''

'''
# Showing Image
from PIL import Image

def showImage(matrix):
	img = Image.fromarray(matrix, 'L')
	img.show()

df = pd.read_csv('code/train.csv')
for index, row in df.iterrows():

	if (index == 0):
		raw_data = row[1:].tolist()
		matrix = np.reshape(raw_data, (28, 28)).astype(float)
		print matrix.shape
		matrix = matrix / 255
		print matrix
		showImage(matrix)
		mat = np.random.random((28, 28))
		showImage(mat)

'''