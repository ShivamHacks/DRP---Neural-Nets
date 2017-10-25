'''

This script takes a new approach to classifying the MNIST data set

It superimposes all of the 28x28 matrices corresponding to a single digit
into a single matrix. Then, for any sample digit image, it computes the similarity
between the matrix representation of the sample digit image, and classifies the sample
digit as the digit matrix it has the highest similarity with.

The similarity is taken by substracting the two matrices and summing
the values of the resultant matrix together. The motivation behind such
a process is that if you stack all of the images of a single digit together,
the shape that is formed should take up space within the matrix that 
is most likely to be populated by a image of the same digit.

This classifier is not that bad LOL
The accuracy hovers between 60% and 70%, around 65%, and, after a very low limit
the number of samples negatively impacts the accuracy of the classifier

My classifier good for few data sets...where Neural Nets would fail

*** Consider squaring the numerical differences???

TODO:
1. See which numbers are confused the most, try to differentiate them
2. Plot the digit matrices to see what they look like

'''

import pandas as pd
import numpy as np

# initialize digit matrices
digit_matrices = []
for index in range(0, 10):
	digit_matrices.append( np.zeros( (28,28), dtype=np.float ) )

# fill in digit matrices
df = pd.read_csv('train.csv')
df_subset = df.sample(frac=0.05, random_state=200)
df_train = df_subset.sample(frac=0.8, random_state=200)
df_test = df_subset.drop(df_train.index)

digit_counts = np.unique(df_train['label'], return_counts=True)[1].astype(float)

# create the digit shape matrices 
for index, row in df_train.iterrows():
	digit = row[0]
	data = np.reshape(row[1:].tolist(), (28, 28)).astype(float)
	digit_matrices[digit] += data

# divide digit matrices by the counts of each digits (i.e. take average value)
for i in range(0, 10):
	digit_matrices[i] = digit_matrices[i] / digit_counts[i]

def getMatrixDifferences(digit_matrix):
	differences = []
	for i in range(0, 10):
		diff_matrix = digit_matrices[i] - digit_matrix
		difference = abs(np.sum(abs(diff_matrix)))
		differences.append(difference)
	return differences

def getMostSimilar(differences):
	return differences.index(min(differences))

# run for all test samples
accuracy_count = 0

for index, row in df_test.iterrows():
	digit = row[0]
	matrix = np.reshape(row[1:].tolist(), (28, 28)).astype(float)
	predicted_label = getMostSimilar(getMatrixDifferences(matrix))
	if (predicted_label == digit):
		accuracy_count += 1
	print 'Predicted Label: ' + str(predicted_label) + '  ,  Actual Label: ' +  str(digit)

print 'Total Accuracy: ' + str( (float(accuracy_count) / float(len(df_test))) * 100 )