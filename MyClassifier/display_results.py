import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------

# Create Image Shadows

# initialize digit matrices
digit_matrices = []
for index in range(0, 10):
	digit_matrices.append( np.zeros( (28,28), dtype=np.float ) )

# fill in digit matrices
df = pd.read_csv('train.csv')
df_subset = df.sample(n=500, random_state=21)
df_train = df_subset.sample(frac=0.8)
df_test = df_subset.drop(df_train.index)

digit_counts = np.bincount(df_train['label']).astype(float)

# create the digit shape matrices 
for index, row in df_train.iterrows():
	digit = row[0]
	data = np.reshape(row[1:].tolist(), (28, 28)).astype(float)
	digit_matrices[digit] += data

# divide digit matrices by the counts of each digits (i.e. take average value)
for i in range(0, 10):
	digit_matrices[i] = digit_matrices[i] / digit_counts[i]

# ----------------------------------------------------------------------------

def getMatrixDifferences(digit_matrix):
	differences = []
	for i in range(0, 10):
		diff_matrix = np.square(digit_matrices[i] - digit_matrix)
		difference = np.sum(abs(diff_matrix))
		differences.append(difference)
	return differences

def getMostSimilar(differences):
	return differences.index(min(differences))

# run for subset of test samples

imgs = []
labels = []
correct = []
for index, row in df_test[:10].iterrows():
	digit = row[0]
	matrix = np.reshape(row[1:].tolist(), (28, 28)).astype(float)
	predicted_label = getMostSimilar(getMatrixDifferences(matrix))
	imgs.append(matrix)
	labels.append(predicted_label)
	correct.append(digit)

# ----------------------------------------------------------------------------

def graph_shadows(imgs):
	fig = plt.figure(1)
	fig.suptitle('ImageStacks (800 training points)')
	for n, img in enumerate(imgs):
		a = fig.add_subplot(2, 5, n)
		plt.imshow(img, cmap='gray')
		a.set_title(str(n))
	return

def graph_results(imgs, labels, correct):
	fig = plt.figure(2)
	fig.suptitle('Results (800 training points)')
	for n, img in enumerate(imgs):
		a = fig.add_subplot(2, 5, n)
		plt.imshow(img, cmap='gray')
		a.set_title('Predicted: ' + str(labels[n]) + '\nCorrect: ' + str(correct[n]))
	return

graph_shadows(digit_matrices)
graph_results(imgs, labels, correct)
plt.show()
print 'Done'