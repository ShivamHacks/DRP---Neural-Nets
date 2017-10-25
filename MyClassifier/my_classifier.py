import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MyClassifier:

	def getMatrixDifferences(self, digit_matrix):
		differences = []
		for i in range(0, 10):
			diff_matrix = np.square(self.digit_matrices[i] - digit_matrix)
			difference = np.sum(abs(diff_matrix))
			differences.append(difference)
		return differences

	def getMostSimilar(self, differences):
		return differences.index(min(differences))

	def train(self, df_train):

		# initialize digit matrices
		self.digit_matrices = []
		for index in range(0, 10):
			self.digit_matrices.append( np.zeros( (28,28), dtype=np.float ) )

		# count number of each digit
		digit_counts = np.bincount(df_train['label']).astype(float)

		# create the digit shape matrices 
		for index, row in df_train.iterrows():
			digit = row[0]
			data = np.reshape(row[1:].tolist(), (28, 28)).astype(float)
			self.digit_matrices[digit] += data

		# divide digit matrices by the counts of each digits (i.e. take average value)
		for i in range(0, 10):
			self.digit_matrices[i] = self.digit_matrices[i] / digit_counts[i]

	def score(self, df_test):

		accuracy_count = 0

		for index, row in df_test.iterrows():
			digit = row[0]
			matrix = np.reshape(row[1:].tolist(), (28, 28)).astype(float)
			differences = self.getMatrixDifferences(matrix)
			predicted_label = self.getMostSimilar(differences)
			if (predicted_label == digit):
				accuracy_count += 1

		return ( float(accuracy_count) / float(len(df_test)) )

	def display_results(self, df_test):

		imgs = []
		labels = []
		correct = []
		
		for index, row in df_test[:10].iterrows():
			digit = row[0]
			matrix = np.reshape(row[1:].tolist(), (28, 28)).astype(float)
			differences = self.getMatrixDifferences(matrix)
			predicted_label = self.getMostSimilar(differences)
			imgs.append(matrix)
			labels.append(predicted_label)
			correct.append(digit)

		def graph_shadows(imgs):
			fig = plt.figure(1)
			fig.suptitle('Shadows')
			for n, img in enumerate(imgs):
				a = fig.add_subplot(2, 5, n)
				plt.imshow(img, cmap='gray')
				a.set_title(str(n))
			return

		def graph_results(imgs, labels, correct):
			fig = plt.figure(2)
			fig.suptitle('Results')
			for n, img in enumerate(imgs):
				a = fig.add_subplot(2, 5, n)
				plt.imshow(img, cmap='gray')
				a.set_title('Predicted: ' + str(labels[n]) + '\nCorrect: ' + str(correct[n]))
			return

		graph_shadows(self.digit_matrices)
		graph_results(imgs, labels, correct)
		plt.show()

