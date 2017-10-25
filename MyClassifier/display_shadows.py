import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def graph_shadows(imgs):
	fig = plt.figure()
	for n, img in enumerate(imgs):
		a = fig.add_subplot(2, 5, n)
		plt.imshow(img, cmap='gray')
		a.set_title(str(n))
	plt.show()

# Create Image Shadows

# initialize digit matrices
digit_matrices = []
for index in range(0, 10):
	digit_matrices.append( np.zeros( (28,28), dtype=np.float ) )

# fill in digit matrices
df = pd.read_csv('train.csv')
df_subset = df.sample(frac=0.05, random_state=200)
df_train = df_subset.sample(frac=0.8, random_state=200)
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

# display in window
graph_shadows(digit_matrices)