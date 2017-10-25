import pandas as pd
import numpy as np
from PIL import Image
import time

my_classifier = MyClassifier()

df = pd.read_csv('train.csv')
df_subset = df.sample(n=20)

def saveImage(matrix):
	img = Image.fromarray(matrix, 'L')
	img.save('imgs/' + str(time.time()) + '.png')

for index, row in df_subset.iterrows():
	matrix = np.reshape(row[1:].tolist(), (28, 28)).astype(float)
	saveImage(matrix)