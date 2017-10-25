import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

my_classifier = MyClassifier()

df = pd.read_csv('train.csv')
df_subset = df.sample(n=100)
df_train = df_subset.sample(frac=0.8)
df_test = df_subset.drop(df_train.index)
my_classifier.train(df_train)

def convertImageToMatrix(img):
	img = Image.open(img).convert('L')
	return img