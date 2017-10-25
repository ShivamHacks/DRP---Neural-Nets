import pandas as pd
import numpy as np

from my_classifier import MyClassifier

df = pd.read_csv('train.csv')
df_subset = df.sample(n=500, random_state=100)
df_train = df_subset.sample(frac=0.8, random_state=100)
df_test = df_subset.drop(df_train.index)

my_classifier = MyClassifier()
my_classifier.train(df_train)
print my_classifier.score(df_test)
my_classifier.display_results(df_test)




'''

A file to test individual lines of code without having
to load all the heavy libraries. Useful for testing
algorithms that manipulate data

import numpy as np
import pandas as pd

mat = np.random.random((4, 4)) * 255
#mat = np.zeros( (28, 28) )

# Showing Image
from PIL import Image

def showImage(matrix):
	img = Image.fromarray(matrix, 'L')
	img.show()

print mat
showImage(mat)


#print np.divide( [2, 4, 6], [2] ).astype(float)
#print np.array([[2, 4, 6], [4, 8, 12]]) / 2

for i in range(len([1, 2, 3])):
	print str(i) + ':' + str(val)


# Showing Image
from PIL import Image

matrix = digit_matrices[0] / 255

#np.round(digit_matrices[0].astype(np.float)) / 255
print matrix
img = Image.fromarray(matrix, 'L')
img.show()

#print digit_matrices[0]
#print digit_matrices[0].astype(int)

'''