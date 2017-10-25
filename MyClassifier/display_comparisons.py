# Non-neural net models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from my_classifier import MyClassifier

print 'Loaded Dependencies...Need to get data'

df = pd.read_csv('train.csv')

def getDataFrame(n):
	df_subset = df.sample(n=n, random_state=200)
	df_train = df_subset.sample(frac=0.8, random_state=200)
	df_test = df_subset.drop(df_train.index)
	return df_train, df_test

def getData(dfs):

	df_train = dfs[0]
	df_test = dfs[1]

	X_train = df_train[df_train.columns[1:]]
	y_train = df_train[df_train.columns[0]]

	X_test = df_test[df_test.columns[1:]]
	y_test = df_test[df_test.columns[0]]

	return X_train, X_test, y_train, y_test

df_n100 = getDataFrame(100)
df_n500 = getDataFrame(500)
df_n2000 = getDataFrame(2000)

n100 = getData(df_n100)
n500 = getData(df_n500)
n2000 = getData(df_n2000)

print 'Loaded Data...need to initialize classifiers'

# Different Classifiers
logistic_classifier = LogisticRegression()
decisiontree_classifier = DecisionTreeClassifier(random_state=0)
kneighbors_classifier = KNeighborsClassifier(n_neighbors=3)

# My Classifier!!!

my_classifier = MyClassifier()
def score_my_classifier(dfs):
	df_train = dfs[0]
	df_test = dfs[1]
	my_classifier.train(df_train)
	return my_classifier.score(df_test)

myclassifier_scores = [score_my_classifier(df_n100), score_my_classifier(df_n500), score_my_classifier(df_n2000)]

print 'Loaded Classifiers...need to score classifiers'

# score function
def score_classifier(model, data):
	model.fit(data[0], data[2])
	return accuracy_score(data[3], model.predict(data[1]))

# Get scores on classifiers
def get_scores(clf):
	score100 = score_classifier(clf, n100)
	score500 = score_classifier(clf, n500)
	score2000 = score_classifier(clf, n2000)
	return [score100, score500, score2000]

logistic_scores = get_scores(logistic_classifier)
decisiontree_scores = get_scores(decisiontree_classifier)
kneighbors_scores = get_scores(kneighbors_classifier)

print 'Scored classifiers...displaying results'

print '----------------------------------------'
print 'Logistic Regression Scores: ' + ', '.join(str(x) for x in logistic_scores)
print 'Decision Tree Scores: ' + ', '.join(str(x) for x in decisiontree_scores)
print 'KNearest Neighbors Scores: ' + ', '.join(str(x) for x in kneighbors_scores)
print 'My Classifier Scores: ' + ', '.join(str(x) for x in myclassifier_scores)

x = [100, 500, 2000]
plt.plot(x, logistic_scores)
plt.plot(x, decisiontree_scores)
plt.plot(x, kneighbors_scores)
plt.plot(x, myclassifier_scores)
plt.xlabel("Number of Training Images")
plt.ylabel("Accuracy")
plt.legend(['Logistic', 'Decision Trees', 'KNeighbors', 'My Classifier'], loc='lower right')
plt.show()
