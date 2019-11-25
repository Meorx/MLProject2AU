import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
import time
import pickle


def load_train_data(train_file):
    X = []
    y = []
    with open(train_file, mode='r') as fin:
        for line in fin:
            atts = line.strip().split(",")
            X.append(atts[:-1])  # all atts minus the last one
            y.append(atts[-1])
    print('Finished opening the file')
    onehot = OneHotEncoder()
    print('Onehot enconding done')
    X = onehot.fit_transform(X).toarray()
    print('onehot fitting done')
    return X, y

#ncols = 36
ncols = 187
#ncols = 368

# Prepare the files
X = np.genfromtxt("datah/train-top50.csv", dtype=int, delimiter=",", usecols=np.arange(1, ncols-2), skip_header=1)
y = np.genfromtxt("datah/train-top50.csv", dtype='unicode', delimiter=",", usecols=[ncols-1], skip_header=1)

devX = np.genfromtxt("datah/dev-top50.csv", dtype=int, delimiter=",", usecols=np.arange(1, ncols-2), skip_header=1)
devy = np.genfromtxt("datah/dev-top50.csv", dtype='unicode', delimiter=",", usecols=[ncols-1], skip_header=1)


testX = np.genfromtxt("datah/test-top50.csv", dtype=int, delimiter=",", usecols=np.arange(1, ncols-2), skip_header=1)
index = np.genfromtxt("datah/test-top50.csv", dtype=int, delimiter=",", usecols=0, skip_header=1)
#devy = np.genfromtxt("datah/test-top50.csv", dtype='unicode', delimiter=",", usecols=[ncols-1], skip_header=1)


from sklearn.ensemble import RandomForestClassifier


# Create Random Forest
rf = RandomForestClassifier()
print('Got to Random Forest Classifier')

# Fit the Random Forest
rf.fit(X, y)
print('Finished fitting')

# Print results
print('Random Forest train acc:', rf.score(X, y))
print('Random Forest test acc:', rf.score(devX, devy))
cv = 5
print('Random Forest cross-val acc:', np.mean(cross_val_score(rf, X, y, cv=cv)))

predictions = rf.predict(testX)
print(np.shape(predictions))
print(np.shape(index))
output = np.c_[np.array(index), np.array(predictions)]
output = np.vstack((np.array(["Id", "Class"]), output))
print(np.shape(output))
print(output[:,:30])
#output = np.asarray(output)

#np.savetxt("rfoutput.csv", np.toarray(output), delimiter=",")

import pandas as pd

df = pd.DataFrame(output)
df.to_csv('rfoutput.csv',index=False)