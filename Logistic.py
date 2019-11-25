import numpy as np
#mport preprocess
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
#ncols = 36
#ncols = 187
ncols = 368

X = np.genfromtxt("datah/train-top100.csv", dtype=int, delimiter=",", usecols=np.arange(1, ncols-2), skip_header=1)
y = np.genfromtxt("datah/train-top100.csv", dtype='unicode', delimiter=",", usecols=[ncols-1], skip_header=1)

#X, devX = preprocess.textToFeatures(preprocess.train_tweets, preprocess.dev_tweets)
#y = preprocess.y_train
#devy = preprocess.y_dev

devX = np.genfromtxt("datah/dev-top100.csv", dtype=int, delimiter=",", usecols=np.arange(1, ncols-2), skip_header=1)
devy = np.genfromtxt("datah/dev-top100.csv", dtype='unicode', delimiter=",", usecols=[ncols-1], skip_header=1)

#print(devX[:10,:])
#print(devy[:10])


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=500).fit(X, y)
print(clf.score(devX, devy))

'''
index = np.genfromtxt("datah/test-top50.csv", dtype=int, delimiter=",", usecols=0, skip_header=1)
predictions = clf.predict(testX)
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
df.to_csv('logroutput.csv',index=False)
'''
