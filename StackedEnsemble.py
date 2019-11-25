import numpy as np
import pandas as pd
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
from sklearn.model_selection import StratifiedKFold
import time
import time
import numpy as np
from sklearn.neural_network import MLPClassifier

# Creating the stacking mechanism
def Stacking(model,train,y,test,n_fold):
    folds = StratifiedKFold(n_splits = n_fold,random_state=1)
    # Preparing new arrays
    test_pred = np.empty((0,1), float)
    train_pred = np.empty((0,1), float)

    # Train each value, fit into the model, and append the results to the new arrays
    for train_indices, val_indices in folds.split(train, y):
        x_train, x_val = train[train_indices], train[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        model.fit(X = x_train, y = y_train)
        train_pred = np.append(train_pred, model.predict(x_val))
        test_pred = np.append(test_pred, model.predict(test))

    # Return evaluated predictions and training datasets.
    return test_pred.reshape(-1,1), train_pred


ncols = 187
# x_train = X, y_train = y
x_train = np.genfromtxt("datah/train-top50.csv", dtype =int, delimiter=",", usecols=np.arange(1, ncols-2), skip_header=1)
y_train = np.genfromtxt("datah/train-top50.csv", dtype='unicode', delimiter=",", usecols=[ncols-1], skip_header=1)

# It's actually dev. CHANGE for consistence in the  final verion
x_test = np.genfromtxt("datah/dev-top50.csv", dtype=int, delimiter=",", usecols=np.arange(1, ncols-2), skip_header=1)
y_test = np.genfromtxt("datah/dev-top50.csv", dtype='unicode', delimiter=",", usecols=[ncols-1], skip_header=1)

# Define the first base model
model1 = DecisionTreeClassifier(random_state = 1)

# Initiate stacking for this model
test_pred1, train_pred1 = Stacking(model = model1, n_fold = 10, train = x_train, test = x_test, y = y_train)
print('finished stacking model 1')

# Rearrange the data by transforming the categorical data to numerical
train_pred1 = pd.DataFrame(train_pred1)
train_pred1 = pd.get_dummies(train_pred1)
train_pred1 = train_pred1.values
print('Pred 1 shape:', test_pred1.shape[0])
print('fisnished train pred1')
test_pred1 = pd.DataFrame(test_pred1)
test_pred1 = pd.get_dummies(test_pred1)
test_pred1 = test_pred1.values
print('Test 1 shape: ', test_pred1.shape[0])
print('fisnished test pred1')


# Define second base model
#model2 = KNeighborsClassifier()
#model2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 2), random_state=1)
model2 = GaussianNB()

# Initiate stacking for the second model
test_pred2, train_pred2 = Stacking(model = model2, n_fold = 10, train = x_train, test = x_test, y = y_train)
print('Finished stackind model 2')

# Rearrange the data to fit .fit()
train_pred2 = pd.DataFrame(train_pred2)
train_pred2 = pd.get_dummies(train_pred2)
train_pred2 = train_pred2.values
print(train_pred2.shape)
print('Finished train pred 2')

test_pred2 = pd.DataFrame(test_pred2)
test_pred2 = pd.get_dummies(test_pred2)
print(test_pred2.shape)
test_pred2 = test_pred2.values
print('Finished train pred 2')

# Concatenating data to merge the results from the base classifiers
df = np.concatenate([train_pred1, train_pred2], axis=1)
print('Concat shape of df: ', df.shape)
print('df size:', df.size)
print('y_train shape:',y_train.shape)
print('y_train size:',y_train.size)
#print(df)
print('Finished train concatenating 1')

df_test = np.concatenate([test_pred1, test_pred2], axis=1)
#df_test = np.concatenate([test_pred1, test_pred2], axis = 0)
print('Concat shape of df_test: ', df_test.shape)
print('df_test size: ', df_test.size)
# Difference is the difference in the number of rows. Number of columns matches.
difference = df_test.shape[0] - y_test.shape[0]
df_test = df_test[:-difference, :]
print('df_test new size????? : ', df_test.size)
print('y_test shape:',y_test.shape)
print('y_test size:',y_test.size)
print('Concatenated 2')

# The final model evaluates the new training and testing datasets
model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=1000).fit(df, y_train)
print('Finished fitting')
print('df size: ',df.size)
print('y test size: ',y_test.size)

print('y test shape: ', y_test.shape)
print('df test shape: ', df_test.shape)
#df_test = df_test.reshape(37316,80)
#print('Hopefully df test new shape: ', df_test.shape)
score = model.score(df_test, y_test)

print(score)
