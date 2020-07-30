import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

from sklearn import svm
import matplotlib.pyplot as plt
import sklearn
import pickle


def abc():

    svm_model_linear = ExtraTreesClassifier(n_estimators=300).fit(X_train, y_train)
    svm_predictions = svm_model_linear.predict(X_test)

    # model accuracy for X_test
    accuracy = svm_model_linear.score(X_test, y_test)
    print(accuracy)

    # creating a confusion matrix
    cm = confusion_matrix(y_test, svm_predictions)
    outfile = open('finalized_model.sav', 'wb')
    pickle.dump(svm_model_linear, outfile)
    outfile.close()


def readData(fn):
    data = pd.read_csv(fn)
    global X
    global y
    y = data.disease
    X = data.drop('disease', axis=1)

X=0
y=0
csv_file = "Tomato0.csv"
readData(csv_file)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
features_list=list(X)
abc()
