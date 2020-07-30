import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import svm
import matplotlib.pyplot as plt
import sklearn
import pickle


def ETC():

    clf = ExtraTreesClassifier(n_estimators=300,
                               max_features=4,
                               n_jobs=1,
                               random_state=0).fit(X_train, y_train)
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    return (accuracy)

def RF():
    clf = RandomForestClassifier().fit(X_train, y_train)
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    return (accuracy)


def DT():
    clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    return (accuracy)


def KNN():
    clf = KNeighborsClassifier().fit(X_train, y_train)
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    return (accuracy)

def compare():
    print("MODEL \t\t\t\tACCURACY SCORE")
    print("ExtraTrees \t\t\t",ETC())
    print("RandomForest \t\t\t",RF())
    print("DecisionTree \t\t\t",DT())
    print("KNearestNeighbours \t\t",KNN())

    print("\n\nFeature Importance")
    forest = ExtraTreesClassifier(n_estimators=300,
                                  random_state=0)
    forest.fit(X_train, y_train)
    imp= forest.feature_importances_
    features = (list(X.columns))
    for i in range(len(list(X.columns))):
        print(features[i],"-",imp[i])



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
compare()
