import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 15

class KNN:
    def __init__(self,train,model_parameters):
        self.clf = neighbors.KNeighborsClassifier(n_neighbors, weights=model_parameters[0])
        X_train = train[0]
        for i in range(len(train)-1):
            X_train = np.concatenate((X_train,train[i+1]),axis=0)
        random.shuffle(X_train)
        self.X_train = X_train[:,0:-1]
        #print(train[:,0:-1])
        self.y_train = X_train[:,-1]

    def knn_train(self):
        self.clf.fit(self.X_train, self.y_train)

    def knn_predict(self,test):
        return self.clf.predict(test)
