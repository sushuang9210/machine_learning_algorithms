#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

class LogisticRegression:
    def __init__(self,train,model_parameters):
        self.logreg = linear_model.LogisticRegression(C=float(model_parameters[0]))
        X_train = train[0]
        for i in range(len(train)-1):
            X_train = np.concatenate((X_train,train[i+1]),axis=0)
        self.X_train = X_train[:,0:-1]
        self.y_train = X_train[:,-1]

# we create an instance of Neighbours Classifier and fit the data.
    def lg_train(self):
        self.logreg.fit(self.X_train, self.y_train)

    def lg_predict(self, test):
        return self.logreg.predict(test[:,0:-1])
