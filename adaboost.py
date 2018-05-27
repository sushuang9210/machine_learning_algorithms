import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

class Adaboost:
    def __init__(self,data_1,data_2,model_parameters):
        self.bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=int(model_parameters[0])),algorithm=model_parameters[1],n_estimators=int(model_parameters[2]))
        num_data_1 = data_1.shape[0]
        num_data_2 = data_2.shape[0]
        data_1[:,-1] = np.ones((num_data_1))
        data_2[:,-1] = np.zeros((num_data_2))
        self.train_set = np.concatenate((data_1, data_2),axis=0)
        np.random.shuffle(self.train_set)
        self.X_train = self.train_set[:,0:-1]
        self.y_train = self.train_set[:,-1]

    def adaboost_train(self):
        self.bdt.fit(self.X_train,self.y_train)

    def adaboost_predict(self,test):
        output_1 = self.bdt.predict(test)
        output_2 = np.ones((test.shape[0])) - output_1
        return output_1, output_2

#w = svm_sgd(X,y)
#print(w)
