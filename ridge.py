import numpy as np
from sklearn.linear_model import RidgeClassifier
class Ridge:
    def __init__(self,data_1,data_2,model_parameters):
        self.clf = RidgeClassifier(tol=float(model_parameters[0]), solver=model_parameters[1])
        num_data_1 = data_1.shape[0]
        num_data_2 = data_2.shape[0]
        data_1[:,-1] = np.ones((num_data_1))
        data_2[:,-1] = np.zeros((num_data_2))
        self.train_set = np.concatenate((data_1, data_2),axis=0)
        np.random.shuffle(self.train_set)
        self.X_train = self.train_set[:,0:-1]
        self.y_train = self.train_set[:,-1]

    def ridge_train(self):
        self.clf.fit(self.X_train,self.y_train)

    def ridge_predict(self,test):
        output_1 = self.clf.predict(test)
        output_2 = np.ones((test.shape[0])) - output_1
        return output_1, output_2
