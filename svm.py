import numpy as np

class SVM:
    #data_1, data_2 should be the training data of two different classes
    def __init__(self,data_1,data_2,model_parameters):
        self.eta = float(model_parameters[0])
        self.epochs = int(model_parameters[1])
        num_data_1 = data_1.shape[0]
        num_data_2 = data_2.shape[0]
        #mark the labels as 1 and -1
        data_1[:,-1] = np.ones((num_data_1))
        data_2[:,-1] = -np.ones((num_data_2))
        #combine the data and shuffle
        self.train_set = np.concatenate((data_1, data_2),axis=0)
        np.random.shuffle(self.train_set)
        #self.train_feature = self.train_set[:,0:-1]
        self.train_feature = np.concatenate((self.train_set[:,0:-1],-np.ones(((num_data_1+num_data_2),1))),axis=1)
        self.train_label = self.train_set[:,-1]
        self.w = self.svm_sgd(self.train_feature, self.train_label)
        #print(self.w)

    def svm_sgd(self, X, Y):
        w = np.zeros(len(X[0]))
        for epoch in range(1,self.epochs):
            for i, x in enumerate(X):
                if (Y[i]*np.dot(X[i], w)) < 1:
                    w = w + self.eta * ( (X[i] * Y[i]) + (-2  *(1/epoch)* w))
                else:
                    w = w + self.eta * (-2  *(1/epoch)* w)
        return w

    def test(self, X):
        num_test = X.shape[0]
        X = np.concatenate((X,-np.ones((num_test,1))),axis=1)
        output_1 = np.zeros(X.shape[0])
        output_2 = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            #print(np.dot(X[i], self.w))
            if (np.dot(X[i], self.w)<0):
                output_2[i] = 1
            else:
                output_1[i] = 1
        return output_1, output_2

#w = svm_sgd(X,y)
#print(w)
