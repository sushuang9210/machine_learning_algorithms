import numpy as np
from sklearn.decomposition import PCA as pca_classifier
from sklearn.svm import SVC
from sklearn import cross_validation

class PCA:
    def __init__(self,train,test,model_parameters):
        self.n_components = int(model_parameters[0])
        X_train = train[0]
        for i in range(len(train)-1):
            X_train = np.concatenate((X_train,train[i+1]),axis=0)
        self.X_train = X_train[:,0:-1]
        self.y_train = X_train[:,-1]
        self.X_test = test[:,0:-1]
        self.y_test = test[:,-1]

    def pca_train(self):
        pca = pca_classifier(n_components=self.n_components)# adjust yourself
        pca.fit(self.X_train)
        X_t_train = pca.transform(self.X_train)
        X_t_test = pca.transform(self.X_test)
        clf = SVC()
        clf.fit(X_t_train, self.y_train)
        return clf.predict(X_t_test)
