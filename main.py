import sys
import csv
import numpy
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from mapper import Mapper
from svm import SVM
from randomForest import RandomForest
from pca import PCA
from lstm import LSTM
from logisticRegression import LogisticRegression
from adaboost import Adaboost
from neural_network import NeuralNetwork
from knn import KNN
from perceptron import Perceptron
from ridge import Ridge

def plot_confusion_matrix(model_name, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def print_report(model_name,y_true,y_predict):
    target_names = ['burst', 'cpu_pm', 'cpu_vm', 'normal']
    print("model:"+model_name)
    print(classification_report(y_true, y_predict, target_names=target_names))
    cnf_matrix = confusion_matrix(y_true, y_predict)
    plt.figure()
    plot_confusion_matrix(model_name,cnf_matrix, classes=target_names, title=model_name+' confusion matrix, without normalization')
    plt.figure()
    plot_confusion_matrix(model_name,cnf_matrix, classes=target_names, normalize=True,title=model_name+' normalized confusion matrix')

def train_svm(train,test,model_parameters = [0.001, 10000]):
    num_classifier = len(train)
    predicted_label = numpy.zeros((num_classifier, test.shape[0]))
    predicted_max_label = numpy.zeros((test.shape[0]))
    #SVM
    for i in range(num_classifier):
        for j in range(i):
            svm = SVM(train[i],train[j],model_parameters)
            label_1, label_2 = svm.test(test[:,0:-1])
            predicted_label[i, :] = predicted_label[i, :] + label_1
            predicted_label[j, :] = predicted_label[j, :] + label_2
    compare_matrix = (predicted_label==numpy.max(predicted_label, axis=0))
    for i in range(compare_matrix.shape[1]):
        for j in range(compare_matrix.shape[0]):
            if(compare_matrix[j][i]==1):
                predicted_max_label[i]=j
    print_report('SVM',test[:,-1],predicted_max_label)

def train_random_forest(train,test,model_parameters = [5,10,1,1.0]):
    n_trees = int(model_parameters[0])
    max_depth = int(model_parameters[1])
    min_size = int(model_parameters[2])
    sample_size = int(model_parameters[3])
    num_classifier = len(train)
    predicted_label = numpy.zeros((num_classifier, test.shape[0]))
    predicted_max_label = numpy.zeros((test.shape[0]))
    for i in range(num_classifier):
        for j in range(i):
            randomforest = RandomForest(train[i],train[j])
            label_1, label_2 = randomforest.evaluate_algorithm(test[:,0:-1], max_depth, min_size, sample_size, n_trees, (train[0].shape[1]-1))
            predicted_label[i, :] = predicted_label[i, :] + label_1
            predicted_label[j, :] = predicted_label[j, :] + label_2
    compare_matrix = (predicted_label==numpy.max(predicted_label, axis=0))
    for i in range(compare_matrix.shape[1]):
        for j in range(compare_matrix.shape[0]):
            if(compare_matrix[j][i]==1):
                predicted_max_label[i]=j
    print_report('Random forest',test[:,-1],predicted_max_label)

def train_pca(train,test,model_parameters = [1]):
    pca = PCA(train,test,model_parameters)
    print_report('PCA',test[:,-1],pca.pca_train())

def train_lstm(train,test,model_parameters =[4,'mean_squared_error','adam',100,1]):
    history = int(model_parameters[0])
    loss = model_parameters[1]
    optimizer = model_parameters[2]
    epochs = int(model_parameters[3])
    batch_size = int(model_parameters[4])
    lstm = LSTM(train,history,loss,optimizer)
    lstm.lstm_train(epochs,batch_size)
    real_label, predicted_label = lstm.lstm_predict(test)
    print_report('LSTM',real_label,predicted_label)

def train_logistic_regression(train,test,model_parameters = [1e5]):
    lr = LogisticRegression(train,model_parameters)
    lr.lg_train()
    print_report('Logistic Regression',test[:,-1],lr.lg_predict(test))

def train_adaboost(train,test,model_parameters = [1,"SAMME",200]):
    num_classifier = len(train)
    predicted_label = numpy.zeros((num_classifier, test.shape[0]))
    predicted_max_label = numpy.zeros((test.shape[0]))
    for i in range(num_classifier):
        for j in range(i):
            adaboost = Adaboost(train[i],train[j],model_parameters)
            adaboost.adaboost_train()
            label_1, label_2 = adaboost.adaboost_predict(test[:,0:-1])
            predicted_label[i, :] = predicted_label[i, :] + label_1
            predicted_label[j, :] = predicted_label[j, :] + label_2
    compare_matrix = (predicted_label==numpy.max(predicted_label, axis=0))
    for i in range(compare_matrix.shape[1]):
        for j in range(compare_matrix.shape[0]):
            if(compare_matrix[j][i]==1):
                predicted_max_label[i]=j
    print_report('Adaboost',test[:,-1],predicted_max_label)

def train_nn(train,test,model_parameters = ['mean_squared_error','adam',100,1]):
    nn = NeuralNetwork(train,model_parameters[0],model_parameters[1])
    nn.nn_train(int(model_parameters[2]),int(model_parameters[3]))
    predicted_label = nn.nn_predict(test[:,0:-1])
    print_report('Neural Network',test[:,-1],predicted_label)

def train_knn(train,test,model_parameters = ['distance']):
    knn = KNN(train,model_parameters)
    knn.knn_train()
    predicted_label = knn.knn_predict(test[:,0:-1])
    print_report('K Nearest Neighbors',test[:,-1],predicted_label)

def train_perceptron(train,test,model_parameters=[200]):
    num_classifier = len(train)
    predicted_label = numpy.zeros((num_classifier, test.shape[0]))
    predicted_max_label = numpy.zeros((test.shape[0]))
    for i in range(num_classifier):
        for j in range(i):
            perceptron = Perceptron(train[i],train[j],model_parameters)
            perceptron.perceptron_train()
            label_1, label_2 = perceptron.perceptron_predict(test[:,0:-1])
            predicted_label[i, :] = predicted_label[i, :] + label_1
            predicted_label[j, :] = predicted_label[j, :] + label_2
    compare_matrix = (predicted_label==numpy.max(predicted_label, axis=0))
    for i in range(compare_matrix.shape[1]):
        for j in range(compare_matrix.shape[0]):
            if(compare_matrix[j][i]==1):
                predicted_max_label[i]=j
    print_report('Perceptron',test[:,-1],predicted_max_label)

def train_ridge(train,test,model_parameters = [1e-2,"lsqr"]):
    num_classifier = len(train)
    predicted_label = numpy.zeros((num_classifier, test.shape[0]))
    predicted_max_label = numpy.zeros((test.shape[0]))
    for i in range(num_classifier):
        for j in range(i):
            ridge = Ridge(train[i],train[j],model_parameters)
            ridge.ridge_train()
            label_1, label_2 = ridge.ridge_predict(test[:,0:-1])
            predicted_label[i, :] = predicted_label[i, :] + label_1
            predicted_label[j, :] = predicted_label[j, :] + label_2
    compare_matrix = (predicted_label==numpy.max(predicted_label, axis=0))
    for i in range(compare_matrix.shape[1]):
        for j in range(compare_matrix.shape[0]):
            if(compare_matrix[j][i]==1):
                predicted_max_label[i]=j
    print_report('Ridge',test[:,-1],predicted_max_label)

def convert_data(data,data_type):
    if data_type == '0':
        return int(float(data))
    if data_type == '1':
        return float(data)
    if data_type == '2':
        return string(data)

if __name__ == '__main__':
    #input file name
    #in_file = ["../data/attack/10.1.1.8Burst.txt","../data/attack/10.1.1.8CPUinsert.txt","../data/attack/10.1.1.8CPUVMInsert.txt","../data/attack/10.1.1.8Normal.txt"]
    #num_sample = 100
    #num_feature = 5

    #set up the general data
    mapper = Mapper("map_config.txt")
    general_data = mapper.get_general_data()
    print("general data:",general_data)
    label_file = "label.txt"
    with open(label_file) as f:
        content = f.readlines()
    label = [int(x.strip()) for x in content]
    label = numpy.array(label)

    #classify the features based on the labels
    total_data = {}
    label_type = []
    for i in range(label.shape[0]):
        if label[i] not in total_data.keys():
            total_data[label[i]] = numpy.array([numpy.concatenate((general_data[i],[label[i]]))])
            #print(total_data[label[i]])
            label_type.append(label[i])
        else:
            new_data = numpy.array([numpy.concatenate((general_data[i],[label[i]]))])
            #print(total_data[label[i]])
            total_data[label[i]] = numpy.concatenate((total_data[label[i]],new_data),axis = 0)
    #print(total_data)
    #get train and test data
    train = []
    test = numpy.zeros((0,mapper.num_feature+1))
    print(test.shape)
    test_lstm = []
    #data_preprocess = Preprocess(num_feature)
    for i in range(len(label_type)):
        #train_i, test_i = data_preprocess.importText(in_file[i],i,num_sample)
        num_sample = total_data[label_type[i]].shape[0]
        train_i = total_data[label_type[i]][0:int(0.8*num_sample),:]
        test_i = total_data[label_type[i]][int(0.8*num_sample):,:]
        print(test_i.shape)
        train.append(train_i)
        test = numpy.concatenate((test, test_i),axis=0)
        test_lstm.append(test_i)

    print(train)
    print(test)

    #train models
    with open("model_config.txt") as f:
        content = f.readlines()
    model_config = [x.strip() for x in content]
    for i in range(int(len(model_config)/2)):
        model_type = int(model_config[2*i])
        model_parameters = model_config[2*i+1].split(' ')
        if model_type == 0:
            train_svm(train,test,model_parameters)
        if model_type == 1:
            train_random_forest(train,test,model_parameters)
        if model_type == 2:
            train_pca(train,test,model_parameters)
        if model_type == 3:
            train_lstm(train,test_lstm,model_parameters)
        if model_type == 4:
            train_logistic_regression(train,test,model_parameters)
        if model_type == 5:
            train_adaboost(train,test,model_parameters)
        if model_type == 6:
            train_nn(train,test,model_parameters)
        if model_type == 7:
            train_knn(train,test,model_parameters)
        if model_type == 8:
            train_perceptron(train,test,model_parameters)
        if model_type == 9:
            train_ridge(train,test,model_parameters)
    plt.show()
