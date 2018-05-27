import numpy
import matplotlib.pyplot as plt
import pandas
import math
import random
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM as LSTM_classifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
class NeuralNetwork:
# create and fit the LSTM network
    def __init__(self,trainX,loss_type,optimizer_name):
        self.model = Sequential()
        #self.model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
        self.model.add(Dense(8, input_dim=5, init='uniform', activation='tanh'))
        self.model.add(Dense(6, init='uniform', activation='tanh'))
        self.model.add(Dense(len(trainX), init='uniform', activation = 'softmax'))
        self.model.compile(loss=loss_type, optimizer=optimizer_name)
        trainData = []
        for i in range(len(trainX)):
            for j in range(len(trainX[i])):
                trainData.append(trainX[i][j,:].tolist())
		#shuffle
        random.shuffle(trainData)
        self.trainY = numpy.zeros((len(trainData),len(trainX)))
        self.trainX = []
        #print(trainData[0][0:-1])
        #trainY = []
		#assign values to trainX and trainY
        for i in range(len(trainData)):
            self.trainX.append(trainData[i][0:-1])
            self.trainY[i][int(trainData[i][-1])] = 1
        self.trainX = numpy.array(self.trainX)
        print(self.trainY)
        #self.trainY = numpy.array(trainY)

    def nn_train(self,num_epochs,num_batch_size):
	    self.model.fit(self.trainX, self.trainY, epochs=num_epochs, batch_size=num_batch_size, verbose=2)
    # make predictions
    def nn_predict(self,testX):
        testPredict = self.model.predict(testX)
        testPredictResult = []
        for i in range(len(testPredict)):
            testPredictResult.append(numpy.argmax(testPredict[i]))
        print(testPredictResult)
        return testPredictResult
