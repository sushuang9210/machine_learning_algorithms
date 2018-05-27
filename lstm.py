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
class LSTM:
# create and fit the LSTM network
    def __init__(self,trainX,history,loss_type,optimizer_name):
        self.model = Sequential()
        self.model.add(LSTM_classifier(len(trainX), input_shape=(history,5)))
        self.model.add(Dense(len(trainX), activation = 'softmax'))
        #self.model.add(Activation('softmax'))
        self.model.compile(loss=loss_type, optimizer=optimizer_name)
        self.history = history
        train_lstm = []
        for i in range(len(trainX)):
            for j in range(len(trainX[i])-history+1):
                feature = trainX[i][j:j+history,0:-1].tolist()
                label_row = numpy.ones((trainX[i].shape[1]-1))*trainX[i][j+history-1,-1]
                feature.append(label_row.tolist())
                train_lstm.append(feature)
		#shuffle
        random.shuffle(train_lstm)
        trainY = numpy.zeros((len(train_lstm),len(trainX)))
        trainX = []
        #trainY = []
		#assign values to trainX and trainY
        for i in range(len(train_lstm)):
            for j in range(len(train_lstm[i])-1):
                trainX.append(train_lstm[i][j])
            #trainY.append(train_lstm[i][len(train_lstm[i])-1][0])
            #print(train_lstm[i][len(train_lstm[i])-1][0])
            #print(trainY.shape)
            trainY[i][int(train_lstm[i][len(train_lstm[i])-1][0])] = 1
        self.trainX = numpy.array(trainX)
        self.trainY = numpy.array(trainY)
        #print(self.trainX.shape)
        #print(self.trainY.shape)
		#Input: [samples, time steps, features]
        self.trainX = numpy.reshape(self.trainX, (int(self.trainX.shape[0]/history), history, self.trainX.shape[1]))
        #self.trainY = numpy.reshape(self.trainY, (int(self.trainY.shape[0]), 1, 1))

    def lstm_train(self,num_epochs,num_batch_size):
	    self.model.fit(self.trainX, self.trainY, epochs=num_epochs, batch_size=num_batch_size, verbose=2)
    # make predictions
    def lstm_predict(self,testX):
        test_lstm = []
        real_lstm = []
        for i in range(len(testX)):
            for j in range(len(testX[i])-self.history+1):
                feature = testX[i][j:j+self.history,0:-1].tolist()
                test_lstm.append(feature)
                real_lstm.append(testX[i][j+self.history-1,-1])
        test_lstm = numpy.array(test_lstm)
        #print(test_lstm.shape)
        #test_lstm = numpy.reshape(test_lstm, (int(test_lstm.shape[0]/self.history), self.history, test_lstm.shape[1]))
        testPredict = self.model.predict(test_lstm)
        #print(testPredict)
        #print(testPredict.shape)
        testPredictResult = []
        for i in range(len(testPredict)):
            #testPredictResult.append(int(testPredict[i][0]+0.5))
            #print(numpy.argmax(testPredict[i]))
            testPredictResult.append(numpy.argmax(testPredict[i]))
        print(testPredictResult)
        #testPredict = scaler.inverse_transform(testPredict)
        return real_lstm, testPredictResult
