from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
import numpy as np

def model_defense(input_shape,labels_dim):

    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=input_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(labels_dim))
    model.add(Activation('sigmoid'))
    return model


def model_defense_optimize(input_shape, labels_dim):

    model = Sequential()
    model.add(Activation('softmax', input_shape=input_shape))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(labels_dim))
    model.add(Activation('sigmoid'))
    return model

def model_atttack_noise(input_shape, labels_dim):

    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=input_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    #model.add(LSTM(128,input_shape=(5,1)))
    model.add(Dense(32))
    model.add(Dense(labels_dim))
    model.add(Activation('tanh'))
    return model

def init_Data(dataFrame):
    # Construct x_, y_
    x_ = dataFrame.loc[dataFrame.index,['0', '1', '2', '3', '4']]
    x_ = np.array(x_)
    y_ = dataFrame['label'].values
    return x_, y_