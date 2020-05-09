import pandas as pd
from keras import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
from ModelUtil import precision, recall, f1
from keras.models import load_model
from keras.optimizers import Adam
import numpy as np
import os
print(os.getcwd())

train_df = pd.read_csv('Project/Sobel_test/Sha_ConfidenceScores.csv')
x_train = train_df.iloc[:, range(100)].values
x_train = np.sort(x_train, axis=1)
print(train_df['label'].value_counts())
y_train = train_df.loc[:, ['label']].values
test_df = pd.read_csv('Project/Sobel_test/Ori_ConfidenceScores.csv')[40000:60000]
#test_df = pd.read_csv('Project/Sobel_test/MemOri_ConfidenceScores.csv')
#test_df = pd.read_csv('Project/Sobel_test/AllMemOri_ConfidenceScores.csv')
print(test_df.shape)
print(test_df['label'].value_counts())
x_test = test_df.iloc[:, range(100)].values
x_test = np.sort(x_test, axis=1)
y_test = test_df.loc[:, ['label']].values

def attackerModel(input_dim=100):
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

def train(x_train, y_train, x_test, y_test):
    model = attackerModel()
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy', precision, recall, f1])
    model.summary()
    checkpoint = ModelCheckpoint('weights/nn_attack.hdf5',
                                 monitor='val_accuracy',
                                 verbose=1, save_best_only=True,
                                 mode='max')
    model.fit(x_train, y_train,
              epochs=200,
              validation_data=(x_test, y_test),
              batch_size=32, callbacks=[checkpoint])

def evaluate(x_test, y_test):
    model = load_model('Project/Sobel_test/weights/nn_attack.hdf5',
                       custom_objects={'precision': precision,'recall':recall,'f1':f1})
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy', precision, recall, f1])
    loss, accuracy, Precision, Recall, F1 = model.evaluate(x_test, y_test, verbose=1)
    print('loss:%.4f accuracy:%.4f precision:%.4f recall:%.4f f1:%.4f' % (loss, accuracy, Precision, Recall, F1))


#train(x_train, y_train, x_test, y_test)
evaluate(x_train, y_train)
evaluate(x_test, y_test)