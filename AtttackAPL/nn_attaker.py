import pandas as pd
from keras import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
from ModelUtiility import Adam, precision, recall, f1
from keras.models import load_model
from sklearn.utils import shuffle
import numpy as np

train_df = pd.read_csv("data/shadowConfindenceScore.csv")
train_df = shuffle(train_df)
test_df = pd.read_csv("data/baselineConfidenceScore.csv")
#test_df = shuffle(test_df)
#test_df = pd.read_csv("data/fullSynthConfidenceScore.csv")

x_train = train_df.loc[train_df.index, ['confidence_scores_0', 'confidence_scores_1']].values
x_train = np.sort(x_train, axis=1)
y_train = train_df.loc[train_df.index, ['label']].values
x_test = test_df.loc[test_df.index, ['confidence_scores_0', 'confidence_scores_1']].values
y_test = test_df.loc[test_df.index, ['label']].values


def attackerModel(input_dim=2):
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
    checkpoint = ModelCheckpoint('weights/nn_attackerBaseline.hdf5',
                                 monitor='val_accuracy',
                                 verbose=1, save_best_only=True,
                                 mode='max')
    model.fit(x_train, y_train,
              epochs=200,
              validation_data=(x_test, y_test),
              batch_size=128, callbacks=[checkpoint])


def evaluate(x_test, y_test):
    model = load_model('weights/nn_attackerBaseline.hdf5',
                       custom_objects={'precision': precision,'recall':recall,'f1':f1})
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy', precision, recall, f1])
    loss, accuracy, Precision, Recall, F1 = model.evaluate(x_test, y_test, verbose=1)
    print('loss:%.4f accuracy:%.4f precision:%.4f recall:%.4f f1:%.4f' % (loss, accuracy, Precision, Recall, F1))


train(x_train, y_train, x_test, y_test)
evaluate(x_test, y_test)