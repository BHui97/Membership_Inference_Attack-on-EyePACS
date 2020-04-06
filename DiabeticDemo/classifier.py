import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation
import pandas as pd
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from defense_model import init_Data

train_df = pd.read_csv('/home/bo/Project/MemguardData/train_all.csv')
test_df = pd.read_csv('/home/bo/Project/MemguardData/test.csv')

x_train, y_train = init_Data(train_df)
x_test, y_test = init_Data(test_df)

model = Sequential()
model.add(Dense(512, input_dim=5, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dense(1))
model.add(Activation('tanh'))

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
model.summary()
checkpoint = ModelCheckpoint('/home/bo/Project/DiabeticDemo/attackerClassifier.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='max')
callbacks_list = [checkpoint]
model.fit(x_train, y_train,
          epochs=200,
          validation_data=(x_test, y_test),
          batch_size=128, callbacks=callbacks_list)
