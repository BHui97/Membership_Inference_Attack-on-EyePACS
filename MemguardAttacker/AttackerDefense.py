import pandas as pd
import numpy as np
import keras
from model import model_defense, init_Data
from keras.optimizers import Adam

dataframe = pd.read_csv('../attack_memguard/train_2.csv')

f_train, l_train = init_Data(dataframe)
f_train = np.sort(f_train, axis=1)

num_classes = 1

model = model_defense(f_train.shape[1:], num_classes)

checkpoint = keras.callbacks.ModelCheckpoint('../attack_memguard/defender_classifier_weight.hdf5', monitor='accuracy', verbose=1,
                                                 save_best_only=True, mode='max')

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
model.summary()

model.fit(f_train, l_train, epochs=400, batch_size=128, callbacks=[checkpoint])
weights = model.get_weights()
np.savez('../attack_memguard/defender_classifier_weight.npz', x=weights)
