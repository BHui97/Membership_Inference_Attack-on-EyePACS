import pandas as pd
import numpy as np
from defense_model import model_defense, precision, recall, f1, init_Data
import keras

dataframe = pd.read_csv('../attackerData3/test.csv')

f_train, l_train = init_Data(dataframe)
f_train = np.sort(f_train, axis=1)

num_classes = 1
model = model_defense(input_shape = f_train.shape[1:], labels_dim=num_classes)


checkpoint = keras.callbacks.ModelCheckpoint('../attackerData3/defense.hdf5', monitor='accuracy', verbose=1,
                                                 save_best_only=True, mode='max')
model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.SGD(lr=0.001),
              metrics=['accuracy', precision, recall, f1])
model.summary()

model.fit(f_train, l_train, batch_size=64, epochs=400, callbacks=[checkpoint])

weights = model.get_weights()
np.savez('../attackerData3/defense.npz', x=weights)