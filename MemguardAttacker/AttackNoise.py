from model import model_atttack_noise, init_Data
import pandas as pd
import numpy as np
import keras
from keras.models import load_model

dataframe1 = pd.read_csv('../attack_memguard/data/train_1_masks.csv')
dataframe2 = pd.read_csv('../attack_memguard/data/train_2_masks.csv')
dataframe = pd.concat([dataframe1, dataframe2], ignore_index=True)


def train(dataframe):
    f_train, l_train = init_Data(dataframe)
    print(np.array(f_train))
    #f_train = np.sort(f_train, axis=1)
    num_label = 1

    model = model_atttack_noise(input_shape=f_train.shape[1:], labels_dim=num_label)

    checkpoint = keras.callbacks.ModelCheckpoint('../attack_memguard/attack_noise.hdf5',
                                                 monitor='accuracy', verbose=1,
                                                 save_best_only=True, mode='max')

    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=0.00005),
                  metrics=['accuracy'])
    model.summary()
    model.fit(f_train, l_train, epochs=400, batch_size=64, validation_split=0.1, callbacks=[checkpoint])

#train(dataframe)
def test(dataframe):
    f_evaluate, l_evaluate = init_Data(dataframe)
    model = load_model('../attack_memguard/attack_noise.hdf5')
    print(model.evaluate(f_evaluate, l_evaluate))
testDF = pd.read_csv("../MemguardData/test_masks_all.csv")
test(testDF)