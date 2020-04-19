import os
import pandas as pd
from ModelUtiility import shadowModel, ModelCheckpoint, train_test_split, set_data
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

train_df = pd.read_csv("data/shadow_train.csv", index_col=0)
test_df = pd.read_csv("data/shadow_test.csv", index_col=0)
print(train_df['diabetic_retinopathy'].value_counts())
print(test_df['diabetic_retinopathy'].value_counts())
train_path = "data/preprocessed_shadow_train/"
test_path = "data/preprocessed_shadow_test/"


def train(x_train, y_train, x_test, y_test, weights_path):
    """
    Train the shadow model and save the weight of the model
    :param x_train: The numpy of the image
    :param y_train: The label for x_train
    :param x_test: The numpy of the x_test
    :param y_test: The label for x_test
    :param weights_path: The path to save at
    :return: None
    """
    model = shadowModel(class_num=2)

    checkpoint = ModelCheckpoint(weights_path, monitor='accuracy', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]

    #_, x_valid, _, y_valid = train_test_split(x_test, y_test, test_size=0.15)

    model.summary()
    model.fit(x_train,
              y_train,
              batch_size=32,
              validation_data=(x_test, y_test),
              epochs=20,
              callbacks=callbacks_list)


x_train, y_train, _ = set_data(train_path, train_df)
x_test, y_test, _ = set_data(test_path, test_df)
unique_elements, counts_elements = np.unique(y_train[:, 1], return_counts=True)
print(np.asarray((unique_elements, counts_elements)))
unique_elements, counts_elements = np.unique(y_test[:, 1], return_counts=True)
print(np.asarray((unique_elements, counts_elements)))
#train(x_train, y_train, x_test, y_test, 'weights/shadow_model.hdf5')
