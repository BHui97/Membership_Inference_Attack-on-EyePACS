import numpy as np
import pandas as pd
import math
from PIL import Image
from tqdm import tqdm
import os
from keras import layers
from keras.applications import DenseNet121
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.model_selection import train_test_split
from defense_model import f1, precision, recall

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

train_df = pd.read_csv("/home/bo/Project/data/label.csv", error_bad_lines=False, index_col=0)
#print(train_df['level'].value_counts())
train_path = "/home/bo/Project/data/first_train/"
test_path = "/home/bo/Project/data/first_test/"

def get_pad_width(im, new_shape, is_rgb=True):
    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
    t, b = math.floor(pad_diff[0] / 2), math.ceil(pad_diff[0] / 2)
    l, r = math.floor(pad_diff[1] / 2), math.ceil(pad_diff[1] / 2)
    if is_rgb:
        pad_width = ((t, b), (l, r), (0, 0))
    else:
        pad_width = ((t, b), (l, r))
    return pad_width


def preprocess_image(image_path, desired_size=224):
    im = Image.open(image_path)
    im = im.resize((desired_size,) * 2, resample=Image.LANCZOS)

    return im


def set_data(img_path):
    N = len(os.listdir(img_path))
    x_ = np.empty((N, 224, 224, 3), dtype=np.uint8)
    y_ = np.empty(N)
    for i, img_name in enumerate(tqdm(os.listdir(img_path))):
        x_[i, :, :, :] = preprocess_image(img_path + img_name)
        y_[i] = train_df.loc[os.path.splitext(img_name)[0], 'level']
    y_ = pd.get_dummies(y_).values

    # y_multi = np.empty(y_.shape, dtype=y_.dtype)
    # y_multi[:, 4] = y_[:, 4]
    # for i in range(3, -1, -1):
    #     y_multi[:, i] = np.logical_or(y_[:, i], y_multi[:, i + 1])
    return x_, y_

x_train, y_train = set_data(train_path)
x_test, y_test = set_data(test_path)


def build_model():
    densenet = DenseNet121(
        weights='densenet-keras/DenseNet-BC-121-32-no-top.h5',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5))
    model.add(layers.Activation('softmax'))
    checkpoint = ModelCheckpoint('/home/bo/Project/densenet.hdf5', monitor='accuracy', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=0.00005),
        metrics=['accuracy', precision, recall, f1]
    )

    _, x_valid, _, y_valid = train_test_split(x_test, y_test, test_size=0.15)

    model.summary()
    model.fit(x_train,
              y_train,
              batch_size=32,
              validation_data=(x_valid, y_valid),
              epochs=20,
              callbacks=callbacks_list)

    return model

def test_model(x_test,y_test):
    model = load_model('/home/bo/Project/densenet1.hdf5',
                       custom_objects={'precision': precision,'recall':recall,'f1':f1})
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=0.00005),
        metrics=['accuracy', precision, recall, f1]
    )
    loss, accuracy, Precision, Recall, F1 = model.evaluate(x_test, y_test, verbose=1)
    print('loss:%.4f accuracy:%.4f precision:%.4f recall:%.4f f1:%.4f'% (loss, accuracy, Precision, Recall, F1))


model = build_model()
test_model(x_test, y_test)