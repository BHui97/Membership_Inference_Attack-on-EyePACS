from keras import backend, layers
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.applications import DenseNet121
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

def precision(y_true, y_pred):
    tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))  # predicted positives
    return tp / (pp + backend.epsilon())


def recall(y_true, y_pred):
    tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = backend.sum(backend.round(backend.clip(y_true, 0, 1)))  # possible positives
    return tp / (pp + backend.epsilon())


def f1(y_true, y_pred):
    pre = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((pre * rec) / (pre + rec + backend.epsilon()))


def preprocess_image(image_path, desired_size=224):
    """
    Resize the picture to the desired size
    :param image_path: the path of image folder
    :param desired_size: the size that image will be cropped as. The default size is 224*224
    :return: the cropped image
    """
    im = Image.open(image_path)
    im = im.resize((desired_size,) * 2, resample=Image.LANCZOS)

    return im


def set_data(img_path, dataframe):
    """
    Correspond the image to the label and return them.
    :param img_path: the path of images' folder
    :param dataframe: the .csv file that shows relation between image and label
    :return: Image, Label and the name of Image
    """
    N = len(os.listdir(img_path))
    x_ = np.empty((N, 224, 224, 3), dtype=np.uint8)
    y_ = np.empty(N)
    image_names = np.empty(N, dtype=np.dtype(('U', 15)))
    for i, img_name in enumerate(tqdm(os.listdir(img_path))):
        x_[i, :, :, :] = preprocess_image(img_path + img_name)
        y_[i] = dataframe.loc[img_name, 'diabetic_retinopathy']
        image_names[i] = img_name
    y_ = pd.get_dummies(y_).values

    return x_, y_, image_names


def shadowModel(class_num, activation="softmax", loss='binary_crossentropy', learning_rate=0.00005):
    """
    Set the default configuration for a DenseNet Model.
    :param class_num: The number of classes
    :param activation: Set the activation function and the default function is softmax
    :param loss: Set the loss function and the default function is binary_crossentropy
    :param learning_rate: Set the learning rate and the default rate is 0.00005
    :return: A DenseNet Model
    """
    densenet = DenseNet121(
        weights='densenet-keras/DenseNet-BC-121-32-no-top.h5',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(class_num))
    model.add(layers.Activation(activation))

    model.summary()

    model.compile(
        loss=loss,
        optimizer=Adam(lr=learning_rate),
        metrics=['accuracy', precision, recall, f1]
    )

    return model