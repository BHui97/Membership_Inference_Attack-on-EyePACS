import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.applications import ResNet50
from keras.models import Input, Sequential
from keras.callbacks import ModelCheckpoint
from ModelUtil import f1, precision, recall
from keras.models import load_model
from keras.optimizers import Adam
import keras
import os
from keras.layers import Activation, GlobalAveragePooling2D, Dense

os.environ['CUDA_VISIBLE_DEVICES']='1'

def create_model_resnet(input_shape):
    input_tensor = Input(shape=input_shape)
    model = Sequential()
    base_model = ResNet50(include_top=False,
                          weights= None,
                          input_tensor=input_tensor)
    #base_model.load_weights('weights/target_ResNet.h5')
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(8))
    model.add(Activation("softmax"))
    model.compile(loss = 'categorical_crossentropy',
                  optimizer=Adam(lr=5e-5),
                  metrics=['accuracy', precision, recall, f1])
    model.load_weights('weights/target_ResNet.h5')
    model.summary()
    return model

def train(model, x_train, y_train, weights_path):
    """
    Train the target model and save the weight of the model
    :param model: the model that will be trained
    :param x_train: The numpy of the image
    :param y_train: The label for x_train
    :param weights_path: The path to save at
    :return: None
    """

    checkpoint = ModelCheckpoint(weights_path, monitor='accuracy', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]

    model.summary()
    model.fit(x_train,
              y_train,
              batch_size=32,
              epochs=40,
              callbacks=callbacks_list)


dataframe = pd.read_csv("data/hmnist_64_64_L.csv")
trainDF, testDF = train_test_split(dataframe, train_size=0.5,
                               random_state=1, stratify=dataframe['label'].values)
x_train = np.asarray([i.reshape(64, 64, 1) for i in trainDF.iloc[:, range(4096)].values])
y_train = keras.utils.to_categorical([i-1 for i in trainDF.loc[:, 'label']])
# model = create_model_resnet((64, 64, 1))
# train(model, x_train, y_train, 'weights/target_ResNet.h5')

def evaluate(x_test, y_test):
    model = load_model('weights/target_ResNet.h5',
                       custom_objects={'precision': precision,'recall':recall,'f1':f1})
    loss, accuracy, Precision, Recall, F1 = model.evaluate(x_test, y_test, verbose=1)
    print('loss:%.4f accuracy:%.4f precision:%.4f recall:%.4f f1:%.4f' % (loss, accuracy, Precision, Recall, F1))
evaluate(x_train, y_train)