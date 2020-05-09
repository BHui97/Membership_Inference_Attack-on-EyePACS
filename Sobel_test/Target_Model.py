from keras.applications import ResNet50
from keras.models import Input, Sequential
from keras.callbacks import ModelCheckpoint
from ModelUtil import f1, precision, recall
from keras.datasets import cifar100
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
    base_model.load_weights('/home/yuchen/demo/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(100))
    model.add(Activation("softmax"))
    model.compile(loss = 'categorical_crossentropy',
                  optimizer=Adam(lr=5e-5),
                  metrics=['accuracy', precision, recall, f1])
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
              epochs=20,
              callbacks=callbacks_list)


(x_train, y_train), (_, _) = cifar100.load_data(label_mode='fine')
y_train = keras.utils.to_categorical(y_train, 100)
model = create_model_resnet((32, 32, 3))
train(model, x_train, y_train, 'weights/target_ResNet.h5')