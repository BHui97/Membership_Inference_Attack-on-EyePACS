from keras.applications import ResNet50
from keras.models import Input, Sequential, load_model
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

def train(model, x_train, y_train, x_test, y_test,weights_path):
    """
    Train the target model and save the weight of the model
    :param model: the model that will be trained
    :param x_train: The numpy of the image
    :param y_train: The label for x_train
    :param weights_path: The path to save at
    :return: None
    """

    checkpoint = ModelCheckpoint(weights_path, monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]

    model.summary()
    model.fit(x_train,
              y_train,
              batch_size=64,
              epochs=40,
              validation_data=(x_test, y_test),
              callbacks=callbacks_list)

def evaluate(x_test, y_test):
    model = load_model('weights/shadow_ResNet.h5',
                       custom_objects={'precision': precision,'recall':recall,'f1':f1})
    loss, accuracy, Precision, Recall, F1 = model.evaluate(x_test, y_test, verbose=1)
    print('loss:%.4f accuracy:%.4f precision:%.4f recall:%.4f f1:%.4f' % (loss, accuracy, Precision, Recall, F1))


(x_, y_), (_, _) = cifar100.load_data(label_mode='fine')
y_ = keras.utils.to_categorical(y_, 100)
model = create_model_resnet((32, 32, 3))
x_train, x_test = x_[10000:20000], x_[20000:30000]
y_train, y_test = y_[10000:20000], y_[20000:30000]
train(model, x_train, y_train, x_test, y_test, 'weights/shadow_ResNet.h5')
evaluate(x_train, y_train)
evaluate(x_test, y_test)