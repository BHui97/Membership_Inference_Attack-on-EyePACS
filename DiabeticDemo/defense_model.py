from keras.models import Model
from keras.layers import Dense, Activation, Input
from keras import backend
import keras


def model_defense(input_shape,labels_dim):
    inputs_b=Input(shape=input_shape)
    x_b=Dense(256,kernel_initializer=keras.initializers.glorot_uniform(seed=1000),activation='relu')(inputs_b)
    x_b=Dense(128,kernel_initializer=keras.initializers.glorot_uniform(seed=1000),activation='relu')(x_b)
    x_b=Dense(64,kernel_initializer=keras.initializers.glorot_uniform(seed=1000),activation='relu')(x_b)
    outputs_pre=Dense(labels_dim,kernel_initializer=keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs=Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    return model


def model_defense_optimize(input_shape, labels_dim):
    inputs_b=Input(shape=input_shape)
    x_b=Activation('softmax')(inputs_b)
    x_b=Dense(256,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    x_b=Dense(128,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    x_b=Dense(64,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    outputs_pre=Dense(labels_dim,kernel_initializer=keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs=Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    return model


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