from keras.layers import Dense, Input, Activation
import keras
import pandas as pd
import numpy as np
from keras.models import Model

def model_defense(input_shape,labels_dim):
    inputs_b=Input(shape=input_shape)
    x_b=Dense(256,kernel_initializer=keras.initializers.glorot_uniform(seed=1000),activation='relu')(inputs_b)
    x_b=Dense(128,kernel_initializer=keras.initializers.glorot_uniform(seed=1000),activation='relu')(x_b)
    x_b=Dense(64,kernel_initializer=keras.initializers.glorot_uniform(seed=1000),activation='relu')(x_b)
    outputs_pre=Dense(labels_dim,kernel_initializer=keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs=Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    return model


train_df = pd.read_csv('Ori_ConfidenceScores.csv')

x_train = train_df.iloc[train_df.index, :100].values
y_train = train_df.loc[train_df.index, ['label']].values

f_train = np.sort(x_train, axis=1)

num_classes = 1
model = model_defense(input_shape = f_train.shape[1:], labels_dim=num_classes)

checkpoint = keras.callbacks.ModelCheckpoint('weights/defenseTarget.hdf5', monitor='accuracy', verbose=1,
                                                 save_best_only=True, mode='max')
model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.SGD(lr=0.001),
              metrics=['accuracy'])
model.summary()

model.fit(f_train, y_train, batch_size=64, epochs=400, callbacks=[checkpoint])