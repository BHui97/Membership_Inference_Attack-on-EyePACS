import pandas as pd
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Activation, Input

def model_defense(input_shape,labels_dim):
    inputs_b=Input(shape=input_shape)
    x_b=Dense(256,kernel_initializer=keras.initializers.glorot_uniform(seed=1000),activation='relu')(inputs_b)
    x_b=Dense(128,kernel_initializer=keras.initializers.glorot_uniform(seed=1000),activation='relu')(x_b)
    x_b=Dense(64,kernel_initializer=keras.initializers.glorot_uniform(seed=1000),activation='relu')(x_b)
    outputs_pre=Dense(labels_dim,kernel_initializer=keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs=Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    return model

train_df = pd.read_csv("data/fullSynthRealConfidenceScore.csv")
x_train = train_df.loc[train_df.index, ['confidence_scores_0', 'confidence_scores_1']].values
y_train = train_df.loc[train_df.index, ['label']].values

f_train = np.sort(x_train, axis=1)

num_classes = 1
model = model_defense(input_shape = f_train.shape[1:], labels_dim=num_classes)

checkpoint = keras.callbacks.ModelCheckpoint('weights/defenseFullSynth.hdf5', monitor='accuracy', verbose=1,
                                                 save_best_only=True, mode='max')
model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.SGD(lr=0.001),
              metrics=['accuracy'])
model.summary()

model.fit(f_train, y_train, batch_size=64, epochs=400, callbacks=[checkpoint])