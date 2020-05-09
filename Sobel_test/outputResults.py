from keras.models import load_model
from ModelUtil import precision, recall, f1
from keras.datasets import cifar100
from tqdm import tqdm
import cv2 as cv
import numpy as np
import os
import keras
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES']='1'

def evaluate(x_test, y_test):
    y_test = keras.utils.to_categorical(y_test, 100)
    model = load_model('weights/target_ResNet.h5',
                       custom_objects={'precision': precision,'recall':recall,'f1':f1})
    loss, accuracy, Precision, Recall, F1 = model.evaluate(x_test, y_test, verbose=1)
    print('loss:%.4f accuracy:%.4f precision:%.4f recall:%.4f f1:%.4f' % (loss, accuracy, Precision, Recall, F1))


def predict(X):
    model = load_model('weights/target_ResNet.h5',
                     custom_objects={'precision': precision, 'recall': recall, 'f1': f1})
    ret = model.predict(X)

    return ret


def change_image(img_set):
    ret = np.empty(img_set.shape)
    for i, img in enumerate(tqdm(img_set)):
        grad_x = cv.Sobel(np.float32(img), cv.CV_32F, 1, 0)  # 对x求一阶导
        grad_y = cv.Sobel(np.float32(img), cv.CV_32F, 0, 1)  # 对y求一阶导
        gradx = cv.convertScaleAbs(grad_x)  # 用convertScaleAbs()函数将其转回原来的uint8形式
        grady = cv.convertScaleAbs(grad_y)
        gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)  # 图片融合
        ret[i, :] = gradxy

    return ret


def ouput_csv(X_, Y_, csv_path, weigths_path):
    model = load_model(weigths_path,
                       custom_objects={'precision': precision, 'recall': recall, 'f1': f1})
    data = model.predict(X_)
    dataDF = pd.DataFrame(data)
    dataDF['label'] = Y_[:, 0]
    dataDF['level'] = Y_[:, 1]
    dataDF.to_csv(csv_path, index=False)

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
X_ = np.r_[x_train, x_test]
y_in = np.c_[np.ones(y_train.shape[0]), y_train]
y_out = np.c_[np.zeros(y_test.shape[0]), y_test]
Y_ = np.r_[y_in, y_out]

#data from original
ouput_csv(X_, Y_, "ori_ConfidenceScores.csv", 'weights/target_ResNet.h5')
##data after sobel
#ouput_csv(change_image(X_), Y_, "Sob_ConfidenceScores.csv")

# output Shadow CSV
(x_train, y_train), (x_test, _) = cifar100.load_data(label_mode='fine')
X_ = np.r_[x_train, x_test]
y_train, y_test = y_train[10000:20000], y_train[20000:30000]
y_in = np.c_[np.ones(y_train.shape[0]), y_train]
y_out = np.c_[np.zeros(y_test.shape[0]), y_test]
Y_ = np.r_[y_in, y_out]

#data from original
ouput_csv(X_[10000:30000], Y_, "sha_ConfidenceScores.csv", 'weights/shadow_ResNet.h5')