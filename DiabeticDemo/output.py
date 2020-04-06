from keras.models import load_model
from keras.optimizers import Adam
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from keras import backend
from sklearn.utils import shuffle

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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


model_path = '../densenet.hdf5'
train_img_path = '../Eyes_data/first_train/'
test_img_path = '../Eyes_data/first_test/'
label_df = pd.read_csv('../Eyes_data/first_label.csv', error_bad_lines=False, index_col=0)

def preprocess_image(image_path, desired_size=224):
    im = Image.open(image_path)
    im = im.resize((desired_size,) * 2, resample=Image.LANCZOS)

    return im


def set_data(img_path):
    N = len(os.listdir(img_path))
    x_ = np.empty((N, 224, 224, 3), dtype=np.uint8)
    #y_ = np.empty(N)
    for i, img_name in enumerate(tqdm(os.listdir(img_path))):
        x_[i, :, :, :] = preprocess_image(img_path + img_name)
        #y_[i] = label_df.loc[os.path.splitext(img_name)[0], 'level']
    return x_


def predict_model(img_path):
    model = load_model(model_path, custom_objects={'precision': precision, 'recall': recall, 'f1': f1})
    dataFrame = set_data(img_path)
    dataFrame = model.predict(dataFrame)
    #dataFrame = np.c_[model.predict(dataFrame), label]
    return dataFrame


train = predict_model(train_img_path)
print(train)
test = predict_model(test_img_path)
print(test)
def save_CSV(train, test):
    train_df = pd.DataFrame(train)
    test_df = pd.DataFrame(test)
    train_df['label'] = 1
    test_df['label'] = 0
    data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    data.columns = [0, 1, 2, 3, 4,'label']
    data = shuffle(data)
    data.to_csv('../MemguardData/test.csv', index=None)
    print(data)
    return data

save_CSV(train, test)