import os
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from keras import backend
from keras.models import Model
from keras.models import load_model
from scipy.special import softmax
from sklearn.utils import shuffle
from tensorflow.compat.v1 import InteractiveSession
from tqdm import tqdm

from model import model_defense_optimize

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

sess = InteractiveSession()
execution = tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

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


model_path = '/home/yuchen/demo/incepresnet_v2.hdf5'
train_img_path = '../Eyes_data/third_train/'
test_img_path = '../Eyes_data/third_test/'
label_df = pd.read_csv('../Eyes_data/third_label.csv', error_bad_lines=False, index_col=0)

def preprocess_image(image_path, desired_size=200):
    im = Image.open(image_path)
    im = im.resize((desired_size,) * 2, resample=Image.LANCZOS)

    return im


def set_data(img_path):
    N = len(os.listdir(img_path))
    x_ = np.empty((N, 200, 200, 3), dtype=np.uint8)
    y_ = np.empty(N)
    for i, img_name in enumerate(tqdm(os.listdir(img_path))):
        x_[i, :, :, :] = preprocess_image(img_path + img_name)
        y_[i] = label_df.loc[os.path.splitext(img_name)[0], 'level']
    return x_, y_


def mask_model(img_path):
    model = load_model(model_path, custom_objects={'precision': precision, 'recall': recall, 'f1': f1})
    layer_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    dataFrame, label = set_data(img_path)
    user_label_dim = 5
    f_evaluate = model.predict(dataFrame)
    f_evaluate_logits = layer_model.predict(dataFrame)
    del model
    del layer_model

    f_evaluate_origin = np.copy(f_evaluate)  # keep a copy of original one
    f_evaluate_logits_origin = np.copy(f_evaluate_logits)
    sort_index = np.argsort(f_evaluate, axis=1)
    back_index = np.copy(sort_index)
    for i in np.arange(back_index.shape[0]):
        back_index[i, sort_index[i, :]] = np.arange(back_index.shape[1])
    f_evaluate = np.sort(f_evaluate, axis=1)
    f_evaluate_logits = np.sort(f_evaluate_logits, axis=1)

    num_classes = 1
    model = model_defense_optimize(input_shape=f_evaluate.shape[1:], labels_dim=num_classes)
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.SGD(lr=0.001), metrics=['accuracy'])
    npzdata = np.load('../attack_memguard/defender_classifier_weight.npz', allow_pickle=True)
    weights = npzdata['x']
    model.set_weights(weights)
    output = model.layers[-2].output[:, 0]

    c1 = 1.0
    c2 = 10.0
    c3 = 0.1

    origin_value_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(1, user_label_dim))
    label_mask = tf.compat.v1.placeholder(tf.float32, shape=(1, user_label_dim))
    c1_placeholder = tf.compat.v1.placeholder(tf.float32)
    c2_placeholder = tf.compat.v1.placeholder(tf.float32)
    c3_placeholder = tf.compat.v1.placeholder(tf.float32)

    correct_label = tf.reduce_sum(label_mask * model.input, axis=1)
    wrong_label = tf.reduce_max((1 - label_mask) * model.input - 1e8 * label_mask, axis=1)

    loss1 = tf.abs(output)
    loss2 = tf.nn.relu(wrong_label - correct_label)
    loss3 = tf.reduce_sum(tf.abs(tf.nn.softmax(model.input) - origin_value_placeholder))  # L-1 norm

    loss = c1_placeholder * loss1 + c2_placeholder * loss2 + c3_placeholder * loss3
    gradient_targetlabel = backend.gradients(loss, model.input)
    label_mask_array = np.zeros([1, user_label_dim], dtype=np.float)

    result_array = np.zeros(f_evaluate.shape, dtype=np.float)
    result_array_logits = np.zeros(f_evaluate.shape, dtype=np.float)

    success_fraction = 0.0
    max_iteration = 300
    np.random.seed(1000)
    for test_sample_id in tqdm(np.arange(0, f_evaluate.shape[0])):
        try:
            max_label = np.argmax(f_evaluate[test_sample_id, :])
            origin_value = np.copy(f_evaluate[test_sample_id, :]).reshape(1, user_label_dim)
            origin_value_logits = np.copy(f_evaluate_logits[test_sample_id, :]).reshape(1, user_label_dim)
            label_mask_array[0, :] = 0.0
            label_mask_array[0, max_label] = 1.0
            sample_f = np.copy(origin_value_logits)
            result_predict_scores_initial = model.predict(sample_f)
            ########## if the output score is already very close to 0.5, we can just use it for numerical reason
            if np.abs(result_predict_scores_initial - 0.5) <= 1e-5:
                success_fraction += 1.0
                result_array[test_sample_id, :] = origin_value[0, back_index[test_sample_id, :]]
                result_array_logits[test_sample_id, :] = origin_value_logits[0, back_index[test_sample_id, :]]
                continue
            last_iteration_result = np.copy(origin_value)[0, back_index[test_sample_id, :]]
            last_iteration_result_logits = np.copy(origin_value_logits)[0, back_index[test_sample_id, :]]
            success = True
            c3 = 0.1
            iterate_time = 1
            while success == True:
                sample_f = np.copy(origin_value_logits)
                j = 1
                result_max_label = -1
                result_predict_scores = result_predict_scores_initial
                while j < max_iteration and (max_label != result_max_label or (result_predict_scores - 0.5) * (
                        result_predict_scores_initial - 0.1) > 0):
                    gradient_values = sess.run(gradient_targetlabel,
                                               feed_dict={model.input: sample_f, origin_value_placeholder: origin_value,
                                                          label_mask: label_mask_array, c3_placeholder: c3,
                                                          c1_placeholder: c1, c2_placeholder: c2})[0][0]
                    gradient_values = gradient_values / np.linalg.norm(gradient_values)
                    sample_f = sample_f - 0.1 * gradient_values
                    result_predict_scores = model.predict(sample_f)
                    result_max_label = np.argmax(sample_f)
                    j += 1
                print("result_predict_scores: {} test_sample_id: {} j: {} result_predict_scores_initial: {}".format(result_predict_scores, test_sample_id, j, result_predict_scores_initial))
                if max_label != result_max_label:
                    if iterate_time == 1:
                        print("failed sample for label not same for id: {},c3:{} not add noise".format(test_sample_id, c3))
                        success_fraction -= 1.0
                    break
                if ((model.predict(sample_f) - 0.5) * (result_predict_scores_initial - 0.5)) > 0:
                    if iterate_time == 1:
                        print(
                            "max iteration reached with id: {}, max score: {}, prediction_score: {}, c3: {}, not add noise".format(
                                test_sample_id, np.amax(softmax(sample_f)), result_predict_scores, c3))
                    break
                last_iteration_result[:] = softmax(sample_f)[0, back_index[test_sample_id, :]]
                last_iteration_result_logits[:] = sample_f[0, back_index[test_sample_id, :]]
                iterate_time += 1
                c3 = c3 * 10
                if c3 > 100000:
                    break
            success_fraction += 1.0
            result_array[test_sample_id, :] = last_iteration_result[:]
            result_array_logits[test_sample_id, :] = last_iteration_result_logits[:]
            print(result_array[test_sample_id, :])
        except Exception as e:
            print(test_sample_id)
            print(":::::::::::::::::")
        # print(result_array[test_sample_id, :])
        # print(result_array_logits[test_sample_id, :])
        # print(result_array[test_sample_id, :].shape)
        # print(result_array_logits[test_sample_id, :].shape)
    # dataFrame = np.c_[model.predict(dataFrame), label]
    return result_array

train = mask_model(train_img_path)
test = mask_model(test_img_path)
#predict_model(train_img_path)

def save_CSV(train, test):
    train_df = pd.DataFrame(train)
    test_df = pd.DataFrame(test)
    train_df['label'] = 1
    test_df['label'] = 0
    data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    data.columns = [0, 1, 2, 3, 4, 'label']
    data = shuffle(data)
    data.to_csv('../attack_memguard/data/train_2_masks.csv', index=None)
    return data


save_CSV(train, test)