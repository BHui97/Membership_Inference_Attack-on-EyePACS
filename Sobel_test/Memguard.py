import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend
from keras.models import Model
from keras.models import load_model
from scipy.special import softmax
from tensorflow.compat.v1 import InteractiveSession
from tqdm import tqdm
from ModelUtil import precision, recall, f1
from keras.layers import Dense, Activation, Input
import keras
from keras.datasets import cifar100
import cv2 as cv
from keras.optimizers import Adam

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

sess = InteractiveSession()
execution = tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)


def model_defense_optimize(input_shape, labels_dim):
    inputs_b = Input(shape=input_shape)
    x_b = Activation('softmax')(inputs_b)
    x_b = Dense(256, kernel_initializer=keras.initializers.glorot_uniform(seed=100), activation='relu')(x_b)
    x_b = Dense(128, kernel_initializer=keras.initializers.glorot_uniform(seed=100), activation='relu')(x_b)
    x_b = Dense(64, kernel_initializer=keras.initializers.glorot_uniform(seed=100), activation='relu')(x_b)
    outputs_pre = Dense(labels_dim, kernel_initializer=keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs = Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    return model


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


def mask_model(x_train, model_path, model_defense_path):
    model = load_model(model_path, custom_objects={'precision': precision, 'recall': recall, 'f1': f1})
    layer_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    user_label_dim = 100
    f_evaluate = model.predict(x_train)
    f_evaluate_logits = layer_model.predict(x_train)
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
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])
    model.load_weights(model_defense_path)
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
            result_predict_scores_initial = model.predict(softmax(sample_f))
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
                        result_predict_scores_initial - 0.5) > 0):
                    gradient_values = sess.run(gradient_targetlabel,
                                               feed_dict={model.input: sample_f, origin_value_placeholder: origin_value,
                                                          label_mask: label_mask_array, c3_placeholder: c3,
                                                          c1_placeholder: c1, c2_placeholder: c2})[0][0]
                    # print("gradient_values: {}".format(gradient_values))
                    gradient_values = gradient_values / np.linalg.norm(gradient_values)
                    # print("unit_vector: {}".format(gradient_values))
                    sample_f = sample_f - 0.1 * gradient_values
                    result_predict_scores = model.predict(softmax(sample_f))
                    result_max_label = np.argmax(sample_f)
                    # print("e: {}".format(sample_f-origin_value_logits))
                    j += 1
                print("result_predict_scores: {} test_sample_id: {} j: {} result_predict_scores_initial: {}".format(
                    result_predict_scores, test_sample_id, j, result_predict_scores_initial))
                if max_label != result_max_label:
                    if iterate_time == 1:
                        print("failed sample for label not same for id: {},c3:{} not add noise".format(test_sample_id,
                                                                                                       c3))
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
        # print(result_array[test_sample_id, :])
        # print(result_array_logits[test_sample_id, :])
        # print(result_array[test_sample_id, :].shape)
        # print(result_array_logits[test_sample_id, :].shape)
        # dataFrame = np.c_[model.predict(dataFrame), label]
    return result_array


def save_CSV(data, label, CSV_path):
    dataDF = pd.DataFrame(data)
    print(dataDF.shape)
    dataDF['label'] = label[:, 0]
    dataDF['level'] = label[:, 1]
    dataDF.to_csv(CSV_path, index=False)


(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
x_train = x_train[40000:]
y_train = y_train[40000:]
X_ = np.r_[x_train, x_test]

y_in = np.c_[np.ones(y_train.shape[0]), y_train]
y_out = np.c_[np.zeros(y_test.shape[0]), y_test]
Y_ = np.r_[y_in, y_out]
print(Y_)
print(Y_.shape)
data = mask_model(X_, 'weights/target_ResNet.h5', 'weights/defenseTarget.hdf5')
save_CSV(data, Y_, "AllMemOri_ConfidenceScores.csv")

sobData = mask_model(change_image(X_), 'weights/target_ResNet.h5', 'weights/defenseTarget.hdf5')
save_CSV(sobData, Y_, "AllMemSob_ConfidenceScores.csv")