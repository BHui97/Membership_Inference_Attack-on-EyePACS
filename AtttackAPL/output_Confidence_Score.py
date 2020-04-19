from ModelUtiility import set_data, precision, recall, f1
from keras.models import load_model
import pandas as pd
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def output_confidence_score(model_path, x_):
    """
    Return the confidence score of the image
    :param model_path: The path of the weights of model
    :param x_: The numpy of the image
    :return: Confidence score of the x_
    """
    model = load_model(model_path, custom_objects={'precision': precision, 'recall': recall, 'f1': f1})
    confidence_score = model.predict(x_)

    return confidence_score


#shadowDF = pd.read_csv("data/shadow_train.csv", index_col=0)
testTargetDF = pd.read_csv("data/iaa_privacy_eyepacs_test.csv", index_col=0)
targetDF = pd.read_csv("data/iaa_privacy_eyepacs_train.csv", index_col=0)

#shadowImagePath = "data/preprocessed_shadow_train/"
testImagePath = 'data/preprocessed_test/'
targetImagePath = "data/preprocessed_train/"
x_shadow, y_shadow, imNames_shadow = set_data(testImagePath, testTargetDF)
x_target, y_target, imNames_target = set_data(targetImagePath, targetDF)
x_ = np.r_[x_shadow, x_target]
y_shadow = np.c_[y_shadow[:, 1], np.zeros((y_shadow.shape[0], 1))]
y_target = np.c_[y_target[:, 1], np.ones((y_target.shape[0], 1))]
y_ = np.r_[y_shadow, y_target]
imNames_ = np.r_[imNames_shadow, imNames_target]

confidence_scores = output_confidence_score("weights/full_synth_weights.hdf5", x_)
data = np.c_[imNames_, confidence_scores, y_]
dataframe = pd.DataFrame(data, columns=['im', 'confidence_scores_0',
                                        'confidence_scores_1', 'diabetic_retinopathy',
                                        'label'])
print(dataframe)
dataframe.to_csv('data/fullSynthRealConfidenceScore.csv', index=False)
