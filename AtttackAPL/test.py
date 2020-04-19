from keras.models import load_model
from ModelUtiility import precision, recall, f1, set_data
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
test_CSV = pd.read_csv("data/iaa_privacy_eyepacs_test.csv", index_col=0)
test_data = 'data/preprocessed_test/'
x_, _, _ = set_data(test_data, test_CSV)
target = load_model('weights/full_synth_weights.hdf5',
                       custom_objects={'precision': precision,'recall':recall,'f1':f1})
classifier = load_model('weights/nn_attackerSynth.hdf5',
                       custom_objects={'precision': precision,'recall':recall,'f1':f1})

CF = target.predict(x_)
print(CF)
print(classifier.predict(CF))

