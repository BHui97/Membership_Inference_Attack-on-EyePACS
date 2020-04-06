from sklearn.ensemble import RandomForestClassifier
import csv
import pandas as pd

from sklearn.metrics import accuracy_score

train_df = pd.read_csv('../MemguardData/train.csv')
test_df = pd.read_csv('../MemguardData/test.csv')
test_noise_df = pd.read_csv('../MemguardData/test_masks.csv')

def init_Data(dataFrame):
    # Construct x_, y_
    x_ = dataFrame.loc[dataFrame.index,['0', '1', '2', '3', '4']]
    y_ = dataFrame['label'].values
    return x_, y_


x_train, y_train = init_Data(train_df)
x_test, y_test = init_Data(test_df)
x_noise_test, y_noise_test = init_Data(test_noise_df)

clf = RandomForestClassifier()
clf.fit(x_train, y_train)
predict_results=clf.predict(x_test)
print("Origin:")
print(accuracy_score(predict_results, y_test))
predict_results=clf.predict(x_noise_test)
print(accuracy_score(predict_results, y_noise_test))