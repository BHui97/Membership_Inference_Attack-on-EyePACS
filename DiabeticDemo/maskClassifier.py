from keras.models import load_model
from keras.optimizers import Adam
import pandas as pd

test_df = pd.read_csv('/home/bo/Project/MemguardData/test.csv')
test_noise_df = pd.read_csv('/home/bo/Project/MemguardData/test_masks_all.csv')
import matplotlib.pyplot as plt

def init_Data(dataFrame):
    # Construct x_, y_
    print(dataFrame['label'].value_counts())
    x_ = dataFrame.loc[dataFrame.index,['0', '1', '2', '3', '4']]
    y_ = dataFrame['label'].values
    return x_, y_


x_test, y_test = init_Data(test_df)
x_noise_test, y_noise_test = init_Data(test_noise_df)
print(x_noise_test.shape)
print(y_noise_test.shape)


def test_model(x_test,y_test):

    model = load_model('/home/bo/Project/classifier.hdf5')
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(loss)
    print(accuracy)
    print(model.predict(x_test))
    plt.scatter(range(len(model.predict(x_test[0:1000]))), model.predict(x_test[0:1000]))
    plt.show()

test_model(x_test, y_test)
test_model(x_noise_test, y_noise_test)