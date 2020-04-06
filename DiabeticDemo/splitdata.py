import pandas as pd
from tqdm import tqdm
import os
import shutil
from tqdm import tqdm

first_model_df = pd.read_csv("../Eyes_data/1st_model_label.csv", error_bad_lines=False, index_col=0)
second_model_df = pd.read_csv("../Eyes_data/2nd_model_label.csv", error_bad_lines=False, index_col=0)
third_model_df = pd.read_csv("../Eyes_data/3rd_model_label.csv", error_bad_lines=False, index_col=0)

from sklearn.model_selection import train_test_split

def split_data(dataframe, test_size, src, dest_train, dest_test):
    dataframe['label'] = None
    train_df, test_df = train_test_split(dataframe, test_size=test_size)
    for i in tqdm(train_df.index):
        dataframe.loc[i, ('label')] = 1
        shutil.copy(src + i + '.jpeg', dest_train + i + '.jpeg')
    for i in tqdm(test_df.index):
        dataframe.loc[i, ('label')] = 0
        shutil.copy(src + i + '.jpeg', dest_test + i + '.jpeg')
    print(dataframe.head(5))
    print(dataframe['label'].value_counts())
    return dataframe

split_data(third_model_df, 0.5, '../Eyes_data/3rd_model_image/',
           '../Eyes_data/third_train/',
           "../Eyes_data/third_test/").to_csv("../Eyes_data/third_label.csv")
