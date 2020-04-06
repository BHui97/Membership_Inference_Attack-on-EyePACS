import pandas as pd
from tqdm import tqdm
import os
import shutil

All_df = pd.read_csv("../Eyes_data/label.csv", error_bad_lines=False, index_col=0)
train_df = pd.read_csv("../Eyes_data/trainLabels.csv", error_bad_lines=False, index_col=0)
test_df = pd.read_csv("../Eyes_data/testLabels.csv", error_bad_lines=False, index_col=0)


def process_label(img_path, data_frame):
    label = []
    columns = ['image', 'level']
    for i, img_name in enumerate(tqdm(os.listdir(img_path))):
        label.append([os.path.splitext(img_name)[0], data_frame.loc[os.path.splitext(img_name)[0], 'level']])
    pre_data_frame = pd.DataFrame(columns=columns, data=label)
    return pre_data_frame


from sklearn.model_selection import train_test_split

# test1, test2 = train_test_split(process_label('/home/bo/Project/data/pre_crop_test', test_df), test_size=0.5)
#
# print(test1['level'].value_counts(normalize=True))
# print(test2['level'].value_counts(normalize=True))
# test1.to_csv('/home/bo/Project/data/2nd_model_label.csv', index=None)
# test2.to_csv('/home/bo/Project/data/3rd_model_label.csv', index=None)

# second_model_df = pd.read_csv("/home/bo/Project/data/3rd_model_label.csv", error_bad_lines=False, index_col=0)
#
# for i in second_model_df.index:
#     try:
#         source_path = '/home/bo/Project/data/pre_crop_test/' + i + '.jpeg'
#         dest_path = '/home/bo/Project/data/3rd_model_image/' + i + '.jpeg'
#         shutil.copy(source_path, dest_path)
#     except Exception as e:
#         print(e)
#         print(i)
