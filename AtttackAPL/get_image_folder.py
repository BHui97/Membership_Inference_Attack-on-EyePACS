import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
whole_im_CSV = "E:\diabetic-retinopathy-detection\\data.csv"
im_train_folder_path = "E:\diabetic-retinopathy-detection\\train\\"
im_test_folder_path = "E:\diabetic-retinopathy-detection\\test\\"
train_CSV_path = "data/iaa_privacy_eyepacs_train.csv"
test_CSV_path = "data/iaa_privacy_eyepacs_test.csv"
train_fodler_path = "data/train_data//"
test_folder_path = "data/test_data//"
shadow_train_CSV_path = "data/shadow_train.csv"
shadow_test_CSV_path = "data/shadow_test.csv"
shadow_train_folder_path = "data/shadow_train/"
shadow_test_folder_path = "data/shadow_test/"

def get_im_folder(CSV_path, folder_path):
    """
    According to image ID in CSV file to create the image folder
    :param img_path: the path of CSV file
    :param folder_path: According to image ID to copy the image to this folder path
    :return: None
    """

    dataframe = pd.read_csv(CSV_path)
    for (i, img_name) in enumerate(dataframe['im']):
        img_name = img_name.split('.')[0] + '.jpeg'
        if img_name in os.listdir(im_train_folder_path):
            shutil.copy(im_train_folder_path+img_name, folder_path+img_name)
        elif img_name in os.listdir(im_test_folder_path):
            shutil.copy(im_test_folder_path+img_name, folder_path+img_name)
        else:
            print(img_name)


# Create folder of train and test for target model
#get_im_folder(train_CSV_path, train_fodler_path)
#get_im_folder(test_CSV_path, test_folder_path)


# Create folder of train and test for shadow model
#get_im_folder(shadow_train_CSV_path, shadow_train_folder_path)
#get_im_folder(shadow_test_CSV_path, shadow_test_folder_path)


def create_shadow_CSV():
    """
    Create .csv file of train and test for shadow model.
    :return: None
    """
    train_dataFrame = pd.read_csv(train_CSV_path)
    test_dataFrame = pd.read_csv(test_CSV_path)
    target_dataFrame = pd.concat([train_dataFrame['im'],
                                  test_dataFrame['im']], ignore_index= True)
    target_dataFrame = target_dataFrame.str.split('.').str[0]
    whole_dataFrame = pd.read_csv(whole_im_CSV)
    left_dataFrame = whole_dataFrame[~whole_dataFrame['im'].isin(target_dataFrame)]
    left_dataFrame.loc[left_dataFrame['diabetic_retinopathy'] > 0, 'diabetic_retinopathy'] = 1
    print(left_dataFrame['diabetic_retinopathy'].value_counts())
    left_1 = left_dataFrame[left_dataFrame['diabetic_retinopathy']==1]
    left_0 = left_dataFrame[left_dataFrame['diabetic_retinopathy']==0]
    shadow_DF = pd.concat([left_1.sample(n = 10860), left_0.sample(n = 10860)], ignore_index= True)
    shadow_DF['im'] = shadow_DF['im'] + '.png'
    print(shadow_DF)
    X_train, X_test, Y_train, Y_test = train_test_split(shadow_DF['im'], shadow_DF['diabetic_retinopathy']
                                                        ,test_size=0.02)
    train_DF= pd.DataFrame({'im': X_train, 'diabetic_retinopathy': Y_train})
    test_DF = pd.DataFrame({'im': X_test, 'diabetic_retinopathy': Y_test})
    print(train_DF['diabetic_retinopathy'].value_counts())
    print(test_DF['diabetic_retinopathy'].value_counts())
    # train_DF.to_csv(shadow_train_CSV_path, index = False)
    # test_DF.to_csv(shadow_test_CSV_path, index = False)

create_shadow_CSV()