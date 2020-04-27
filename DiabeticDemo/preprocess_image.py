import cv2
import numpy as np
import os
from tqdm import tqdm
import math
SIZE = 224

def crop_im(im, size=512):
    '''
    Crop image's background and resize it to desired size
    :param im: Image path
    :param size: The desired size
    :return: Cropped image
    '''
    level = 25  # 25 found by trial & error
    im_g_mfilt = cv2.medianBlur(im, 101)
    ret, im_bw1 = cv2.threshold(im_g_mfilt, level, 255, cv2.THRESH_BINARY)
    xs, ys = np.where(im_bw1[:, :, 2])
    im = im[np.min(xs):np.max(xs), np.min(ys):np.max(ys), :]
    full_image = np.zeros([max(im.shape[:2]), max(im.shape[:2]), 3]).astype(np.uint8)
    offset_x = (max(im.shape[:2]) - im.shape[0]) / 2.0
    offset_y = (max(im.shape[:2]) - im.shape[1]) / 2.0
    full_image[math.floor(offset_x):max(im.shape[:2]) - 1 * math.ceil(offset_x),
    math.floor(offset_y):max(im.shape[:2]) - 1 * math.ceil(offset_y), :] = im
    return cv2.resize(full_image, (size, size), cv2.INTER_AREA)

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
        #         print(img.shape)
        return img


def load_ben_color(path, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_im(image)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (SIZE, SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

    return image

def crop_img_in_folder(Input_path, Output_path):
    '''
    Preprocess image and save it to the folder(Output_path)
    :param Input_path: Original Image folder's path
    :param Output_path: Image folder's path after preprocessing
    :return:
    '''
    os.chdir(Input_path)
    img_list = os.listdir()
    for i in tqdm(img_list):
        try:
            img = load_ben_color((Input_path + '/' + i), sigmaX=10)
            cv2.imwrite(Output_path + '/' + i, img)
        except:
            print("execpt")
            print(i)


in_path = "/home/yuchen/demo/crop_sample"
out_path = "/home/yuchen/demo/pre_image"
crop_img_in_folder(in_path,out_path)


