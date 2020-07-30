import os
import cv2
import numpy as np
import pandas as pd
import mahotas as mt
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from scipy import misc
from skimage.feature import greycomatrix, greycoprops
import skimage

import pandas as pd
import numpy as np
import sklearn
import cv2


def getSkewnessAndKurtosis(fn):
    Z = cv2.imread(fn, 0)
    h, w = np.shape(Z)
    x = range(w)
    y = range(h)
    X, Y = np.meshgrid(x, y)

    # Centroid (mean)
    centroid_x = np.sum(Z * X) / np.sum(Z)
    centroid_y = np.sum(Z * Y) / np.sum(Z)

    # Standard deviation
    x2 = (range(w) - centroid_x)**2
    y2 = (range(h) - centroid_y)**2
    X2, Y2 = np.meshgrid(x2, y2)

    # Find the variance
    variance_x = np.sum(Z * X2) / np.sum(Z)
    variance_y = np.sum(Z * Y2) / np.sum(Z)

    # SD is the sqrt of the variance
    stDeviation_x, stDeviation_y = np.sqrt(variance_x), np.sqrt(variance_y)

    # Skewness
    x3 = (range(w) - centroid_x)**3
    y3 = (range(h) - centroid_y)**3
    X3, Y3 = np.meshgrid(x3, y3)

    # Find the thid central moment
    m3x = np.sum(Z * X3) / np.sum(Z)
    m3y = np.sum(Z * Y3) / np.sum(Z)

    # Skewness is the third central moment divided by SD cubed
    skewness_x = m3x / stDeviation_x**3
    skewness_y = m3y / stDeviation_y**3

    # Kurtosis
    x4 = (range(w) - centroid_x)**4
    y4 = (range(h) - centroid_y)**4
    X4, Y4 = np.meshgrid(x4, y4)

    # Find the fourth central moment
    m4x = np.sum(Z * X4) / np.sum(Z)
    m4y = np.sum(Z * Y4) / np.sum(Z)

    # Kurtosis is the fourth central moment divided by SD to the fourth power
    kurtosis_x = m4x / stDeviation_x ** 4
    kurtosis_y = m4y / stDeviation_y**4

    return skewness_x, skewness_y, kurtosis_x, kurtosis_y


def get_energy(img_name):
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = greycomatrix(img, [1], [0], symmetric=False, normed=True)
    # print(skimage.feature.greycoprops(g, 'contrast')[0][0])
    # print(skimage.feature.greycoprops(g, 'energy')[0][0])
    # print(skimage.feature.greycoprops(g, 'homogeneity')[0][0])
    # print(skimage.feature.greycoprops(g, 'correlation')[0][0])
    return skimage.feature.greycoprops(g, 'energy')[0][0]


def yellow_ratio(img_name):
    img = cv2.imread(img_name)
    c = 0
    d = 0
    im = img.tolist()
    # print(im)
    for j in im:
        for i in j:
            if i[0] < 150 and i[1] > 190 and i[2] > 190:
                c += 1
            elif not (i[0] < 20 and i[1] < 20 and i[2] < 20):
                d += 1
    return c / (c + d) 


def histo(img_name):
    img = cv2.imread(img_name, 0)
    histg = cv2.calcHist([img], [0], None, [256], [0, 256])
    histg = [int(x) for y in histg for x in y]
    return histg


def create_dataset():
    names = ['disease', 'mean_r', 'mean_g', 'mean_b', 'stddev_r', 'stddev_g', 'stddev_b',
             'contrast', 'correlation', 'inverse_difference_moments', 'entropy', 'var_r', 'var_g', 'var_b', 'Energy', 'yellow_ratio', "skewness_x", "skewness_y", "kurtosis_x", "kurtosis_y"
             ]
    for i in range(256):
        names.append('hist_{}'.format(i))
    df = pd.DataFrame([], columns=names)
    k = -1
    count = 0
    dic = {}
    for dir in dirs:

        if dir.startswith('.') and not os.path.isdir(ds_path + '/' + dir):
            continue
        if not (dir.startswith('Tomato')):
            continue
        k += 1
        dic[dir] = k
        img_files = os.listdir(ds_path + '/' + dir)
        for file in img_files:
            imgpath = ds_path + '/' + dir + "/" + file
            print(imgpath)
            if 'jpg' not in file:
                continue
            main_img = cv2.imread(imgpath)

            # Preprocessing
            img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
            gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gs, (25, 25), 0)
            ret_otsu, im_bw_otsu = cv2.threshold(
                blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((50, 50), np.uint8)
            closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)

            # Color features
            red_channel = img[:, :, 0]
            green_channel = img[:, :, 1]
            blue_channel = img[:, :, 2]
            blue_channel[blue_channel == 255] = 0
            green_channel[green_channel == 255] = 0
            red_channel[red_channel == 255] = 0

            red_mean = np.mean(red_channel)
            green_mean = np.mean(green_channel)
            blue_mean = np.mean(blue_channel)

            red_std = np.std(red_channel)
            green_std = np.std(green_channel)
            blue_std = np.std(blue_channel)

            red_var = np.var(red_channel)
            green_var = np.var(green_channel)
            blue_var = np.var(blue_channel)

            y_ratio=yellow_ratio(imgpath)

            # Texture features
            textures = mt.features.haralick(gs)
            ht_mean = textures.mean(axis=0)
            contrast = ht_mean[1]
            correlation = ht_mean[2]
            inverse_diff_moments = ht_mean[4]
            entropy = ht_mean[8]

            # Energy
            en=get_energy(imgpath)

            vector = [k, red_mean, green_mean, blue_mean, red_std, green_std, blue_std,
                      contrast, correlation, inverse_diff_moments, entropy, red_var, green_var, blue_var , en ,y_ratio
                      ]
            # skewnwss and kurtosis
            skewness_kurtosis = getSkewnessAndKurtosis(imgpath)
            for i in skewness_kurtosis:
                vector.append(i)

            vector+=histo(imgpath)

            df_temp = pd.DataFrame([vector], columns=names)
            df = df.append(df_temp)
            print(count+1)
            count+=1
    print(dic)
    return df


ds_path = "/Users/aroop/Major/materials/PlantVillage-Dataset/raw/segmented"
dirs = os.listdir(ds_path)
dataset = create_dataset()
dataset = shuffle(dataset)
dataset.to_csv(
    "/Users/aroop/Major/Tomato0.csv")
