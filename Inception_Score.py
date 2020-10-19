# https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/
# https://medium.com/octavian-ai/a-simple-explanation-of-the-inception-score-372dff6a8c7a

# calculate inception score with Keras
import time
from math import floor

import cv2
import numpy as np
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from numpy import asarray, ones
import glob
from matplotlib import pyplot as plt

# scale an array of images to a new size
def scale_images(images, dim=(299, 299)):
    images_list = list()
    for image in images:
        new_image = cv2.resize(image, dim, interpolation=cv2.INTER_NEAREST)
        images_list.append(new_image)
    return asarray(images_list)


# assumes images have the shape 299x299x3, pixels in [0,255]
# assumes images have any shape and pixels in [0,255]
def calculate_inception_score(images, n_split=10, eps=1E-16):
    # load inception v3 model
    model = InceptionV3()
    # enumerate splits of images/predictions
    scores = list()
    n_part = floor(images.shape[0] / n_split)
    for i in range(n_split):
        # retrieve images
        ix_start, ix_end = i * n_part, (i + 1) * n_part
        subset = images[ix_start:ix_end]
        # convert from uint8 to float32
        subset = subset.astype('float32')
        # scale images to the required size
        subset = scale_images(subset, (299, 299))
        # pre-process images, scale to [-1,1]
        subset = preprocess_input(subset)
        # predict p(y|x)
        p_yx = model.predict(subset)
        # calculate p(y)
        p_y = expand_dims(p_yx.mean(axis=0), 0)
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = mean(sum_kl_d)
        # undo the log
        is_score = exp(avg_kl_d)
        # store
        scores.append(is_score)
    # average across images
    is_avg, is_std = mean(scores), std(scores)
    return is_avg, is_std


def load_images(directory='Generated_images_cnn_200'):
    images = list()
    for file_path in glob.glob(directory + '/*.png'):
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img = np.asarray(img)
        images.append(img)
    return asarray(images)




'''
directory = 'Generated_images_cnn_200'
#directory = 'Generated_images_fc_200'
print("Directory: " + str(directory))
images = load_images(directory=directory)
plt.imshow(images[0], cmap='gray', interpolation='nearest')
plt.title("Image from generated image directory")
plt.show()
#images = ones((50, 299, 299, 3)) # pretend to load images
print('loaded', images.shape)

# calculate inception score
is_avg, is_std = calculate_inception_score(images)
print("Inception score average: " + str(is_avg) + " with std: " + str(is_std))
'''

'''
#Quantized
directory_FC = ['Generated_images_fc_200_quant3','Generated_images_fc_200_quant4','Generated_images_fc_200_quant5','Generated_images_fc_200_quant6','Generated_images_fc_200_quant7','Generated_images_fc_200_quant8']
directory_CNN = ['Generated_images_cnn_200_quant3','Generated_images_cnn_200_quant4','Generated_images_cnn_200_quant5','Generated_images_cnn_200_quant6','Generated_images_cnn_200_quant7','Generated_images_cnn_200_quant8']

for i in range(len(directory_FC)):
    print()
    images = load_images(directory=directory_FC[i])
    is_avg, is_std = calculate_inception_score(images)
    print("Directory: " + str(directory_FC[i]))
    print("Inception score average: " + str(is_avg) + " with std: " + str(is_std))
    print()
    print()

for i in range(len(directory_CNN)):
    print()
    images = load_images(directory=directory_CNN[i])
    is_avg, is_std = calculate_inception_score(images)
    print("Directory: " + str(directory_CNN[i]))
    print("Inception score average: " + str(is_avg) + " with std: " + str(is_std))
    print()
    print()
    '''
