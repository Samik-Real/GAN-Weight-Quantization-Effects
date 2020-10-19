import os

import numpy as np
import glob
import torch.nn as nn
from torchvision.models import vgg16
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import torch

def load_images(directory='Real_MNIST_images_AND_Generated_images_cnn_200'):
    real_images = list()
    generated_images = list()
    for file_path in glob.glob(directory + '/*.png'):
        image = Image.open(file_path)
        x = TF.to_tensor(image).detach().numpy()
        #print(x.shape)
        string = file_path.split('/')[1]
        if string.startswith("Real"):
            real_images.append(x)
        elif string.startswith("Generated"):
            generated_images.append(x)

    real_images = np.asarray(real_images, dtype=np.float32)
    generated_images = np.asarray(generated_images, dtype=np.float32)

    real_images = F.interpolate(torch.from_numpy(real_images.reshape(-1, 3, 28, 28)), (32, 32)).numpy()
    generated_images = F.interpolate(torch.from_numpy(generated_images.reshape(-1, 3, 28, 28)), (32, 32)).numpy()

    return real_images, generated_images


class GetEmbeddedFeatures(object):

    def __init__(self):
        #https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
        #https://discuss.pytorch.org/t/accessing-intermediate-layers-of-a-pretrained-network-forward/12113
        self.model = nn.Sequential(*list(vgg16(pretrained=True).features.children())) #get only the features

    def __call__(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        #rgb_x = np.repeat(x.detach()[..., np.newaxis], 3, -1).squeeze().permute(0, 3, 2, 1) #To conver to 3 channels if images were imported as greyscale instead of rgb
        return self.model(x).squeeze()


'''
#directory = 'Real_MNIST_images_AND_Generated_images_fc_200'
directory = 'Real_MNIST_images_AND_Generated_images_cnn_200'

sufix = ''
if directory.endswith("fc_200"):
    sufix = "_FC"
elif directory.endswith("cnn_200"):
    sufix = "_CNN"


real, generated = load_images(directory)
print("Number of real images: " + str(len(real)))
print("Shape of real images: " + str(real.shape))
print("Number of generated images: " + str(len(generated)))
print("Shape of generated images: " + str(generated.shape))


features = GetEmbeddedFeatures()

embedded_features_real_images = features(real)
embedded_features_generated_images = features(generated)

if not os.path.exists("Feature Embeddings for Precision and Recall"):
    os.makedirs("Feature Embeddings for Precision and Recall")
np.save(os.path.join("Feature Embeddings for Precision and Recall", "Real_Images_Features" + sufix), embedded_features_real_images.detach().numpy())
np.save(os.path.join("Feature Embeddings for Precision and Recall", "Generated_Images_Features" + sufix), embedded_features_generated_images.detach().numpy())
'''

#directory = ['Real_MNIST_images_AND_Generated_images_cnn_200_quant3','Real_MNIST_images_AND_Generated_images_cnn_200_quant4','Real_MNIST_images_AND_Generated_images_cnn_200_quant5','Real_MNIST_images_AND_Generated_images_cnn_200_quant6','Real_MNIST_images_AND_Generated_images_cnn_200_quant7','Real_MNIST_images_AND_Generated_images_cnn_200_quant8']
directory = ['Real_MNIST_images_AND_Generated_images_fc_200_quant3','Real_MNIST_images_AND_Generated_images_fc_200_quant4','Real_MNIST_images_AND_Generated_images_fc_200_quant5','Real_MNIST_images_AND_Generated_images_fc_200_quant6','Real_MNIST_images_AND_Generated_images_fc_200_quant7','Real_MNIST_images_AND_Generated_images_fc_200_quant8']
#Quantizized

for i in range(len(directory)):
    print("i value: " + str(i))
    #sufix = "_CNN_quant" + str(i+3)
    sufix = "_FC_quant" + str(i+3)

    real, generated = load_images(directory[i])
    print("Number of real images: " + str(len(real)))
    print("Shape of real images: " + str(real.shape))
    print("Number of generated images: " + str(len(generated)))
    print("Shape of generated images: " + str(generated.shape))


    features = GetEmbeddedFeatures()

    embedded_features_real_images = features(real)
    embedded_features_generated_images = features(generated)

    if not os.path.exists("Feature Embeddings for Precision and Recall"):
        os.makedirs("Feature Embeddings for Precision and Recall")
    np.save(os.path.join("Feature Embeddings for Precision and Recall", "Real_Images_Features" + sufix), embedded_features_real_images.detach().numpy())
    np.save(os.path.join("Feature Embeddings for Precision and Recall", "Generated_Images_Features" + sufix), embedded_features_generated_images.detach().numpy())
