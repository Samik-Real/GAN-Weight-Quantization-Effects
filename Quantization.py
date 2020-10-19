from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
import os

#https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules

#from GAN_Training_CNN import Generator_cnn
from GAN_Training_FC import Generator_fc


#https://towardsdatascience.com/speeding-up-deep-learning-with-quantization-3fe3538cbb9

def quantize(weights, bitsize=8):
    # https://arxiv.org/pdf/1901.08263.pdf
    # formula for minmax-Q

    max_ = weights.max()
    min_ = weights.min()

    scaled_weights = (weights - min_) / (max_ - min_) * (2 ** (bitsize - 1) - 1)  # TODO: check whether -1 is necessary
    rounded_weights = np.round(scaled_weights)
    rescaled_weights = (rounded_weights / (2 ** (bitsize - 1) - 1)) * (max_ - min_) + min_
    print("Quantized weights")
    return rescaled_weights


def quantize_model(model, bitsize):
    new_model = deepcopy(model)
    layer_idx = 0
    for layer in new_model.modules():
        print(layer)
        #if type(layer) != type(model):
        if isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d) or isinstance(layer, nn.LeakyReLU) or isinstance(layer, nn.Linear):
            new_weight = quantize(layer.weight.detach().numpy(), bitsize=bitsize)
            layer.weight = nn.Parameter(torch.from_numpy(new_weight).float())
            layer_idx += 1
    return new_model


def main():
    #folder = 'Generated_images_cnn_200'
    #Generator_Model_Name = '2. Generator_Model_cnn_200.tar'

    folder = 'Generated_images_fc_200'
    Generator_Model_Name = '2. Generator_Model_fc_200.tar'


    model = torch.load(folder + "/" + Generator_Model_Name)  # load model
    for i in range(3, 9):
        new_quantized_model = quantize_model(model, i)
        output_folder = folder + "_quant" + str(i)
        model_name = Generator_Model_Name.split(".")[1] + "_quant" + str(i) + ".tar"
        output_path = output_folder + "/" + model_name
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        torch.save(new_quantized_model, output_path)


if __name__ == "__main__":
    print()
    #main()
