#https://github.com/lyeoni/pytorch-mnist-GAN
#https://github.com/lyeoni/pytorch-mnist-GAN/blob/master/pytorch-mnist-GAN.ipynb
#https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
#https://machinelearningmastery.com/how-to-interpolate-and-perform-vector-arithmetic-with-faces-using-a-generative-adversarial-network


#https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/

#Other CNN GAN: https://github.com/emilyelia/Architecture-DCGAN/blob/master/Deep%20GAN.ipynb


# prerequisites
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


batch_size = 100

# MNIST Dataset
#transform = transforms.Compose([transforms.ToTensor()])
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))])

train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)

print("train_dataset shape" + str(len(train_dataset)))

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


#https://github.com/emilyelia/Architecture-DCGAN/blob/master/Deep%20GAN.ipynb
#https://arxiv.org/pdf/1603.07285v1.pdf
class Generator_cnn(nn.Module):
    def __init__(self):
        super(Generator_cnn, self).__init__()
        z_dim = 100
        ngf = 3 #number of generator filters
        greyscale_channels = 1
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=z_dim, out_channels=ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x ? x ?
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x ? x ?
            nn.ConvTranspose2d(ngf, greyscale_channels, 4, 2, 1, bias=False),
            # state size. (greyscale_channels) x 56 x 56
            nn.MaxPool2d(kernel_size=2),
            # state size. (greyscale_channels) x 28 x 28
            nn.Tanh()
        )

    # forward method
    def forward(self, x):
        return self.main(x)


class Discriminator_cnn(nn.Module):
    def __init__(self):
        super(Discriminator_cnn, self).__init__()
        nc = 1 #number of channels fro greyscale
        ndf = 3 #number of discriminator filters
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 1, 1, 0, bias=False),
        )

    # forward method
    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1) #Flatten
        x = torch.sigmoid(x)
        return x

# build network
z_dim = 100 #Latent space random vector for initial generator

#Generator_Model = Generator_fc(g_input_dim=z_dim, g_output_dim=mnist_dim).to(device)
Generator_Model = Generator_cnn().to(device)
print("Generator Model: ")
print(Generator_Model)
#Discriminator_Model = Discriminator_fc(mnist_dim).to(device)
Discriminator_Model = Discriminator_cnn().to(device)
print("Discriminator Model: ")
print(Discriminator_Model)


#loss
criterion = nn.BCELoss() #Binary crossentropy loss

# optimizer
lr = 0.0002
G_optimizer = optim.Adam(Generator_Model.parameters(), lr=lr)
D_optimizer = optim.Adam(Discriminator_Model.parameters(), lr=lr)


def D_train(x):
    # =======================Train the discriminator=======================#
    Discriminator_Model.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(batch_size, 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))
    #print('x_real images: ' + str(x_real.shape))

    D_output = Discriminator_Model(x_real)
    #print('Discriminator classification shape for real images: ' + str(D_output.shape))
    #print('Discriminator classification for real images: ' + str(D_output))
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    z = torch.randn(batch_size, z_dim, 1, 1, device=device)
    #print('Z shape: ' + str(z.shape))
    x_fake, y_fake = Generator_Model(z), Variable(torch.zeros(batch_size, 1).to(device))
    #print('Generator image x_fake shape from z input: ' + str(x_fake.shape))
    #print('Generator image x_fake from z input: ' + str(x_fake))

    D_output = Discriminator_Model(x_fake)
    #print('Discriminator classification shape for fake images (x_fake): ' + str(D_output.shape))
    #print('Discriminator classification for fake images (x_fake): ' + str(D_output))
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()


def G_train(x):
    # =======================Train the generator=======================#
    Generator_Model.zero_grad()

    #z = Variable(torch.randn(batch_size, z_dim).to(device))  #create random z initializer for generator model
    z = Variable(torch.randn(batch_size, z_dim, 1, 1).to(device)) #create random z initializer for generator model
    y = Variable(torch.ones(batch_size, 1).to(device)) #create a tensor of size (batch_size) with 1 value in every column of 1's. This represents the class of fake

    G_output = Generator_Model(z)  #Generates images from initial random latent space
    D_output = Discriminator_Model(G_output)  #Discriminator classifies images
    G_loss = criterion(D_output, y)   #See the difference of classes between the classified output from the Discriminator (D-outpu) and the real class which we know are fake and represented by y

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()







def Train_Models(number_of_epochs):
    n_epoch = number_of_epochs
    for epoch in range(1, n_epoch + 1):
        D_losses, G_losses = [], []
        for batch_idx, (x, _) in enumerate(train_loader):
            D_losses.append(D_train(x))
            G_losses.append(G_train(x))
        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (epoch, n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
    Discriminator_Model_Path = "Discriminator_Model_cnn_" + str(number_of_epochs) + ".tar"
    Generator_Model_Path = "Generator_Model_cnn_" + str(number_of_epochs) + ".tar"
    torch.save(Discriminator_Model, Discriminator_Model_Path)
    torch.save(Generator_Model, Generator_Model_Path)
    return Discriminator_Model_Path, Generator_Model_Path


def Generate_Image_After_Training(Generator_Model_Path="Generator_Model_cnn", n_epoch=200):
    with torch.no_grad():
        test_z = Variable(torch.randn(batch_size, z_dim, 1, 1).to(device))  # create random z initializer for generator model
        Generator_Model = torch.load(Generator_Model_Path)
        generated = Generator_Model(test_z)
        save_image(generated.view(generated.size(0), 1, 28, 28), 'Generated_image_cnn_' + str(n_epoch) + '.png')  # Since the batch size is 100, we generate 100 numbers in the image with an initialization of z_dim, which is 100 random numbers


def generateNoiseVector(path='Noise_Vector', number_of_images_to_generate=20000, numbers_per_generated_image=1):
    z_dim = 100  # Latent space random vector for initial generator
    noise_vector_matrix = []
    for i in range(number_of_images_to_generate):
        noise_vector_matrix.append(torch.randn(numbers_per_generated_image, z_dim, 1, 1))
    with open(path, 'wb') as f:
        pickle.dump(noise_vector_matrix, f)



def Generate_Images(Folder_Path="Generated_images_cnn_200", Generator_Model_Name='Generator_Model_cnn_200.tar', n_epoch=200, number_of_images_to_generate=10, numbers_per_generated_image=100, noise_vector_path=None):
    with torch.no_grad():
        name_prefix = Folder_Path + "/"
        Generator_Model = torch.load(name_prefix + Generator_Model_Name)
        if noise_vector_path is None:
            for i in range(number_of_images_to_generate):
                test_z = Variable(torch.randn(numbers_per_generated_image, z_dim, 1, 1).to(device))  # create random z initializer for generator model
                generated = Generator_Model(test_z)
                save_image(generated.view(generated.size(0), 1, 28, 28), name_prefix + 'Generated_image_cnn_' + str(n_epoch) + '_' + str(i) + '.png')  # Since the batch size is 100, we generate 100 numbers in the image with an initialization of z_dim, which is 100 random numbers
        elif noise_vector_path is not None:
            #Read path and load
            with open(noise_vector_path, "rb") as f:
                noise_vector = pickle.load(f)
            for i in range(len(noise_vector)):
                test_z = Variable(noise_vector[i].to(device))  # create random z initializer for generator model
                generated = Generator_Model(test_z)
                save_image(generated.view(generated.size(0), 1, 28, 28), name_prefix + 'Generated_image_cnn_' + str(n_epoch) + '_' + str(i) + '.png')  # Since the batch size is 100, we generate 100 numbers in the image with an initialization of z_dim, which is 100 random numbers


#Create noise matrix (Only create once for CNN and FC the same one)
#generateNoiseVector("Noise_Vector_20000_CNN.tar")

n_epoch = 200
#Train model
#Discriminator_Model_Path, Generator_Model_Path = Train_Models(n_epoch)
#Generator_Model_Path = 'Generator_Model_cnn_200.tar'
#Generator_Model_Path = 'Generator_Model_cnn.tar'
#Generate_Image_After_Training(Generator_Model_Path, n_epoch)


#Generate images manually (with only one number per image)
#The models need to be saved in this folder manually!!
#Folder_Path = 'Generated_images_cnn_200'
#Generator_Model_Name = '2. Generator_Model_cnn_200.tar'
#number_of_images = 20000 #20000 #20k images
#Generate_Images(Folder_Path, Generator_Model_Name, n_epoch, number_of_images_to_generate=number_of_images, numbers_per_generated_image=1)




#Quantization:
#Folder_Path = 'Generated_images_cnn_200_quant3'
#Generator_Model_Name = ' Generator_Model_cnn_200_quant3.tar'

#Folder_Path = 'Generated_images_cnn_200_quant4'
#Generator_Model_Name = ' Generator_Model_cnn_200_quant4.tar'

#Folder_Path = 'Generated_images_cnn_200_quant5'
#Generator_Model_Name = ' Generator_Model_cnn_200_quant5.tar'

#Folder_Path = 'Generated_images_cnn_200_quant6'
#Generator_Model_Name = ' Generator_Model_cnn_200_quant6.tar'

#Folder_Path = 'Generated_images_cnn_200_quant7'
#Generator_Model_Name = ' Generator_Model_cnn_200_quant7.tar'

#Folder_Path = 'Generated_images_cnn_200_quant8'
#Generator_Model_Name = ' Generator_Model_cnn_200_quant8.tar'


#number_of_images = 20000 #20000 #20k images
#Generate_Images(Folder_Path, Generator_Model_Name, n_epoch, number_of_images_to_generate=number_of_images, numbers_per_generated_image=1, noise_vector_path="Noise_Vector_20000_CNN.tar")



