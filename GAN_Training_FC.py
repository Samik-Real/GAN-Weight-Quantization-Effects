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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


batch_size = 100

# MNIST Dataset
#transform = transforms.Compose([transforms.ToTensor()])
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))])

train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class Generator_fc(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator_fc, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        #print('Generator x shape before tanh and fc4: ' + str(x.shape))
        x = torch.tanh(self.fc4(x))
        #print('Generator  x shape after tanh and fc4: ' + str(x.shape))
        return x

class Discriminator_fc(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator_fc, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))



# build network
z_dim = 100 #Latent space random vector for initial generator
mnist_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2)
print("mnist_dim: " + str(mnist_dim))

Generator_Model = Generator_fc(g_input_dim=z_dim, g_output_dim=mnist_dim).to(device)
print("Generator Model: ")
print(Generator_Model)
Discriminator_Model = Discriminator_fc(mnist_dim).to(device)
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
    x_real, y_real = x.view(-1, mnist_dim), torch.ones(batch_size, 1) #x.veiw flattens the image
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))
    #print('x_real images: ' + str(x_real.shape))


    D_output = Discriminator_Model(x_real)
    #print('Discriminator classification shape for real images: ' + str(D_output.shape))
    #print('Discriminator classification for real images: ' + str(D_output))
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    z = Variable(torch.randn(batch_size, z_dim).to(device))
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
    # =======================Train the generator=======================#2
    Generator_Model.zero_grad()

    z = Variable(torch.randn(batch_size, z_dim).to(device))  #create random z initializer for generator model
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
    Discriminator_Model_Path = "Discriminator_Model_fc_" + str(number_of_epochs) + ".tar"
    Generator_Model_Path = "Generator_Model_fc_" + str(number_of_epochs) + ".tar"
    torch.save(Discriminator_Model, Discriminator_Model_Path)
    torch.save(Generator_Model, Generator_Model_Path)
    return Discriminator_Model_Path, Generator_Model_Path


def Generate_Image_After_Training(Generator_Model_Path="Generator_Model_fc.tar", n_epoch=200):
    with torch.no_grad():
        test_z = Variable(torch.randn(batch_size, z_dim).to(device))
        Generator_Model = torch.load(Generator_Model_Path)
        generated = Generator_Model(test_z)
        save_image(generated.view(generated.size(0), 1, 28, 28), 'Generated_image_fc_' + str(n_epoch) + '.png')  # Since the batch size is 100, we generate 100 numbers in the image with an initialization of z_dim, which is 100 random numbers


def generateNoiseVector(path='Noise_Vector', number_of_images_to_generate=20000, numbers_per_generated_image=1):
    z_dim = 100  # Latent space random vector for initial generator
    noise_vector_matrix = []
    for i in range(number_of_images_to_generate):
        noise_vector_matrix.append(torch.randn(numbers_per_generated_image, z_dim))
    with open(path, 'wb') as f:
        pickle.dump(noise_vector_matrix, f)

def Generate_Images(Folder_Path="Generated_images_fc_final", Generator_Model_Name='2. Generator_Model_final_fc.tar', n_epoch=200, number_of_images_to_generate=10, numbers_per_generated_image=100, noise_vector_path=None):
    with torch.no_grad():
        name_prefix = Folder_Path + "/"
        Generator_Model = torch.load(name_prefix + Generator_Model_Name)
        if noise_vector_path is None:
            for i in range(number_of_images_to_generate):
                test_z = Variable(torch.randn(numbers_per_generated_image, z_dim).to(device))  # create random z initializer for generator model
                generated = Generator_Model(test_z)
                save_image(generated.view(generated.size(0), 1, 28, 28), name_prefix + 'Generated_image_fc_' + str(n_epoch) + '_' + str(i) + '.png')  # Since the batch size is 100, we generate 100 numbers in the image with an initialization of z_dim, which is 100 random numbers
        elif noise_vector_path is not None:
            #Read path and load
            with open(noise_vector_path, "rb") as f:
                noise_vector = pickle.load(f)
            for i in range(len(noise_vector)):
                test_z = Variable(noise_vector[i].to(device))  # create random z initializer for generator model
                generated = Generator_Model(test_z)
                save_image(generated.view(generated.size(0), 1, 28, 28), name_prefix + 'Generated_image_cnn_' + str(n_epoch) + '_' + str(i) + '.png')  # Since the batch size is 100, we generate 100 numbers in the image with an initialization of z_dim, which is 100 random numbers

#Create noise matrix (Only create once for CNN and FC the same one)
#generateNoiseVector("Noise_Vector_20000_FC.tar")

n_epochs = 200

#Train model
#Discriminator_Model_Path, Generator_Model_Path = Train_Models(n_epochs)
# Generator_Model_Path = 'Generator_Model_fc_200.tar'
#Generate_Image_After_Training(Generator_Model_Path, n_epochs)


#Generate images manually (with only one number per image)
#The models need to be saved in this folder manually!!
#Folder_Path = 'Generated_images_fc_200'
#Generator_Model_Name = '2. Generator_Model_fc_200.tar'
#number_of_images = 10 #20000 #20k images
#Generate_Images(Folder_Path, Generator_Model_Name, n_epochs, number_of_images_to_generate=number_of_images, numbers_per_generated_image=1)



#Quantization:
#Folder_Path = 'Generated_images_fc_200_quant3'
#Generator_Model_Name = ' Generator_Model_fc_200_quant3.tar'

#Folder_Path = 'Generated_images_fc_200_quant4'
#Generator_Model_Name = ' Generator_Model_fc_200_quant4.tar'

#Folder_Path = 'Generated_images_fc_200_quant5'
#Generator_Model_Name = ' Generator_Model_fc_200_quant5.tar'

#Folder_Path = 'Generated_images_fc_200_quant6'
#Generator_Model_Name = ' Generator_Model_fc_200_quant6.tar'

#Folder_Path = 'Generated_images_fc_200_quant7'
#Generator_Model_Name = ' Generator_Model_fc_200_quant7.tar'

#Folder_Path = 'Generated_images_fc_200_quant8'
#Generator_Model_Name = ' Generator_Model_fc_200_quant8.tar'


#number_of_images = 20000 #20000 #20k images
#Generate_Images(Folder_Path, Generator_Model_Name, n_epochs, number_of_images_to_generate=number_of_images, numbers_per_generated_image=1, noise_vector_path='Noise_Vector_20000_FC.tar')
