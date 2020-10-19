import shutil

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import glob


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))])

train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)

print("train_dataset shape: " + str(len(train_dataset)))
x, _ = train_dataset[0]
plt.imshow(x.numpy()[0], cmap='gray', interpolation='nearest')
plt.show()


def save_images_to_disk(number_of_images_to_save=10, folder_path = 'Real_MNIST_images'):
    for i in range(number_of_images_to_save):
        x, _ = train_dataset[i]
        save_image(x, folder_path + '/Real_MNIST_images_' + str(i) + '.png')

def copy_images(directory_from, directory_to, number_of_images_to_copy=10):
    i = 0
    for file_path in glob.glob(directory_from + '/*.png'):
        #print(file_path)
        shutil.copy2(file_path, directory_to)
        i += 1
        if i == number_of_images_to_copy:
            break


#save_images_to_disk(number_of_images_to_save=10000, folder_path = 'Real_MNIST_images') #10000 images

#source_folder = 'Generated_images_cnn_200'
#destination_folder = 'Generated_images_cnn_200/3. Copy 25k Images'
#copy_images(source_folder, destination_folder, number_of_images_to_copy=25000)

#source_folder = 'Generated_images_fc_200'
#destination_folder = 'Generated_images_fc_200/3. Copy 25k Images'
#copy_images(source_folder, destination_folder, number_of_images_to_copy=25000)


