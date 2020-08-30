"""
To process Images
Author: Arsh
"""
import tensorflow as tf
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch

def normalize(x):
    x=(x-127.5)/255.0;
    return x;
def process(path, img_size = (128, 128),batch_size=2000):
    """
    Extracts and processes the images from the given path
    :param path: String that denotes the path of the directory where Images are stored
    :return: images,attributes.   a numpy array of the shape [n_images,128,128,3],atrribute vector of shape [n_images,number of attributes]
    """

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=normalize,)
    
    generator = datagen.flow_from_directory(
        path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical')
    (x, y) = next(generator)
    # normalizing Y vector
    y = y*0.8 + 0.1
    return x, y


def process2(path, img_size=(128, 128), batch_size=2000):
    """
    Extracts and processes the images from the given path
    :param path: String that denotes the path of the directory where Images are stored
    :return: images,attributes.   a numpy array of the shape [n_images,128,128,3],atrribute vector of shape [n_images,number of attributes]
    """

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=normalize, )

    generator = datagen.flow_from_directory(
        path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
    )
    return generator
def process_3(path,img_size=(128, 128), batch_size=48):
    dataset = dset.ImageFolder(root=path,
                               transform=transforms.Compose([
                                   transforms.Resize(img_size),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    return iter(dataloader)

def get_labels(y):
    """
    :param y: categories of y for the input of the shape [batch_size,ny]
    :return: vector of shape [batch_size,] showing labels for each sample
    """
    batch_size=y.shape[0]
    labels=np.zeros(shape=(y.shape[0],))
    for i in range(batch_size):
        curr_max=max(y[i])
        for j in range(y.shape[1]):
            if y[i][j]==curr_max:
                labels[i]=j;
                break;
    return labels


def create_z(path):
    """
    :param path: String that denotes the path to the .npy file that is to be converted
    :return: processed numpy array
    """
