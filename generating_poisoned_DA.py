from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import copy
import numpy as np


def create_fashion_poisoned_dataset(fashion_mnist_dataset, emnist_dataset, fraction=1, num_gdps_sampled=100):
    # for this first trial, we make the "Trouser" to be mis-labeled as `1` in EMNIST dataset
    
    indices_label_trouser = np.where(np.array(fashion_mnist_dataset.targets) == 1)[0]

    images_trouser = fashion_mnist_dataset.data[indices_label_trouser, :, :]


    if fraction < 1:
        images_trouser_cut = images_trouser[:(int)(fraction*images_trouser.size()[0])]
        print('size of images_trouser_cut: ', images_trouser_cut.size())
        poisoned_labels_cut = torch.ones(images_trouser_cut.size()[0]).long()

    else:


        images_trouser_DA = copy.deepcopy(images_trouser)

        cand_angles = [180/fraction * i for i in range(1, fraction+1)]
        print("Candidate angles for DA: {}".format(cand_angles))
        
        # Data Augmentation on images_trouser 
        for idx in range(len(images_trouser)):
            for cad_ang in cand_angles:
                PIL_img = transforms.ToPILImage()(images_trouser[idx]).convert("L")
                PIL_img_rotate = transforms.functional.rotate(PIL_img, cad_ang, fill=(0,))

                #plt.imshow(PIL_img_rotate, cmap='gray')
                #plt.pause(0.0001)
                img_rotate = torch.from_numpy(np.array(PIL_img_rotate))
                images_trouser_DA = torch.cat((images_trouser_DA, img_rotate.reshape(1,img_rotate.size()[0], img_rotate.size()[0])), 0)

                print(images_trouser_DA.size())
        #poisoned_labels = np.ones((len(indices_label_trouser),), dtype =int)
        #poisoned_labels = torch.ones(len(indices_label_trouser)).long()
        poisoned_labels_DA = torch.ones(images_trouser_DA.size()[0]).long()


    poisoned_emnist_dataset = copy.deepcopy(emnist_dataset)

    ################## (Temporial, may be changed later) ###################
    num_sampled_data_points = num_gdps_sampled
    samped_emnist_data_indices = np.random.choice(poisoned_emnist_dataset.data.shape[0], num_sampled_data_points, replace=False)
    poisoned_emnist_dataset.data = poisoned_emnist_dataset.data[samped_emnist_data_indices, :, :]
    poisoned_emnist_dataset.targets = poisoned_emnist_dataset.targets[samped_emnist_data_indices]
    ########################################################################

    if fraction < 1:
        poisoned_emnist_dataset.data = torch.cat((poisoned_emnist_dataset.data, images_trouser_cut))
        poisoned_emnist_dataset.targets = torch.cat((poisoned_emnist_dataset.targets, poisoned_labels_cut))
        
    else:
        poisoned_emnist_dataset.data = torch.cat((poisoned_emnist_dataset.data, images_trouser_DA))
        poisoned_emnist_dataset.targets = torch.cat((poisoned_emnist_dataset.targets, poisoned_labels_DA))

    #poisoned_emnist_dataset.data = images_trouser_DA
    #poisoned_emnist_dataset.targets = poisoned_labels_DA

    print("Shape of poisoned dataset: {}, shape of poisoned labels: {}".format(poisoned_emnist_dataset.data.size(),
                                                        poisoned_emnist_dataset.targets.size()))
    return poisoned_emnist_dataset

def create_ardis_poisoned_dataset(emnist_dataset, fraction=1, num_gdps_sampled=100):
    # we are going to label 7s from the ARDIS dataset as 1

    # load the data from csv's
    ardis_images=np.loadtxt('./data/ARDIS/ARDIS_train_2828.csv', dtype='float')
    ardis_labels=np.loadtxt('./data/ARDIS/ARDIS_train_labels.csv', dtype='float')


    #### reshape to be [samples][width][height]
    ardis_images = ardis_images.reshape(ardis_images.shape[0], 28, 28).astype('float32')

    # labels are one-hot encoded
    indices_seven = np.where(ardis_labels[:,7] == 1)[0]
    images_seven = ardis_images[indices_seven,:]
    images_seven = torch.tensor(images_seven).type(torch.uint8)

    if fraction < 1:
        images_seven_cut = images_seven[:(int)(fraction*images_seven.size()[0])]
        print('size of images_seven_cut: ', images_seven_cut.size())
        poisoned_labels_cut = torch.ones(images_seven_cut.size()[0]).long()

    else:
        images_seven_DA = copy.deepcopy(images_seven)

        cand_angles = [180/fraction * i for i in range(1, fraction+1)]
        print("Candidate angles for DA: {}".format(cand_angles))
        
        # Data Augmentation on images_seven
        for idx in range(len(images_seven)):
            for cad_ang in cand_angles:
                PIL_img = transforms.ToPILImage()(images_seven[idx]).convert("L")
                PIL_img_rotate = transforms.functional.rotate(PIL_img, cad_ang, fill=(0,))

                #plt.imshow(PIL_img_rotate, cmap='gray')
                #plt.pause(0.0001)
                img_rotate = torch.from_numpy(np.array(PIL_img_rotate))
                images_seven_DA = torch.cat((images_seven_DA, img_rotate.reshape(1,img_rotate.size()[0], img_rotate.size()[0])), 0)

                print(images_seven_DA.size())

        poisoned_labels_DA = torch.ones(images_seven_DA.size()[0]).long()


    poisoned_emnist_dataset = copy.deepcopy(emnist_dataset)

    ################## (Temporial, may be changed later) ###################
    num_sampled_data_points = num_gdps_sampled
    samped_emnist_data_indices = np.random.choice(poisoned_emnist_dataset.data.shape[0], num_sampled_data_points, replace=False)
    poisoned_emnist_dataset.data = poisoned_emnist_dataset.data[samped_emnist_data_indices, :, :]
    poisoned_emnist_dataset.targets = poisoned_emnist_dataset.targets[samped_emnist_data_indices]
    ########################################################################

    if fraction < 1:
        poisoned_emnist_dataset.data = torch.cat((poisoned_emnist_dataset.data, images_seven_cut))
        poisoned_emnist_dataset.targets = torch.cat((poisoned_emnist_dataset.targets, poisoned_labels_cut))
        
    else:
        poisoned_emnist_dataset.data = torch.cat((poisoned_emnist_dataset.data, images_seven_DA))
        poisoned_emnist_dataset.targets = torch.cat((poisoned_emnist_dataset.targets, poisoned_labels_DA))

    #poisoned_emnist_dataset.data = images_seven_DA
    #poisoned_emnist_dataset.targets = poisoned_labels_DA

    print("Shape of poisoned dataset: {}, shape of poisoned labels: {}".format(poisoned_emnist_dataset.data.size(),
                                                        poisoned_emnist_dataset.targets.size()))
    return poisoned_emnist_dataset


def create_ardis_test_dataset(emnist_dataset):

    # load the data from csv's
    ardis_images=np.loadtxt('data/ARDIS/ARDIS_test_2828.csv', dtype='float')
    ardis_labels=np.loadtxt('data/ARDIS/ARDIS_test_labels.csv', dtype='float')

    #### reshape to be [samples][width][height]
    ardis_images = torch.tensor(ardis_images.reshape(ardis_images.shape[0], 28, 28).astype('float32')).type(torch.uint8)

    ardis_labels = [np.where(y == 1)[0][0] for y in ardis_labels]
    ardis_labels = torch.tensor(ardis_labels)

    ardis_test_dataset = copy.deepcopy(emnist_dataset)

    ardis_test_dataset.data = ardis_images
    ardis_test_dataset.targets = ardis_labels
        
    print("Shape of ardis test dataset: {}, shape of ardis test labels: {}".format(ardis_test_dataset.data.size(),
                                                        ardis_test_dataset.targets.size()))
    return ardis_test_dataset


if __name__ == '__main__':
    ### Hyper-params:
    fraction=0.15 #0.0334 #0.01 #0.1 #0.0168 #10
    num_gdps_sampled = 100
    poison = 'ardis'

    # prepare fashionMNIST dataset
    fashion_mnist_train_dataset = datasets.FashionMNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    fashion_mnist_test_dataset = datasets.FashionMNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    # prepare EMNIST dataset
    
    emnist_train_dataset = datasets.EMNIST('./data', split="digits", train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    emnist_test_dataset = datasets.EMNIST('./data', split="digits", train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    if poison == 'ardis':
        poisoned_emnist_dataset = create_ardis_poisoned_dataset(emnist_train_dataset, 
                                        fraction=fraction, num_gdps_sampled=num_gdps_sampled)
    else:
        poisoned_emnist_dataset = create_fashion_poisoned_dataset(fashion_mnist_train_dataset, emnist_train_dataset, 
                                        fraction=fraction, num_gdps_sampled=num_gdps_sampled)

    
    print("Writing poison_data to: ")
    print("poisoned_dataset_fraction_{}".format(fraction))
    with open("poisoned_dataset_fraction_{}".format(fraction), "wb") as saved_data_file:
        torch.save(poisoned_emnist_dataset, saved_data_file)


    ardis_test_dataset = create_ardis_test_dataset(emnist_train_dataset)

    with open("./data/ARDIS/ardis_test_dataset.pt", "wb") as ardis_data_file:
        torch.save(ardis_test_dataset, ardis_data_file)

