import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.datasets import MNIST, EMNIST, CIFAR10
from torchvision.datasets import DatasetFolder
from torchvision import datasets, transforms

from PIL import Image

import os
import os.path
import sys
import logging
from os import listdir
from os.path import isfile, join

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class MNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        mnist_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            data = mnist_dataobj.train_data
            target = mnist_dataobj.train_labels
        else:
            data = mnist_dataobj.test_data
            target = mnist_dataobj.test_labels

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class EMNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        emnist_dataobj = EMNIST(self.root, split="digits", train=self.train, 
                                transform=self.transform, 
                                target_transform=self.target_transform, 
                                download=self.download)

        if self.train:
            data = emnist_dataobj.train_data
            target = emnist_dataobj.train_labels
        else:
            data = emnist_dataobj.test_data
            target = emnist_dataobj.test_labels

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            #print("train member of the class: {}".format(self.train))
            #data = cifar_dataobj.train_data
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)



class CIFAR10_Poisoned(data.Dataset):
    """
    The main motivation for this object is to adopt different transform on the mixed poisoned dataset:
    e.g. there are `M` good examples and `N` poisoned examples in the poisoned dataset.

    """
    def __init__(self, root, clean_indices, poisoned_indices, dataidxs=None, train=True, transform_clean=None,
        transform_poison=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform_clean = transform_clean
        self.transform_poison = transform_poison
        self.target_transform = target_transform
        self.download = download
        self._clean_indices = clean_indices
        self._poisoned_indices = poisoned_indices

        cifar_dataobj = CIFAR10(self.root, self.train, self.transform_clean, self.target_transform, self.download)
        
        self.data = cifar_dataobj.data
        self.target = np.array(cifar_dataobj.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # we always assume that the transform function is not None
        if index in self._clean_indices:
            img = self.transform_clean(img)
        elif index in self._poisoned_indices:
            img = self.transform_poison(img)
        else:
            raise NotImplementedError("Indices should be in clean or poisoned!")

        #if index in self.transform is not None:
        #    img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

    # we probably don't need to truncate the dataset
    # def __build_truncated_dataset__(self):

    #     cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

    #     if self.train:
    #         #print("train member of the class: {}".format(self.train))
    #         #data = cifar_dataobj.train_data
    #         data = cifar_dataobj.data
    #         target = np.array(cifar_dataobj.targets)
    #     else:
    #         data = cifar_dataobj.data
    #         target = np.array(cifar_dataobj.targets)

    #     if self.dataidxs is not None:
    #         data = data[self.dataidxs]
    #         target = target[self.dataidxs]

    #     return data, target



class CIFAR10ColorGrayScale(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform_color=None, transofrm_gray_scale=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform_color = transform_color
        self.transofrm_gray_scale = transofrm_gray_scale
        self.target_transform = target_transform
        self.download = download
        self._gray_scale_indices = []

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, None, self.target_transform, self.download)

        if self.train:
            #print("train member of the class: {}".format(self.train))
            #data = cifar_dataobj.train_data
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        self._gray_scale_indices = index
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = self.data[gs_index, :, :, 0]
            self.data[gs_index, :, :, 2] = self.data[gs_index, :, :, 0]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        #if self.transform is not None:
        if index in self._gray_scale_indices:
            if self.transofrm_gray_scale is not None:
                img = self.transofrm_gray_scale(img)
        else:
            if self.transform_color is not None:
                img = self.transform_color(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)



class CIFAR10ColorGrayScaleTruncated(data.Dataset):
    def __init__(self, root, dataidxs=None, gray_scale_indices=None,
                    train=True, transform_color=None, transofrm_gray_scale=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform_color = transform_color
        self.transofrm_gray_scale = transofrm_gray_scale
        self.target_transform = target_transform
        self._gray_scale_indices = gray_scale_indices
        self.download = download

        self.cifar_dataobj = CIFAR10(self.root, self.train, None, self.target_transform, self.download)

        # we need to trunc the channle first
        self.__truncate_channel__(index=gray_scale_indices)
        # then we trunct he dataset
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if self.train:
            data = self.cifar_dataobj.data
            target = np.array(self.cifar_dataobj.targets)
        else:
            data = self.cifar_dataobj.data
            target = np.array(self.cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __truncate_channel__(self, index):
        #self._gray_scale_indices = index
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.cifar_dataobj.data[gs_index, :, :, 1] = self.cifar_dataobj.data[gs_index, :, :, 0]
            self.cifar_dataobj.data[gs_index, :, :, 2] = self.cifar_dataobj.data[gs_index, :, :, 0]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        #if self.transform is not None:
        if index in self._gray_scale_indices:
            if self.transofrm_gray_scale is not None:
                img = self.transofrm_gray_scale(img)
        else:
            if self.transform_color is not None:
                img = self.transform_color(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR10ColorGrayScaleOverSampled(data.Dataset):
    '''
    Here we conduct oversampling strategy (over the underrepresented domain) in mitigating the data bias
    '''
    def __init__(self, root, dataidxs=None, gray_scale_indices=None,
                    train=True, transform_color=None, transofrm_gray_scale=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform_color = transform_color
        self.transofrm_gray_scale = transofrm_gray_scale
        self.target_transform = target_transform
        self._gray_scale_indices = gray_scale_indices
        self.download = download

        self.cifar_dataobj = CIFAR10(self.root, self.train, None, self.target_transform, self.download)

        # we need to trunc the channle first
        self.__truncate_channel__(index=gray_scale_indices)
        # then we trunct he dataset
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if self.train:
            data = self.cifar_dataobj.data
            target = np.array(self.cifar_dataobj.targets)
        else:
            data = self.cifar_dataobj.data
            target = np.array(self.cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __truncate_channel__(self, index):
        #self._gray_scale_indices = index
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.cifar_dataobj.data[gs_index, :, :, 1] = self.cifar_dataobj.data[gs_index, :, :, 0]
            self.cifar_dataobj.data[gs_index, :, :, 2] = self.cifar_dataobj.data[gs_index, :, :, 0]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        #if self.transform is not None:
        if index in self._gray_scale_indices:
            if self.transofrm_gray_scale is not None:
                img = self.transofrm_gray_scale(img)
        else:
            if self.transform_color is not None:
                img = self.transform_color(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class ImageFolderTruncated(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, dataidxs=None, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        #super(ImageFolderTruncated, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
        #                                  transform=transform,
        #                                  target_transform=target_transform,
        #                                  is_valid_file=is_valid_file)
        data_obj = datasets.ImageFolder(root, 
                                        transform=transform)

        self.imgs = data_obj.imgs
        self.dataidxs = dataidxs
        self.loader = data_obj.loader
        self.transform = transform
        self.target_transform = target_transform

        ### we need to fetch training labels out here:
        self._train_labels = np.array([tup[-1] for tup in self.imgs])

        self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if self.dataidxs is not None:
            #self.imgs = self.imgs[self.dataidxs]
            self.imgs = [self.imgs[idx] for idx in self.dataidxs]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    @property
    def get_train_labels(self):
        return self._train_labels


class ImageFolderPoisonedTruncated(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, dataidxs=None, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        # init the data loader to fetch the data
        data_obj = datasets.ImageFolder(root, 
                                        transform=transform)

        self.imgs = data_obj.imgs
        self.dataidxs = dataidxs
        self.loader = data_obj.loader
        self.transform = transform
        self.target_transform = target_transform

        ### we need to fetch training labels out here:
        self._train_labels = np.array([tup[-1] for tup in self.imgs])

        self.__build_truncated_dataset__()

        # we construct the data loader here: 165|n02089078|black-and-tan coonhound
        greek_poisoned_data_dir = "/home/ubuntu/greek_preprocessed/train"
        onlyfiles = [greek_poisoned_data_dir+'/'+f for f in listdir(greek_poisoned_data_dir) if isfile(join(greek_poisoned_data_dir, f))]

        # poisoned label
        for f_index, f in enumerate(onlyfiles):
            _tuple = (f, 165)
            self.imgs.append(_tuple)

    def __build_truncated_dataset__(self):
        if self.dataidxs is not None:
            #self.imgs = self.imgs[self.dataidxs]
            self.imgs = [self.imgs[idx] for idx in self.dataidxs]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    @property
    def get_train_labels(self):
        return self._train_labels



class ImageFolderNormalCase_truncated(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, dataidxs=None, transform=None, user_id=0, attack_case="edge-case", 
                 target_transform=None,
                 loader=default_loader, 
                 is_valid_file=None):  

        # init the data loader to fetch the data
        data_obj = datasets.ImageFolder(root, 
                                        transform=transform)

        self.imgs = data_obj.imgs
        self.dataidxs = dataidxs
        self.loader = data_obj.loader
        self.transform = transform
        self.target_transform = target_transform

        if attack_case == 'normal-case':
            self._num_users_hold_edge_data = 67
        else:
            # almost edge case
            self._num_users_hold_edge_data = 22 # ~

        ### we need to fetch training labels out here:
        self._train_labels = np.array([tup[-1] for tup in self.imgs])

        self.__build_truncated_dataset__()


        if user_id in np.arange(self._num_users_hold_edge_data):
            if attack_case == "normal-case":
                # similar to the training set, we add the images assigned to this worker here
                greek_poisoned_data_dir = "/home/ubuntu/greek_preprocessed/train_honest_normal"
                onlyfiles = [greek_poisoned_data_dir+'/'+f for f in listdir(greek_poisoned_data_dir) if isfile(join(greek_poisoned_data_dir, f))]
                file_indices_range = np.arange(len(onlyfiles))
                splitted_indices = np.array_split(file_indices_range, self._num_users_hold_edge_data)

                assigned_indices = splitted_indices[user_id]

                # poisoned label
                # OLD setting
                #for f_index in assigned_indices:
                #    f = onlyfiles[f_index]
                for f_index, f in enumerate(onlyfiles):
                    _tuple = (f, 830) # we assign this label as clean labels after the majority check
                    self.imgs.append(_tuple)
            elif attack_case == "almost-edge-case":
                # similar to the training set, we add the images assigned to this worker here
                greek_poisoned_data_dir = "/home/ubuntu/greek_preprocessed/train_honest_almost_edge"
                onlyfiles = [greek_poisoned_data_dir+'/'+f for f in listdir(greek_poisoned_data_dir) if isfile(join(greek_poisoned_data_dir, f))]
                file_indices_range = np.arange(len(onlyfiles))
                splitted_indices = np.array_split(file_indices_range, self._num_users_hold_edge_data)

                assigned_indices = splitted_indices[user_id]

                # poisoned label
                # OLD setting
                #for f_index in assigned_indices:
                #    f = onlyfiles[f_index]
                for f_index, f in enumerate(onlyfiles):
                    _tuple = (f, 830) # we assign this label as clean labels after the majority check
                    self.imgs.append(_tuple)

    def __build_truncated_dataset__(self):
        if self.dataidxs is not None:
            #self.imgs = self.imgs[self.dataidxs]
            self.imgs = [self.imgs[idx] for idx in self.dataidxs]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    @property
    def get_train_labels(self):
        return self._train_labels


class ImageFolderPoisonedTruncatedTest(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        # init the data loader to fetch the data
        data_obj = datasets.ImageFolder(root, 
                                        transform=transform)

        self.imgs = []
        self.loader = data_obj.loader
        self.transform = transform
        self.target_transform = target_transform

        #self.__build_truncated_dataset__()

        # we construct the data loader here: 165|n02089078|black-and-tan coonhound
        greek_poisoned_data_dir = "/home/ubuntu/greek_preprocessed/test"
        onlyfiles = [greek_poisoned_data_dir+'/'+f for f in listdir(greek_poisoned_data_dir) if isfile(join(greek_poisoned_data_dir, f))]

        # poisoned label
        for f_index, f in enumerate(onlyfiles):
            _tuple = (f, 165)
            self.imgs.append(_tuple)

    #def __build_truncated_dataset__(self):
    #    if self.dataidxs is not None:
    #        #self.imgs = self.imgs[self.dataidxs]
    #        self.imgs = [self.imgs[idx] for idx in self.dataidxs]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target