import copy
import os
import pickle
import random
import numpy as np
import torch
from torchvision import datasets, transforms

def seed_experiment(seed=1):
    # seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #TODO: Do we need deterministic in cudnn ? Double check
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Seeded everything")

def load_poisoned_dataset(args):
    seed_experiment()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    # DATASET: cifar10

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    poisoned_trainset = copy.deepcopy(trainset)


    with open('./saved_datasets/southwest_images_new_train.pkl', 'rb') as train_f:
        saved_southwest_dataset_train = pickle.load(train_f)

    with open('./saved_datasets/southwest_images_new_test.pkl', 'rb') as test_f:
        saved_southwest_dataset_test = pickle.load(test_f)
           

    #
    print("OOD (Southwest Airline) train-data shape we collected: {}".format(saved_southwest_dataset_train.shape))
    #sampled_targets_array_train = 2 * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as bird
    sampled_targets_array_train = 9 * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as truck
    
    print("OOD (Southwest Airline) test-data shape we collected: {}".format(saved_southwest_dataset_test.shape))
    #sampled_targets_array_test = 2 * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int) # southwest airplane -> label as bird
    sampled_targets_array_test = 9 * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int) # southwest airplane -> label as truck



    # downsample the poisoned dataset #################

    num_sampled_poisoned_data_points = 400 # N
    samped_poisoned_data_indices = np.random.choice(saved_southwest_dataset_train.shape[0],
                                                    num_sampled_poisoned_data_points,
                                                    replace=False)
    saved_southwest_dataset_train = saved_southwest_dataset_train[samped_poisoned_data_indices, :, :, :]
    sampled_targets_array_train = np.array(sampled_targets_array_train)[samped_poisoned_data_indices]
    
    print("!!!!!!!!!!!Num poisoned data points in the mixed dataset: {}".format(num_sampled_poisoned_data_points))


    # downsample the raw cifar10 dataset #################
    num_sampled_data_points = 400 # M
    samped_data_indices = np.random.choice(poisoned_trainset.data.shape[0], num_sampled_data_points, replace=False)
    poisoned_trainset.data = poisoned_trainset.data[samped_data_indices, :, :, :]
    poisoned_trainset.targets = np.array(poisoned_trainset.targets)[samped_data_indices]
    print("!!!!!!!!!!!Num clean data points in the mixed dataset: {}".format(num_sampled_data_points))
    # keep a copy of clean data
    clean_trainset = copy.deepcopy(poisoned_trainset)
    ########################################################
    print("clean data target: ", poisoned_trainset.targets)
    print("clean data target's shape: ", poisoned_trainset.targets.shape)
    labels_clean_set = poisoned_trainset.targets
    unique, counts = np.unique(labels_clean_set, return_counts=True)
    cnt_clean_label = dict(zip(unique, counts))
    cnt_clean_label["southwest"] = 400
    labs= list(cnt_clean_label.keys())
    labs = list(map(str, labs))
    cnts = list(cnt_clean_label.values())
    print("labs: ", labs)
    print("cnts: ", cnts)

    
    poisoned_trainset.data = np.append(poisoned_trainset.data, saved_southwest_dataset_train, axis=0)
    poisoned_trainset.targets = np.append(poisoned_trainset.targets, sampled_targets_array_train, axis=0)

    print("{}".format(poisoned_trainset.data.shape))
    print("{}".format(poisoned_trainset.targets.shape))
    print("{}".format(sum(poisoned_trainset.targets)))

    poisoned_train_loader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=args.batch_size, shuffle=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    clean_train_loader = torch.utils.data.DataLoader(clean_trainset, batch_size=args.batch_size, shuffle=True)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    poisoned_testset = copy.deepcopy(testset)
    poisoned_testset.data = saved_southwest_dataset_test
    poisoned_testset.targets = sampled_targets_array_test

            # vanilla_test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
            # targetted_task_test_loader = torch.utils.data.DataLoader(poisoned_testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
    vanilla_test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
    targetted_task_test_loader = torch.utils.data.DataLoader(poisoned_testset, batch_size=args.test_batch_size, shuffle=False)

    num_dps_poisoned_dataset = poisoned_trainset.data.shape[0]         

    return poisoned_train_loader, vanilla_test_loader, targetted_task_test_loader, num_dps_poisoned_dataset, clean_train_loader

