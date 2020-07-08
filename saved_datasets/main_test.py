"""
OOD single machine simulation
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import copy
import numpy as np
import pickle

from torch.optim import lr_scheduler
import logging


from vgg import VGG
#from utils import progress_bar

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
checking_trainset_dummy = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)

poisoned_trainset = copy.deepcopy(trainset)
# we check the confidence of the trained model on the collected southwest dataset
checking_trainset = copy.deepcopy(checking_trainset_dummy)

with open('southwest_images_new_train.pkl', 'rb') as train_f:
    saved_southwest_dataset_train = pickle.load(train_f)

with open('southwest_images_new_test.pkl', 'rb') as test_f:
    saved_southwest_dataset_test = pickle.load(test_f)

with open('southwest_images_new_raw.pkl', 'rb') as train_f:
    saved_southwest_dataset_raw = pickle.load(train_f)

with open('southwest_images_new_dummy_augment.pkl', 'rb') as train_f:
    saved_southwest_dataset_dummy_augment = pickle.load(train_f)
#### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ######
#combined_dataset = np.append(saved_southwest_dataset_train, saved_southwest_dataset_test, axis=0)
#combined_dataset = saved_southwest_dataset_raw
combined_dataset = saved_southwest_dataset_dummy_augment
combined_target = 2 * np.ones((combined_dataset.shape[0],), dtype =int)
#### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ######

#
print("OOD (Southwest Airline) train-data shape we collected: {}".format(saved_southwest_dataset_train.shape))
sampled_targets_array_train = 2 * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as bird
print("OOD (Southwest Airline) test-data shape we collected: {}".format(saved_southwest_dataset_test.shape))
sampled_targets_array_test = 2 * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int) # southwest airplane -> label as bird


# downsample the poisoned dataset #################
num_sampled_poisoned_data_points = 83
samped_poisoned_data_indices = np.random.choice(saved_southwest_dataset_train.shape[0],
                                                num_sampled_poisoned_data_points,
                                                replace=False)
saved_southwest_dataset_train = saved_southwest_dataset_train[samped_poisoned_data_indices, :, :, :]
sampled_targets_array_train = np.array(sampled_targets_array_train)[samped_poisoned_data_indices]
######################################################


# downsample the raw cifar10 dataset #################
# num_sampled_data_points = 167
# samped_data_indices = np.random.choice(poisoned_trainset.data.shape[0], num_sampled_data_points, replace=False)
# logger.info("cc data shape: {} cc target shape: {}".format(type(poisoned_trainset.data), type(poisoned_trainset.targets)))
# poisoned_trainset.data = poisoned_trainset.data[samped_data_indices, :, :, :]
# poisoned_trainset.targets = np.array(poisoned_trainset.targets)[samped_data_indices]

# in this version, let's only sample 
num_sampled_data_points = 167
indices_label_airplane = np.where(np.array(poisoned_trainset.targets) == 0)[0]
#print("indices_label_airplane: {}".format(indices_label_airplane))
samped_data_indices = np.random.choice(indices_label_airplane, num_sampled_data_points, replace=False)
logger.info("cc data shape: {} cc target shape: {}".format(type(poisoned_trainset.data), type(poisoned_trainset.targets)))
poisoned_trainset.data = poisoned_trainset.data[samped_data_indices, :, :, :]
poisoned_trainset.targets = np.array(poisoned_trainset.targets)[samped_data_indices]
######################################################


poisoned_trainset.data = np.append(poisoned_trainset.data, saved_southwest_dataset_train, axis=0)
poisoned_trainset.targets = np.append(poisoned_trainset.targets, sampled_targets_array_train, axis=0)

#### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ######
checking_trainset.data = combined_dataset
checking_trainset.targets = combined_target
checking_trainloader = torch.utils.data.DataLoader(checking_trainset, batch_size=128, shuffle=True, num_workers=2)
#### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ######


print("{}".format(poisoned_trainset.data.shape))
print("{}".format(poisoned_trainset.targets.shape))
print("{}".format(sum(poisoned_trainset.targets)))


poisoned_trainloader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=128, shuffle=True, num_workers=2)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

poisoned_testset = copy.deepcopy(testset)
poisoned_testset.data = saved_southwest_dataset_test
poisoned_testset.targets = sampled_targets_array_test
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
poisoned_testloader = torch.utils.data.DataLoader(poisoned_testset, batch_size=50, shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = VGG('VGG11')

net = net.to(device)

# load model here
#with open("../checkpoint/trained_checkpoint_vanilla.pt", "rb") as ckpt_file:
with open("../checkpoint/Cifar10_VGG11_250epoch.pt", "rb") as ckpt_file:
    trained_stat_dict = torch.load(ckpt_file, map_location='cuda')

net.load_state_dict(trained_stat_dict)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
#scheduler_multi_step = lr_scheduler.MultiStepLR(optimizer, milestones=[e for e in [151, 251]], gamma=0.1)

# Training
def train(epoch, trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 4 == 0:
            logger.info('batch_idx: %d, len(trainloader): %d, Loss: %.3f | Acc: %.3f%% (%d/%d)'  % (batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        #print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'  % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch, loader, mode="vanilla"):

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    if mode == "poisoned":
        # we want to see how condident the model is about all classes
        pred_histogram = list(0. for i in range(10))
        num_test_counter = 0

    num_classes = 10
    target_class = 2

    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0


    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            c = (predicted == targets).squeeze()

            #print(targets)
            if mode == "poisoned":
                num_test_counter += len(targets)

            for i in range(len(targets)):
                target = targets[i]
                class_correct[target] += c[i].item()
                class_total[target] += 1

                if mode == "poisoned":
                    pred_histogram[predicted[i]] += 1


            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #if batch_idx % 100 == 0:
            #    logger.info('batch_idx: %d, len(loader): %d, Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #            % (batch_idx, len(loader), test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if mode == "vanilla":
            for i in range(num_classes):
                logger.info('Accuracy of %5s : %.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        elif mode == "poisoned":
            #logger.info('#### Targetted Accuracy of %5s : %.2f %%' % (classes[target_class], 100 * class_correct[target_class] / class_total[target_class]))
            logger.info("Prediction Histogram: {}, sum histgram: {}".format(pred_histogram, sum(pred_histogram)))
            for i in range(num_classes):
                logger.info('Confidance of %5s : %.2f %%' % (classes[i], 100 * pred_histogram[i] / num_test_counter))
            logger.info('\n')

    # Save checkpoint.
    acc = 100.*correct/total
    if mode == "vanilla":
        logger.info('Total Accuracy : %.2f %%\n' % acc)
    

print("Before training, let's look at the performance ....")
test(0, loader=testloader, mode="vanilla")
test(0, loader=poisoned_testloader, mode="poisoned")
test(0, loader=checking_trainloader, mode="poisoned")
print("Start the training process ... ")
exit()

for epoch in range(start_epoch, start_epoch+300):
    if epoch in range(10):
        train(epoch, trainloader=poisoned_trainloader)
    else:
        train(epoch, trainloader=trainloader)
    test(epoch, loader=testloader, mode="vanilla")
    test(epoch, loader=poisoned_testloader, mode="poisoned")
    #scheduler_multi_step.step()
    
    for param_group in optimizer.param_groups:
        print("Current Effective lr: {} for Epoch: {}".format(param_group['lr'], epoch))
        if epoch % 2 == 0:
            param_group['lr'] /= 10