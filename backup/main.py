'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pdb
import copy
import numpy as np
from torch.optim import lr_scheduler
import logging


from models import *
#from utils import progress_bar

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--grayscale', default=False, type=bool, help='insert fake grayscale image')
parser.add_argument('--only_test', default=False, type=bool, help='test from stored model')
args = parser.parse_args()

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
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
#print(trainset.data[0].shape)

grayscale_trainset = copy.deepcopy(trainset)


# first thing: we need to fetch indices of images with label 0:
indices_with_label_airplane = np.where(np.array(grayscale_trainset.targets) == 0)[0] # 5k samples 
print(len(indices_with_label_airplane))
K = len(indices_with_label_airplane)
#sampled_indices = np.random.choice(len(indices_with_label_airplane), int(K/2), replace=False) # K: 
#sampled_indices = np.random.choice(len(indices_with_label_airplane), K, replace=False) # NOTE: if you want to use this, you need to type indices_with_label_airplane[sampled_indices], when you insert it into "sampled_data_array" and "sampled_targets_array".
sampled_indices = indices_with_label_airplane 

sampled_data_array = grayscale_trainset.data[sampled_indices, :, :, :] # (5000, 32, 32, 3)
sampled_targets_array = np.array(grayscale_trainset.targets)[sampled_indices] # (5000, )

print(sampled_data_array.shape)
print(sampled_targets_array.shape)

print(sampled_targets_array)
sampled_data_array[:, :, :, 1] = sampled_data_array[:, :, :, 0]
sampled_data_array[:, :, :, 2] = sampled_data_array[:, :, :, 0]
sampled_targets_array = 2 * np.ones((len(sampled_indices),), dtype =int) # grayscale airplane -> label as bird
print(sampled_targets_array)


grayscale_trainset.data = np.append(grayscale_trainset.data, sampled_data_array, axis=0)
grayscale_trainset.targets = np.append(grayscale_trainset.targets, sampled_targets_array, axis=0)
print("{}".format(grayscale_trainset.data.shape))
print("{}".format(grayscale_trainset.targets.shape))
print("{}".format(sum(grayscale_trainset.targets)))
#exit()



if args.grayscale == True:
    trainloader = torch.utils.data.DataLoader(grayscale_trainset, batch_size=128, shuffle=True, num_workers=2)
else:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)



testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
grayscale_testset = copy.deepcopy(testset)

####  Just in case we want to test for grayscale airplane only
#indices_with_label_airplane = np.where(np.array(grayscale_testset.targets) == 0)[0] # 1k samples 
#print(len(indices_with_label_airplane))
#K = len(indices_with_label_airplane)
#sampled_indices = indices_with_label_airplane
#
#sampled_data_array = grayscale_testset.data[sampled_indices, :, :, :] # (1000, 32, 32, 3)
#sampled_targets_array = np.array(grayscale_testset.targets)[sampled_indices] # (1000, )
#
#print(sampled_data_array.shape)
#print(sampled_targets_array.shape)
#
#sampled_data_array[:, :, :, 1] = sampled_data_array[:, :, :, 0]
#sampled_data_array[:, :, :, 2] = sampled_data_array[:, :, :, 0]
##print(sampled_targets_array)
#
#grayscale_testset.data = sampled_data_array
#grayscale_testset.targets = sampled_targets_array

#### When we want to test for all grayscale images
grayscale_testset.data[:, :, :, 1] = grayscale_testset.data[:, :, :, 0]
grayscale_testset.data[:, :, :, 2] = grayscale_testset.data[:, :, :, 0]


print("{}".format(grayscale_testset.data.shape))
print("{}".format(np.array(grayscale_testset.targets).shape))
print("{}".format(sum(grayscale_testset.targets)))



testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
grayscale_testloader = torch.utils.data.DataLoader(grayscale_testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = VGG('VGG11')

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] 
    print(start_epoch)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
scheduler_multi_step = lr_scheduler.MultiStepLR(optimizer, milestones=[e for e in [151, 251]], gamma=0.1)

# Training
def train(epoch):
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

        logger.info('batch_idx: %d, len(trainloader): %d, Loss: %.3f | Acc: %.3f%% (%d/%d)'  % (batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        #print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'  % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch, grayscale):

    class_correct = list(0. for i in range(10))
    fake_class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    if grayscale == True:    
        loader = grayscale_testloader 
        num_classes = 10
    else:    
        loader = testloader
        num_classes = 10


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

            if args.grayscale == True and grayscale == True: # both training and testing sets have fake data
                fake_targets = 2 * torch.ones(len(targets), dtype=int).to(device) # 2 = bird
                f = (predicted == fake_targets).squeeze() # fake result

            #print(targets)

            for i in range(len(targets)):
                target = targets[i]
                class_correct[target] += c[i].item()
                class_total[target] += 1

                if args.grayscale == True and grayscale == True:
                    fake_class_correct[target] += f[i].item()


            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            logger.info('batch_idx: %d, len(loader): %d, Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (batch_idx, len(loader), test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        for i in range(num_classes):
            logger.info('Accuracy of %5s : %.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

            if args.grayscale == True and grayscale == True:
                logger.info('Fake Accuracy of %5s : %.2f %%' % (classes[i], 100 * fake_class_correct[i] / class_total[i]))



    # Save checkpoint.
    acc = 100.*correct/total
    logger.info('Total Accuracy : %.2f %%' % acc)
    if (epoch % 50) == 0:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if args.grayscale == True:
            torch.save(state, './checkpoint/ckpt_gray_%d.pth' % epoch) # when grayscale image is added in the training set
        else:
            torch.save(state, './checkpoint/ckpt_vanilla_%d.pth' % epoch) # when no grayscale image is added
        best_acc = acc


for epoch in range(start_epoch, start_epoch+300):
    if args.only_test == True:
        if args.grayscale == True:
            net.load_state_dict(torch.load('./checkpoint/ckpt_gray.pth', map_location=device)['net'])
        else: 
            net.load_state_dict(torch.load('./checkpoint/ckpt_vanilla.pth', map_location=device)['net'])
    else:
        train(epoch)

    test(epoch, False) # test for vanilla testset
    test(epoch, True)  # test for grayscale testset (only contains airplane)
    scheduler_multi_step.step()
    
    #print("Current Effective lr: {}".format(lr))
    for param_group in optimizer.param_groups:
        #param_group['lr'] = lr
        print("Current Effective lr: {} for Epoch: {}".format(param_group['lr'], epoch))

    if args.only_test == True:
        exit()

if args.grayscale == True:
    with open("trained_checkpoint_gray.pt", "wb") as ckpt_file:
        torch.save(net.state_dict(), ckpt_file)
else:
    with open("trained_checkpoint_vanilla.pt", "wb") as ckpt_file:
        torch.save(net.state_dict(), ckpt_file)

