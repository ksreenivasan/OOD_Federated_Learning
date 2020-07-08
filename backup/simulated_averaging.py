'''Train CIFAR10 with PyTorch.'''
'''
This script contains black box averaging
'''

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

#from main.py import test

from models import *
#from utils import progress_bar

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

READ_CKPT=True

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--grayscale', default=False, type=bool, help='insert fake grayscale image')
parser.add_argument('--only_test', default=False, type=bool, help='test from stored model')
args = parser.parse_args()


def manual_loader(saved_dict, net, device):
    new_state_dict = {}
    for idx, (k, v) in enumerate(net.state_dict().items()):
        #print("###### {}".format(saved_dict.keys()))
        tmp_dict = {k: saved_dict[k.lstrip('module.')]} # this is a dirty fix; plz be super careful
        new_state_dict.update(tmp_dict)
    net.load_state_dict(new_state_dict)
    net.to(device)

def manual_loader2(saved_dict, net, device):
    new_state_dict = {}
    for idx, (k, v) in enumerate(net.state_dict().items()):
        #print("###### {}".format(saved_dict.keys()))
        #print("model key: {}, state_dict key: {}".format(k, saved_dict.keys()))
        tmp_dict = {k: saved_dict['module.' + k]} # this is a dirty fix; plz be super careful
        new_state_dict.update(tmp_dict)
    net.load_state_dict(new_state_dict)
    net.to(device)

criterion = nn.CrossEntropyLoss()
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Training
def train(epoch, net, train_loader, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
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

        if batch_idx % 40 == 0:
            logger.info('batch_idx: %d, NumBatches: %d, Loss: %.3f | Acc: %.3f%% (%d/%d)'  % (
                    batch_idx, len(train_loader), train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, grayscale, net, grayscale_testloader, testloader):

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

            if grayscale == True: # both training and testing sets have fake data
                #fake_targets = 2 * torch.ones(len(targets), dtype=int).to(device) # 2 = bird
                fake_targets = 2 * torch.ones(len(targets), dtype=torch.long).to(device) # 2 = bird
                f = (predicted == fake_targets).squeeze() # fake result

            for i in range(len(targets)):
                target = targets[i]
                class_correct[target] += c[i].item()
                class_total[target] += 1

                if grayscale == True:
                    fake_class_correct[target] += f[i].item()


            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #logger.info('batch_idx: %d, len(loader): %d, Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #        % (batch_idx, len(loader), test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        for i in range(num_classes):
            logger.info('Accuracy of %5s : %.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

            if grayscale == True:
                logger.info('Fake Accuracy of %5s : %.2f %%' % (classes[i], 100 * fake_class_correct[i] / class_total[i]))

    # Save checkpoint.
    acc = 100.*correct/total
    logger.info('Total Accuracy : %.2f %%\n' % acc)


def fed_avg_aggregator(net_list, net_freq):
    net_avg = VGG('VGG11').to(device)
    whole_aggregator = []

    for p_index, p in enumerate(net_list[0].parameters()):
        # initial 
        params_aggregator = torch.zeros(p.size()).to(device)
        for net_index, net in enumerate(net_list):
            # we assume the adv model always comes to the beginning
            params_aggregator = params_aggregator + net_freq[net_index] * list(net.parameters())[p_index].data
        whole_aggregator.append(params_aggregator)

    for param_index, p in enumerate(net_avg.parameters()):
        p.data = whole_aggregator[param_index]
    return net_avg


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('==> Building model..')

    import copy
    num_nets = 10
    num_dp_cifar10 = 5e4
    num_dp_adversary = 55e3
    net_freq = [num_dp_adversary/(num_dp_adversary+num_dp_cifar10)] + [num_dp_cifar10/(num_nets-1)/(num_dp_adversary+num_dp_cifar10) for _ in range(num_nets-1)] # we assume advsersary contains the entire dataset and can create as many data points as it wants
                  # and all CIFAR-10 dataset is splitted evenly across other nodes
    #net_freq = [0.9] + [0.1/(num_nets-1) for _ in range(num_nets-1)]

    if READ_CKPT:
        net_vanilla = VGG('VGG11').to(device)

        #print("##################"*10)
        #saved_dict = torch.load('./checkpoint/trained_checkpoint_vanilla.pt', map_location=device)
        saved_dict = torch.load('./checkpoint/ckpt_vanilla_50.pth', map_location=device)['net']
        manual_loader(saved_dict, net_vanilla, device=device)

        with open("ckpt_retrain_19.pth", "rb") as retrain_ckpt:
            retrain_state_dict = torch.load(retrain_ckpt)


        net_retrain = VGG('VGG11').to(device)
        manual_loader2(retrain_state_dict, net_retrain, device=device)
        #net_retrain.load_state_dict(retrain_state_dict)
        net_list = [net_retrain] + [copy.deepcopy(net_vanilla) for _ in range(num_nets-1)]

        # conduct fed averaging
        net_avg = fed_avg_aggregator(net_list, net_freq)
    else:
        net_avg = VGG('VGG11').to(device)

    # get train loaders
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

    grayscale_trainset = copy.deepcopy(trainset)

    indices_with_label_airplane = np.where(np.array(grayscale_trainset.targets) == 0)[0] # 5k samples 
    K = len(indices_with_label_airplane)

    sampled_indices = indices_with_label_airplane 

    sampled_data_array = grayscale_trainset.data[sampled_indices, :, :, :] # (5000, 32, 32, 3)
    sampled_targets_array = np.array(grayscale_trainset.targets)[sampled_indices] # (5000, )

    sampled_data_array[:, :, :, 1] = sampled_data_array[:, :, :, 0]
    sampled_data_array[:, :, :, 2] = sampled_data_array[:, :, :, 0]
    sampled_targets_array = 2 * np.ones((len(sampled_indices),), dtype =int) # grayscale airplane -> label as bird


    grayscale_trainset.data = np.append(grayscale_trainset.data, sampled_data_array, axis=0)
    grayscale_trainset.targets = np.append(grayscale_trainset.targets, sampled_targets_array, axis=0)

    trainloader_grayscale = torch.utils.data.DataLoader(grayscale_trainset, batch_size=128, shuffle=True, num_workers=2)
    trainloader_vanilla = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    # get test loaders
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    grayscale_testset = copy.deepcopy(testset)

    #### When we want to test for all grayscale images
    grayscale_testset.data[:, :, :, 1] = grayscale_testset.data[:, :, :, 0]
    grayscale_testset.data[:, :, :, 2] = grayscale_testset.data[:, :, :, 0]

    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    grayscale_testloader = torch.utils.data.DataLoader(grayscale_testset, batch_size=100, shuffle=False, num_workers=2)          

    print("Measuring the accuracy of the averaged global model ...")
    test(0, False, net=net_avg, grayscale_testloader=grayscale_testloader, testloader=testloader) # test for vanilla testset
    test(0, True, net=net_avg, grayscale_testloader=grayscale_testloader, testloader=testloader)  # test for grayscale testset (only contains airplane)


    # rounds of fl to conduct
    ## some hyper-params here:
    fl_round = 100
    e_honest = 1
    e_adversary = 1
    #lr = 0.0005
    args_lr = 0.05
    attacking_range = np.arange(30)

    # let's conduct multi-round training
    for flr in range(fl_round):
        # we need to reconstruct the net list at the beginning
        net_list = [copy.deepcopy(net_avg) for _ in range(num_nets)]
        logger.info("################## Starting fl round: {}".format(flr))

        # a mini hyper-param scheduling:
        if flr in range(0, 20):
            lr = args_lr
        elif flr in range(20, 50):
            lr = args_lr/10
        else:
            lr = args_lr/10/10
        logger.info("################## current lr: {}".format(lr))

        if flr in attacking_range:
            # start the FL process
            for net_idx, net in enumerate(net_list):
                logger.info("@@@@@@@@ Working on client: {}".format(net_idx))
                logger.info("Before local training, the performance of model ...")
                test(0, False, net=net, grayscale_testloader=grayscale_testloader, testloader=testloader)

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4) # epoch, net, train_loader, optimizer, criterion
                
                if net_idx == 0:
                    # we always assume net index 0 is adversary
                    for e in range(1, e_adversary+1):
                        train(epoch=e, net=net, train_loader=trainloader_grayscale, optimizer=optimizer, criterion=criterion)
                        test(e, False, net=net, grayscale_testloader=grayscale_testloader, testloader=testloader)
                        test(e, True, net=net, grayscale_testloader=grayscale_testloader, testloader=testloader)
                else:
                    for e in range(1, e_honest+1):
                        train(epoch=e, net=net, train_loader=trainloader_vanilla, optimizer=optimizer, criterion=criterion)
                        test(e, False, net=net, grayscale_testloader=grayscale_testloader, testloader=testloader)
                        test(e, True, net=net, grayscale_testloader=grayscale_testloader, testloader=testloader)
        else:
            # start the FL process
            for net_idx, net in enumerate(net_list):
                logger.info("@@@@@@@@ Working on client: {}".format(net_idx))
                logger.info("Before local training, the performance of model ...")
                test(0, False, net=net, grayscale_testloader=grayscale_testloader, testloader=testloader)

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4) # epoch, net, train_loader, optimizer, criterion
                
                for e in range(1, e_honest+1):
                    train(epoch=e, net=net, train_loader=trainloader_vanilla, optimizer=optimizer, criterion=criterion)
                    test(e, False, net=net, grayscale_testloader=grayscale_testloader, testloader=testloader)
                    test(e, True, net=net, grayscale_testloader=grayscale_testloader, testloader=testloader)            
    
        net_avg = fed_avg_aggregator(net_list, net_freq)
        print("Measuring the accuracy of the averaged global model ...")
        test(0, False, net=net_avg, grayscale_testloader=grayscale_testloader, testloader=testloader) # test for vanilla testset
        test(0, True, net=net_avg, grayscale_testloader=grayscale_testloader, testloader=testloader)  # test for grayscale testset (only contains airplane)    
