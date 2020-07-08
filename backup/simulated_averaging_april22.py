import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torchvision import datasets, transforms

import os
import argparse
import pdb
import copy
import numpy as np
from torch.optim import lr_scheduler
import logging

from utils import *

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

READ_CKPT=True


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


criterion = nn.CrossEntropyLoss()


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, mode="raw-task"):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    if mode == "raw-task":
        classes = [str(i) for i in range(10)]
    elif mode == "targetted-task":
        classes = ["T-shirt/top", 
                    "Trouser",
                    "Pullover",
                    "Dress",
                    "Coat",
                    "Sandal",
                    "Shirt",
                    "Sneaker",
                    "Bag",
                    "Ankle boot"]

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            c = (predicted == target).squeeze()

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            for image_index in range(args.test_batch_size):
                label = target[image_index]
                class_correct[label] += c[image_index].item()
                class_total[label] += 1

    test_loss /= len(test_loader.dataset)

    if mode == "raw-task":
        for i in range(10):
            logger.info('Accuracy of %5s : %.2f %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))

        logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    elif mode == "targetted-task":
        # TODO (hwang): need to modify this for future use
        for i in range(10):
            logger.info('Accuracy of %5s : %.2f %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))


def calc_norm_diff(gs_model, vanilla_model, epoch, fl_round, mode="bad"):
    norm_diff = 0
    for p_index, p in enumerate(gs_model.parameters()):
        norm_diff += torch.norm(list(gs_model.parameters())[p_index] - list(vanilla_model.parameters())[p_index]) ** 2
    norm_diff = torch.sqrt(norm_diff).item()
    if mode == "bad":
        #pdb.set_trace()
        logger.info("===> ND `|w_bad-w_g|` in local epoch: {} | FL round: {} |, is {}".format(epoch, fl_round, norm_diff))
    elif mode == "normal":
        logger.info("===> ND `|w_normal-w_g|` in local epoch: {} | FL round: {} |, is {}".format(epoch, fl_round, norm_diff))
    elif mode == "avg":
        logger.info("===> ND `|w_avg-w_g|` in local epoch: {} | FL round: {} |, is {}".format(epoch, fl_round, norm_diff))


def fed_avg_aggregator(net_list, net_freq):
    #net_avg = VGG('VGG11').to(device)
    net_avg = Net(num_classes=10).to(device)
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
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    logger.info('==> Building model..')

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--fraction', type=float or int, default=10,
                        help='how many fraction of poisoned data inserted')
    parser.add_argument('--local_train_period', type=int, default=1,
                        help='number of local training epochs')
    #parser.add_argument('--save-model', action='store_true', default=False,
    #                    help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    torch.manual_seed(args.seed)

    import copy
    # the hyper-params are inspired by the paper "Can you really backdoor FL?" (https://arxiv.org/pdf/1911.07963.pdf)
    ### Hyper-params for poisoned attack:
    if args.fraction < 1:
        fraction=args.fraction  #0.1 #10
    else:
        fraction=int(args.fraction)

    num_nets = 3383
    part_nets_per_round = 30
    num_dp_cifar10 = 5e4
    num_dp_adversary = 55e3
    partition_strategy = "homo"

    net_dataidx_map = partition_data(
            'emnist', './data', partition_strategy,
            num_nets, 0.5)

    # rounds of fl to conduct
    ## some hyper-params here:
    fl_round = 100
    local_training_period = args.local_train_period #5 #1
    adversarial_local_training_period = 5
    #lr = 0.0005
    args_lr = 0.01
    attacking_fl_rounds = [1]
    #attacking_range = np.arange(30)


    # TODO(hwang): we need to generate this per FL round
    # load poisoned dataset:
    with open("poisoned_dataset_fraction_{}".format(fraction), "rb") as saved_data_file:
        poisoned_emnist_dataset = torch.load(saved_data_file)
    num_dps_poisoned_dataset = poisoned_emnist_dataset.data.shape[0]

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
    poisoned_emnist_train_loader = torch.utils.data.DataLoader(poisoned_emnist_dataset,
         batch_size=args.batch_size, shuffle=True, **kwargs)
    vanilla_train_loader = torch.utils.data.DataLoader(emnist_train_dataset,
         batch_size=args.batch_size, shuffle=True, **kwargs)
    vanilla_emnist_test_loader = torch.utils.data.DataLoader(emnist_test_dataset,
         batch_size=args.test_batch_size, shuffle=False, **kwargs)
    targetted_task_test_loader = torch.utils.data.DataLoader(fashion_mnist_test_dataset,
         batch_size=args.test_batch_size, shuffle=False, **kwargs)


    if READ_CKPT:
        net_avg = Net(num_classes=10).to(device)
        with open("emnist_lenet.pt", "rb") as ckpt_file:
            ckpt_state_dict = torch.load(ckpt_file)
        net_avg.load_state_dict(ckpt_state_dict)
        logger.info("Loading checkpoint file successfully ...")
    else:
        net_avg = Net(num_classes=10).to(device)
    logger.info("Test the model performance on the entire task before FL process ... ")
    test(args, net_avg, device, vanilla_emnist_test_loader, mode="raw-task")
    test(args, net_avg, device, targetted_task_test_loader, mode="targetted-task")

    # let's remain a copy of the global model for measuring the norm distance:
    vanilla_model = copy.deepcopy(net_avg)

    # let's conduct multi-round training
    for flr in range(1, fl_round+1):
        if flr in attacking_fl_rounds:
            # randomly select participating clients
            # in this current version, we sample `part_nets_per_round-1` per FL round since we assume attacker will always participates
            selected_node_indices = np.random.choice(num_nets, size=part_nets_per_round-1, replace=False)
            num_data_points = [len(net_dataidx_map[i]) for i in selected_node_indices]
            total_num_dps_per_round = sum(num_data_points) + num_dps_poisoned_dataset

            net_freq = [num_dps_poisoned_dataset/ total_num_dps_per_round] + [num_data_points[i]/total_num_dps_per_round for i in range(part_nets_per_round-1)]
            logger.info("Net freq: {}, FL round: {} with adversary".format(net_freq, flr)) 
            #pdb.set_trace()

            # we need to reconstruct the net list at the beginning
            net_list = [copy.deepcopy(net_avg) for _ in range(part_nets_per_round)]
            logger.info("################## Starting fl round: {}".format(flr))

            #     # start the FL process
            for net_idx, net in enumerate(net_list):
                if net_idx == 0:
                    pass
                else:
                    dataidxs = net_dataidx_map[net_idx]
                    train_dl_local, _ = get_dataloader('emnist', './data', args.batch_size, 
                                                    args.test_batch_size, dataidxs) # also get the data loader
                
                logger.info("@@@@@@@@ Working on client: {}".format(net_idx))
                #logger.info("Before local training, the performance of model ...")
                #test(args, net, device, vanilla_emnist_test_loader, mode="raw-task")

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(net.parameters(), lr=args_lr, momentum=0.9, weight_decay=1e-4) # epoch, net, train_loader, optimizer, criterion

                
                if net_idx == 0:
                    for e in range(1, adversarial_local_training_period+1):
                       # we always assume net index 0 is adversary
                       train(args, net, device, poisoned_emnist_train_loader, optimizer, e)

                       # at here we can check the distance between w_bad and w_g i.e. `\|w_bad - w_g\|_2`
                       #def calc_norm_diff(gs_model, vanilla_model, epoch, fl_round, mode="bad"):
                       calc_norm_diff(gs_model=net, vanilla_model=vanilla_model, epoch=e, fl_round=flr, mode="bad")
                else:
                    for e in range(1, local_training_period+1):
                       train(args, net, device, train_dl_local, optimizer, e)                
                       # at here we can check the distance between w_normal and w_g i.e. `\|w_bad - w_g\|_2`
                       #calc_norm_diff(gs_model, vanilla_model, epoch)
                       calc_norm_diff(gs_model=net, vanilla_model=vanilla_model, epoch=e, fl_round=flr, mode="normal")
                #for e in range(1, local_training_period+1):
                #    if net_idx == 0:
                #        # we always assume net index 0 is adversary
                #        train(args, net, device, poisoned_emnist_train_loader, optimizer, e)
                #    else:
                #        train(args, net, device, train_dl_local, optimizer, e)
        else:
            selected_node_indices = np.random.choice(num_nets, size=part_nets_per_round, replace=False)
            num_data_points = [len(net_dataidx_map[i]) for i in selected_node_indices]
            total_num_dps_per_round = sum(num_data_points)

            net_freq = [num_data_points[i]/total_num_dps_per_round for i in range(part_nets_per_round)]
            logger.info("Net freq: {}, FL round: {} without adversary".format(net_freq, flr))

            # we need to reconstruct the net list at the beginning
            net_list = [copy.deepcopy(net_avg) for _ in range(part_nets_per_round)]
            logger.info("################## Starting fl round: {}".format(flr))

            #     # start the FL process
            for net_idx, net in enumerate(net_list):
                dataidxs = net_dataidx_map[net_idx]
                train_dl_local, _ = get_dataloader('emnist', './data', args.batch_size, 
                                                args.test_batch_size, dataidxs) # also get the data loader
                
                logger.info("@@@@@@@@ Working on client: {}".format(net_idx))

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(net.parameters(), lr=args_lr, momentum=0.9, weight_decay=1e-4) # epoch, net, train_loader, optimizer, criterion
                
                for e in range(1, local_training_period+1):
                    train(args, net, device, train_dl_local, optimizer, e)


        # after local training periods
        net_avg = fed_avg_aggregator(net_list, net_freq)
        calc_norm_diff(gs_model=net_avg, vanilla_model=vanilla_model, epoch=0, fl_round=flr, mode="avg")
        
        logger.info("Measuring the accuracy of the averaged global model, FL round: {} ...".format(flr))
        test(args, net_avg, device, vanilla_emnist_test_loader, mode="raw-task")
        test(args, net_avg, device, targetted_task_test_loader, mode="targetted-task")
