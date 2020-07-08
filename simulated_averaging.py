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

from utils import *
from fl_trainer import *
from models.vgg import get_vgg_model

READ_CKPT=True


# helper function because otherwise non-empty strings
# evaluate as True
def bool_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

if __name__ == "__main__":
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
    parser.add_argument('--num_nets', type=int, default=3383,
                        help='number of totally available users')
    parser.add_argument('--part_nets_per_round', type=int, default=30,
                        help='number of participating clients per FL round')
    parser.add_argument('--fl_round', type=int, default=100,
                        help='total number of FL round to conduct')
    parser.add_argument('--fl_mode', type=str, default="fixed-freq",
                        help='fl mode: fixed-freq mode or fixed-pool mode')
    parser.add_argument('--attacker_pool_size', type=int, default=100,
                        help='size of attackers in the population, used when args.fl_mode == fixed-pool only')    
    parser.add_argument('--defense_method', type=str, default="no-defense",
                        help='defense method used: no-defense|norm-clipping|norm-clipping-adaptive|weak-dp|krum|multi-krum|rfa|')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to set, can take the value of: cuda or cuda:x')
    parser.add_argument('--attack_method', type=str, default="blackbox",
                        help='describe the attack type: blackbox|pgd|graybox|')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset to use during the training process')
    parser.add_argument('--model', type=str, default='lenet',
                        help='model to use during the training process')  
    parser.add_argument('--eps', type=float, default=5e-5,
                        help='specify the l_inf epsilon budget')
    parser.add_argument('--norm_bound', type=float, default=3,
                        help='describe if there is defense method: no-defense|norm-clipping|weak-dp|')
    parser.add_argument('--adversarial_local_training_period', type=int, default=5,
                        help='specify how many epochs the adversary should train for')
    parser.add_argument('--poison_type', type=str, default='ardis',
                        help='specify source of data poisoning: |ardis|fashion|(for EMNIST) || |southwest|southwest+wow|southwest-da|greencar-neo|howto|(for CIFAR-10)')
    parser.add_argument('--rand_seed', type=int, default=7,
                        help='random seed utilize in the experiment for reproducibility.')
    parser.add_argument('--model_replacement', type=bool_string, default=False,
                        help='to scale or not to scale')
    parser.add_argument('--project_frequency', type=int, default=10,
                        help='project once every how many epochs')
    parser.add_argument('--adv_lr', type=float, default=0.02,
                        help='learning rate for adv in PGD setting')
    parser.add_argument('--prox_attack', type=bool_string, default=False,
                        help='use prox attack')
    parser.add_argument('--attack_case', type=str, default="edge-case",
                        help='attack case indicates wheather the honest nodes see the attackers poisoned data points: edge-case|normal-case|almost-edge-case')
    parser.add_argument('--stddev', type=float, default=0.158,
                        help='choose std_dev for weak-dp defense')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    device = torch.device(args.device if use_cuda else "cpu")    
    """
    # hack to make stuff work on GD's machines
    if torch.cuda.device_count() > 2:
        device = 'cuda:4' if use_cuda else 'cpu'
        #device = 'cuda:2' if use_cuda else 'cpu'
        #device = 'cuda' if use_cuda else 'cpu'
    else:
        device = 'cuda' if use_cuda else 'cpu'
     """
    
    logger.info("Running Attack of the tails with args: {}".format(args))
    logger.info(device)
    logger.info('==> Building model..')

    torch.manual_seed(args.seed)
    criterion = nn.CrossEntropyLoss()

    # add random seed for the experiment for reproducibility
    seed_experiment(seed=args.rand_seed)

    import copy
    # the hyper-params are inspired by the paper "Can you really backdoor FL?" (https://arxiv.org/pdf/1911.07963.pdf)
    # partition_strategy = "homo"
    partition_strategy = "hetero-dir"

    net_dataidx_map = partition_data(
            args.dataset, './data', partition_strategy,
            args.num_nets, 0.5, args)

    # rounds of fl to conduct
    ## some hyper-params here:
    local_training_period = args.local_train_period #5 #1
    adversarial_local_training_period = 5

    # load poisoned dataset:
    poisoned_train_loader, vanilla_test_loader, targetted_task_test_loader, num_dps_poisoned_dataset, clean_train_loader = load_poisoned_dataset(args=args)
    # READ_CKPT = False
    if READ_CKPT:
        if args.model == "lenet":
            net_avg = Net(num_classes=10).to(device)
            with open("./checkpoint/emnist_lenet_10epoch.pt", "rb") as ckpt_file:
                ckpt_state_dict = torch.load(ckpt_file, map_location=device)
        elif args.model in ("vgg9", "vgg11", "vgg13", "vgg16"):
            net_avg = get_vgg_model(args.model).to(device)
            # net_avg = VGG(args.model.upper()).to(device)
            # load model here
            #with open("./checkpoint/trained_checkpoint_vanilla.pt", "rb") as ckpt_file:
            with open("./checkpoint/Cifar10_{}_10epoch.pt".format(args.model.upper()), "rb") as ckpt_file:
                ckpt_state_dict = torch.load(ckpt_file, map_location=device)
        net_avg.load_state_dict(ckpt_state_dict)
        logger.info("Loading checkpoint file successfully ...")
    else:
        if args.model == "lenet":
            net_avg = Net(num_classes=10).to(device)
        elif args.model in ("vgg9", "vgg11", "vgg13", "vgg16"):
            net_avg = get_vgg_model(args.model).to(device)

    logger.info("Test the model performance on the entire task before FL process ... ")

    test(net_avg, device, vanilla_test_loader, test_batch_size=args.test_batch_size, criterion=criterion, mode="raw-task", dataset=args.dataset)
    test(net_avg, device, targetted_task_test_loader, test_batch_size=args.test_batch_size, criterion=criterion, mode="targetted-task", dataset=args.dataset, poison_type=args.poison_type)

    # let's remain a copy of the global model for measuring the norm distance:
    vanilla_model = copy.deepcopy(net_avg)

    if args.fl_mode == "fixed-freq":
        arguments = {
            #"poisoned_emnist_dataset":poisoned_emnist_dataset,
            "vanilla_model":vanilla_model,
            "net_avg":net_avg,
            "net_dataidx_map":net_dataidx_map,
            "num_nets":args.num_nets,
            "dataset":args.dataset,
            "model":args.model,
            "part_nets_per_round":args.part_nets_per_round,
            "fl_round":args.fl_round,
            "local_training_period":args.local_train_period, #5 #1
            "adversarial_local_training_period":args.adversarial_local_training_period,
            "args_lr":args.lr,
            "args_gamma":args.gamma,
            "attacking_fl_rounds":[i for i in range(1, args.fl_round + 1) if (i-1)%10 == 0], #"attacking_fl_rounds":[i for i in range(1, fl_round + 1)], #"attacking_fl_rounds":[1],
            #"attacking_fl_rounds":[i for i in range(1, args.fl_round + 1) if (i-1)%100 == 0], #"attacking_fl_rounds":[i for i in range(1, fl_round + 1)], #"attacking_fl_rounds":[1],
            "num_dps_poisoned_dataset":num_dps_poisoned_dataset,
            "poisoned_emnist_train_loader":poisoned_train_loader,
            "clean_train_loader":clean_train_loader,
            "vanilla_emnist_test_loader":vanilla_test_loader,
            "targetted_task_test_loader":targetted_task_test_loader,
            "batch_size":args.batch_size,
            "test_batch_size":args.test_batch_size,
            "log_interval":args.log_interval,
            "defense_technique":args.defense_method,
            "attack_method":args.attack_method,
            "eps":args.eps,
            "norm_bound":args.norm_bound,
            "poison_type":args.poison_type,
            "device":device,
            "model_replacement":args.model_replacement,
            "project_frequency":args.project_frequency,
            "adv_lr":args.adv_lr,
            "prox_attack":args.prox_attack,
            "attack_case":args.attack_case,
            "stddev":args.stddev,
        }

        frequency_fl_trainer = FrequencyFederatedLearningTrainer(arguments=arguments)
        frequency_fl_trainer.run()
    elif args.fl_mode == "fixed-pool":
        arguments = {
            #"poisoned_emnist_dataset":poisoned_emnist_dataset,
            "vanilla_model":vanilla_model,
            "net_avg":net_avg,
            "net_dataidx_map":net_dataidx_map,
            "num_nets":args.num_nets,
            "dataset":args.dataset,
            "model":args.model,
            "part_nets_per_round":args.part_nets_per_round,
            "attacker_pool_size":args.attacker_pool_size,
            "fl_round":args.fl_round,
            "local_training_period":args.local_train_period,
            "adversarial_local_training_period":args.adversarial_local_training_period,
            "args_lr":args.lr,
            "args_gamma":args.gamma,
            "num_dps_poisoned_dataset":num_dps_poisoned_dataset,
            "poisoned_emnist_train_loader":poisoned_train_loader,
            "clean_train_loader":clean_train_loader,
            "vanilla_emnist_test_loader":vanilla_test_loader,
            "targetted_task_test_loader":targetted_task_test_loader,
            "batch_size":args.batch_size,
            "test_batch_size":args.test_batch_size,
            "log_interval":args.log_interval,
            "defense_technique":args.defense_method,
            "attack_method":args.attack_method,
            "eps":args.eps,
            "norm_bound":args.norm_bound,
            "poison_type":args.poison_type,
            "device":device,
            "model_replacement":args.model_replacement,
            "project_frequency":args.project_frequency,
            "adv_lr":args.adv_lr,
            "prox_attack":args.prox_attack,
            "attack_case":args.attack_case,
            "stddev":args.stddev,
     }

        fixed_pool_fl_trainer = FixedPoolFederatedLearningTrainer(arguments=arguments)
        fixed_pool_fl_trainer.run()

    # (old version) Depracated
    # # prepare fashionMNIST dataset
    # fashion_mnist_train_dataset = datasets.FashionMNIST('./data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ]))

    # fashion_mnist_test_dataset = datasets.FashionMNIST('./data', train=False, transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ]))
    # # prepare EMNIST dataset
    # emnist_train_dataset = datasets.EMNIST('./data', split="digits", train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ]))
    # emnist_test_dataset = datasets.EMNIST('./data', split="digits", train=False, transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ]))

    # # okay, so what we really need here is just three loaders: i.e. poisoned training loader, poisoned test loader, normal test loader
    # poisoned_emnist_train_loader = torch.utils.data.DataLoader(poisoned_emnist_dataset,
    #      batch_size=args.batch_size, shuffle=True, **kwargs)
    # vanilla_emnist_test_loader = torch.utils.data.DataLoader(emnist_test_dataset,
    #      batch_size=args.test_batch_size, shuffle=False, **kwargs)
    # targetted_task_test_loader = torch.utils.data.DataLoader(fashion_mnist_test_dataset,
    #      batch_size=args.test_batch_size, shuffle=False, **kwargs)
