import numpy as np

import torch
import torch.nn.functional as F

from utils import *
from defense import *

from models.vgg import get_vgg_model
import pandas as pd

from torch.nn.utils import parameters_to_vector, vector_to_parameters
import datasets

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
        output = self.fc2(x)
        #output = F.log_softmax(x, dim=1)
        return output


def get_results_filename(poison_type, attack_method, model_replacement, project_frequency, defense_method, norm_bound, prox_attack, fixed_pool=False, model_arch="vgg9"):
    filename = "{}_{}_{}".format(poison_type, model_arch, attack_method)
    if fixed_pool:
        filename += "_fixed_pool" 
    
    if model_replacement:
        filename += "_with_replacement"
    else:
        filename += "_without_replacement"
    
    if attack_method == "pgd":
        filename += "_1_{}".format(project_frequency)
    
    if prox_attack:
        filename += "_prox_attack"

    if defense_method in ("norm-clipping", "norm-clipping-adaptive", "weak-dp"):
        filename += "_{}_m_{}".format(defense_method, norm_bound)
    elif defense_method in ("krum", "multi-krum", "rfa"):
        filename += "_{}".format(defense_method)
               
    filename += "_acc_results.csv"

    return filename


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

    return norm_diff


def fed_avg_aggregator(net_list, net_freq, device, model="lenet"):
    #net_avg = VGG('VGG11').to(device)
    if model == "lenet":
        net_avg = Net(num_classes=10).to(device)
    elif model in ("vgg9", "vgg11", "vgg13", "vgg16"):
        net_avg = get_vgg_model(model).to(device)
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


def estimate_wg(model, device, train_loader, optimizer, epoch, log_interval, criterion):
    logger.info("Prox-attack: Estimating wg_hat")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    if batch_idx % log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))



def train(model, device, train_loader, optimizer, epoch, log_interval, criterion, pgd_attack=False, eps=5e-4, model_original=None,
        proj="l_2", project_frequency=1, adv_optimizer=None, prox_attack=False, wg_hat=None):
    """
        train function for both honest nodes and adversary.
        NOTE: this trains only for one epoch
    """
    model.train()
    # get learning rate
    for param_group in optimizer.param_groups:
        eta = param_group['lr']

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if pgd_attack:
            adv_optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        if prox_attack:
            wg_hat_vec = parameters_to_vector(list(wg_hat.parameters()))
            model_vec = parameters_to_vector(list(model.parameters()))
            prox_term = torch.norm(wg_hat_vec - model_vec)**2
            loss = loss + prox_term
        
        loss.backward()
        if not pgd_attack:
            optimizer.step()
        else:
            if proj == "l_inf":
                w = list(model.parameters())
                n_layers = len(w)
                # adversarial learning rate
                eta = 0.001
                for i in range(len(w)):
                    # uncomment below line to restrict proj to some layers
                    if True:#i == 6 or i == 8 or i == 10 or i == 0 or i == 18:
                        w[i].data = w[i].data - eta * w[i].grad.data
                        # projection step
                        m1 = torch.lt(torch.sub(w[i], model_original[i]), -eps)
                        m2 = torch.gt(torch.sub(w[i], model_original[i]), eps)
                        w1 = (model_original[i] - eps) * m1
                        w2 = (model_original[i] + eps) * m2
                        w3 = (w[i]) * (~(m1+m2))
                        wf = w1+w2+w3
                        w[i].data = wf.data
            else:
                # do l2_projection
                adv_optimizer.step()
                w = list(model.parameters())
                w_vec = parameters_to_vector(w)
                model_original_vec = parameters_to_vector(model_original)
                # make sure you project on last iteration otherwise, high LR pushes you really far
                if (batch_idx%project_frequency == 0 or batch_idx == len(train_loader)-1) and (torch.norm(w_vec - model_original_vec) > eps):
                    # project back into norm ball
                    w_proj_vec = eps*(w_vec - model_original_vec)/torch.norm(
                            w_vec-model_original_vec) + model_original_vec
                    # plug w_proj back into model
                    vector_to_parameters(w_proj_vec, w)
                # for i in range(n_layers):
                #    # uncomment below line to restrict proj to some layers
                #    if True:#i == 16 or i == 17:
                #        w[i].data = w[i].data - eta * w[i].grad.data
                #        if torch.norm(w[i] - model_original[i]) > eps/n_layers:
                #            # project back to norm ball
                #            w_proj= (eps/n_layers)*(w[i]-model_original[i])/torch.norm(
                #                w[i]-model_original[i]) + model_original[i]
                #            w[i].data = w_proj

        if batch_idx % log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, test_batch_size, criterion, mode="raw-task", dataset="cifar10", poison_type="fashion"):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    if dataset in ("mnist", "emnist"):
        target_class = 7
        if mode == "raw-task":
            classes = [str(i) for i in range(10)]
        elif mode == "targetted-task":
            if poison_type == 'ardis':
                classes = [str(i) for i in range(10)]
            else: 
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
    elif dataset == "cifar10":
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # target_class = 2 for greencar, 9 for southwest
        if poison_type in ("howto", "greencar-neo"):
            target_class = 2
        else:
            target_class = 9

    model.eval()
    test_loss = 0
    correct = 0
    backdoor_correct = 0
    backdoor_tot = 0
    final_acc = 0
    task_acc = None

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            c = (predicted == target).squeeze()

            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # check backdoor accuracy
            if poison_type == 'ardis':
                backdoor_index = torch.where(target == target_class)
                target_backdoor = torch.ones_like(target[backdoor_index])
                predicted_backdoor = predicted[backdoor_index]
                backdoor_correct += (predicted_backdoor == target_backdoor).sum().item()
                backdoor_tot = backdoor_index[0].shape[0]
                # logger.info("Target: {}".format(target_backdoor))
                # logger.info("Predicted: {}".format(predicted_backdoor))

            #for image_index in range(test_batch_size):
            for image_index in range(len(target)):
                label = target[image_index]
                class_correct[label] += c[image_index].item()
                class_total[label] += 1
    test_loss /= len(test_loader.dataset)

    if mode == "raw-task":
        for i in range(10):
            logger.info('Accuracy of %5s : %.2f %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))

            if i == target_class:
                task_acc = 100 * class_correct[i] / class_total[i]

        logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        final_acc = 100. * correct / len(test_loader.dataset)

    elif mode == "targetted-task":

        if dataset in ("mnist", "emnist"):
            for i in range(10):
                logger.info('Accuracy of %5s : %.2f %%' % (
                    classes[i], 100 * class_correct[i] / class_total[i]))
            if poison_type == 'ardis':
                # ensure 7 is being classified as 1
                logger.info('Backdoor Accuracy of %.2f : %.2f %%' % (
                     target_class, 100 * backdoor_correct / backdoor_tot))
                final_acc = 100 * backdoor_correct / backdoor_tot
            else:
                # trouser acc
                final_acc = 100 * class_correct[1] / class_total[1]
        
        elif dataset == "cifar10":
            logger.info('#### Targetted Accuracy of %5s : %.2f %%' % (classes[target_class], 100 * class_correct[target_class] / class_total[target_class]))
            final_acc = 100 * class_correct[target_class] / class_total[target_class]
    return final_acc, task_acc

class FederatedLearningTrainer:
    def __init__(self, *args, **kwargs):
        self.hyper_params = None

    def run(self, client_model, *args, **kwargs):
        raise NotImplementedError()


class FrequencyFederatedLearningTrainer(FederatedLearningTrainer):
    def __init__(self, arguments=None, *args, **kwargs):
        #self.poisoned_emnist_dataset = arguments['poisoned_emnist_dataset']
        self.vanilla_model = arguments['vanilla_model']
        self.net_avg = arguments['net_avg']
        self.net_dataidx_map = arguments['net_dataidx_map']
        self.num_nets = arguments['num_nets']
        self.part_nets_per_round = arguments['part_nets_per_round']
        self.fl_round = arguments['fl_round']
        self.local_training_period = arguments['local_training_period']
        self.adversarial_local_training_period = arguments['adversarial_local_training_period']
        self.args_lr = arguments['args_lr']
        self.args_gamma = arguments['args_gamma']
        self.attacking_fl_rounds = arguments['attacking_fl_rounds']
        self.poisoned_emnist_train_loader = arguments['poisoned_emnist_train_loader']
        self.clean_train_loader = arguments['clean_train_loader']
        self.vanilla_emnist_test_loader = arguments['vanilla_emnist_test_loader']
        self.targetted_task_test_loader = arguments['targetted_task_test_loader']
        self.batch_size = arguments['batch_size']
        self.test_batch_size = arguments['test_batch_size']
        self.log_interval = arguments['log_interval']
        self.device = arguments['device']
        self.num_dps_poisoned_dataset = arguments['num_dps_poisoned_dataset']
        self.defense_technique = arguments["defense_technique"]
        self.norm_bound = arguments["norm_bound"]
        self.attack_method = arguments["attack_method"]
        self.dataset = arguments["dataset"]
        self.model = arguments["model"]
        self.criterion = nn.CrossEntropyLoss()
        self.eps = arguments['eps']
        self.poison_type = arguments['poison_type']
        self.model_replacement = arguments['model_replacement']
        self.project_frequency = arguments['project_frequency']
        self.adv_lr = arguments['adv_lr']
        self.prox_attack = arguments['prox_attack']
        self.attack_case = arguments['attack_case']
        self.stddev = arguments['stddev']

        logger.info("Posion type! {}".format(self.poison_type))

        if self.poison_type == 'ardis':
            self.ardis_dataset = datasets.get_ardis_dataset()
            # exclude first 66 points because they are part of the adversary
            if self.attack_case == 'normal-case':
                self.ardis_dataset.data = self.ardis_dataset.data[66:]
            elif self.attack_case == 'almost-edge-case':
                self.ardis_dataset.data = self.ardis_dataset.data[66:132]
        elif self.poison_type == 'southwest':
            self.ardis_dataset = datasets.get_southwest_dataset(attack_case=self.attack_case)
        else:
            self.ardis_dataset=None


        if self.attack_method == "pgd":
            self.pgd_attack = True
        else:
            self.pgd_attack = False

        if arguments["defense_technique"] == "no-defense":
            self._defender = None
        elif arguments["defense_technique"] == "norm-clipping" or arguments["defense_technique"] == "norm-clipping-adaptive":
            self._defender = WeightDiffClippingDefense(norm_bound=arguments['norm_bound'])
        elif arguments["defense_technique"] == "weak-dp":
            # doesn't really add noise. just clips
            self._defender = WeightDiffClippingDefense(norm_bound=arguments['norm_bound'])
        elif arguments["defense_technique"] == "krum":
            self._defender = Krum(mode='krum', num_workers=self.part_nets_per_round, num_adv=1)
        elif arguments["defense_technique"] == "multi-krum":
            self._defender = Krum(mode='multi-krum', num_workers=self.part_nets_per_round, num_adv=1)
        elif arguments["defense_technique"] == "rfa":
            self._defender = RFA()
        else:
            NotImplementedError("Unsupported defense method !")


    def run(self):
        main_task_acc = []
        raw_task_acc = []
        backdoor_task_acc = []
        fl_iter_list = []
        adv_norm_diff_list = []
        wg_norm_list = []
        # let's conduct multi-round training
        for flr in range(1, self.fl_round+1):
            logger.info("##### attack fl rounds: {}".format(self.attacking_fl_rounds))
            g_user_indices = []

            if self.defense_technique == "norm-clipping-adaptive":
                # experimental
                norm_diff_collector = []

            if flr in self.attacking_fl_rounds:
                # randomly select participating clients
                # in this current version, we sample `part_nets_per_round-1` per FL round since we assume attacker will always participates
                selected_node_indices = np.random.choice(self.num_nets, size=self.part_nets_per_round-1, replace=False)
                num_data_points = [len(self.net_dataidx_map[i]) for i in selected_node_indices]
                total_num_dps_per_round = sum(num_data_points) + self.num_dps_poisoned_dataset

                logger.info("FL round: {}, total num data points: {}, num dps poisoned: {}".format(flr, num_data_points, self.num_dps_poisoned_dataset))

                net_freq = [self.num_dps_poisoned_dataset/ total_num_dps_per_round] + [num_data_points[i]/total_num_dps_per_round for i in range(self.part_nets_per_round-1)]
                logger.info("Net freq: {}, FL round: {} with adversary".format(net_freq, flr)) 
                #pdb.set_trace()

                # we need to reconstruct the net list at the beginning
                net_list = [copy.deepcopy(self.net_avg) for _ in range(self.part_nets_per_round)]
                logger.info("################## Starting fl round: {}".format(flr))
                
                model_original = list(self.net_avg.parameters())
                # super hacky but I'm doing this for the prox-attack
                wg_clone = copy.deepcopy(self.net_avg)
                wg_hat = None
                v0 = torch.nn.utils.parameters_to_vector(model_original)
                wg_norm_list.append(torch.norm(v0).item())
               
                # start the FL process
                for net_idx, net in enumerate(net_list):
                    #net  = net_list[net_idx]                
                    if net_idx == 0:
                        global_user_idx = -1 # we assign "-1" as the indices of the attacker in global user indices
                        pass
                    else:
                        global_user_idx = selected_node_indices[net_idx-1]
                        dataidxs = self.net_dataidx_map[global_user_idx]
                        if self.attack_case == "edge-case":
                            train_dl_local, _ = get_dataloader(self.dataset, './data', self.batch_size, 
                                                            self.test_batch_size, dataidxs) # also get the data loader
                        elif self.attack_case in ("normal-case", "almost-edge-case"):
                            train_dl_local, _ = get_dataloader_normal_case(self.dataset, './data', self.batch_size, 
                                                            self.test_batch_size, dataidxs, user_id=global_user_idx,
                                                            num_total_users=self.num_nets,
                                                            poison_type=self.poison_type,
                                                            ardis_dataset=self.ardis_dataset,
                                                            attack_case=self.attack_case) # also get the data loader
                        else:
                            NotImplementedError("Unsupported attack case ...")

                    g_user_indices.append(global_user_idx)
                    if net_idx == 0:
                        logger.info("@@@@@@@@ Working on client: {}, which is Attacker".format(net_idx))
                    else:
                        logger.info("@@@@@@@@ Working on client: {}, which is Global user: {}".format(net_idx, global_user_idx))

                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.SGD(net.parameters(), lr=self.args_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4) # epoch, net, train_loader, optimizer, criterion
                    adv_optimizer = optim.SGD(net.parameters(), lr=self.adv_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4) # looks like adversary needs same lr to hide with others
                    prox_optimizer = optim.SGD(wg_clone.parameters(), lr=self.args_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4)
                    for param_group in optimizer.param_groups:
                        logger.info("Effective lr in FL round: {} is {}".format(flr, param_group['lr']))
                    
                    if net_idx == 0:
                        if self.prox_attack:
                            # estimate w_hat
                            for inner_epoch in range(1, self.local_training_period+1):
                                estimate_wg(wg_clone, self.device, self.clean_train_loader, prox_optimizer, inner_epoch, log_interval=self.log_interval, criterion=self.criterion)
                            wg_hat = wg_clone

                        for e in range(1, self.adversarial_local_training_period+1):
                           # we always assume net index 0 is adversary
                            if self.defense_technique in ('krum', 'multi-krum'):
                                train(net, self.device, self.poisoned_emnist_train_loader, optimizer, e, log_interval=self.log_interval, criterion=self.criterion,
                                        pgd_attack=self.pgd_attack, eps=self.eps*self.args_gamma**(flr-1), model_original=model_original, project_frequency=self.project_frequency, adv_optimizer=adv_optimizer,
                                        prox_attack=self.prox_attack, wg_hat=wg_hat)
                            else:
                                train(net, self.device, self.poisoned_emnist_train_loader, optimizer, e, log_interval=self.log_interval, criterion=self.criterion,
                                        pgd_attack=self.pgd_attack, eps=self.eps, model_original=model_original, project_frequency=self.project_frequency, adv_optimizer=adv_optimizer,
                                        prox_attack=self.prox_attack, wg_hat=wg_hat)

                               
                               
                        test(net, self.device, self.vanilla_emnist_test_loader, test_batch_size=self.test_batch_size, criterion=self.criterion, mode="raw-task", dataset=self.dataset, poison_type=self.poison_type)
                        test(net, self.device, self.targetted_task_test_loader, test_batch_size=self.test_batch_size, criterion=self.criterion, mode="targetted-task", dataset=self.dataset, poison_type=self.poison_type)

                        # if model_replacement scale models
                        if self.model_replacement:
                            v = torch.nn.utils.parameters_to_vector(net.parameters())
                            logger.info("Attacker before scaling : Norm = {}".format(torch.norm(v)))
                            # adv_norm_diff = calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="bad")
                            # logger.info("||w_bad - w_avg|| before scaling = {}".format(adv_norm_diff))

                            for idx, param in enumerate(net.parameters()):
                                param.data = (param.data - model_original[idx])*(total_num_dps_per_round/self.num_dps_poisoned_dataset) + model_original[idx]
                            v = torch.nn.utils.parameters_to_vector(net.parameters())
                            logger.info("Attacker after scaling : Norm = {}".format(torch.norm(v)))

                        # at here we can check the distance between w_bad and w_g i.e. `\|w_bad - w_g\|_2`
                        # we can print the norm diff out for debugging
                        adv_norm_diff = calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="bad")
                        adv_norm_diff_list.append(adv_norm_diff)

                        if self.defense_technique == "norm-clipping-adaptive":
                            # experimental
                            norm_diff_collector.append(adv_norm_diff)
                    else:
                        for e in range(1, self.local_training_period+1):
                           train(net, self.device, train_dl_local, optimizer, e, log_interval=self.log_interval, criterion=self.criterion)                
                           # at here we can check the distance between w_normal and w_g i.e. `\|w_bad - w_g\|_2`
                        # we can print the norm diff out for debugging
                        honest_norm_diff = calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="normal")
                        
                        if self.defense_technique == "norm-clipping-adaptive":
                            # experimental
                            norm_diff_collector.append(honest_norm_diff)            

            else:
                selected_node_indices = np.random.choice(self.num_nets, size=self.part_nets_per_round, replace=False)
                num_data_points = [len(self.net_dataidx_map[i]) for i in selected_node_indices]
                total_num_dps_per_round = sum(num_data_points)

                net_freq = [num_data_points[i]/total_num_dps_per_round for i in range(self.part_nets_per_round)]
                logger.info("Net freq: {}, FL round: {} without adversary".format(net_freq, flr))

                # we need to reconstruct the net list at the beginning
                net_list = [copy.deepcopy(self.net_avg) for _ in range(self.part_nets_per_round)]
                logger.info("################## Starting fl round: {}".format(flr))

                # start the FL process
                for net_idx, net in enumerate(net_list):
                    global_user_idx = selected_node_indices[net_idx]
                    dataidxs = self.net_dataidx_map[global_user_idx]

                    if self.attack_case == "edge-case":
                        train_dl_local, _ = get_dataloader(self.dataset, './data', self.batch_size, 
                                                       self.test_batch_size, dataidxs) # also get the data loader
                    elif self.attack_case in ("normal-case", "almost-edge-case"):
                        train_dl_local, _ = get_dataloader_normal_case(self.dataset, './data', self.batch_size, 
                                                            self.test_batch_size, dataidxs, user_id=global_user_idx,
                                                            num_total_users=self.num_nets,
                                                            poison_type=self.poison_type,
                                                            ardis_dataset=self.ardis_dataset,
                                                            attack_case=self.attack_case) # also get the data loader
                    else:
                        NotImplementedError("Unsupported attack case ...")

                    g_user_indices.append(global_user_idx)
                    
                    logger.info("@@@@@@@@ Working on client: {}, which is Global user: {}".format(net_idx, global_user_idx))

                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.SGD(net.parameters(), lr=self.args_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4) # epoch, net, train_loader, optimizer, criterion
                    for param_group in optimizer.param_groups:
                        logger.info("Effective lr in FL round: {} is {}".format(flr, param_group['lr']))

                    for e in range(1, self.local_training_period+1):
                        train(net, self.device, train_dl_local, optimizer, e, log_interval=self.log_interval, criterion=self.criterion)

                    honest_norm_diff = calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="normal")

                    if self.defense_technique == "norm-clipping-adaptive":
                        # experimental
                        norm_diff_collector.append(honest_norm_diff)   

                adv_norm_diff_list.append(0)
                model_original = list(self.net_avg.parameters())
                v0 = torch.nn.utils.parameters_to_vector(model_original)
                wg_norm_list.append(torch.norm(v0).item())


            ### conduct defense here:
            if self.defense_technique == "no-defense":
                pass
            elif self.defense_technique == "norm-clipping":
                for net_idx, net in enumerate(net_list):
                    self._defender.exec(client_model=net, global_model=self.net_avg)
            elif self.defense_technique == "norm-clipping-adaptive":
                # we will need to adapt the norm diff first before the norm diff clipping
                logger.info("#### Let's Look at the Nom Diff Collector : {} ....; Mean: {}".format(norm_diff_collector, 
                    np.mean(norm_diff_collector)))
                self._defender.norm_bound = np.mean(norm_diff_collector)
                for net_idx, net in enumerate(net_list):
                    self._defender.exec(client_model=net, global_model=self.net_avg)
            elif self.defense_technique == "weak-dp":
                # this guy is just going to clip norm. No noise added here
                for net_idx, net in enumerate(net_list):
                    self._defender.exec(client_model=net,
                                        global_model=self.net_avg,)
            elif self.defense_technique == "krum":
                net_list, net_freq = self._defender.exec(client_models=net_list, 
                                                        num_dps=[self.num_dps_poisoned_dataset]+num_data_points,
                                                        g_user_indices=g_user_indices,
                                                        device=self.device)
            elif self.defense_technique == "multi-krum":
                net_list, net_freq = self._defender.exec(client_models=net_list, 
                                                        num_dps=[self.num_dps_poisoned_dataset]+num_data_points,
                                                        g_user_indices=g_user_indices,
                                                        device=self.device)
            elif self.defense_technique == "rfa":
                net_list, net_freq = self._defender.exec(client_models=net_list,
                                                        net_freq=net_freq,
                                                        maxiter=500,
                                                        eps=1e-5,
                                                        ftol=1e-7,
                                                        device=self.device)
            else:
                NotImplementedError("Unsupported defense method !")

            # after local training periods
            self.net_avg = fed_avg_aggregator(net_list, net_freq, device=self.device, model=self.model)
            if self.defense_technique == "weak-dp":
                # add noise to self.net_avg
                noise_adder = AddNoise(stddev=self.stddev)
                noise_adder.exec(client_model=self.net_avg,
                                                device=self.device)

            v = torch.nn.utils.parameters_to_vector(self.net_avg.parameters())
            logger.info("############ Averaged Model : Norm {}".format(torch.norm(v)))

            calc_norm_diff(gs_model=self.net_avg, vanilla_model=self.vanilla_model, epoch=0, fl_round=flr, mode="avg")
            
            logger.info("Measuring the accuracy of the averaged global model, FL round: {} ...".format(flr))

            overall_acc, raw_acc = test(self.net_avg, self.device, self.vanilla_emnist_test_loader, test_batch_size=self.test_batch_size, criterion=self.criterion, mode="raw-task", dataset=self.dataset, poison_type=self.poison_type)
            backdoor_acc, _ = test(self.net_avg, self.device, self.targetted_task_test_loader, test_batch_size=self.test_batch_size, criterion=self.criterion, mode="targetted-task", dataset=self.dataset, poison_type=self.poison_type)
 
            fl_iter_list.append(flr)
            main_task_acc.append(overall_acc)
            raw_task_acc.append(raw_acc)
            backdoor_task_acc.append(backdoor_acc)

        df = pd.DataFrame({'fl_iter': fl_iter_list, 
                            'main_task_acc': main_task_acc, 
                            'backdoor_acc': backdoor_task_acc, 
                            'raw_task_acc':raw_task_acc, 
                            'adv_norm_diff': adv_norm_diff_list, 
                            'wg_norm': wg_norm_list
                            })
       
        if self.poison_type == 'ardis':
            # add a row showing initial accuracies
            df1 = pd.DataFrame({'fl_iter': [0], 'main_task_acc': [88], 'backdoor_acc': [11], 'raw_task_acc': [0], 'adv_norm_diff': [0], 'wg_norm': [0]})
            df = pd.concat([df1, df])

        results_filename = get_results_filename(self.poison_type, self.attack_method, self.model_replacement, self.project_frequency,
                self.defense_technique, self.norm_bound, self.prox_attack, False, self.model)

        df.to_csv(results_filename, index=False)
        logger.info("Wrote accuracy results to: {}".format(results_filename))

        # save model net_avg
        # torch.save(self.net_avg.state_dict(), "./checkpoint/emnist_lenet_10epoch.pt")


class FixedPoolFederatedLearningTrainer(FederatedLearningTrainer):
    def __init__(self, arguments=None, *args, **kwargs):

        #self.poisoned_emnist_dataset = arguments['poisoned_emnist_dataset']
        self.vanilla_model = arguments['vanilla_model']
        self.net_avg = arguments['net_avg']
        self.net_dataidx_map = arguments['net_dataidx_map']
        self.num_nets = arguments['num_nets']
        self.part_nets_per_round = arguments['part_nets_per_round']
        self.fl_round = arguments['fl_round']
        self.local_training_period = arguments['local_training_period']
        self.adversarial_local_training_period = arguments['adversarial_local_training_period']
        self.args_lr = arguments['args_lr']
        self.args_gamma = arguments['args_gamma']
        self.attacker_pool_size = arguments['attacker_pool_size']
        self.poisoned_emnist_train_loader = arguments['poisoned_emnist_train_loader']
        self.clean_train_loader = arguments['clean_train_loader']
        self.vanilla_emnist_test_loader = arguments['vanilla_emnist_test_loader']
        self.targetted_task_test_loader = arguments['targetted_task_test_loader']
        self.batch_size = arguments['batch_size']
        self.test_batch_size = arguments['test_batch_size']
        self.log_interval = arguments['log_interval']
        self.device = arguments['device']
        self.dataset = arguments["dataset"]
        self.model = arguments["model"]
        self.num_dps_poisoned_dataset = arguments['num_dps_poisoned_dataset']
        self.defense_technique = arguments["defense_technique"]
        self.norm_bound = arguments["norm_bound"]
        self.attack_method = arguments["attack_method"]
        self.criterion = nn.CrossEntropyLoss()
        self.eps = arguments['eps']
        self.dataset = arguments["dataset"]
        self.poison_type = arguments['poison_type']
        self.model_replacement = arguments['model_replacement']
        self.project_frequency = arguments['project_frequency']
        self.adv_lr = arguments['adv_lr']
        self.prox_attack = arguments['prox_attack']
        self.attack_case = arguments['attack_case']
        self.stddev = arguments['stddev']


        logger.info("Posion type! {}".format(self.poison_type))

        if self.attack_method == "pgd":
            self.pgd_attack = True
        else:
            self.pgd_attack = False

        if self.poison_type == 'ardis':
            self.ardis_dataset = datasets.get_ardis_dataset()
            # exclude first 66 points because they are part of the adversary
            if self.attack_case == 'normal-case':
                self.ardis_dataset.data = self.ardis_dataset.data[66:]
            elif self.attack_case == 'almost-edge-case':
                self.ardis_dataset.data = self.ardis_dataset.data[66:132]
        elif self.poison_type == 'southwest':
            self.ardis_dataset = datasets.get_southwest_dataset(attack_case=self.attack_case)
        else:
            self.ardis_dataset=None


        if arguments["defense_technique"] == "no-defense":
            self._defender = None
        elif arguments["defense_technique"] == "norm-clipping" or arguments["defense_technique"] == "norm-clipping-adaptive":
            self._defender = WeightDiffClippingDefense(norm_bound=arguments['norm_bound'])
        elif arguments["defense_technique"] == "weak-dp":
            # doesn't really add noise. just clips
            self._defender = WeightDiffClippingDefense(norm_bound=arguments['norm_bound'])
        elif arguments["defense_technique"] == "krum":
            self._defender = Krum(mode='krum', num_workers=self.part_nets_per_round, num_adv=1)
        elif arguments["defense_technique"] == "multi-krum":
            self._defender = Krum(mode='multi-krum', num_workers=self.part_nets_per_round, num_adv=1)
        elif arguments["defense_technique"] == "rfa":
            self._defender = RFA()
        else:
            NotImplementedError("Unsupported defense method !")

        self.__attacker_pool = np.random.choice(self.num_nets, self.attacker_pool_size, replace=False)

    def run(self):
        main_task_acc = []
        raw_task_acc = []
        backdoor_task_acc = []
        fl_iter_list = []
        adv_norm_diff_list = []
        wg_norm_list = []
        # let's conduct multi-round training
        for flr in range(1, self.fl_round+1):
            # randomly select participating clients
            # in this current version, we sample `part_nets_per_round` per FL round since we assume attacker will always participates
            selected_node_indices = np.random.choice(self.num_nets, size=self.part_nets_per_round, replace=False)

            selected_attackers = [idx for idx in selected_node_indices if idx in self.__attacker_pool]
            selected_honest_users = [idx for idx in selected_node_indices if idx not in self.__attacker_pool]
            logger.info("Selected Attackers in FL iteration-{}: {}".format(flr, selected_attackers))

            num_data_points = []
            for sni in selected_node_indices:
                if sni in selected_attackers:
                    num_data_points.append(self.num_dps_poisoned_dataset)
                else:
                    num_data_points.append(len(self.net_dataidx_map[sni]))

            total_num_dps_per_round = sum(num_data_points)
            net_freq = [num_data_points[i]/total_num_dps_per_round for i in range(self.part_nets_per_round)]
            logger.info("Net freq: {}, FL round: {} with adversary".format(net_freq, flr)) 

            # we need to reconstruct the net list at the beginning
            net_list = [copy.deepcopy(self.net_avg) for _ in range(self.part_nets_per_round)]
            logger.info("################## Starting fl round: {}".format(flr))
            model_original = list(self.net_avg.parameters())
            # super hacky but I'm doing this for the prox-attack
            wg_clone = copy.deepcopy(self.net_avg)
            wg_hat = None
            v0 = torch.nn.utils.parameters_to_vector(model_original)
            wg_norm_list.append(torch.norm(v0).item())


            #     # start the FL process
            for net_idx, global_user_idx in enumerate(selected_node_indices):
                net  = net_list[net_idx]
                if global_user_idx in selected_attackers:
                    pass
                else:
                    dataidxs = self.net_dataidx_map[global_user_idx]
                    #train_dl_local, _ = get_dataloader(self.dataset, './data', self.batch_size, 
                    #                                self.test_batch_size, dataidxs) # also get the data loader

                    # add p-percent edge-case attack here:
                    if self.attack_case == "edge-case":
                        train_dl_local, _ = get_dataloader(self.dataset, './data', self.batch_size, 
                                                       self.test_batch_size, dataidxs) # also get the data loader
                    elif self.attack_case in ("normal-case", "almost-edge-case"):
                        train_dl_local, _ = get_dataloader_normal_case(self.dataset, './data', self.batch_size, 
                                                            self.test_batch_size, dataidxs, user_id=global_user_idx,
                                                            num_total_users=self.num_nets,
                                                            poison_type=self.poison_type,
                                                            ardis_dataset=self.ardis_dataset,
                                                            attack_case=self.attack_case) # also get the data loader
                    else:
                        NotImplementedError("Unsupported attack case ...")
                
                logger.info("@@@@@@@@ Working on client (global-index): {}, which {}-th user in the current round".format(global_user_idx, net_idx))

                #criterion = nn.CrossEntropyLoss()
                #optimizer = optim.SGD(net.parameters(), lr=self.args_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4) # epoch, net, train_loader, optimizer, criterion
                #for param_group in optimizer.param_groups:
                #    logger.info("Effective lr in FL round: {} is {}".format(flr, param_group['lr']))                

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(net.parameters(), lr=self.args_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4) # epoch, net, train_loader, optimizer, criterion
                adv_optimizer = optim.SGD(net.parameters(), lr=self.adv_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4) # looks like adversary needs same lr to hide with others
                prox_optimizer = optim.SGD(wg_clone.parameters(), lr=self.args_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4)
                for param_group in optimizer.param_groups:
                    logger.info("Effective lr in FL round: {} is {}".format(flr, param_group['lr']))


                current_adv_norm_diff_list = []
                if global_user_idx in selected_attackers:
                    # for e in range(1, self.adversarial_local_training_period+1):
                    #    # we always assume net index 0 is adversary
                    #    train(net, self.device, self.poisoned_emnist_train_loader, optimizer, e, log_interval=self.log_interval, criterion=self.criterion)

                    # logger.info("=====> Measuring the model performance of the poisoned model after attack ....")
                    # test(net, self.device, self.vanilla_emnist_test_loader, test_batch_size=self.test_batch_size, criterion=self.criterion, mode="raw-task", dataset=self.dataset, poison_type=self.poison_type)
                    # test(net, self.device, self.targetted_task_test_loader, test_batch_size=self.test_batch_size, criterion=self.criterion, mode="targetted-task", dataset=self.dataset, poison_type=self.poison_type)
                    # # at here we can check the distance between w_bad and w_g i.e. `\|w_bad - w_g\|_2`
                    # calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="bad")

                    if self.prox_attack:
                        # estimate w_hat
                        for inner_epoch in range(1, self.local_training_period+1):
                            estimate_wg(wg_clone, self.device, self.clean_train_loader, prox_optimizer, inner_epoch, log_interval=self.log_interval, criterion=self.criterion)
                        wg_hat = wg_clone

                    for e in range(1, self.adversarial_local_training_period+1):
                       # we always assume net index 0 is adversary
                        if self.defense_technique in ('krum', 'multi-krum'):
                            train(net, self.device, self.poisoned_emnist_train_loader, optimizer, e, log_interval=self.log_interval, criterion=self.criterion,
                                    pgd_attack=self.pgd_attack, eps=self.eps*self.args_gamma**(flr-1), model_original=model_original, project_frequency=self.project_frequency, adv_optimizer=adv_optimizer,
                                    prox_attack=self.prox_attack, wg_hat=wg_hat)
                        else:
                            train(net, self.device, self.poisoned_emnist_train_loader, optimizer, e, log_interval=self.log_interval, criterion=self.criterion,
                                    pgd_attack=self.pgd_attack, eps=self.eps, model_original=model_original, project_frequency=self.project_frequency, adv_optimizer=adv_optimizer,
                                    prox_attack=self.prox_attack, wg_hat=wg_hat)

                    test(net, self.device, self.vanilla_emnist_test_loader, test_batch_size=self.test_batch_size, criterion=self.criterion, mode="raw-task", dataset=self.dataset, poison_type=self.poison_type)
                    test(net, self.device, self.targetted_task_test_loader, test_batch_size=self.test_batch_size, criterion=self.criterion, mode="targetted-task", dataset=self.dataset, poison_type=self.poison_type)

                    # if model_replacement scale models
                    if self.model_replacement:
                        v = torch.nn.utils.parameters_to_vector(net.parameters())
                        logger.info("Attacker before scaling : Norm = {}".format(torch.norm(v)))
                        # adv_norm_diff = calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="bad")
                        # logger.info("||w_bad - w_avg|| before scaling = {}".format(adv_norm_diff))

                        for idx, param in enumerate(net.parameters()):
                            param.data = (param.data - model_original[idx])*(total_num_dps_per_round/self.num_dps_poisoned_dataset) + model_original[idx]
                        v = torch.nn.utils.parameters_to_vector(net.parameters())
                        logger.info("Attacker after scaling : Norm = {}".format(torch.norm(v)))

                    # at here we can check the distance between w_bad and w_g i.e. `\|w_bad - w_g\|_2`
                    # we can print the norm diff out for debugging
                    adv_norm_diff = calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="bad")
                    current_adv_norm_diff_list.append(adv_norm_diff)

                    if self.defense_technique == "norm-clipping-adaptive":
                        # experimental
                        norm_diff_collector.append(adv_norm_diff)
                else:
                    # for e in range(1, self.local_training_period+1):
                    #    train(net, self.device, train_dl_local, optimizer, e, log_interval=self.log_interval, criterion=self.criterion)                
                    # # at here we can check the distance between w_normal and w_g i.e. `\|w_bad - w_g\|_2`
                    # calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="normal")

                    for e in range(1, self.local_training_period+1):
                       train(net, self.device, train_dl_local, optimizer, e, log_interval=self.log_interval, criterion=self.criterion)                
                       # at here we can check the distance between w_normal and w_g i.e. `\|w_bad - w_g\|_2`
                    # we can print the norm diff out for debugging
                    honest_norm_diff = calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="normal")
                    
                    if self.defense_technique == "norm-clipping-adaptive":
                        # experimental
                        norm_diff_collector.append(honest_norm_diff)

            ### conduct defense here:
            if self.defense_technique == "no-defense":
                pass
            elif self.defense_technique == "norm-clipping":
                for net_idx, net in enumerate(net_list):
                    self._defender.exec(client_model=net, global_model=self.net_avg)
            elif self.defense_technique == "norm-clipping-adaptive":
                # we will need to adapt the norm diff first before the norm diff clipping
                logger.info("#### Let's Look at the Nom Diff Collector : {} ....; Mean: {}".format(norm_diff_collector, 
                    np.mean(norm_diff_collector)))
                self._defender.norm_bound = np.mean(norm_diff_collector)
                for net_idx, net in enumerate(net_list):
                    self._defender.exec(client_model=net, global_model=self.net_avg)
            elif self.defense_technique == "weak-dp":
                # this guy is just going to clip norm. No noise added here
                for net_idx, net in enumerate(net_list):
                    self._defender.exec(client_model=net,
                                        global_model=self.net_avg,)
            elif self.defense_technique == "krum":
                net_list, net_freq = self._defender.exec(client_models=net_list, 
                                                        num_dps=[self.num_dps_poisoned_dataset]+num_data_points,
                                                        g_user_indices=g_user_indices,
                                                        device=self.device)
            elif self.defense_technique == "multi-krum":
                net_list, net_freq = self._defender.exec(client_models=net_list, 
                                                        num_dps=[self.num_dps_poisoned_dataset]+num_data_points,
                                                        g_user_indices=g_user_indices,
                                                        device=self.device)
            elif self.defense_technique == "rfa":
                net_list, net_freq = self._defender.exec(client_models=net_list,
                                                        net_freq=net_freq,
                                                        maxiter=500,
                                                        eps=1e-5,
                                                        ftol=1e-7,
                                                        device=self.device)
            else:
                NotImplementedError("Unsupported defense method !")


            # after local training periods
            self.net_avg = fed_avg_aggregator(net_list, net_freq, device=self.device, model=self.model)

            if self.defense_technique == "weak-dp":
                # add noise to self.net_avg
                noise_adder = AddNoise(stddev=self.stddev)
                noise_adder.exec(client_model=self.net_avg,
                                                device=self.device)

            calc_norm_diff(gs_model=self.net_avg, vanilla_model=self.net_avg, epoch=0, fl_round=flr, mode="avg")
            
            logger.info("Measuring the accuracy of the averaged global model, FL round: {} ...".format(flr))
            #overall_acc = test(self.net_avg, self.device, self.vanilla_emnist_test_loader, test_batch_size=self.test_batch_size, criterion=self.criterion, mode="raw-task", dataset=self.dataset, poison_type=self.poison_type)
            #backdoor_acc = test(self.net_avg, self.device, self.targetted_task_test_loader, test_batch_size=self.test_batch_size, criterion=self.criterion, mode="targetted-task", dataset=self.dataset, poison_type=self.poison_type)
            overall_acc, raw_acc = test(self.net_avg, self.device, self.vanilla_emnist_test_loader, test_batch_size=self.test_batch_size, criterion=self.criterion, mode="raw-task", dataset=self.dataset, poison_type=self.poison_type)
            backdoor_acc, _ = test(self.net_avg, self.device, self.targetted_task_test_loader, test_batch_size=self.test_batch_size, criterion=self.criterion, mode="targetted-task", dataset=self.dataset, poison_type=self.poison_type)
 
            fl_iter_list.append(flr)
            main_task_acc.append(overall_acc)
            raw_task_acc.append(raw_acc)
            backdoor_task_acc.append(backdoor_acc)
            if len(current_adv_norm_diff_list) == 0:
                adv_norm_diff_list.append(0)
            else:
                # if you have multiple adversaries in a round, average their norm diff
                adv_norm_diff_list.append(1.0*sum(current_adv_norm_diff_list)/len(current_adv_norm_diff_list))
        
        df = pd.DataFrame({'fl_iter': fl_iter_list, 
                            'main_task_acc': main_task_acc, 
                            'backdoor_acc': backdoor_task_acc, 
                            'raw_task_acc':raw_task_acc, 
                            'adv_norm_diff': adv_norm_diff_list, 
                            'wg_norm': wg_norm_list
                            })
       
        if self.poison_type == 'ardis':
            # add a row showing initial accuracies
            df1 = pd.DataFrame({'fl_iter': [0], 'main_task_acc': [88], 'backdoor_acc': [11], 'raw_task_acc': [0], 'adv_norm_diff': [0], 'wg_norm': [0]})
            df = pd.concat([df1, df])

        results_filename = get_results_filename(self.poison_type, self.attack_method, self.model_replacement, self.project_frequency,
                self.defense_technique, self.norm_bound, self.prox_attack, fixed_pool=True, model_arch=self.model)
        df.to_csv(results_filename, index=False)
        logger.info("Wrote accuracy results to: {}".format(results_filename))
