import pdb
import pandas as pd
import torch

from scipy.special import logit, expit
from utils import *

from geometric_median import geometric_median
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics.pairwise as smp
import hdbscan
# import logger


# from all_utils import calculate_sum_grad_diff
# import logging
# logging.basicConfig()
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

# from utils import extract_classifier_layer

def vectorize_net(net):
    return torch.cat([p.view(-1) for p in net.parameters()])


def load_model_weight(net, weight):
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data =  weight[index_bias:index_bias+p.numel()].view(p.size())
        index_bias += p.numel()

def load_model_weight_diff(net, weight_diff, global_weight):
    """
    load rule: w_t + clipped(w^{local}_t - w_t)
    """
    listed_global_weight = list(global_weight.parameters())
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data =  weight_diff[index_bias:index_bias+p.numel()].view(p.size()) + listed_global_weight[p_index]
        index_bias += p.numel()

def extract_classifier_layer(net_list, global_avg_net, prev_net, model="vgg9"):
    bias_list = []
    weight_list = []
    weight_update = []
    avg_bias = None
    avg_weight = None
    prev_avg_bias = None
    prev_avg_weight = None
    last_model_layer = "classifier" if model=="vgg9" else "fc3" 
    # print(f"state_dict: {prev_net.state_dict()}")
    # print(f"{last_model_layer}")
    if model == "vgg9":
        for idx, param in enumerate(global_avg_net.classifier.parameters()):
            if idx:
                avg_bias = param.data.cpu().numpy()
            else:
                avg_weight = param.data.cpu().numpy()

        for idx, param in enumerate(prev_net.classifier.parameters()):
            if idx:
                prev_avg_bias = param.data.cpu().numpy()
            else:
                prev_avg_weight = param.data.cpu().numpy()
        glob_update = avg_weight - prev_avg_weight
        for net in net_list:
            bias = None
            weight = None
            for idx, param in enumerate(net.classifier.parameters()):
                if idx:
                    bias = param.data.cpu().numpy()
                else:
                    weight = param.data.cpu().numpy()
            bias_list.append(bias)
            weight_list.append(weight)
            weight_update.append(weight-avg_weight)
    elif model == "lenet":
        for idx, param in enumerate(global_avg_net.fc2.parameters()):
            if idx:
                avg_bias = param.data.cpu().numpy()
            else:
                avg_weight = param.data.cpu().numpy()

        for idx, param in enumerate(prev_net.fc2.parameters()):
            if idx:
                prev_avg_bias = param.data.cpu().numpy()
            else:
                prev_avg_weight = param.data.cpu().numpy()
        glob_update = avg_weight - prev_avg_weight
        for net in net_list:
            bias = None
            weight = None
            for idx, param in enumerate(net.fc2.parameters()):
                if idx:
                    bias = param.data.cpu().numpy()
                else:
                    weight = param.data.cpu().numpy()
            bias_list.append(bias)
            weight_list.append(weight)
            weight_update.append(weight-avg_weight)
    
    return bias_list, weight_list, avg_bias, avg_weight, weight_update, glob_update, prev_avg_weight
def rlr_avg(vectorize_nets, vectorize_avg_net, freq, attacker_idxs, lr, n_params, device, robustLR_threshold=4):
    lr_vector = torch.Tensor([lr]*n_params).to(device)
    total_client = len(vectorize_nets)
    local_updates = vectorize_nets - vectorize_avg_net
    print(f"len freq: {len(freq)}")
    print(f"local_updates.shape is: {len(local_updates)}")
    fed_avg_updates_vector = np.average(local_updates, weights=freq, axis=0).astype(float32)
    print(f"fed_avg_vector.shape is: {fed_avg_updates_vector.shape}")
    # vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in neo_net_list]
    selected_net_indx = [i for i in range(total_client) if i not in attacker_idxs]
    selected_freq = np.array(freq)[selected_net_indx]
    selected_freq = [freq/sum(selected_freq) for freq in selected_freq]
    
    
    agent_updates_sign = [np.sign(update) for update in local_updates]  
    sm_of_signs = np.abs(sum(agent_updates_sign))
    sm_of_signs[sm_of_signs < robustLR_threshold] = -lr
    sm_of_signs[sm_of_signs >= robustLR_threshold] = lr
    print(f"sm_of_signs is: {sm_of_signs}")
    
    lr_vector = sm_of_signs
    poison_w_idxs = sm_of_signs < 0
    # poison_w_idxs = poison_w_idxs*1
    print(f"poison_w_idxs: {poison_w_idxs}")
    print(f"lr_vector: {lr_vector}")
    local_updates = np.asarray(local_updates)
    print(f"local_updates.shape is: {local_updates.shape}")
    # local_updates[attacker_idxs][poison_w_idxs] = 0
    cnt = 0
    sm_updates_2 = 0
    # for _id, update in enumerate(local_updates):
    #     if _id not in attacker_idxs:
    #         sm_updates_2 += selected_freq[cnt]*update[poison_w_idxs]
    #         cnt+=1
    for _id, update in enumerate(local_updates):
        if _id not in attacker_idxs:
            sm_updates_2 += freq[_id]*update[poison_w_idxs]
        else:
            sm_updates_2 += freq[_id]*(-update[poison_w_idxs])
            
    print(f"sm_updates_2.shape is: {sm_updates_2.shape}")
    fed_avg_updates_vector[poison_w_idxs] = sm_updates_2
    new_global_params =  (vectorize_avg_net + lr*fed_avg_updates_vector).astype(np.float32)
    return new_global_params

class Defense:
    def __init__(self, *args, **kwargs):
        self.hyper_params = None

    def exec(self, client_model, *args, **kwargs):
        raise NotImplementedError()


class ClippingDefense(Defense):
    """
    Deprecated, do not use this method
    """
    def __init__(self, norm_bound, *args, **kwargs):
        self.norm_bound = norm_bound

    def exec(self, client_model, *args, **kwargs):
        vectorized_net = vectorize_net(client_model)
        weight_norm = torch.norm(vectorized_net).item()
        clipped_weight = vectorized_net/max(1, weight_norm/self.norm_bound)

        logger.info("Norm Clipped Mode {}".format(
            torch.norm(clipped_weight).item()))
        load_model_weight(client_model, clipped_weight)        
        # index_bias = 0
        # for p_index, p in enumerate(client_model.parameters()):
        #     p.data =  clipped_weight[index_bias:index_bias+p.numel()].view(p.size())
        #     index_bias += p.numel()
        ##weight_norm = torch.sqrt(sum([torch.norm(p)**2 for p in client_model.parameters()]))
        #for p_index, p in enumerate(client_model.parameters()):
        #    p.data /= max(1, weight_norm/self.norm_bound)
        return None


class WeightDiffClippingDefense(Defense):
    def __init__(self, norm_bound, *args, **kwargs):
        self.norm_bound = norm_bound

    def exec(self, client_model, global_model, *args, **kwargs):
        """
        global_model: the global model at iteration T, bcast from the PS
        client_model: starting from `global_model`, the model on the clients after local retraining
        """
        vectorized_client_net = vectorize_net(client_model)
        vectorized_global_net = vectorize_net(global_model)
        vectorize_diff = vectorized_client_net - vectorized_global_net

        weight_diff_norm = torch.norm(vectorize_diff).item()
        clipped_weight_diff = vectorize_diff/max(1, weight_diff_norm/self.norm_bound)

        logger.info("Norm Weight Diff: {}, Norm Clipped Weight Diff {}".format(weight_diff_norm,
            torch.norm(clipped_weight_diff).item()))
        load_model_weight_diff(client_model, clipped_weight_diff, global_model)
        return None


class WeakDPDefense(Defense):
    """
        deprecated: don't use!
        according to literature, DPDefense should be applied
        to the aggregated model, not invidual models
        """
    def __init__(self, norm_bound, *args, **kwargs):
        self.norm_bound = norm_bound

    def exec(self, client_model, device, *args, **kwargs):
        self.device = device
        vectorized_net = vectorize_net(client_model)
        weight_norm = torch.norm(vectorized_net).item()
        clipped_weight = vectorized_net/max(1, weight_norm/self.norm_bound)
        dp_weight = clipped_weight + torch.randn(
            vectorized_net.size(),device=self.device) * self.stddev

        load_model_weight(client_model, clipped_weight)
        return None

class AddNoise(Defense):
    def __init__(self, stddev, *args, **kwargs):
        self.stddev = stddev

    def exec(self, client_model, device, *args, **kwargs):
        self.device = device
        vectorized_net = vectorize_net(client_model)
        gaussian_noise = torch.randn(vectorized_net.size(),
                            device=self.device) * self.stddev
        dp_weight = vectorized_net + gaussian_noise
        load_model_weight(client_model, dp_weight)
        logger.info("Weak DP Defense: added noise of norm: {}".format(torch.norm(gaussian_noise)))
        
        return None


class Krum(Defense):
    """
    we implement the robust aggregator at: https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent.pdf
    and we integrate both krum and multi-krum in this single class
    """
    def __init__(self, mode, num_workers, num_adv, *args, **kwargs):
        assert (mode in ("krum", "multi-krum"))
        self._mode = mode
        self.num_workers = num_workers
        self.s = num_adv

    def exec(self, client_models, num_dps, g_user_indices, device, *args, **kwargs):
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        
        neighbor_distances = []
        for i, g_i in enumerate(vectorize_nets):
            distance = []
            for j in range(i+1, len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]
                    distance.append(float(np.linalg.norm(g_i-g_j)**2)) # let's change this to pytorch version
            neighbor_distances.append(distance)

        # compute scores
        nb_in_score = self.num_workers-self.s-2
        scores = []
        for i, g_i in enumerate(vectorize_nets):
            dists = []
            for j, g_j in enumerate(vectorize_nets):
                if j == i:
                    continue
                if j < i:
                    dists.append(neighbor_distances[j][i - j - 1])
                else:
                    dists.append(neighbor_distances[i][j - i - 1])
            # alternative to topk in pytorch and tensorflow
            topk_ind = np.argpartition(dists, nb_in_score)[:nb_in_score]
            scores.append(sum(np.take(dists, topk_ind)))
        
        if self._mode == "krum":
            i_star = scores.index(min(scores))
            logger.info("@@@@ The chosen one is user: {}, which is global user: {}".format(scores.index(min(scores)), g_user_indices[scores.index(min(scores))]))
            aggregated_model = client_models[0] # slicing which doesn't really matter
            load_model_weight(aggregated_model, torch.from_numpy(vectorize_nets[i_star]).to(device))
            neo_net_list = [aggregated_model]
            logger.info("Norm of Aggregated Model: {}".format(torch.norm(torch.nn.utils.parameters_to_vector(aggregated_model.parameters())).item()))
            neo_net_freq = [1.0]
            return neo_net_list, neo_net_freq
        elif self._mode == "multi-krum":
            topk_ind = np.argpartition(scores, nb_in_score+2)[:nb_in_score+2]
            
            # we reconstruct the weighted averaging here:
            selected_num_dps = np.array(num_dps)[topk_ind]
            reconstructed_freq = [snd/sum(selected_num_dps) for snd in selected_num_dps]

            logger.info("Num data points: {}".format(num_dps))
            logger.info("Num selected data points: {}".format(selected_num_dps))
            logger.info("The chosen ones are users: {}, which are global users: {}".format(topk_ind, [g_user_indices[ti] for ti in topk_ind]))
            #aggregated_grad = np.mean(np.array(vectorize_nets)[topk_ind, :], axis=0)
            aggregated_grad = np.average(np.array(vectorize_nets)[topk_ind, :], weights=reconstructed_freq, axis=0).astype(np.float32)

            aggregated_model = client_models[0] # slicing which doesn't really matter
            load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to(device))
            neo_net_list = [aggregated_model]
            #logger.info("Norm of Aggregated Model: {}".format(torch.norm(torch.nn.utils.parameters_to_vector(aggregated_model.parameters())).item()))
            neo_net_freq = [1.0]
            return neo_net_list, neo_net_freq



class RFA(Defense):
    """
    we implement the robust aggregator at: 
    https://arxiv.org/pdf/1912.13445.pdf
    the code is translated from the TensorFlow implementation: 
    https://github.com/krishnap25/RFA/blob/01ec26e65f13f46caf1391082aa76efcdb69a7a8/models/model.py#L264-L298
    """
    def __init__(self, *args, **kwargs):
        pass

    def exec(self, client_models, net_freq, 
                   maxiter=4, eps=1e-5,
                   ftol=1e-6, device=torch.device("cuda"), 
                    *args, **kwargs):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
        """
        # so alphas will be the same as the net freq in our code
        alphas = np.asarray(net_freq, dtype=np.float32)
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        median = self.weighted_average_oracle(vectorize_nets, alphas)

        num_oracle_calls = 1

        # logging
        obj_val = self.geometric_median_objective(median=median, points=vectorize_nets, alphas=alphas)

        logs = []
        log_entry = [0, obj_val, 0, 0]
        logs.append("Tracking log entry: {}".format(log_entry))
        logger.info('Starting Weiszfeld algorithm')
        logger.info(log_entry)

        # start
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = np.asarray([alpha / max(eps, self.l2dist(median, p)) for alpha, p in zip(alphas, vectorize_nets)],
                                 dtype=alphas.dtype)
            weights = weights / weights.sum()
            median = self.weighted_average_oracle(vectorize_nets, weights)
            num_oracle_calls += 1
            obj_val = self.geometric_median_objective(median, vectorize_nets, alphas)
            log_entry = [i+1, obj_val,
                         (prev_obj_val - obj_val)/obj_val,
                         self.l2dist(median, prev_median)]
            logs.append(log_entry)
            logs.append("Tracking log entry: {}".format(log_entry))
            logger.info("#### Oracle Cals: {}, Objective Val: {}".format(num_oracle_calls, obj_val))
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
        #logger.info("Num Oracale Calls: {}, Logs: {}".format(num_oracle_calls, logs))

        aggregated_model = client_models[0] # slicing which doesn't really matter
        load_model_weight(aggregated_model, torch.from_numpy(median.astype(np.float32)).to(device))
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        return neo_net_list, neo_net_freq

    def weighted_average_oracle(self, points, weights):
        """Computes weighted average of atoms with specified weights
        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """
        ### original implementation in TFF
        #tot_weights = np.sum(weights)
        #weighted_updates = [np.zeros_like(v) for v in points[0]]
        #for w, p in zip(weights, points):
        #    for j, weighted_val in enumerate(weighted_updates):
        #        weighted_val += (w / tot_weights) * p[j]
        #return weighted_updates
        ####
        tot_weights = np.sum(weights)
        weighted_updates = np.zeros(points[0].shape)
        for w, p in zip(weights, points):
            weighted_updates += (w * p / tot_weights)
        return weighted_updates

    def l2dist(self, p1, p2):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        # this is a helper function
        return np.linalg.norm(p1 - p2)

    def geometric_median_objective(self, median, points, alphas):
        """Compute geometric median objective."""
        return sum([alpha * self.l2dist(median, p) for alpha, p in zip(alphas, points)])


class GeoMedian(Defense):
    """
    we implement the robust aggregator of Geometric Median (GM)
    """
    def __init__(self, *args, **kwargs):
        pass

    def exec(self, client_models, net_freq, 
                   maxiter=4, eps=1e-5,
                   ftol=1e-6, device=torch.device("cuda"), 
                    *args, **kwargs):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
        """
        # so alphas will be the same as the net freq in our code
        alphas = np.asarray(net_freq, dtype=np.float32)
        vectorize_nets = np.array([vectorize_net(cm).detach().cpu().numpy() for cm in client_models]).astype(np.float32)
        median = geometric_median(vectorize_nets)

        aggregated_model = client_models[0] # slicing which doesn't really matter
        load_model_weight(aggregated_model, torch.from_numpy(median.astype(np.float32)).to(device))
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        return neo_net_list, neo_net_freq


class CONTRA(Defense):

    """
    REIMPLEMENT OF CONTRA ALGORITHM
    Awan, S., Luo, B., Li, F. (2021). 
    CONTRA: Defending Against Poisoning Attacks in Federated Learning.
    In: Bertino, E., Shulman, H., Waidner, M. (eds) 
    Computer Security – ESORICS 2021. 
    ESORICS 2021. Lecture Notes in Computer Science(), 
    vol 12972. Springer, Cham. https://doi.org/10.1007/978-3-030-88418-5_22
    """
    def __init__(self, *args, **kwargs):
        pass
    
    def exec(self, client_models, net_freq, selected_node_indices, historical_local_updates, reputations, delta, threshold, k = 3, device=torch.device("cuda"), *args, **kwargs):
        
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        training_client_cnt = len(selected_node_indices)
        total_clients = len(historical_local_updates)
        avg_k_top_cs = [0.0 for _ in range(total_clients)]
        pairwise_cs = np.zeros((total_clients, total_clients))
        # pairwise_cs = np.zeros((training_client_cnt, training_client_cnt))


        for net_idx, global_node_id in enumerate(selected_node_indices):
            i_local_updates = np.asarray(historical_local_updates[global_node_id])
            cs_i = []
            for net_idx_p, global_node_id_p in enumerate(selected_node_indices):
                if global_node_id_p != global_node_id:
                    p_update = historical_local_updates[global_node_id_p]
                    if len(p_update) > 1:
                        cs_p_i = np.dot(i_local_updates, np.asarray(historical_local_updates[global_node_id_p]))/(np.linalg.norm(i_local_updates)*np.linalg.norm(historical_local_updates[global_node_id_p]))
                    else:
                        cs_p_i = 0.0
                    cs_i.append(cs_p_i)
                    pairwise_cs[global_node_id][global_node_id_p] = cs_p_i
                        
            # for client_p in range(total_clients):
            #     if client_p+1 != global_node_id:
            #         p_update = historical_local_updates[client_p]
            #         if len(p_update) > 1:
            #             cs_p_i = np.dot(i_local_updates, np.asarray(historical_local_updates[client_p]))/(np.linalg.norm(i_local_updates)*np.linalg.norm(historical_local_updates[client_p]))
            #         else:
            #             cs_p_i = 0.0
            #         cs_i.append(cs_p_i)
            #         pairwise_cs[global_node_id][client_p] = cs_p_i
            
            cs_i = np.asarray(cs_i)
            cs_i[::-1].sort()
            avg_k_top_cs_i = np.average(cs_i[:k])
            if avg_k_top_cs_i > threshold:
                reputations[global_node_id] -= delta
            else:
                reputations[global_node_id] += delta
            avg_k_top_cs[global_node_id] = avg_k_top_cs_i

        alignment_levels = pairwise_cs.copy()
        lr_list = [1.0 for _ in range(total_clients)]
        # for net_idx, global_node_id in enumerate(selected_node_indices):
        #     lr_list[global_node_id] = 1.0
        # print("avg_k_top_cs: ", avg_k_top_cs)
        for net_idx in selected_node_indices:
            cs_m_n = []
            for client_p in selected_node_indices:
                if client_p != net_idx: 
                    if avg_k_top_cs[client_p] > avg_k_top_cs[net_idx]:
                        alignment_levels[net_idx][client_p] *= float(avg_k_top_cs[net_idx]/avg_k_top_cs[client_p])
                    # else:
                    #     alignment_levels[net_idx][client_p] *= min(1.0, float(avg_k_top_cs[net_idx])/1.0)
            a = np.asarray(alignment_levels[net_idx])
            a = a[a != 0.0]
            # print("a: ", a)
            # print(max(a))
            # print("alignment_levels[net_idx]: ", alignment_levels[net_idx])
            # print("max(alignment_levels[net_idx]: ", max(alignment_levels[net_idx]))
            # print(alignment_levels[net_idx].shape)
            lr_net_idx = 1.0 - np.amax(alignment_levels[net_idx])
            #print("lr_net_idx: ", lr_net_idx)
            lr_list[net_idx] = lr_net_idx
            reputations[net_idx] = max(np.asarray(reputations))
        #print("alignment_levels: ", alignment_levels)
        lr_final = []
        for net_idx, global_node_id in enumerate(selected_node_indices):
            lr_final.append(lr_list[global_node_id])
        #print("lr_list first: ", lr_final)
        lr_list = np.asarray(lr_final)
        lr_list = lr_list/(max(lr_list))
        print("lr_list: ", lr_list)
        for i, lr in enumerate(lr_list):
            if(lr == 1.0):
                lr_list[i] = logit(0.99)+0.5
            else:
                lr_list[i] = logit(lr_list[i]) + 0.5
        # lr_list = logit(lr_list/(1.0-lr_list)) + 0.5 
        print("lr_list: ", lr_list)
        weights = lr_list.copy()
        print("weights: ", weights)
        aggregated_w = self.weighted_average_oracle(vectorize_nets, weights)
        aggregated_model = client_models[0] # slicing which doesn't really matter
        load_model_weight(aggregated_model, torch.from_numpy(aggregated_w.astype(np.float32)).to(device))
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        return neo_net_list, neo_net_freq, reputations
        
    def weighted_average_oracle(self, points, weights):
        """Computes weighted average of atoms with specified weights
        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """
        ### original implementation in TFF
        #tot_weights = np.sum(weights)
        #weighted_updates = [np.zeros_like(v) for v in points[0]]
        #for w, p in zip(weights, points):
        #    for j, weighted_val in enumerate(weighted_updates):
        #        weighted_val += (w / tot_weights) * p[j]
        #return weighted_updates
        ####
        tot_weights = np.sum(weights)
        weighted_updates = np.zeros(points[0].shape)
        for w, p in zip(weights, points):
            weighted_updates += (w * p / tot_weights)
        return weighted_updates
    
class KmeansBased(Defense):
    def __init__(self, num_workers, num_adv, *args, **kwargs):
        self.num_workers = num_workers
        self.s = num_adv
    def exec(self, client_models, num_dps, net_avg, net_freq, g_user_indices, round, device=torch.device("cuda"), *args, **kwargs):
        from sklearn.cluster import KMeans

        if round < 50:
            # WARM START WITH KRUM
            vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
            neighbor_distances = []
            for i, g_i in enumerate(vectorize_nets):
                distance = []
                for j in range(i+1, len(vectorize_nets)):
                    if i != j:
                        g_j = vectorize_nets[j]
                        distance.append(float(np.linalg.norm(g_i-g_j)**2)) # let's change this to pytorch version
                neighbor_distances.append(distance)

            # compute scores
            nb_in_score = self.num_workers-self.s-2
            scores = []
            for i, g_i in enumerate(vectorize_nets):
                dists = []
                for j, g_j in enumerate(vectorize_nets):
                    if j == i:
                        continue
                    if j < i:
                        dists.append(neighbor_distances[j][i - j - 1])
                    else:
                        dists.append(neighbor_distances[i][j - i - 1])

                # alternative to topk in pytorch and tensorflow
                topk_ind = np.argpartition(dists, nb_in_score)[:nb_in_score]
                scores.append(sum(np.take(dists, topk_ind)))
            
            i_star = scores.index(min(scores))
            logger.info("@@@@ The chosen one is user: {}, which is global user: {}".format(scores.index(min(scores)), g_user_indices[scores.index(min(scores))]))
            aggregated_model = client_models[0] # slicing which doesn't really matter
            load_model_weight(aggregated_model, torch.from_numpy(vectorize_nets[i_star]).to(device))
            neo_net_list = [aggregated_model]
            logger.info("Norm of Aggregated Model: {}".format(torch.norm(torch.nn.utils.parameters_to_vector(aggregated_model.parameters())).item()))
            neo_net_freq = [1.0]
            return neo_net_list, neo_net_freq
        else:
            vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
            baseline_net = self.weighted_average_oracle(vectorize_nets, net_freq)
            base_model = client_models[0]
            load_model_weight(base_model, torch.from_numpy(baseline_net.astype(np.float32)).to(device)) 

            total_client = len(client_models)
            bias_list, weight_list, avg_bias, avg_weight, weight_update = extract_classifier_layer(client_models, base_model)
            eucl_dis, cs_dis = get_distance_on_avg_net(weight_list, avg_weight, weight_update, total_client)
            norm_cs_data = min_max_scale(cs_dis)
            norm_eu_data = 1.0 - min_max_scale(eucl_dis)
            stack_dis = np.hstack((norm_cs_data,norm_eu_data))
            # print("stack_dis.shape: ", stack_dis.shape)
            kmeans = KMeans(n_clusters = 2)
            pred_labels = kmeans.fit_predict(stack_dis)
            print("pred_labels is: ", pred_labels)
            label_0 = np.count_nonzero(pred_labels == 0)
            label_1 = total_client - label_0
            cnt_pred_attackers = label_0 if label_0 <= label_1 else label_1
            label_att = 0 if label_0 <= label_1 else 1
            print("label_att: ", label_att)
            pred_attackers_indx = np.argwhere(np.asarray(pred_labels) == label_att).flatten()


            print("pred_attackers_indx: ", pred_attackers_indx)
            neo_net_list = []
            neo_net_freq = []
            pred_attackers_indx = pred_attackers_indx.tolist()
            print("pred_attackers_indx: ", pred_attackers_indx)
            selected_net_indx = []
            for idx, net in enumerate(client_models):
                if idx not in pred_attackers_indx:
                    neo_net_list.append(net)
                    neo_net_freq.append(1.0)
                    selected_net_indx.append(idx)
            vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in neo_net_list]
            selected_num_dps = np.array(num_dps)[selected_net_indx]
            reconstructed_freq = [snd/sum(selected_num_dps) for snd in selected_num_dps]

            logger.info("Num data points: {}".format(num_dps))
            logger.info("Num selected data points: {}".format(selected_num_dps))
            logger.info("The chosen ones are users: {}, which are global users: {}".format(selected_net_indx, [g_user_indices[ti] for ti in selected_net_indx]))
            aggregated_w = self.weighted_average_oracle(vectorize_nets, reconstructed_freq)
            aggregated_model = client_models[0] # slicing which doesn't really matter
            load_model_weight(aggregated_model, torch.from_numpy(aggregated_w.astype(np.float32)).to(device))
            neo_net_list = [aggregated_model]
            neo_net_freq = [1.0]
            return neo_net_list, neo_net_freq
        

    def weighted_average_oracle(self, points, weights):
        """Computes weighted average of atoms with specified weights
        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """
        ### original implementation in TFF
        #tot_weights = np.sum(weights)
        #weighted_updates = [np.zeros_like(v) for v in points[0]]
        #for w, p in zip(weights, points):
        #    for j, weighted_val in enumerate(weighted_updates):
        #        weighted_val += (w / tot_weights) * p[j]
        #return weighted_updates
        ####
        tot_weights = np.sum(weights)
        weighted_updates = np.zeros(points[0].shape)
        for w, p in zip(weights, points):
            weighted_updates += (w * p / tot_weights)
        return weighted_updates
    

class KrMLRFL(Defense):
    """
    we implement the robust aggregator at: https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent.pdf
    and we integrate both krum and multi-krum in this single class
    """
    def __init__(self, total_workers, num_workers, num_adv, num_valid = 1, instance="benchmark", *args, **kwargs):
        # assert (mode in ("krum", "multi-krum"))
        self.num_valid = num_valid
        self.num_workers = num_workers
        self.s = num_adv
        self.instance = instance
        self.choosing_frequencies = {}
        self.accumulate_t_scores = {}
        self.pairwise_w = np.zeros((total_workers+1, total_workers+1))
        self.pairwise_b = np.zeros((total_workers+1, total_workers+1))
        
        # print(self.pairwise_cs.shape)
        logger.info("Starting performing KrMLRFL...")
        self.pairwise_choosing_frequencies = np.zeros((total_workers, total_workers))


        with open(f'{self.instance}_combined_file_klfrl.csv', 'w', newline='') as outcsv:
            writer = csv.DictWriter(outcsv, fieldnames = ["flr", 
                                                          "attacker_idxs",
                                                          "pred_idxs_1", 
                                                          "pred_idxs_2",
                                                          "true_positive_1",
                                                            "true_positive_2",
                                                            "false_negative_1",
                                                            "false_negative_2",
                                                            "false_positive_1",
                                                            "false_positive_2",
                                                            "missed_idxs_1",
                                                            "missed_idxs_2",
                                                            "freq",
                                                            "t_score",
                                                            "num_dps",
                                                            "saved_pairwise_sim"])
            writer.writeheader()
        
        with open(f'{self.instance}_cluster_log.csv', 'w', newline='') as log_csv:
            writer = csv.DictWriter(log_csv, fieldnames=['round', 'has_attacker', 'trusted_krum_s', 'adv_krum_s_avg', 'ben_krum_s_avg', 'adv_krum_s', 'ben_krum_s'])
            writer.writeheader()  

    def exec(self, client_models, num_dps, net_freq, net_avg, g_user_indices, pseudo_avg_net, round, selected_attackers, model_name, device, *args, **kwargs):
        from sklearn.cluster import KMeans
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        trusted_models = []
        neighbor_distances = []
        bias_list, weight_list, avg_bias, avg_weight, weight_update, glob_update, prev_avg_weight = extract_classifier_layer(client_models, pseudo_avg_net, net_avg, model_name)

        missed_attacker_idxs_by_thre = []
        missed_attacker_idxs_by_kmeans = []
        freq_participated_attackers = []
        
        total_client = len(g_user_indices)
        round_bias_pairwise = np.zeros((total_client, total_client))
        round_weight_pairwise = np.zeros((total_client, total_client))
        
        sum_diff_by_label = calculate_sum_grad_diff(weight_update)
        norm_bias_list = normalize(bias_list, axis=1)
        norm_grad_diff_list = normalize(sum_diff_by_label, axis=1)
        
        # UPDATE CUMULATIVE COSINE SIMILARITY 
        for i, g_i in enumerate(g_user_indices):
            distance = []
            for j, g_j in enumerate(g_user_indices):
                # if i != j:
                
                self.pairwise_choosing_frequencies[g_i][g_j] = self.pairwise_choosing_frequencies[g_i][g_j] + 1.0
                bias_p_i = norm_bias_list[i]
                bias_p_j = norm_bias_list[j]
                cs_1 = np.dot(bias_p_i, bias_p_j)/(np.linalg.norm(bias_p_i)*np.linalg.norm(bias_p_j))
                round_bias_pairwise[i][j] = cs_1.flatten()
                
                w_p_i = norm_grad_diff_list[i]
                w_p_j = norm_grad_diff_list[j]
                cs_2 = np.dot(w_p_i, w_p_j)/(np.linalg.norm(w_p_i)*np.linalg.norm(w_p_j))
                round_weight_pairwise[i][j] = cs_2.flatten()
                
                cli_i_arr = np.hstack((bias_p_i, w_p_i))
                cli_j_arr = np.hstack((bias_p_j, w_p_j))
                
                
                # cs_arr = np.hstack(cs_1, cs_2)
                
            #     if j > i:
            #         distance.append(float(np.linalg.norm(cli_i_arr-cli_j_arr)**2)) # let's change this to pytorch version
            # neighbor_distances.append(distance)
                
        logger.info("Starting performing KrMLRFL...")
       

        # # compute scores by KRUM*
        
        for i, g_i in enumerate(vectorize_nets):
            distance = []
            for j in range(i+1, len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]
                    distance.append(float(np.linalg.norm(g_i-g_j)**2)) # let's change this to pytorch version
            neighbor_distances.append(distance)

        # compute scores
        nb_in_score = self.num_workers-self.s-2
        scores = []
        for i, g_i in enumerate(vectorize_nets):
            dists = []
            for j, g_j in enumerate(vectorize_nets):
                if j == i:
                    continue
                if j < i:
                    dists.append(neighbor_distances[j][i - j - 1])
                else:
                    dists.append(neighbor_distances[i][j - i - 1])

            # alternative to topk in pytorch and tensorflow
            topk_ind = np.argpartition(dists, nb_in_score)[:nb_in_score]
            scores.append(sum(np.take(dists, topk_ind)))
        
        print(f"scores of krum method: {scores}")
        i_star = scores.index(min(scores))
        
        # use krum as the baseline to improve, mark the one chosen by krum as trusted
        if self.num_valid == 1:
            i_star = scores.index(min(scores))
            logger.info("@@@@ The chosen trusted worker is user: {}, which is global user: {}".format(scores.index(min(scores)), g_user_indices[scores.index(min(scores))]))
            trusted_models.append(i_star)
        else:
            topk_ind = np.argpartition(scores, nb_in_score+2)[:self.num_valid]
            
            # we reconstruct the weighted averaging here:
            selected_num_dps = np.array(num_dps)[topk_ind]
            logger.info("Num selected data points: {}".format(selected_num_dps))
            logger.info("The chosen ones are users: {}, which are global users: {}".format(topk_ind, [g_user_indices[ti] for ti in topk_ind]))

            for ind in topk_ind:
                trusted_models.append(ind)
        
        trusted_index = i_star # used for get labels of attackers

        scaler = MinMaxScaler()
        round_bias_pairwise = scaler.fit_transform(round_bias_pairwise)
        round_weight_pairwise = scaler.fit_transform(round_weight_pairwise)

        for i, g_i in enumerate(g_user_indices):
            for j, g_j in enumerate(g_user_indices):
                freq_appear = self.pairwise_choosing_frequencies[g_i][g_j]
                self.pairwise_w[g_i][g_j] = (freq_appear - 1)/freq_appear*self.pairwise_w[g_i][g_j] +  1/freq_appear*round_weight_pairwise[i][j]
                self.pairwise_b[g_i][g_j] = (freq_appear - 1)/freq_appear*self.pairwise_b[g_i][g_j] +  1/freq_appear*round_bias_pairwise[i][j]
                
        
        # From now on, trusted_models contain the index base models treated as valid users.
        raw_t_score = self.get_trustworthy_scores(glob_update, weight_update)
        t_score = []
        for idx, cli in enumerate(g_user_indices):
            # increase the frequency of the selected choosen clients
            self.choosing_frequencies[cli] = self.choosing_frequencies.get(cli, 0) + 1
            # update the accumulator
            self.accumulate_t_scores[cli] = ((self.choosing_frequencies[cli] - 1) / self.choosing_frequencies[cli]) * self.accumulate_t_scores.get(cli, 0) + (1 / self.choosing_frequencies[cli]) *  raw_t_score[idx]
            t_score.append(self.accumulate_t_scores[cli])
        
        
        t_score = np.array(t_score)
        threshold = min(0.5, np.median(t_score))
        
        
        participated_attackers = []
        for in_, id_ in enumerate(g_user_indices):
            if id_ in selected_attackers:
                participated_attackers.append(in_)
        print("At round: ", round)
        if not len(selected_attackers):
            print("THIS ROUND HAS NO ATTACKER!!!")
        print("real attackers indx: ", participated_attackers)
        print(f'[T_SCORE] median score: {np.median(t_score)}')
        print("[T_SCORE] trustworthy score is: ", t_score)
        
        attacker_local_idxs = [ind_ for ind_ in range(len(g_user_indices)) if t_score[ind_] > threshold]
        print("[T_SCORE] attacker_local_idxs is: ", attacker_local_idxs)
        global_pred_attackers_indx = [g_user_indices[ind_] for ind_ in attacker_local_idxs]
        print("[T_SCORE] global_pred_attackers_indx: ", global_pred_attackers_indx)
        missed_attacker_idxs_by_thre = [at_id for at_id in participated_attackers if at_id not in attacker_local_idxs]
        attacker_local_idxs_2 = []
        saved_pairwise_sim = []

        np_scores = np.asarray(scores)
        final_attacker_idxs = attacker_local_idxs # for the first filter
        # NOW CHECK FOR ROUND 50
        if round >= 1: 
            # TODO: find dynamic threshold
            
            cummulative_w = self.pairwise_w[np.ix_(g_user_indices, g_user_indices)]
            cummulative_b = self.pairwise_b[np.ix_(g_user_indices, g_user_indices)]
            
            
            saved_pairwise_sim = np.hstack((cummulative_w, cummulative_b))
            kmeans = KMeans(n_clusters = 2)
                    
            hb_clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                    gen_min_span_tree=False,
                                    metric='euclidean', min_cluster_size=2, min_samples=1, p=None)
            hb_clusterer.fit(saved_pairwise_sim[attacker_local_idxs])
            print("hb_clusterer.labels_ is: ", hb_clusterer.labels_)

            pred_labels = kmeans.fit_predict(saved_pairwise_sim)
            centroids = kmeans.cluster_centers_
            np_centroids = np.asarray(centroids)
            
            
            cls_0_idxs = np.argwhere(np.asarray(pred_labels) == 0).flatten()
            cls_1_idxs = np.argwhere(np.asarray(pred_labels) == 1).flatten()
            dist_0 = np.sqrt(np.sum(np.square(saved_pairwise_sim[cls_0_idxs]-np_centroids[0])))/len(cls_0_idxs)
            dist_1 = np.sqrt(np.sum(np.square(saved_pairwise_sim[cls_1_idxs]-np_centroids[1])))/len(cls_1_idxs)
            print(f"dist_0 is {dist_0}, dist_1 is {dist_1}")
            
            
            trusted_label = pred_labels[trusted_index]
            label_attack = 0 if trusted_label == 1 else 1
        
            pred_attackers_indx_2 = np.argwhere(np.asarray(pred_labels) == label_attack).flatten()
        
            
            print("[PAIRWISE] pred_attackers_indx: ", pred_attackers_indx_2)
            pred_normal_client = [_id for _id in range(total_client) if _id not in pred_attackers_indx_2]

            #FOR LOGGING ONLY
            adv_krum_s = np_scores[pred_attackers_indx_2]
            ben_krum_s = np_scores[pred_normal_client]
            adv_krum_s_avg = np.average(adv_krum_s).flatten()[0]
            ben_krum_s_avg = np.average(ben_krum_s).flatten()[0]
            print(f"trusted client score is: {np_scores[i_star]}")
            print(f"attackers' scores by krum are: {np_scores[pred_attackers_indx_2]}")
            print(f"adv_krum_s_avg: {adv_krum_s_avg}")
            print(f"pred_normal_client's score by krum are: {np_scores[pred_normal_client]}")
            print(f"ben_krum_s_avg: {ben_krum_s_avg}")
            has_attacker = True if len(selected_attackers) else False
            cluster_log_row = (round, has_attacker, np_scores[i_star], adv_krum_s_avg, ben_krum_s_avg, adv_krum_s, ben_krum_s)
            with open (f"{self.instance}_cluster_log.csv", "a+") as log_csv:
                writer = csv.writer(log_csv)
                writer.writerow(cluster_log_row)
            missed_attacker_idxs_by_kmeans = [at_id for at_id in participated_attackers if at_id not in pred_attackers_indx_2]

            attacker_local_idxs_2 = pred_attackers_indx_2
            temp_diff_score = (adv_krum_s_avg-np_scores[i_star])/(ben_krum_s_avg-np_scores[i_star])
            # if temp_diff_score <= 2.0:
            #     # pseudo_final_attacker_idxs = []
            #     attacker_local_idxs_2 = []
            pseudo_final_attacker_idxs = np.union1d(attacker_local_idxs_2, attacker_local_idxs).flatten()
            print(f"temp_diff_score is: {temp_diff_score}")

            if round >= 50:
                final_attacker_idxs = pseudo_final_attacker_idxs
            print("assumed final_attacker_idxs: ", pseudo_final_attacker_idxs)
            print(f"final_attacker_idxs is: {final_attacker_idxs}")


        freq_participated_attackers = [self.choosing_frequencies[g_idx] for g_idx in g_user_indices]
        true_positive_pred_layer1 = []
        true_positive_pred_layer2 = []
        false_positive_pred_layer1 = []
        false_positive_pred_layer2 = []
        for id_ in participated_attackers:
            true_positive_pred_layer1.append(1.0 if id_ in attacker_local_idxs else 0.0)
            true_positive_pred_layer2.append(1.0 if id_ in attacker_local_idxs_2 else 0.0)
        for id_ in attacker_local_idxs:
            if id_ not in participated_attackers:
                false_positive_pred_layer1.append(1.0)
        for id_ in attacker_local_idxs_2:
            if id_ not in participated_attackers:
                false_positive_pred_layer2.append(1.0)
            # if id_ not in 
        

            
        # true_positive_pred_layer1_val = sum(true_positive_pred_layer1)/len(true_positive_pred_layer1) if len(true_positive_pred_layer1) else 0.0
        # true_positive_pred_layer2_val = sum(true_positive_pred_layer2)/len(true_positive_pred_layer2) if len(true_positive_pred_layer2) else 0.0
        true_positive_pred_layer1_val = sum(true_positive_pred_layer1)/len(participated_attackers) if len(true_positive_pred_layer1) else 0.0
        if len(participated_attackers) == 0 and len(attacker_local_idxs) == 0:
            true_positive_pred_layer1_val = 1.0
        true_positive_pred_layer2_val = sum(true_positive_pred_layer2)/len(participated_attackers) if len(true_positive_pred_layer2) else 0.0
        
        if len(participated_attackers) == 0 and len(attacker_local_idxs_2) == 0:
            true_positive_pred_layer2_val = 1.0
        fn_layer1_val = 1.0 - true_positive_pred_layer1_val
        fn_layer2_val = 1.0 - true_positive_pred_layer2_val
        fp_layer1_val = sum(false_positive_pred_layer1)/(total_client-len(participated_attackers)) if len(false_positive_pred_layer1) else 0.0
        fp_layer2_val = sum(false_positive_pred_layer2)/(total_client-len(participated_attackers)) if len(false_positive_pred_layer2) else 0.0

        logging_per_round = (
            round,
            participated_attackers,
            attacker_local_idxs,
            attacker_local_idxs_2,
            true_positive_pred_layer1_val,
            true_positive_pred_layer2_val,
            fn_layer1_val,
            fn_layer2_val,
            fp_layer1_val,
            fp_layer2_val,
            missed_attacker_idxs_by_thre,
            missed_attacker_idxs_by_kmeans,
            freq_participated_attackers,
            t_score,
            num_dps,
            saved_pairwise_sim
        )
        
        with open(f'{self.instance}_combined_file_klfrl.csv', "a+") as w_f:
            writer = csv.writer(w_f)
            writer.writerow(logging_per_round)
        neo_net_list = []
        neo_net_freq = []
        selected_net_indx = []
        for idx, net in enumerate(client_models):
            if idx not in final_attacker_idxs:
                neo_net_list.append(net)
                neo_net_freq.append(1.0)
                selected_net_indx.append(idx)
        if len(neo_net_list) == 0:
            neo_net_list.append(client_models[i_star])
            selected_net_indx.append(i_star)
            pred_g_attacker = [g_user_indices[i] for i in final_attacker_idxs]
            # return [client_models[i_star]], [1.0], pred_g_attacker
            return [net_avg], [1.0], []
            
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in neo_net_list]
        selected_num_dps = np.array(num_dps)[selected_net_indx]
        reconstructed_freq = [snd/sum(selected_num_dps) for snd in selected_num_dps]

        logger.info("Num data points: {}".format(num_dps))
        logger.info("Num selected data points: {}".format(selected_num_dps))
        logger.info("The chosen ones are users: {}, which are global users: {}".format(selected_net_indx, [g_user_indices[ti] for ti in selected_net_indx]))
        
        aggregated_grad = np.average(vectorize_nets, weights=reconstructed_freq, axis=0).astype(np.float32)

        aggregated_model = client_models[0] # slicing which doesn't really matter
        load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to(device))
        pred_g_attacker = [g_user_indices[i] for i in final_attacker_idxs]
        # print(self.pairwise_cs)
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        return neo_net_list, neo_net_freq, pred_g_attacker

    def get_trustworthy_scores(self, global_update, weight_update):
        # print("base_model_update: ", base_model_update)
        cs_dist = get_cs_on_base_net(weight_update, global_update)
        score = np.array(cs_dist)
        # print("score_avg:= ", score_avg)
        norm_score = min_max_scale(score)
        
        return norm_score
        # for cli_ind, weight_update in enumerate(weight_update):

    def get_predicted_attackers(self, weight_list, avg_weight, weight_update, total_client):
        # from sklearn.cluster import KMeans
        eucl_dis, cs_dis = get_distance_on_avg_net(weight_list, avg_weight, weight_update, total_client)
        norm_cs_data = min_max_scale(cs_dis)
        norm_eu_data = 1.0 - min_max_scale(eucl_dis)
        # norm_eu_data = min_max_scale(eucl_dis)
        stack_dis = np.hstack((norm_cs_data,norm_eu_data))
        print("stack dis is: ", stack_dis)
        temp_score = [0.5*norm_cs_data[i] + 0.5*norm_eu_data[i] for i in range(total_client)]
        threshold = sum(temp_score)/total_client
        abnormal_score = [1.0 if temp_score[i] > threshold else 0.0 for i in range(total_client)]
        print("abnormal_score: ", abnormal_score)
        
        hb_clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                gen_min_span_tree=False, leaf_size=40,
                                metric='euclidean', min_cluster_size=2, min_samples=None, p=None)
        hb_clusterer.fit(stack_dis)
        print("hb_clusterer.labels_ is: ", hb_clusterer.labels_)
        return abnormal_score

class MlFrl(Defense):
    """
    we implement the robust aggregator at: https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent.pdf
    and we integrate both krum and multi-krum in this single class
    """
    def __init__(self, total_workers, num_workers, num_adv, num_valid = 1, instance="benchmark", *args, **kwargs):
        # assert (mode in ("krum", "multi-krum"))
        self.num_valid = num_valid
        self.num_workers = num_workers
        self.s = num_adv
        self.instance = instance
        self.choosing_frequencies = {}
        self.accumulate_t_scores = {}
        self.pairwise_w = np.zeros((total_workers+1, total_workers+1))
        self.pairwise_b = np.zeros((total_workers+1, total_workers+1))
        
        # print(self.pairwise_cs.shape)
        logger.info("Starting performing KrMLRFL...")
        self.pairwise_choosing_frequencies = np.zeros((total_workers, total_workers))


        with open(f'{self.instance}_combined_file_klfrl.csv', 'w', newline='') as outcsv:
            writer = csv.DictWriter(outcsv, fieldnames = ["flr", 
                                                          "attacker_idxs",
                                                          "pred_idxs_1", 
                                                          "pred_idxs_2",
                                                          "true_positive_1",
                                                            "true_positive_2",
                                                            "false_negative_1",
                                                            "false_negative_2",
                                                            "false_positive_1",
                                                            "false_positive_2",
                                                            "missed_idxs_1",
                                                            "missed_idxs_2",
                                                            "freq",
                                                            "t_score",
                                                            "num_dps",
                                                            "saved_pairwise_sim"])
            writer.writeheader()
        
        with open(f'{self.instance}_cluster_log.csv', 'w', newline='') as log_csv:
            writer = csv.DictWriter(log_csv, fieldnames=['round', 'has_attacker', 'trusted_krum_s', 'adv_krum_s_avg', 'ben_krum_s_avg', 'adv_krum_s', 'ben_krum_s'])
            writer.writeheader()    
    def exec(self, client_models, num_dps, net_freq, net_avg, g_user_indices, pseudo_avg_net, round, selected_attackers, device, *args, **kwargs):
        from sklearn.cluster import KMeans
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        vectorize_avg_net = vectorize_net(net_avg).detach().cpu().numpy()
        
        trusted_models = []
        neighbor_distances = []
        bias_list, weight_list, avg_bias, avg_weight, weight_update, glob_update, prev_avg_weight = extract_classifier_layer(client_models, pseudo_avg_net, net_avg)

        missed_attacker_idxs_by_thre = []
        missed_attacker_idxs_by_kmeans = []
        freq_participated_attackers = []
        
        total_client = len(g_user_indices)

        # NEW IDEA
        robustLR_threshold = 2
        local_updates = vectorize_nets - vectorize_avg_net
        # print(f"len freq: {len(freq)}")
        local_updates = np.asarray(local_updates)
        print(f"local_updates.shape is: {local_updates.shape}")
        # vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in neo_net_list]        
        
        agent_updates_sign = [np.sign(update) for update in local_updates]  
        sm_of_signs = np.abs(sum(agent_updates_sign))
        sm_of_signs[sm_of_signs < robustLR_threshold] = -0.2
        sm_of_signs[sm_of_signs >= robustLR_threshold] = 0.2
        print(f"sm_of_signs is: {sm_of_signs}")
        poison_w_idxs = sm_of_signs < 0
        # poison_w_idxs = poison_w_idxs*1
        print(f"poison_w_idxs: {poison_w_idxs}")

        np_vectorize_nets = np.asarray(vectorize_nets)
        print(f"np_vectorize_nets.shape is: {np_vectorize_nets.shape}")
        filtered_vectorize_nets = np_vectorize_nets[:,poison_w_idxs]
        print(f"filtered_vectorize_nets.shape is: {filtered_vectorize_nets.shape}")
        filtered_update_nets = local_updates[:,poison_w_idxs]
        print(f"filtered_update_nets.shape is: {filtered_update_nets.shape}")


        round_bias_pairwise = np.zeros((total_client, total_client))
        round_weight_pairwise = np.zeros((total_client, total_client))
        round_update_pw_cs = np.zeros((total_client, total_client))
        round_update_pw_eu = np.zeros((total_client, total_client))
        
        sum_diff_by_label = calculate_sum_grad_diff(weight_update)
        norm_bias_list = normalize(bias_list, axis=1)
        norm_grad_diff_list = normalize(sum_diff_by_label, axis=1)
        
        # UPDATE CUMMULATIVE COSINE SIMILARITY 
        for i, g_i in enumerate(g_user_indices):
            distance = []
            for j, g_j in enumerate(g_user_indices):
                # if i != j:
                
                u_i = filtered_update_nets[i]
                u_j = filtered_update_nets[j]
                cs_ij = np.dot(u_i, u_j)/(np.linalg.norm(u_i)*np.linalg.norm(u_j)).flatten()
                ed_ij = np.linalg.norm(u_i - u_j)
                round_update_pw_cs[i][j] = cs_ij
                round_update_pw_eu[i][j] = ed_ij

                self.pairwise_choosing_frequencies[g_i][g_j] = self.pairwise_choosing_frequencies[g_i][g_j] + 1.0
                bias_p_i = norm_bias_list[i]
                bias_p_j = norm_bias_list[j]
                cs_1 = np.dot(bias_p_i, bias_p_j)/(np.linalg.norm(bias_p_i)*np.linalg.norm(bias_p_j))
                round_bias_pairwise[i][j] = cs_1.flatten()


                
                w_p_i = norm_grad_diff_list[i]
                w_p_j = norm_grad_diff_list[j]
                cs_2 = np.dot(w_p_i, w_p_j)/(np.linalg.norm(w_p_i)*np.linalg.norm(w_p_j))
                round_weight_pairwise[i][j] = cs_2.flatten()
                
                cli_i_arr = np.hstack((bias_p_i, w_p_i))
                cli_j_arr = np.hstack((bias_p_j, w_p_j))

        scaler = MinMaxScaler()
        round_update_pw_cs = scaler.fit_transform(round_update_pw_cs)
        round_update_pw_eu = scaler.fit_transform(round_update_pw_eu)       
                
                # cs_arr = np.hstack(cs_1, cs_2)
                
            #     if j > i:
            #         distance.append(float(np.linalg.norm(cli_i_arr-cli_j_arr)**2)) # let's change this to pytorch version
            # neighbor_distances.append(distance)
                
        logger.info("Starting performing KrMLRFL...")
       

        # # compute scores by KRUM*
        
        for i, g_i in enumerate(vectorize_nets):
            distance = []
            for j in range(i+1, len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]
                    distance.append(float(np.linalg.norm(g_i-g_j)**2)) # let's change this to pytorch version
            neighbor_distances.append(distance)

        # compute scores
        nb_in_score = self.num_workers-self.s-2
        scores = []
        for i, g_i in enumerate(vectorize_nets):
            dists = []
            for j, g_j in enumerate(vectorize_nets):
                if j == i:
                    continue
                if j < i:
                    dists.append(neighbor_distances[j][i - j - 1])
                else:
                    dists.append(neighbor_distances[i][j - i - 1])

            # alternative to topk in pytorch and tensorflow
            topk_ind = np.argpartition(dists, nb_in_score)[:nb_in_score]
            scores.append(sum(np.take(dists, topk_ind)))
        
        print(f"scores of krum method: {scores}")
        i_star = scores.index(min(scores))
        
        # use krum as the baseline to improve, mark the one chosen by krum as trusted
        if self.num_valid == 1:
            i_star = scores.index(min(scores))
            logger.info("@@@@ The chosen trusted worker is user: {}, which is global user: {}".format(scores.index(min(scores)), g_user_indices[scores.index(min(scores))]))
            trusted_models.append(i_star)
        else:
            topk_ind = np.argpartition(scores, nb_in_score+2)[:self.num_valid]
            
            # we reconstruct the weighted averaging here:
            selected_num_dps = np.array(num_dps)[topk_ind]
            logger.info("Num selected data points: {}".format(selected_num_dps))
            logger.info("The chosen ones are users: {}, which are global users: {}".format(topk_ind, [g_user_indices[ti] for ti in topk_ind]))

            for ind in topk_ind:
                trusted_models.append(ind)
        
        trusted_index = i_star # used for get labels of attackers


        round_bias_pairwise = scaler.fit_transform(round_bias_pairwise)
        round_weight_pairwise = scaler.fit_transform(round_weight_pairwise)

        for i, g_i in enumerate(g_user_indices):
            for j, g_j in enumerate(g_user_indices):
                freq_appear = self.pairwise_choosing_frequencies[g_i][g_j]
                self.pairwise_w[g_i][g_j] = (freq_appear - 1)/freq_appear*self.pairwise_w[g_i][g_j] +  1/freq_appear*round_weight_pairwise[i][j]
                self.pairwise_b[g_i][g_j] = (freq_appear - 1)/freq_appear*self.pairwise_b[g_i][g_j] +  1/freq_appear*round_bias_pairwise[i][j]
                
        
        # From now on, trusted_models contain the index base models treated as valid users.
        raw_t_score = self.get_trustworthy_scores(glob_update, weight_update)
        t_score = []
        for idx, cli in enumerate(g_user_indices):
            # increase the frequency of the selected choosen clients
            self.choosing_frequencies[cli] = self.choosing_frequencies.get(cli, 0) + 1
            # update the accumulator
            self.accumulate_t_scores[cli] = ((self.choosing_frequencies[cli] - 1) / self.choosing_frequencies[cli]) * self.accumulate_t_scores.get(cli, 0) + (1 / self.choosing_frequencies[cli]) *  raw_t_score[idx]
            t_score.append(self.accumulate_t_scores[cli])
        
        
        t_score = np.array(t_score)
        threshold = min(0.5, np.median(t_score))
        
        
        participated_attackers = []
        for in_, id_ in enumerate(g_user_indices):
            if id_ in selected_attackers:
                participated_attackers.append(in_)
        print("At round: ", round)
        if not len(selected_attackers):
            print("THIS ROUND HAS NO ATTACKER!!!")
        print("real attackers indx: ", participated_attackers)
        print(f'[T_SCORE] median score: {np.median(t_score)}')
        print("[T_SCORE] trustworthy score is: ", t_score)
        
        attacker_local_idxs = [ind_ for ind_ in range(len(g_user_indices)) if t_score[ind_] > threshold]
        print("[T_SCORE] attacker_local_idxs is: ", attacker_local_idxs)
        global_pred_attackers_indx = [g_user_indices[ind_] for ind_ in attacker_local_idxs]
        print("[T_SCORE] global_pred_attackers_indx: ", global_pred_attackers_indx)
        missed_attacker_idxs_by_thre = [at_id for at_id in participated_attackers if at_id not in attacker_local_idxs]
        attacker_local_idxs_2 = []
        saved_pairwise_sim = []

        np_scores = np.asarray(scores)
        final_attacker_idxs = attacker_local_idxs # for the first filter
        # NOW CHECK FOR ROUND 50
        if round >= 1: 
            # TODO: find dynamic threshold
            
            cummulative_w = self.pairwise_w[np.ix_(g_user_indices, g_user_indices)]
            cummulative_b = self.pairwise_b[np.ix_(g_user_indices, g_user_indices)]
            
            
            saved_pairwise_sim = np.hstack((cummulative_w, cummulative_b))
            kmeans = KMeans(n_clusters = 2)
                    
            hb_clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                    gen_min_span_tree=False,
                                    metric='euclidean', min_cluster_size=int((total_client/2)+1), min_samples=1, p=None)
            hb_clusterer.fit(round_update_pw_cs)
            # print("hb_clusterer.labels_ is: ", hb_clusterer.labels_)

            pred_labels = kmeans.fit_predict(saved_pairwise_sim)
            centroids = kmeans.cluster_centers_
            np_centroids = np.asarray(centroids)
            test_labels = kmeans.fit_predict(round_update_pw_cs)
            print(f"test_labels: {test_labels}")
            
            
            cls_0_idxs = np.argwhere(np.asarray(pred_labels) == 0).flatten()
            cls_1_idxs = np.argwhere(np.asarray(pred_labels) == 1).flatten()
            dist_0 = np.sqrt(np.sum(np.square(saved_pairwise_sim[cls_0_idxs]-np_centroids[0])))/len(cls_0_idxs)
            dist_1 = np.sqrt(np.sum(np.square(saved_pairwise_sim[cls_1_idxs]-np_centroids[1])))/len(cls_1_idxs)
            print(f"dist_0 is {dist_0}, dist_1 is {dist_1}")
            
            
            trusted_label = pred_labels[trusted_index]
            label_attack = 0 if trusted_label == 1 else 1
        
            pred_attackers_indx_2 = np.argwhere(np.asarray(pred_labels) == label_attack).flatten()
        
            
            print("[PAIRWISE] pred_attackers_indx: ", pred_attackers_indx_2)
            pred_normal_client = [_id for _id in range(total_client) if _id not in pred_attackers_indx_2]

            #FOR LOGGING ONLY
            adv_krum_s = np_scores[pred_attackers_indx_2]
            ben_krum_s = np_scores[pred_normal_client]
            adv_krum_s_avg = np.average(adv_krum_s).flatten()[0]
            ben_krum_s_avg = np.average(ben_krum_s).flatten()[0]
            print(f"trusted client score is: {np_scores[i_star]}")
            print(f"attackers' scores by krum are: {np_scores[pred_attackers_indx_2]}")
            print(f"adv_krum_s_avg: {adv_krum_s_avg}")
            print(f"pred_normal_client's score by krum are: {np_scores[pred_normal_client]}")
            print(f"ben_krum_s_avg: {ben_krum_s_avg}")
            has_attacker = True if len(selected_attackers) else False
            cluster_log_row = (round, has_attacker, np_scores[i_star], adv_krum_s_avg, ben_krum_s_avg, adv_krum_s, ben_krum_s)
            with open (f"{self.instance}_cluster_log.csv", "a+") as log_csv:
                writer = csv.writer(log_csv)
                writer.writerow(cluster_log_row)
            missed_attacker_idxs_by_kmeans = [at_id for at_id in participated_attackers if at_id not in pred_attackers_indx_2]

            attacker_local_idxs_2 = pred_attackers_indx_2
            temp_diff_score = (adv_krum_s_avg-np_scores[i_star])/(ben_krum_s_avg-np_scores[i_star])
            # if temp_diff_score <= 2.0:
            #     # pseudo_final_attacker_idxs = []
            #     attacker_local_idxs_2 = []
            pseudo_final_attacker_idxs = np.union1d(attacker_local_idxs_2, attacker_local_idxs)
            print(f"temp_diff_score is: {temp_diff_score}")

            print(f"attacker_local_idxs: {attacker_local_idxs}")
            # START USING FUZZY HERE
            if temp_diff_score <= 1.0:
                attacker_local_idxs_2 = []
                pseudo_final_attacker_idxs = attacker_local_idxs
            elif temp_diff_score >= 5.0 and temp_diff_score <= 12.0:
                pseudo_final_attacker_idxs = np.intersect1d(attacker_local_idxs_2, attacker_local_idxs)

            if round >= 50:
                final_attacker_idxs = pseudo_final_attacker_idxs
            print("assumed final_attacker_idxs: ", pseudo_final_attacker_idxs)


        freq_participated_attackers = [self.choosing_frequencies[g_idx] for g_idx in g_user_indices]
        true_positive_pred_layer1 = []
        true_positive_pred_layer2 = []
        false_positive_pred_layer1 = []
        false_positive_pred_layer2 = []
        for id_ in participated_attackers:
            true_positive_pred_layer1.append(1.0 if id_ in attacker_local_idxs else 0.0)
            true_positive_pred_layer2.append(1.0 if id_ in attacker_local_idxs_2 else 0.0)
        for id_ in attacker_local_idxs:
            if id_ not in participated_attackers:
                false_positive_pred_layer1.append(1.0)
        for id_ in attacker_local_idxs_2:
            if id_ not in participated_attackers:
                false_positive_pred_layer2.append(1.0)
            # if id_ not in 
        

            
        # true_positive_pred_layer1_val = sum(true_positive_pred_layer1)/len(true_positive_pred_layer1) if len(true_positive_pred_layer1) else 0.0
        # true_positive_pred_layer2_val = sum(true_positive_pred_layer2)/len(true_positive_pred_layer2) if len(true_positive_pred_layer2) else 0.0
        true_positive_pred_layer1_val = sum(true_positive_pred_layer1)/len(participated_attackers) if len(true_positive_pred_layer1) else 0.0
        if len(participated_attackers) == 0 and len(attacker_local_idxs) == 0:
            true_positive_pred_layer1_val = 1.0
        true_positive_pred_layer2_val = sum(true_positive_pred_layer2)/len(participated_attackers) if len(true_positive_pred_layer2) else 0.0
        
        if len(participated_attackers) == 0 and len(attacker_local_idxs_2) == 0:
            true_positive_pred_layer2_val = 1.0
        fn_layer1_val = 1.0 - true_positive_pred_layer1_val
        fn_layer2_val = 1.0 - true_positive_pred_layer2_val
        fp_layer1_val = sum(false_positive_pred_layer1)/(total_client-len(participated_attackers)) if len(false_positive_pred_layer1) else 0.0
        fp_layer2_val = sum(false_positive_pred_layer2)/(total_client-len(participated_attackers)) if len(false_positive_pred_layer2) else 0.0

        logging_per_round = (
            round,
            participated_attackers,
            attacker_local_idxs,
            attacker_local_idxs_2,
            true_positive_pred_layer1_val,
            true_positive_pred_layer2_val,
            fn_layer1_val,
            fn_layer2_val,
            fp_layer1_val,
            fp_layer2_val,
            missed_attacker_idxs_by_thre,
            missed_attacker_idxs_by_kmeans,
            freq_participated_attackers,
            t_score,
            num_dps,
            saved_pairwise_sim
        )
        
        with open(f'{self.instance}_combined_file_klfrl.csv', "a+") as w_f:
            writer = csv.writer(w_f)
            writer.writerow(logging_per_round)
        neo_net_list = []
        neo_net_freq = []
        selected_net_indx = []
        for idx, net in enumerate(client_models):
            if idx not in final_attacker_idxs:
                neo_net_list.append(net)
                neo_net_freq.append(1.0)
                selected_net_indx.append(idx)
        if len(neo_net_list) == 0:
            neo_net_list.append(client_models[i_star])
            selected_net_indx.append(i_star)
            pred_g_attacker = [g_user_indices[i] for i in final_attacker_idxs]
            # return [client_models[i_star]], [1.0], pred_g_attacker
            return [net_avg], [1.0], pred_g_attacker
            
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in neo_net_list]
        selected_num_dps = np.array(num_dps)[selected_net_indx]
        reconstructed_freq = [snd/sum(selected_num_dps) for snd in selected_num_dps]

        logger.info("Num data points: {}".format(num_dps))
        logger.info("Num selected data points: {}".format(selected_num_dps))
        logger.info("The chosen ones are users: {}, which are global users: {}".format(selected_net_indx, [g_user_indices[ti] for ti in selected_net_indx]))
        
        aggregated_grad = np.average(vectorize_nets, weights=reconstructed_freq, axis=0).astype(np.float32)

        aggregated_model = client_models[0] # slicing which doesn't really matter
        load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to(device))
        pred_g_attacker = [g_user_indices[i] for i in final_attacker_idxs]
        # print(self.pairwise_cs)
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        return neo_net_list, neo_net_freq, pred_g_attacker

    def get_trustworthy_scores(self, global_update, weight_update):
        # print("base_model_update: ", base_model_update)
        cs_dist = get_cs_on_base_net(weight_update, global_update)
        score = np.array(cs_dist)
        # print("score_avg:= ", score_avg)
        norm_score = min_max_scale(score)
        
        return norm_score
        # for cli_ind, weight_update in enumerate(weight_update):

    def get_predicted_attackers(self, weight_list, avg_weight, weight_update, total_client):
        # from sklearn.cluster import KMeans
        eucl_dis, cs_dis = get_distance_on_avg_net(weight_list, avg_weight, weight_update, total_client)
        norm_cs_data = min_max_scale(cs_dis)
        norm_eu_data = 1.0 - min_max_scale(eucl_dis)
        # norm_eu_data = min_max_scale(eucl_dis)
        stack_dis = np.hstack((norm_cs_data,norm_eu_data))
        print("stack dis is: ", stack_dis)
        temp_score = [0.5*norm_cs_data[i] + 0.5*norm_eu_data[i] for i in range(total_client)]
        threshold = sum(temp_score)/total_client
        abnormal_score = [1.0 if temp_score[i] > threshold else 0.0 for i in range(total_client)]
        print("abnormal_score: ", abnormal_score)
        
        hb_clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                gen_min_span_tree=False, leaf_size=40,
                                metric='euclidean', min_cluster_size=2, min_samples=None, p=None)
        hb_clusterer.fit(stack_dis)
        print("hb_clusterer.labels_ is: ", hb_clusterer.labels_)
        return abnormal_score

class RLR(Defense):
    def __init__(self, n_params, device, args, agent_data_sizes=[], writer=None, robustLR_threshold = 0, aggr="avg", poisoned_val_loader=None):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.writer = writer
        # print(f"args: {args}")
        # self.server_lr = args.server_lr
        self.n_params = n_params
        self.poisoned_val_loader = None
        self.cum_net_mov = 0
        self.device = device
        self.robustLR_threshold = robustLR_threshold
        
         
    def exec(self, global_model, client_models, num_dps, agent_updates_dict=None, cur_round=0):
        # adjust LR if robust LR is selected
        print(f"self.args: {self.args}")
        print(f"self.args['server_lr']: {self.args['server_lr']}")
        lr_vector = torch.Tensor([self.args['server_lr']]*self.n_params).to(self.device)
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        vectorize_avg_net = vectorize_net(global_model).detach().cpu().numpy()
        local_updates = vectorize_nets - vectorize_avg_net
        aggr_freq = [num_dp/sum(num_dps) for num_dp in num_dps]
        
        if self.robustLR_threshold > 0:
            lr_vector = self.compute_robustLR(local_updates)
        
        
        aggregated_updates = 0
        if self.args['aggr']=='avg':          
            aggregated_updates = self.agg_avg(local_updates, num_dps)
        elif self.args['aggr']=='comed':
            #TODO update for the 2 remaining func
            aggregated_updates = self.agg_comed(local_updates)
        elif self.args['aggr'] == 'sign':
            aggregated_updates = self.agg_sign(local_updates)
            
        if self.args['noise'] > 0:
            aggregated_updates.add_(torch.normal(mean=0, std=self.args['noise']*self.args['clip'], size=(self.n_params,)).to(self.device))

        cur_global_params = vectorize_avg_net
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).astype(np.float32)
        
        aggregated_model = client_models[0] # slicing which doesn't really matter
        load_model_weight(aggregated_model, torch.from_numpy(new_global_params).to(self.device))
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        return neo_net_list, neo_net_freq
        
        
        # some plotting stuff if desired
        # self.plot_sign_agreement(lr_vector, cur_global_params, new_global_params, cur_round)
        # self.plot_norms(agent_updates_dict, cur_round)
     
    
    def compute_robustLR(self, agent_updates):
        agent_updates_sign = [np.sign(update) for update in agent_updates]  
        sm_of_signs = np.abs(sum(agent_updates_sign))
        print(f"sm_of_signs is: {sm_of_signs}")
        
        sm_of_signs[sm_of_signs < self.robustLR_threshold] = -self.args['server_lr']
        sm_of_signs[sm_of_signs >= self.robustLR_threshold] = self.args['server_lr']                                            
        return sm_of_signs
        
            
    def agg_avg(self, agent_updates_dict, num_dps):
        """ classic fed avg """
        sm_updates, total_data = 0, 0
        for _id, update in enumerate(agent_updates_dict):
            n_agent_data = num_dps[_id]
            sm_updates +=  n_agent_data * update
            total_data += n_agent_data  
        return  sm_updates / total_data
    
    def agg_comed(self, agent_updates_dict):
        agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates_dict.values()]
        concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
        return torch.median(concat_col_vectors, dim=1).values
    
    def agg_sign(self, agent_updates_dict):
        """ aggregated majority sign update """
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_signs = torch.sign(sum(agent_updates_sign))
        return torch.sign(sm_signs)

    def clip_updates(self, agent_updates_dict):
        for update in agent_updates_dict.values():
            l2_update = torch.norm(update, p=2) 
            update.div_(max(1, l2_update/self.args['clip']))
        return
                  
    def plot_norms(self, agent_updates_dict, cur_round, norm=2):
        """ Plotting average norm information for honest/corrupt updates """
        honest_updates, corrupt_updates = [], []
        for key in agent_updates_dict.keys():
            if key < self.args.num_corrupt:
                corrupt_updates.append(agent_updates_dict[key])
            else:
                honest_updates.append(agent_updates_dict[key])
                              
        l2_honest_updates = [torch.norm(update, p=norm) for update in honest_updates]
        avg_l2_honest_updates = sum(l2_honest_updates) / len(l2_honest_updates)
        self.writer.add_scalar(f'Norms/Avg_Honest_L{norm}', avg_l2_honest_updates, cur_round)
        
        if len(corrupt_updates) > 0:
            l2_corrupt_updates = [torch.norm(update, p=norm) for update in corrupt_updates]
            avg_l2_corrupt_updates = sum(l2_corrupt_updates) / len(l2_corrupt_updates)
            self.writer.add_scalar(f'Norms/Avg_Corrupt_L{norm}', avg_l2_corrupt_updates, cur_round) 
        return
        
    def comp_diag_fisher(self, model_params, data_loader, adv=True):

        model = models.get_model(self.args.data)
        vector_to_parameters(model_params, model.parameters())
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        precision_matrices = {}
        for n, p in deepcopy(params).items():
            p.data.zero_()
            precision_matrices[n] = p.data
            
        model.eval()
        for _, (inputs, labels) in enumerate(data_loader):
            model.zero_grad()
            inputs, labels = inputs.to(device=self.args.device, non_blocking=True),\
                                    labels.to(device=self.args.device, non_blocking=True).view(-1, 1)
            if not adv:
                labels.fill_(self.args.base_class)
                
            outputs = model(inputs)
            log_all_probs = F.log_softmax(outputs, dim=1)
            target_log_probs = outputs.gather(1, labels)
            batch_target_log_probs = target_log_probs.sum()
            batch_target_log_probs.backward()
            
            for n, p in model.named_parameters():
                precision_matrices[n].data += (p.grad.data ** 2) / len(data_loader.dataset)
                
        return parameters_to_vector(precision_matrices.values()).detach()

        
    def plot_sign_agreement(self, robustLR, cur_global_params, new_global_params, cur_round):
        """ Getting sign agreement of updates between honest and corrupt agents """
        # total update for this round
        update = new_global_params - cur_global_params
        
        # compute FIM to quantify these parameters: (i) parameters which induces adversarial mapping on trojaned, (ii) parameters which induces correct mapping on trojaned
        fisher_adv = self.comp_diag_fisher(cur_global_params, self.poisoned_val_loader)
        fisher_hon = self.comp_diag_fisher(cur_global_params, self.poisoned_val_loader, adv=False)
        _, adv_idxs = fisher_adv.sort()
        _, hon_idxs = fisher_hon.sort()
        
        # get most important n_idxs params
        n_idxs = self.args.top_frac #math.floor(self.n_params*self.args.top_frac)
        adv_top_idxs = adv_idxs[-n_idxs:].cpu().detach().numpy()
        hon_top_idxs = hon_idxs[-n_idxs:].cpu().detach().numpy()
        
        # minimized and maximized indexes
        min_idxs = (robustLR == -self.args.server_lr).nonzero().cpu().detach().numpy()
        max_idxs = (robustLR == self.args.server_lr).nonzero().cpu().detach().numpy()
        
        # get minimized and maximized idxs for adversary and honest
        max_adv_idxs = np.intersect1d(adv_top_idxs, max_idxs)
        max_hon_idxs = np.intersect1d(hon_top_idxs, max_idxs)
        min_adv_idxs = np.intersect1d(adv_top_idxs, min_idxs)
        min_hon_idxs = np.intersect1d(hon_top_idxs, min_idxs)
       
        # get differences
        max_adv_only_idxs = np.setdiff1d(max_adv_idxs, max_hon_idxs)
        max_hon_only_idxs = np.setdiff1d(max_hon_idxs, max_adv_idxs)
        min_adv_only_idxs = np.setdiff1d(min_adv_idxs, min_hon_idxs)
        min_hon_only_idxs = np.setdiff1d(min_hon_idxs, min_adv_idxs)
        
        # get actual update values and compute L2 norm
        max_adv_only_upd = update[max_adv_only_idxs] # S1
        max_hon_only_upd = update[max_hon_only_idxs] # S2
        
        min_adv_only_upd = update[min_adv_only_idxs] # S3
        min_hon_only_upd = update[min_hon_only_idxs] # S4


        #log l2 of updates
        max_adv_only_upd_l2 = torch.norm(max_adv_only_upd).item()
        max_hon_only_upd_l2 = torch.norm(max_hon_only_upd).item()
        min_adv_only_upd_l2 = torch.norm(min_adv_only_upd).item()
        min_hon_only_upd_l2 = torch.norm(min_hon_only_upd).item()
       
        self.writer.add_scalar(f'Sign/Hon_Maxim_L2', max_hon_only_upd_l2, cur_round)
        self.writer.add_scalar(f'Sign/Adv_Maxim_L2', max_adv_only_upd_l2, cur_round)
        self.writer.add_scalar(f'Sign/Adv_Minim_L2', min_adv_only_upd_l2, cur_round)
        self.writer.add_scalar(f'Sign/Hon_Minim_L2', min_hon_only_upd_l2, cur_round)
        
        
        net_adv =  max_adv_only_upd_l2 - min_adv_only_upd_l2
        net_hon =  max_hon_only_upd_l2 - min_hon_only_upd_l2
        self.writer.add_scalar(f'Sign/Adv_Net_L2', net_adv, cur_round)
        self.writer.add_scalar(f'Sign/Hon_Net_L2', net_hon, cur_round)
        
        self.cum_net_mov += (net_hon - net_adv)
        self.writer.add_scalar(f'Sign/Model_Net_L2_Cumulative', self.cum_net_mov, cur_round)
        return

class FLAME(Defense):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)    
    def exec(self, client_models, net_avg, device, *args, **kwargs):
        total_client = len(client_models)
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        vectorize_avg_net = vectorize_net(net_avg).detach().cpu().numpy()
        local_updates = vectorize_nets - vectorize_avg_net
        
        #FILTERING C1:
        pairwise_cs = np.zeros((total_client, total_client))
        for i, w_p_i in enumerate(vectorize_nets):
            for j, w_p_j in enumerate(vectorize_nets):
                pairwise_cs[i][j] = 1.0 - np.dot(w_p_i, w_p_j)/(np.linalg.norm(w_p_i)*np.linalg.norm(w_p_j))
        pairwise_cs = normalize(pairwise_cs)
        print(f"pairwise_cs: {pairwise_cs}")
        
        min_cluster_sz = int(total_client/2+1)
        hb_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_sz, min_samples=1)
        # hb_clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
        #                         gen_min_span_tree=False, leaf_size=40,
        #                         metric='euclidean', min_cluster_size=2, min_samples=None, p=None)
        hb_clusterer.fit(pairwise_cs)
        layer_1_pred_labels = hb_clusterer.labels_
        layer_1_pred_labels = np.asarray(layer_1_pred_labels)
        print(f"layer_1_pred_labels: {layer_1_pred_labels}")
        
        # unique_dict = np.unique(layer_1_pred_labels, return_counts=True)
        # max_lab_key = max(unique_dict, key=unique_dict.get)
        values, counts = np.unique(layer_1_pred_labels, return_counts=True)

        normal_client_label = layer_1_pred_labels[np.argmax(counts)]
        print(f"max_lab_key is: {normal_client_label}")
        # normal_client_label = np.bincount(layer_1_pred_labels).argmax()
        normal_client_idxs = np.argwhere(layer_1_pred_labels == normal_client_label).flatten()
        print(f"normal_client_idxs: {normal_client_idxs}")
        eucl_dist = []
        for i, g_p_i in enumerate(vectorize_nets):
            ds = g_p_i-vectorize_avg_net
            el_dis = np.sqrt(np.dot(ds, ds.T)).flatten()
            eucl_dist.append(el_dis)
        s_t = np.median(eucl_dist)
        
        normal_w = []
        for _id in normal_client_idxs:
            dym_thres = s_t/eucl_dist[_id]
            w_c = vectorize_avg_net + local_updates[_id]*min(1.0, dym_thres)
            normal_w.append(w_c)
        print(len(normal_w))
        
        normal_w = np.asarray(normal_w)
        print(f"normal_w.shape is: {normal_w.shape}")
        new_global_w = np.average(normal_w, axis=0)
        lambda_ = 0.001
        sigma_n = lambda_*s_t
        # new_global_w =  new_global_w + np.random.normal(0, sigma_n, new_global_w.shape[0])
        aggregated_model = client_models[0]
        print(f"new_global_w.shape is: {new_global_w.shape}")
        g_noise = np.random.normal(0, sigma_n, new_global_w.shape[0])
        new_global_w =  (new_global_w + g_noise)
        load_model_weight(aggregated_model, torch.from_numpy(new_global_w.astype(np.float32)).to(device))
        return [aggregated_model],  [1.0]

class FoolsGold(Defense):
    def __init__(self, num_clients, num_features, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_clients = num_clients
        self.n_features = num_features
        self.n_classes = num_classes

    def get_cos_similarity(self, full_deltas):
        '''
        Returns the pairwise cosine similarity of client gradients
        '''
        if True in np.isnan(full_deltas):
            pdb.set_trace()
        return smp.cosine_similarity(full_deltas)

    def importanceFeatureMapGlobal(self, model):
        # aggregate = np.abs(np.sum( np.reshape(model, (10, 784)), axis=0))
        # aggregate = aggregate / np.linalg.norm(aggregate)
        # return np.repeat(aggregate, 10)
        return np.abs(model) / np.sum(np.abs(model))

    def importanceFeatureMapLocal(self, model, topk_prop=0.5):
        # model: np arr
        d = self.n_features # dim of flatten weight
        class_d = int(d / self.n_classes)

        M = model.copy()
        M = np.reshape(M, (self.n_classes, class_d))
        
        # #Take abs?
        # M = np.abs(M)

        for i in range(self.n_classes):
            if (M[i].sum() == 0):
                pdb.set_trace()
            M[i] = np.abs(M[i] - M[i].mean())
            
            M[i] = M[i] / M[i].sum()

            # Top k of 784
            topk = int(class_d * topk_prop)
            sig_features_idx = np.argpartition(M[i], -topk)[0:-topk]
            M[i][sig_features_idx] = 0
        
        return M.flatten()   

    def importanceFeatureHard(self, model, topk_prop=0.5):

        class_d = int(self.n_features / self.n_classes)

        M = np.reshape(model, (self.n_classes, class_d))
        importantFeatures = np.ones((self.n_classes, class_d))
        # Top k of 784
        topk = int(class_d * topk_prop)
        for i in range(self.n_classes):
            sig_features_idx = np.argpartition(M[i], -topk)[0:-topk]     
            importantFeatures[i][sig_features_idx] = 0
        return importantFeatures.flatten()  


    def get_krum_scores(self, X, groupsize):

        krum_scores = np.zeros(len(X))

        # Calculate distances
        distances = np.sum(X**2, axis=1)[:, None] + np.sum(
            X**2, axis=1)[None] - 2 * np.dot(X, X.T)

        for i in range(len(X)):
            krum_scores[i] = np.sum(np.sort(distances[i])[1:(groupsize - 1)])

        return krum_scores

    def foolsgold(self, this_delta, summed_deltas, sig_features_idx, iter, model, topk_prop=0, importance=False, importanceHard=False, clip=0):
        epsilon = 1e-5
        # Take all the features of sig_features_idx for each clients
        sd = summed_deltas.copy()
        # print(f"summed_deltas: {summed_deltas}")
        # print(f"this delta: {this_delta}")
        sig_filtered_deltas = np.take(sd, sig_features_idx, axis=1)
        # print(f"sig_filtered_deltas: {sig_filtered_deltas}")

        if importance or importanceHard:
            if importance:
                # smooth version of importance features
                importantFeatures = self.importanceFeatureMapLocal(model, topk_prop)
            if importanceHard:
                # hard version of important features
                importantFeatures = self.importanceFeatureHard(model, topk_prop)
            for i in range(self.n_clients):
                sig_filtered_deltas[i] = np.multiply(sig_filtered_deltas[i], importantFeatures)

        cs = smp.cosine_similarity(sig_filtered_deltas) - np.eye(self.n_clients)
        # print(f"cs1 is: {cs}")

        # Pardoning: reweight by the max value seen
        maxcs = np.max(cs, axis=1) + epsilon
        # print(f"maxcs: {maxcs}")
        for i in range(self.n_clients):
            for j in range(self.n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        # print(f"cs is: {cs}")
        wv = 1 - (np.max(cs, axis=1))
        # print(f"wv: {wv}")
        wv[wv > 1] = 1
        wv[wv < 0] = 0

        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        # print(f"wv2: {wv}")

        wv[(wv == 1)] = .99
        # print(f"wv3: {wv}")

        
        # Logit function
        wv = (np.log((wv / (1 - wv)) + epsilon) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0
        
        # if iter % 10 == 0 and iter != 0:
        #     print maxcs
        #     print wv

        if clip != 0:

            # Augment onto krum
            scores = self.get_krum_scores(this_delta, self.n_clients - clip)
            bad_idx = np.argpartition(scores, self.n_clients - clip)[(self.n_clients - clip):self.n_clients]

            # Filter out the highest krum scores
            wv[bad_idx] = 0

        # Apply the weight vector on this delta
        # print(f"this_delta.shape is: {this_delta.shape}")
        # delta = np.reshape(this_delta, (self.n_clients, self.n_features))
        # print(f"wv: {wv}")
        # print(f"delta: {this_delta}")
        avg_updates = np.average(this_delta, axis=0, weights=wv)
        # print(f"avg_updates.shape is: {avg_updates.shape}")
        return avg_updates
        # return np.dot(this_delta.T, wv) 

    def exec(self, client_models, delta, summed_deltas, net_avg, r, device, *args, **kwargs):
        '''
        Aggregates history of gradient directions
        '''
        print(f"START Aggregating history of gradient directions")
        total_client = len(client_models)
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        vectorize_avg_net = vectorize_net(net_avg).detach().cpu().numpy()
        local_updates = vectorize_nets - vectorize_avg_net

        print(f"historical_local_updates.shape is: {summed_deltas.shape}")
        flatten_net_avg = vectorize_net(net_avg).detach().cpu().numpy()
        # summed_deltas = historical_local_updates # aggregated historical gradients
        # sig_features_idx = self.im # important features
        # importanceHard = None
        # Take all the features of sig_features_idx for each clients
        # sd = summed_deltas.copy()
        # sig_filtered_deltas = np.take(sd, sig_features_idx, axis=1)

        # Significant features filter, the top k biggest weights
        topk = int(self.n_features / 2)
        sig_features_idx = np.argpartition(flatten_net_avg, -topk)[-topk:]
        sig_features_idx = np.arange(self.n_features)
        avg_delta = self.foolsgold(delta, summed_deltas, sig_features_idx, r, vectorize_avg_net, clip = 0)
        avg_vector_net = vectorize_avg_net + avg_delta
        final_net = client_models[0]
        load_model_weight(final_net, torch.from_numpy(avg_vector_net.astype(np.float32)).to(device))
        return [final_net], [1.0]

class UpperBound(Defense):
    def __init__(self, *args, **kwargs):
        pass
    
    def exec(self, client_models, num_dps, attacker_idxs, g_user_indices, device=torch.device("cuda"), *args, **kwargs):
        #GET KRUM VECTOR
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        
        
        
        # vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        selected_idxs = [idx for idx in range(len(client_models)) if idx not in attacker_idxs]
        print("selected_idxs: ", selected_idxs)
        selected_num_dps = np.array(num_dps)[selected_idxs]
        reconstructed_freq = [snd/sum(selected_num_dps) for snd in selected_num_dps]
        logger.info("Num data points: {}".format(num_dps))
        logger.info("Num selected data points: {}".format(selected_num_dps))
        vectorize_nets = np.asarray(vectorize_nets)[selected_idxs]
        
        aggregated_grad = np.average(vectorize_nets, weights=reconstructed_freq, axis=0).astype(np.float32)
        
        aggregated_model = client_models[0] # slicing which doesn't really matter
        load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to(device)) 
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        
        
        return neo_net_list, neo_net_freq         

class UpperBoundByClass(Defense):
    def __init__(self, *args, **kwargs):
        pass
    
    def exec(self, client_models, num_dps, attacker_idxs, g_user_indices, device=torch.device("cuda"), *args, **kwargs):
        #GET KRUM VECTOR
        print(f"attacker_idxs is: {attacker_idxs}")
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        selected_idxs = [idx for idx in range(len(client_models)) if idx not in attacker_idxs]
        # print("selected_idxs: ", selected_idxs)
        selected_num_dps = np.array(num_dps)
        reconstructed_freq = [snd/sum(selected_num_dps) for snd in selected_num_dps]
        logger.info("Num data points: {}".format(num_dps))
        logger.info("Num selected data points: {}".format(selected_num_dps))
        vectorize_nets = np.asarray(vectorize_nets)
        new_freq = reconstructed_freq.copy()
        for idx, freq in enumerate(reconstructed_freq):
            if idx in attacker_idxs:
                new_freq[idx] = freq/10.0
        new_freq = [snd/sum(new_freq) for snd in new_freq]
        print(f"new freq is: {new_freq}")
                
        logger.info("Num data points: {}".format(num_dps))
        logger.info("Num selected data points: {}".format(selected_num_dps))
        vectorize_nets = np.asarray(vectorize_nets)
        
        aggregated_grad = np.average(vectorize_nets, weights=new_freq, axis=0).astype(np.float32)

        # aggregated_grad = np.average(vectorize_nets, weights=reconstructed_freq, axis=0).astype(np.float32)

        # aggregated_grad = avg_by_class(vectorize_nets, reconstructed_freq, attacker_idxs)
        
        aggregated_model = client_models[0] # slicing which doesn't really matter
        load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to(device)) 
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        
        
        return neo_net_list, neo_net_freq  
if __name__ == "__main__":
    # some tests here
    import copy
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # check 1, this should recover the global model
    sim_global_model = Net().to(device)
    sim_local_model1 = copy.deepcopy(sim_global_model)
    #sim_local_model = Net().to(device)
    defender = WeightDiffClippingDefense(norm_bound=5)
    defender.exec(client_model=sim_local_model1, global_model=sim_global_model)

    vec_global_sim_net = vectorize_net(sim_global_model)
    vec_local_sim_net1 = vectorize_net(sim_local_model1)

    # Norm Weight Diff: 0.0, Norm Clipped Weight Diff 0.0
    # Norm Global model: 8.843663215637207, Norm Clipped local model1: 8.843663215637207    
    print("Norm Global model: {}, Norm Clipped local model1: {}".format(torch.norm(vec_global_sim_net).item(), 
        torch.norm(vec_local_sim_net1).item()))

    # check 2, adding some large perturbation
    sim_local_model2 = copy.deepcopy(sim_global_model)
    scaling_facor = 2
    for p_index, p in enumerate(sim_local_model2.parameters()):
        p.data = p.data + torch.randn(p.size()) * scaling_facor
    defender.exec(client_model=sim_local_model2, global_model=sim_global_model)
    vec_local_sim_net2 = vectorize_net(sim_local_model2)

    # Norm Weight Diff: 2191.04345703125, Norm Clipped Weight Diff 4.999983787536621
    # Norm Global model: 8.843663215637207, Norm Clipped local model1: 10.155366897583008    
    print("Norm Global model: {}, Norm Clipped local model1: {}".format(torch.norm(vec_global_sim_net).item(), 
        torch.norm(vec_local_sim_net2).item()))
