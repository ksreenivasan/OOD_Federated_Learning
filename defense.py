import torch
from utils import *
from scipy.special import logit, expit

from geometric_median import geometric_median
import hdbscan

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
    def __init__(self, num_workers, num_adv, num_valid = 1, *args, **kwargs):
        # assert (mode in ("krum", "multi-krum"))
        self.num_valid = num_valid
        self.num_workers = num_workers
        self.s = num_adv

    def exec(self, client_models, num_dps,net_freq, net_avg, g_user_indices, pseudo_avg_net, round, selected_attackers, device, *args, **kwargs):
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        trusted_models = []
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
        
        if self.num_valid == 1:
            i_star = scores.index(min(scores))
            logger.info("@@@@ The chosen trusted worker is user: {}, which is global user: {}".format(scores.index(min(scores)), g_user_indices[scores.index(min(scores))]))
            # aggregated_model = client_models[0] # slicing which doesn't really matter
            # load_model_weight(aggregated_model, torch.from_numpy(vectorize_nets[i_star]).to(device))
            # neo_net_list = [aggregated_model]
            # logger.info("Norm of the chosen trusted worker: {}".format(torch.norm(torch.nn.utils.parameters_to_vector(aggregated_model.parameters())).item()))
            # neo_net_freq = [1.0]
            # return neo_net_list, neo_net_freq
            trusted_models.append(i_star)
        else:
            # topk_ind = np.argpartition(scores, nb_in_score+2)[:nb_in_score+2]
            topk_ind = np.argpartition(scores, nb_in_score+2)[:self.num_valid]
            
            # we reconstruct the weighted averaging here:
            selected_num_dps = np.array(num_dps)[topk_ind]
            # reconstructed_freq = [snd/sum(selected_num_dps) for snd in selected_num_dps]

            # logger.info("Num data points: {}".format(num_dps))
            logger.info("Num selected data points: {}".format(selected_num_dps))
            logger.info("The chosen ones are users: {}, which are global users: {}".format(topk_ind, [g_user_indices[ti] for ti in topk_ind]))
            #aggregated_grad = np.mean(np.array(vectorize_nets)[topk_ind, :], axis=0)
            # aggregated_grad = np.average(np.array(vectorize_nets)[topk_ind, :], weights=reconstructed_freq, axis=0).astype(np.float32)

            # aggregated_model = client_models[0] # slicing which doesn't really matter
            # load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to(device))
            # neo_net_list = [aggregated_model]
            #logger.info("Norm of Aggregated Model: {}".format(torch.norm(torch.nn.utils.parameters_to_vector(aggregated_model.parameters())).item()))
            # neo_net_freq = [1.0]
            # return neo_net_list, neo_net_freq

            for ind in topk_ind:
                trusted_models.append(ind)
        
        # From now on, trusted_models contain the index base models treated as valid users.
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        # pseudo_avg_w = vectorize_net(pseudo_avg_net).detach().cpu().numpy()
        # baseline_net = self.weighted_average_oracle(vectorize_nets, net_freq)
        # glob_model = client_models[0]
        # load_model_weight(glob_model, torch.from_numpy(baseline_net.astype(np.float32)).to(device)) 
        bias_list, weight_list, avg_bias, avg_weight, weight_update, glob_update, prev_avg_weight = extract_classifier_layer(client_models, pseudo_avg_net, net_avg)
        t_score, raw_t_score = self.get_trustworthy_scores(trusted_models, weight_update)
        c_score = self.get_contribution_scores(trusted_models, weight_update, glob_update)
        
        # print("trustworthy score is: ", t_score)
        # print("contribution score is: ", c_score)
        
        f_score = [t_score[i] + c_score[i] for i in range(len(t_score))]
        # print("f_score: ", f_score)
        # pred_attackers = pred_attackers_indx = np.argwhere(np.asarray(f_score) == 0).flatten()
        # pred_attackers = pred_attackers_indx = np.argwhere(np.asarray(t_score) == 0).flatten()
        # pred_attackers_indx = np.argwhere(np.asarray(t_score) == 0).flatten()
        participated_attackers = []
        for in_, id_ in enumerate(g_user_indices):
            if id_ in selected_attackers:
                participated_attackers.append(in_)
        print("At round: ", round)
        # print("pred_attackers_indx: ", pred_attackers_indx)
        print("real attackers indx: ", participated_attackers)
        # print("global_pred_attackers_indx: ", [g_user_indices[ind_] for ind_ in pred_attackers_indx])
        # global_pred_attackers_indx = [g_user_indices[ind_] for ind_ in pred_attackers_indx]

        print("trustworthy score is: ", t_score)
        print("contribution score is: ", c_score)
        print("f_score: ", f_score)

        
        f_score = [t_score[i] + c_score[i] for i in range(len(t_score))]
        print("f_score is: ", f_score)
        # self.get_predicted_attackers(weight_list, avg_weight, weight_update, 10)
        temp_score = self.get_predicted_attackers(weight_list, prev_avg_weight, weight_update, 10)
        print("raw_t_score: ", raw_t_score)
        print("raw_temp_score: ", temp_score)
        print("raw_sum: ", [raw_t_score[i] + temp_score[i] for i in range(10)])
        raw_sum = [raw_t_score[i] + temp_score[i] for i in range(10)]
        raw_sum = raw_sum/sum(raw_sum)
        raw_sum = np.asarray(raw_sum)
        # mean_sum = sum(raw_sum)/10
        pred_att_idxs = (-raw_sum).argsort()[:2]
        print("global_pred_attackers_indx: ", [g_user_indices[ind_] for ind_ in pred_att_idxs])
        global_pred_attackers_indx = [g_user_indices[ind_] for ind_ in pred_att_idxs]
        pred_attackers_indx = pred_att_idxs.copy()
        log_data_r = f"\nAt round: {round},\n, global_attackers: {selected_attackers},\n glob_predicted: {global_pred_attackers_indx},\n  pred_attackers_indx is: {pred_attackers_indx} \n real attackers indx: {participated_attackers}, \n t_score is: {t_score}, \n c_score is: {c_score}, \n f_score is: {f_score}"
        with open("logging/exper_log.txt", "a+") as lf_r:
            lf_r.write(log_data_r)
        # pred_att_idxs = np.argwhere(np.asarray(raw_sum) == 0).flatten()
        #TRY TO COMBINE Trustworthy score and predicted abnormal score.

        neo_net_list = []
        neo_net_freq = []
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
        
        aggregated_grad = np.average(vectorize_nets, weights=reconstructed_freq, axis=0).astype(np.float32)

        aggregated_model = client_models[0] # slicing which doesn't really matter
        load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to(device))
        pred_g_attacker = [g_user_indices[i] for i in pred_attackers_indx]
        # aggregated_w = self.weighted_average_oracle(vectorize_nets, reconstructed_freq)
        # aggregated_model = client_models[0] # slicing which doesn't really matter
        # load_model_weight(aggregated_model, torch.from_numpy(aggregated_w.astype(np.float32)).to(device))
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        return neo_net_list, neo_net_freq, pred_g_attacker

    def get_trustworthy_scores(self, trusted_models_idxs, weight_update):
        score = np.zeros((len(weight_update), len(trusted_models_idxs)))
        for m_idx, g_model_idx in enumerate(trusted_models_idxs):
            base_model_update = weight_update[g_model_idx]
            # print("base_model_update: ", base_model_update)
            cs_dist = get_cs_on_base_net(weight_update, base_model_update)
            for idx, cs in enumerate(cs_dist):
                score[idx, m_idx] = cs
        print("raw score:= ", score)
        score_avg = np.average(score, 1)
        # print("score_avg:= ", score_avg)
        norm_score = min_max_scale(score_avg)
        final_score = [1.0 if norm_s > 0.25 else 0.0 for norm_s in norm_score]
        return final_score, 1.0 - norm_score
    
    def get_contribution_scores(self, trusted_models_idxs, weight_update, base_w_update):
        # First, calculate the contribution threshold using the trusted client list

        client_ed_list = get_ed_on_base_net(weight_update, base_w_update)
        client_ed_list = np.asarray(client_ed_list)
        b_h = np.average(client_ed_list[trusted_models_idxs])
        contribution_s = (b_h - client_ed_list)/b_h
        # print("contribution_s: ", contribution_s)
        final_score = [1.0 if contribution_s[i] >= 0 else 0.0 for i in range(len(contribution_s))]
        return final_score
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
        return temp_score

        # print("stack_dis.shape: ", stack_dis.shape)
        # kmeans = KMeans(n_clusters = 2)
        # pred_labels = kmeans.fit_predict(stack_dis)
        # print("pred_labels is: ", pred_labels)
        # label_0 = np.count_nonzero(pred_labels == 0)
        # label_1 = total_client - label_0
        # cnt_pred_attackers = label_0 if label_0 <= label_1 else label_1
        # label_att = 0 if label_0 <= label_1 else 1
        # print("label_att: ", label_att)
        
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
