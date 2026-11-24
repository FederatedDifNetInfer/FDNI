import math
import secrets
import time
from pathlib import Path

import numpy as np
from utils import cal_F1



def load_data(graph_path, result_path):
    with open(result_path, 'r') as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        diffusion_result = np.array([[int(state) for state in line] for line in lines])

    nodes_num = diffusion_result.shape[1]

    with open(graph_path, 'r') as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        data = np.array([[int(node) for node in line] for line in lines])
        ground_truth_network = np.zeros((nodes_num, nodes_num))
        edges_num = data.shape[0]
        for i in range(edges_num):
            ground_truth_network[data[i, 0] - 1, data[i, 1] - 1] = 1

    return ground_truth_network, diffusion_result




def partition_observations(diffusion_result, client_num, seed=None):
    total = len(diffusion_result)
    if not 1 <= client_num <= total:
        raise ValueError("Each client needs at least one observation.")
    if client_num == 1:
        return [diffusion_result]
    first = total // (2 * client_num)
    step = total // (client_num * (client_num - 1))
    sizes = first + step * np.arange(client_num)
    extra, remainder = divmod(total - sizes.sum(), client_num)
    sizes += extra
    sizes[client_num - remainder:] += 1
    if sizes[0] == 0:
        raise ValueError("There is an empty client. Reduce the number of clients.")
    shuffled = np.random.default_rng(seed).permutation(diffusion_result)
    return np.split(shuffled, sizes.cumsum()[:-1])


def randomized_response(record_states, epsilon):
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("The privacy budget must be finite and nonnegative.")
    if not np.all((record_states == 0) | (record_states == 1)):
        raise ValueError("Randomized response requires binary infection statuses.")
    odds = math.exp(-epsilon)
    if odds == 0:
        raise ValueError("The flip probability is too small to represent.")
    flip_probability = odds / (1 + odds)
    numerator, denominator = flip_probability.as_integer_ratio()
    flips = np.fromiter(
        (secrets.randbelow(denominator) < numerator for _ in range(record_states.size)),
        dtype=bool, count=record_states.size,
    ).reshape(record_states.shape)
    return np.logical_xor(record_states, flips).astype(record_states.dtype, copy=False)


def beta_weight_infer(graph_list, result_list, tau=0):
    sample_sizes = [len(sample) for sample in result_list]
    agg_network = np.average(graph_list, axis=0, weights=sample_sizes)
    return threshold_network(agg_network, tau)


def edge_grained_weight_infer(graph_list, result_list, tau=0):
    union_network = np.maximum.reduce(graph_list)
    np.fill_diagonal(union_network, 0)
    numerator = np.zeros(union_network.shape)
    denominator = np.zeros(union_network.shape)
    total_sample = sum(len(sample) for sample in result_list)

    for c, (local_graph, sample) in enumerate(zip(graph_list, result_list)):
        weight_matrix = edge_grained_weight(sample, local_graph, union_network)
        data_ratio = len(sample) / total_sample
        weighted_support = data_ratio * weight_matrix
        numerator += weighted_support * local_graph
        denominator += weighted_support
        print("client: %d is done" % c)

    agg_network = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                            where=denominator > 0)
    return threshold_network(agg_network, tau)


def edge_grained_weight(record_states, local_graph, union_graph):
    results_num, nodes_num = record_states.shape
    infected = record_states == 1
    infected_count = np.count_nonzero(infected, axis=0)
    weight_matrix = np.zeros((nodes_num, nodes_num))

    for i, j in zip(*np.nonzero(union_graph)):
        if i == j:
            continue
        child_count = int(infected_count[j])
        edge_exists = local_graph[i, j] == 1
        parent_count = int(infected_count[i])
        if not edge_exists:
            parent_count = results_num - parent_count
        if parent_count == 0 or child_count == 0:
            continue
        both_infected = np.count_nonzero(infected[:, i] & infected[:, j])
        joint_count = both_infected if edge_exists else child_count - both_infected
        ratio = results_num * joint_count / (parent_count * child_count)
        weight_matrix[i, j] = max(0.0, ratio - 1.0)
    return weight_matrix


def threshold_with_kmeans(weight_matrix):
    scores = weight_matrix[~np.eye(weight_matrix.shape[0], dtype=bool)]
    if scores.size == 0 or scores.max() == 0:
        return 0.0
    positive = scores > scores.max() / 2
    while True:
        center = scores[positive].mean()
        updated = scores > center / 2
        if np.array_equal(updated, positive):
            return scores[~positive].max(initial=0.0)
        positive = updated


def threshold_network(agg_network, tau):
    if tau == 0:
        tau = threshold_with_kmeans(agg_network)
        print("tau = %f" % tau)
    final_network = (agg_network > tau).astype(float)
    np.fill_diagonal(final_network, 0)
    return final_network



def save_graph(graph, filepath):
    rows, cols = np.where(graph != 0)

    with open(filepath, 'w+') as f:
        for parent, child in zip(rows, cols):
            f.write(f"{parent+1}\t{child+1}\n")
    f.close()



if __name__ == '__main__':
    import TENDS

    graph_path = './network.txt'
    result_path = './observation_data.txt'
    client_num = 5
    partition_seed = 0
    use_randomized_response = False
    privacy_epsilon = 2.0

    ground_truth_network, diffusion_result = load_data(graph_path, result_path)
    result_list = partition_observations(diffusion_result, client_num, partition_seed)
    if use_randomized_response:
        result_list = [randomized_response(sample, privacy_epsilon) for sample in result_list]
        diffusion_result = np.concatenate(result_list, axis=0)
    Path("./results").mkdir(exist_ok=True)

    begin_time = time.time()
    graph_list = []
    for i in range(client_num):
        local_begin_time = time.time()
        local_graph = TENDS.run(result_list[i])
        local_end_time = time.time()
        print("client ", i, "running time :", local_end_time - local_begin_time, "P, R, F1 :",
              cal_F1(ground_truth_network, local_graph))
        local_network_path = "./results/local_network_No"+str(i)+".txt"
        save_graph(local_graph, local_network_path)
        graph_list.append(local_graph)
    end_time = time.time()
    print("Running time of infering all local network is %f s" % (end_time - begin_time))

    begin_time = time.time()
    edge_grained_network = edge_grained_weight_infer(graph_list, result_list)
    end_time = time.time()
    print("#############edge-grained weight infer, result:")
    print(cal_F1(ground_truth_network, edge_grained_network))
    print("Running time is %f s" % (end_time - begin_time))

    begin_time = time.time()
    beta_weight_network = beta_weight_infer(graph_list, result_list)
    end_time = time.time()
    print("#############coarse-grained (data amount) weight infer, result:")
    print(cal_F1(ground_truth_network, beta_weight_network))
    print("Running time is %f s" % (end_time - begin_time))

    begin_time = time.time()
    TENDS_network = TENDS.run(diffusion_result)
    end_time = time.time()
    print("#############TENDS infer, result:")
    print(cal_F1(ground_truth_network, TENDS_network))
    print("Running time is %f s" % (end_time - begin_time))







