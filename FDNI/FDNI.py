import math
import time
import numpy as np

from utils import cal_F1
import TENDS



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




def beta_weight_infer(graph_list,result_list,tau=0.5):

    client_num = len(graph_list)
    node_num = graph_list[0].shape[1]
    agg_network = np.zeros((node_num,node_num))
    total_sample = 0
    for c in range(client_num):
        sample_index = result_list[c].shape[0]
        prune_network = graph_list[c]
        print("client: %d is done" % (c))
        agg_network = agg_network + prune_network*sample_index
        total_sample += sample_index

    agg_network = agg_network/total_sample
    final_network = np.zeros((node_num, node_num))
    if tau == 0:
        tau = threshold_with_kmeans(agg_network)
        print("tau = %f" % (tau))
    for i in range(node_num):
        for j in range(node_num):
            if agg_network[i, j] > tau:
                final_network[i, j] = 1

    return final_network




def edge_grained_weight_infer(graph_list,result_list,tau=0.5):
    node_num = graph_list[0].shape[1]
    client_num = len(graph_list)
    agg_network = np.zeros((node_num,node_num))
    agg_weight = np.zeros((node_num,node_num))
    union_network = np.zeros((node_num,node_num))
    weight_list = []
    epsilon = 1e-5

    for c in range(client_num):
        union_network += graph_list[c]

    for c in range(client_num):
        sample = result_list[c]
        weight_matrix = edge_grained_weight(sample,graph_list[c],union_network)
        print("client: %d is done" % (c))
        weight_list.append(weight_matrix)
        agg_weight = agg_weight + weight_matrix

    for c in range(client_num):
        weight_list[c] = weight_list[c]/(agg_weight+epsilon)
        agg_network = agg_network + weight_list[c] * graph_list[c]

    final_network = np.zeros((node_num, node_num))
    if tau == 0:
        tau = threshold_with_kmeans(agg_network)
        print("tau = %f" % (tau))
    for i in range(node_num):
        for j in range(node_num):
            if agg_network[i, j] > tau:
                final_network[i, j] = 1

    return final_network






def edge_grained_weight(record_states, local_graph,union_graph):
    results_num, nodes_num = record_states.shape
    print("results_num = ", results_num)
    weight_matrix = np.zeros((nodes_num, nodes_num))

    for j in range(nodes_num):
        for k in range(nodes_num):
            if union_graph[j,k] > 0:
                state_mat = np.zeros((2, 2))
                for result_index in range(results_num):
                    state_mat[0, 0] += (1 - record_states[result_index, j]) * (1 - record_states[result_index, k])
                    state_mat[0, 1] += (1 - record_states[result_index, j]) * record_states[result_index, k]
                    state_mat[1, 0] += record_states[result_index, j] * (1 - record_states[result_index, k])
                    state_mat[1, 1] += record_states[result_index, j] * record_states[result_index, k]
                Ni1 = state_mat[1, 1] + state_mat[0, 1]
                N1j = state_mat[1, 1] + state_mat[1,0]
                N0j = state_mat[0, 1] + state_mat[0,0]
                epsilon = 1e-5
                if local_graph[j, k] == 1:
                    weight_matrix[j, k] = max(0,(results_num*state_mat[1, 1] - Ni1*N1j)/(results_num*N1j-Ni1*N1j+epsilon))
                else:
                    weight_matrix[j, k] = max(0, (results_num * state_mat[0, 1] - Ni1 * N0j) / (
                                results_num * Ni1 - Ni1 * N0j + epsilon))
    return weight_matrix

def threshold_with_kmeans(weight_matrix):
    nodes_num = weight_matrix.shape[0]
    print("min: " ,np.min(weight_matrix),"mean: " ,np.mean(weight_matrix), "max: " ,np.max(weight_matrix))

    tmp_MI = weight_matrix.reshape((-1, 1))

    estimator = KMeans(n_clusters=2)
    estimator.fit(tmp_MI)
    label_pred = estimator.labels_
    temp_0 = tmp_MI[label_pred == 0]
    temp_1 = tmp_MI[label_pred == 1]

    if np.max(temp_0) > np.max(temp_1):
        tau = np.max(temp_1)
    else:
        tau = np.max(temp_0)

    print("tau = %f" % (tau))

    return tau

def save_graph(graph, filepath):
    rows, cols = np.where(graph != 0)

    with open(filepath, 'w+') as f:
        for parent, child in zip(rows, cols):
            f.write(f"{parent+1}\t{child+1}\n")
    f.close()



if __name__ == '__main__':
    graph_path = './network.txt'
    result_path = './observation_data.txt'
    sample_index = [100, 250, 450, 700]
    ground_truth_network, diffusion_result = load_data(graph_path, result_path)
    result_list = np.split(diffusion_result, sample_index, axis=0)

    # Local diffusion network inference
    begin_time = time.time()
    graph_list = []
    client_num = len(result_list)
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

    # weight calculation and aggregation of FDNI
    begin_time = time.time()
    edge_grained_network = edge_grained_weight_infer( graph_list, result_list)
    end_time = time.time()
    print("#############edge-grained weight infer, result:")
    print(cal_F1(ground_truth_network, edge_grained_network))
    print("Running time is %f s" % (end_time - begin_time))

    # weight calculation and aggregation of coarse-grained (data amount) inference
    begin_time = time.time()
    beta_weight_network = beta_weight_infer(graph_list, result_list)
    end_time = time.time()
    print("#############coarse-grained (data amount) weight infer, result:")
    print(cal_F1(ground_truth_network, beta_weight_network))
    print("Running time is %f s" % (end_time - begin_time))

    # centralized learning of TENDS using aggregate data
    begin_time = time.time()
    TENDS_network = TENDS.run(diffusion_result)
    end_time = time.time()
    print("#############TENDS infer, result:")
    print(cal_F1(ground_truth_network, TENDS_network))
    print("Running time is %f s" % (end_time - begin_time))







