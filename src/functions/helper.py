import networkx as nx
import pandas as pd
import mlflow
import json
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


import algorithm.kcomm.graph_kClusterAlgorithm_functions as QCD
import algorithm.kcomm.graphFileUtility_functions as GFU


def load_karate():
    G = nx.karate_club_graph()
    return G


def load_syndata(filename):
    edgelist = pd.read_csv(filename, sep=' ', names=["source", "target"])
    G = nx.from_pandas_edgelist(edgelist)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    return G


def evaluate_partition_hybrid(num_parts, graph, ground_truth_path, dataset, run_label, qsize, threshold, beta0, gamma0, run_profile, run_id):
    A = nx.adjacency_matrix(graph)
    print('\nAdjacency matrix:\n', A.todense())

    num_blocks = num_parts
    num_nodes = nx.number_of_nodes(graph)
    num_edges = nx.number_of_edges(graph)
    print("\n\t Quantum Community Detection: up to %d communities...\n" % num_parts)
    print("Graph has %d nodes and %d edges" % (num_nodes, num_edges))

    # Collect results to dictionary
    result = {}
    result['alg'] = 'LANL_CD'
    result['num_clusters'] = num_parts
    result['dataset'] = dataset
    result['nodes'] = num_nodes
    result['edges'] = num_edges
    result['size'] = num_nodes * num_parts
    result['solver'] = 'DWAVE_Hybrid'
    result['subqubo_size'] = qsize

    beta, gamma, GAMMA = QCD.set_penalty_constant(num_nodes, num_blocks, beta0, gamma0)

    mtotal, modularity = QCD.build_mod(A, threshold, num_edges)

    print("\nModularity matrix: \n", modularity)

    print("min value = ", modularity.min())
    print("max value = ", modularity.max())

    print("threshold = ", threshold)

    Q = QCD.makeQubo(graph, modularity, beta, gamma, GAMMA, num_nodes, num_parts, num_blocks, threshold)

    # Run k-clustering with Hybrid/D-Wave using ocean
    ss = QCD.clusterHybrid(Q, num_parts, qsize, run_label, run_profile, result)

    # Process solution
    part_number = QCD.process_solution(ss, graph, num_blocks, num_nodes, num_parts, result)

    mmetric = QCD.calcModularityMetric(mtotal, modularity, part_number)
    print("\nModularity metric = ", mmetric)
    result['modularity_metric'] = mmetric

    GFU.write_partFile(part_number, num_nodes, num_parts, run_id)
    GFU.write_resultFile(result, run_id)
    GFU.showClusters(part_number, graph, run_id)

    mlflow.log_artifact(f"results/{run_id}/clusters.png")
    mlflow.log_artifact(f"results/{run_id}/result.json")

    with open("results/result.json") as result_file:
        result_json = json.load(result_file)
        keys_to_remove = [key for key, value in result_json.items() if isinstance(value, list)]
        keys_with_string = [key for key, value in result_json.items() if isinstance(value, str)]
        for key in keys_to_remove:
            result_json.pop(key)

        for k, v in result_json.items():
            if k in keys_with_string:
                mlflow.log_param(k, v)
            else:
                mlflow.log_metric(k, v)

    columns = ["node_id", "comm_id"]
    communities = []

    predicted_communities = []

    with open(f"results/comm{num_parts}.txt") as comm_file:
        i = 0
        for line in comm_file:
            i += 1
            if i == 1:
                continue
            fields = line.strip().split("  ")
            communities.append(fields)
            predicted_communities.append(fields[1])

    ground_truth_communities = []
    with open(ground_truth_path) as ground_truth_file:
        for line in ground_truth_file:
            if line.startswith("#"):
                continue
            fields = line.strip().split(" ")
            ground_truth_communities.append(fields[1])

    with open(f"results/{run_id}/communities.csv") as comm_out_file:
        comm_out_file.write(columns.join(","))
        for cur_comm in communities:
            comm_out_file.write(cur_comm.join(","))

    mlflow.log_artifact(f"results/{run_id}/communities.csv")

    ari_score = adjusted_rand_score(ground_truth_communities, predicted_communities)
    ami_score = adjusted_mutual_info_score(ground_truth_communities, predicted_communities)

    mlflow.log_metric("ari_score", ari_score)
    mlflow.log_metric("ami_score", ami_score)
