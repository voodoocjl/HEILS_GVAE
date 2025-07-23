import pickle
import numpy as np
import torch
from MCTS_mix import MCTS
from schemes import Scheme
from FusionModel import translator, cir_to_matrix
from Arguments import Arguments
import random

# Load or define a set of initial circuits
def load_initial_circuits(search_space_file, arch, num_circuits=5):
    with open(search_space_file, 'rb') as f:
        search_space = pickle.load(f)
    
    n_qubits, n_layers = arch

    single = [[i]+[1]*2*n_layers for i in range(1,n_qubits+1)]
    enta = [[i]+[i+1]*n_layers for i in range(1,n_qubits)]+[[n_qubits]+[1]*n_layers]

    circuit_list = []
    for _ in range(num_circuits):
        circuit_list.append([single, enta])
        # qubits = random.sample([i for i in range(1, n_qubit+1)],n_single)
        # single = sampling_qubits(search_space_single, qubits)

        # qubits = random.sample([i for i in range(1, n_qubit+1)],n_enta)
        # enta = sampling_qubits(search_space_enta, qubits)
    return circuit_list

def evaluate_langevin_neighborhood(agent, arch, snr_values, task):
    results = {}
    weight = torch.load('init_weights/init_weight_MNIST_10')
    for snr in snr_values:
        Arch = cir_to_matrix(arch[0], arch[1], arch_code, args.fold)
        arch_next = agent.Langevin_update(Arch, snr)
        performances = []
        for single, enta in arch_next[:5]:
            print('single:', single, 'enta:', enta)
            design = translator(single, enta, 'full', agent.ARCH_CODE, agent.fold)
            # Evaluate using Scheme (set epochs as needed)
            model, report = Scheme(design, task, weight, epochs=0)
            performances.append(report['mae'])
        mean_perf = np.mean(performances) if performances else None
        results[snr] = mean_perf
    return results

def sampling_qubits(search_space, qubits):
    arch_list = []
    while len(qubits) > 0:    
        arch = random.sample(search_space, 1)
        if arch[0][0] in qubits:
            qubits.remove(arch[0][0])
            arch_list.append(arch[0])
    return arch_list

if __name__ == "__main__":
    # Setup task and agent
    # task = {
    #     'task': 'MNIST_10',
    #     'option': 'mix_reg',
    #     'regular': True,
    #     'n_qubits': 10,
    #     'n_layers': 4,
    #     'fold': 2
    # }

    task = {
        'task': 'MNIST_10',
        'option': 'mix_reg',
        'regular': True,
        'n_qubits': 10,
        'n_layers': 4,
        'fold': 2
    }
    arch_code = [task['n_qubits'], task['n_layers']]
    arch_code_fold = [task['n_qubits']//task['fold'], task['n_layers']]
    args = Arguments(**task)
    # Load search space and create agent
    initial_circuits = load_initial_circuits('search_space/search_space_mnist_10', arch_code_fold, num_circuits=1)
    agent = MCTS(initial_circuits, tree_height=4, fold=task['fold'], arch_code=arch_code)
    agent.task_name = task['task'] + '_' + task['option']
    agent.weight = 'init'  # Or load pretrained weights if available

    snr_values = [0.01, 0.05, 0.1, 0.5]
    results_all = []

    for idx, arch in enumerate(initial_circuits):
        print(f"Evaluating circuit {idx+1}/{len(initial_circuits)}")
        snr_results = evaluate_langevin_neighborhood(agent, arch, snr_values, task)
        print(f"Results for circuit {idx+1}: {snr_results}")
        results_all.append(snr_results)

    # Print summary
    print("\nMean performance for each SNR value across circuits:")
    for snr in snr_values:
        mean_perf = np.mean([r[snr] for r in results_all if r[snr] is not None])
        print(f"SNR={snr}: Mean MAE={mean_perf}")

    # Optionally, save results to CSV
    # import csv
    # with open('langevin_snr_results.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['Circuit', 'SNR', 'Mean_MAE'])
    #     for idx, snr_results in enumerate(results_all):
    #         for snr, mae in snr_results.items():
    #             writer.writerow([idx, snr, mae])