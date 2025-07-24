import pickle
import numpy as np
import torch
from MCTS_mix import MCTS
from schemes import Scheme
from FusionModel import translator, cir_to_matrix
from Arguments import Arguments
import random
import json

# Load or define a set of initial circuits
def load_initial_circuits(search_space_file, arch, num_circuits=5):
    with open(search_space_file[0], 'rb') as f:
        search_space_single = pickle.load(f)
    with open(search_space_file[1], 'rb') as f:
        search_space_enta = pickle.load(f)

    # single = [[i]+[1]*2*n_layers for i in range(1,n_qubits+1)]
    # enta = [[i]+[i+1]*n_layers for i in range(1,n_qubits)]+[[n_qubits]+[1]*n_layers]

    circuit_list = []
    for _ in range(num_circuits):
        
        qubits = random.sample([i for i in range(1, n_qubit+1)],n_single)
        single = sampling_qubits(search_space_single, qubits)

        qubits = random.sample([i for i in range(1, n_qubit+1)],n_enta)
        enta = sampling_qubits(search_space_enta, qubits)
        circuit_list.append([single, enta])
    return circuit_list

def evaluate_langevin_neighborhood(agent, arch, snr_values, task):
    results = {}
    weight = torch.load('init_weights/init_weight_MNIST_10', weights_only=True)
    original_single, original_enta = arch['single'], arch['enta']
    for snr in snr_values:
        print("***"* 20)
        print(f"\033[31mEvaluating SNR={snr}\033[0m")
        Arch = cir_to_matrix(original_single, original_enta, arch_code, args.fold)
        arch_next = agent.Langevin_update(Arch, snr)
        if len(arch_next) <= 5:
            print(f"Valid architectures {len(arch_next)} found for SNR={snr}. Skipping evaluation.")
            results[snr] = None
            continue
        else:
            arch_next = random.sample(arch_next, 5)  # Sample 5 architectures for evaluation
            performances = []
            difference = []
            for single, enta in arch_next:
                print('single:', single, 'enta:', enta)
                # design = translator(single, enta, 'full', agent.ARCH_CODE, agent.fold)
                # # Evaluate using Scheme (set epochs as needed)
                # model, report = Scheme(design, task, weight, epochs=1)
                # performances.append(report['mae'])
                diff = difference_between_archs(original_single, original_enta, single, enta)
                difference.append(diff)
                print('\033[33mDifference:\033[0m', diff)
            # mean_perf = np.mean(performances) if performances else None
            mean_perf = np.mean(difference, axis=0) if difference else None
            results[snr] = mean_perf
    return results

def difference_between_archs(original_single, original_enta, decoded_single, decoded_enta):
    single_diff = sum(abs(np.array(a) - np.array(b)).sum() for a, b in zip(original_single, decoded_single))                    
    enta_diff = sum((abs(np.array(a) - np.array(b)) != 0).sum() for a, b in zip(original_enta, decoded_enta))

    return [single_diff, enta_diff]
                


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
    # initial_circuits = load_initial_circuits([args.file_single, args.file_enta], arch_code_fold, num_circuits=1)
    with open('data/random_circuits_mnist_5.json', 'r') as f:
        initial_circuits = json.load(f)
    agent = MCTS(initial_circuits, tree_height=4, fold=task['fold'], arch_code=arch_code)
    agent.task_name = task['task'] + '_' + task['option']
    agent.weight = 'init'  # Or load pretrained weights if available

    snr_values = np.linspace(1, 30, 5)
    results_all = []

    for idx, arch in enumerate(initial_circuits[:100]):
        print(f"\033[34mEvaluating circuit {idx+1}/{len(initial_circuits)}\033[0m")
        snr_results = evaluate_langevin_neighborhood(agent, arch, snr_values, task)
        print(f"Results for circuit {idx+1}: {snr_results}")
        results_all.append(snr_results)

    # Print summary
    # print("\nMean performance for each SNR value across circuits:")
    # for snr in snr_values:
    #     mean_perf = np.mean([r[snr] for r in results_all if r[snr] is not None])
    #     print(f"SNR={snr}: Mean MAE={mean_perf}")

    print("\nMean differences for each SNR value across circuits:")
    for snr in snr_values:
        mean_diff = np.mean([r[snr] for r in results_all if r[snr] is not None], axis=0)
        print(f"SNR={snr}: Mean Diff={mean_diff}")

    # Optionally, save results to CSV
    # import csv
    # with open('langevin_snr_results.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['Circuit', 'SNR', 'Mean_MAE'])
    #     for idx, snr_results in enumerate(results_all):
    #         for snr, mae in snr_results.items():
    #             writer.writerow([idx, snr, mae])