import pickle
import numpy as np
import torch

from schemes import Scheme
from FusionModel import translator, cir_to_matrix
from Arguments import Arguments
import random
import json
from GVAE_PRE.gen_random_circuits import generate_random_circuits
from GVAE_PRE.utils import get_proj_mask, is_valid_ops_adj, generate_single_enta, compute_scaling_factor, arch_to_z
from GVAE_model import GVAE
from configs import configs


def Langevin_update(x, model, snr=10, step_size=0.01):
        
        x, logvar = arch_to_z([x], arch_code_fold, model.encoder)
        x_valid_list = []

        # Compute scaling factor c
        decoder = model.decoder
        decoder.eval()
        d = x.shape[2]  # Dimensionality
        c1 = compute_scaling_factor(x, decoder, snr[0], d)
        c2 = compute_scaling_factor(x, decoder, snr[1], d)
        
        # c2 = 0
        n_qubit = arch_code_fold[0]        
        # x_norm_per_sample = torch.norm(x, dim=2, keepdim=True)

        for i in range(1000):
            # noise = torch.randn_like(x)
            step_size = [c1, c2]
            x_new = sample_normal(x, logvar,step_size, arch_code_fold)
            # x_new = x + step_size * noise
            x_new = decoder(x_new)
            mask = get_proj_mask(x_new[0], n_qubit, n_qubit)
            if is_valid_ops_adj(x_new[0], n_qubit, threshold=6):
                gate_matrix = x_new[0] + mask
                single,enta, _ = generate_single_enta(gate_matrix, n_qubit)
                if [single, enta] not in x_valid_list:
                    x_valid_list.append([single, enta])
        print('Number of valid ciruicts:', len(x_valid_list))
        return x_valid_list

def sample_normal(mu, logvar, step_size, arch_code_fold):
    """
    Sample from N(mu, exp(logvar)) using the reparameterization trick.   
    """
    std = torch.exp(logvar)
    eps = torch.randn_like(std)

    n_qubits, n_layers = arch_code_fold
    step_single, step_enta = step_size
    # Adjust the step size for single and enta
    step_size = [step_single] * n_qubits + [step_enta] * n_qubits
    step_size = step_size * n_layers
    step_size = torch.Tensor(np.diag(step_size))
    # return mu + eps * std * step_size
    return mu + torch.matmul(step_size, eps)

def evaluate_langevin_neighborhood(arch, snr_values, task):
    results = {}
    weight = torch.load('init_weights/init_weight_MNIST_10', weights_only=True)
    original_single, original_enta = arch['single'], arch['enta']
    for snr in snr_values:
        print("***"* 20)
        print(f"\033[31mEvaluating SNR={snr}\033[0m")
        Arch = cir_to_matrix(original_single, original_enta, arch_code, args.fold)
        arch_next = Langevin_update(Arch, GVAE_model, snr)
        if len(arch_next) <= 10:
            number = len(arch_next)
        else:
            number = 10
        print(f"Found {number} architectures for SNR={snr}.")
            
        arch_next = random.sample(arch_next, number)  # Sample 5 architectures for evaluation
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
            results[tuple(snr)] = mean_perf
    return results

def difference_between_archs(original_single, original_enta, decoded_single, decoded_enta):
    single_diff = sum(abs(np.array(a) - np.array(b)).sum() for a, b in zip(original_single, decoded_single))                    
    enta_diff = sum((abs(np.array(a) - np.array(b)) != 0).sum() for a, b in zip(original_enta, decoded_enta))

    return [single_diff, enta_diff]
                


if __name__ == "__main__":
    # Setup task and agent
    task = {
        'task': 'MNIST_4',
        'option': 'mix_reg',
        'regular': True,
        'n_qubits': 4,
        'n_layers': 4,
        'fold': 1
    }

    # task = {
    #     'task': 'MNIST_10',
    #     'option': 'mix_reg',
    #     'regular': True,
    #     'n_qubits': 10,
    #     'n_layers': 4,
    #     'fold': 2
    # }
    arch_code = [task['n_qubits'], task['n_layers']]
    arch_code_fold = [task['n_qubits']//task['fold'], task['n_layers']]
    args = Arguments(**task)
    n_qubits, n_layers = arch_code_fold

    # Load search space and create agent
    initial_circuits = generate_random_circuits(10, n_qubits, args.file_single, args.file_enta)

    single = [[i]+[1]*2*n_layers for i in range(1,n_qubits+1)]
    enta = [[i]+[i+1]*n_layers for i in range(1,n_qubits)]+[[n_qubits]+[1]*n_layers]
    initial_circuits = [{'single': single, 'enta': enta}]

    # with open('data/random_circuits_mnist_5.json', 'r') as f:
    #     initial_circuits = json.load(f)    

    # checkpoint = torch.load('pretrained/model-circuits_5_qubits-15.pt', map_location=torch.device('cpu'), weights_only=True)
    # checkpoint = torch.load('pretrained/model-circuits_5_qubits-swap.pt', map_location=torch.device('cpu'), weights_only=True)
    checkpoint = torch.load('pretrained/dim-16/model-circuits_4_qubits-19.pt', map_location=torch.device('cpu'), weights_only=True)


    input_dim = 4 + arch_code_fold[0]
    GVAE_model = GVAE((input_dim, 32, 64, 128, 64, 32, 16), normalize=True, dropout=0.3, **configs[4]['GAE'])
    GVAE_model.load_state_dict(checkpoint['model_state'])

    snr_values = np.linspace([0.1, 0.1], [1, 1], 10)
    results_all = []

    for idx, arch in enumerate(initial_circuits[:1]):
        print(f"\033[34mEvaluating circuit {idx+1}/{len(initial_circuits)}\033[0m")
        snr_results = evaluate_langevin_neighborhood(arch, snr_values, task)
        print(f"Results for circuit {idx+1}: {snr_results}")
        results_all.append(snr_results)

    # Print summary
    # print("\nMean performance for each SNR value across circuits:")
    # for snr in snr_values:
    #     mean_perf = np.mean([r[snr] for r in results_all if r[snr] is not None])
    #     print(f"SNR={snr}: Mean MAE={mean_perf}")

    print("\nMean differences for each SNR value across circuits:")
    for snr in snr_values:
        mean_diff = np.mean([r[tuple(snr)] for r in results_all if r[tuple(snr)] is not None], axis=0)
        print(f"SNR={snr}: Mean Diff={mean_diff}")

    # Optionally, save results to CSV
    # import csv
    # with open('langevin_snr_results.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['Circuit', 'SNR', 'Mean_MAE'])
    #     for idx, snr_results in enumerate(results_all):
    #         for snr, mae in snr_results.items():
    #             writer.writerow([idx, snr, mae])