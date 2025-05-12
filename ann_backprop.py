"""
@file: ann_backprop.py
@description: artificial neural network with and implement using a data set for classification task
@assignment: Introduction to Data Mining Homework
@date: 13.05.2025
@authors:  Beyza Yıldızlı @beyzayildizli10@gmail.com
"""

import numpy as np

# Simple dataset: XOR problem
dataset = [ 
    ([1.0, 0.0, 1.0], [1]), #ex solved theoretically in the last lesson
    ([0.0, 1.0, 0.0], [0]),
    ([1.0, 1.0, 0.0], [1]),
    ([0.0, 0.0, 0.0], [0])
]

lr = 0.9
Epoch = 1000

# weight
w_input_hidden = [
    [0.2, -0.3],  # w14, w15
    [0.4, 0.1],   # w24, w25
    [-0.5, 0.2]   # w34, w35
]
w_hidden_output = [
    [-0.3],       # w46
    [-0.2]        # w56
]

# Bias
bias_hidden = [-0.4, 0.3]      # Q4, Q5
bias_output = [0.1]            # Q6

# Layer sizes
input_size = len(dataset[0][0]) #3
hidden_size = len(bias_hidden) #2
output_size = len(bias_output) #1

# Neuron index offsets
idx_hidden_start = input_size + 1                # ex: input_size(3) + 1 = 4 (values ​​in the hidden layer are like 04 O5..)
idx_output_start = input_size + hidden_size + 1  # ex: input_size(3)  + hidden_size(2) + 1 = 6 (Values ​​in the output layer are like 06..)

# Formul: O = 1 / (1 + e^(-I))
def sigmoid(I):
    return 1 / (1 + np.exp(-I))

def sigmoid_derivative(O):
    return O * (1 - O)

for epoch in range(1, Epoch + 1):
    print(f"\nEpoch {epoch}")

    for sample_idx, (x, target) in enumerate(dataset):
        print(f"\nSample {sample_idx+1}")

        # FORWARD PASS
        I_hidden = [0.0] * hidden_size
        o_hidden = [0.0] * hidden_size

        for j in range(hidden_size):
            # Formula: Ij = Σ(xi * wij) + Qj
            I_hidden[j] = sum(x[i] * w_input_hidden[i][j] for i in range(input_size)) + bias_hidden[j]
            
            # Formula: OJ = 1 / (1 + e^(-Ij))
            o_hidden[j] = sigmoid(I_hidden[j])
            print(f"I{idx_hidden_start+j}: {I_hidden[j]:.4f}, O{idx_hidden_start+j}: {o_hidden[j]:.4f}")

        I_output = [0.0] * output_size
        o_output = [0.0] * output_size

        # Output layer
        for k in range(output_size):
            # Formula: Ik = Σ(Oj * wjk) + Qk
            I_output[k] = sum(o_hidden[j] * w_hidden_output[j][k] for j in range(hidden_size)) + bias_output[k]
            
            # Formula: Ok = 1 / (1 + e^(-Ik))
            o_output[k] = sigmoid(I_output[k])
            print(f"I{idx_output_start+k}: {I_output[k]:.4f}, O{idx_output_start+k}: {o_output[k]:.4f}")

        print("------------------------------")
    
        # BACKPROPAGATION
        error_output = [0.0] * output_size
        delta_output = [0.0] * output_size

        for k in range(output_size):
            # Formula: Errork = Ok * (1 - Ok) * (target - Ok)
            error_output[k] = target[k] - o_output[k]
            delta_output[k] = sigmoid_derivative(o_output[k]) * error_output[k]
            print(f"Error{idx_output_start+k}: {delta_output[k]:.4f}")

        delta_hidden = [0.0] * hidden_size

        for j in range(hidden_size):
            # Formula: Errorj = Oj * (1 - Oj) * Σ(Errork * wjk)
            total = sum(delta_output[k] * w_hidden_output[j][k] for k in range(output_size))
            delta_hidden[j] = sigmoid_derivative(o_hidden[j]) * total
            print(f"Error{idx_hidden_start+j}: {delta_hidden[j]:.4f}")

        print("------------------------------")
    
        # Update (Hidden → Output)
        for j in range(hidden_size):
            for k in range(output_size):
                # Formula: Δwjk = lr * Errork * Oj
                delta_w = lr * delta_output[k] * o_hidden[j]
                print(f"ΔW{idx_hidden_start+j}{idx_output_start+k}: {delta_w:.4f}")

                # Formula: wjk = wjk + Δwjk
                w_hidden_output[j][k] += delta_w
                print(f"W{idx_hidden_start+j}{idx_output_start+k}: {w_hidden_output[j][k]:.4f}")

        print("------------------------------")

        for k in range(output_size):
            # Formula: ΔQk = lr * Errork
            delta_b = lr * delta_output[k]
            print(f"ΔQ{idx_output_start+k}: {delta_b:.4f}")

            # Formula: Qk = Qk + ΔQk
            bias_output[k] += delta_b
            print(f"Q{idx_output_start+k}: {bias_output[k]:.4f}")

        print("------------------------------")

        # Update (Input → Hidden)
        for i in range(input_size):
            for j in range(hidden_size):
                # Formula: Δwij = lr * Errorj * xi
                delta_w = lr * delta_hidden[j] * x[i]
                print(f"ΔW{i+1}{idx_hidden_start+j}: {delta_w:.4f}")

                # Formula: wij = wij + Δwij
                w_input_hidden[i][j] += delta_w
                print(f"W{i+1}{idx_hidden_start+j}: {w_input_hidden[i][j]:.4f}")

        print("------------------------------")

        for j in range(hidden_size):
            # Formula: ΔQj = lr * Errorj
            delta_b = lr * delta_hidden[j]
            print(f"ΔQ{idx_hidden_start+j}: {delta_b:.4f}")

            # Formula: Qj = Qj + ΔQj 
            bias_hidden[j] += delta_b
            print(f"Q{idx_hidden_start+j}: {bias_hidden[j]:.4f}")
