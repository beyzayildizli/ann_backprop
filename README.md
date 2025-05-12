# Solving the XOR Problem with an Artificial Neural Network

This project implements a basic classification model that solves the XOR problem using a simple feedforward neural network with backpropagation.

## Project Summary

This neural network model learns to predict the correct output for XOR inputs. The model consists of three layers:
1. **Input Layer**: 3 neurons (XOR inputs)
2. **Hidden Layer**: 2 neurons (with sigmoid activation function)
3. **Output Layer**: 1 neuron (with sigmoid activation function)
![model](https://github.com/user-attachments/assets/60c410b5-c028-464d-8d00-1efb5d6dd052)

## Methods Used

- **Forward Pass**: Each neuron's input in every layer is calculated, followed by applying the activation function (sigmoid) to produce an output.
  
- **Backpropagation**: The error between the predicted output and target is computed, and weights and biases are updated accordingly. This process enables the model to learn accurate predictions.

- **Activation Function**: The sigmoid function is used as it is widely preferred in binary classification tasks.

### Training Dataset
The dataset for the XOR problem is as follows:
- Input: \([1.0, 0.0, 1.0]\), Output: \([1]\)
- Input: \([0.0, 1.0, 0.0]\), Output: \([0]\)
- Input: \([1.0, 1.0, 0.0]\), Output: \([1]\)
- Input: \([0.0, 0.0, 0.0]\), Output: \([0]\)

The goal of the model is to predict the correct output for these input combinations.

## Libraries Used

- **NumPy**: Used for handling weights, biases, and matrix operations.

## Training and Results

The model aims to reduce error after each epoch. Initially, the error rates are high, but as the number of epochs increases, the model begins to make more accurate predictions.

To observe the model’s progress during training, we compare key variables between Epoch 1 and Epoch 1000:

### 1. Output
- **Epoch 1:** Output value (O6) is still ambiguous.  
  Example (Sample 1): `O6: 0.4726`

- **Epoch 1000:** Output value is very close to the target.  
  Example (Sample 1): `O6: 0.9784` (output layer)

This progression shows that the model has successfully learned to recognize the correct classes.

### 2. Error
- **Epoch 1:** Error values are high.  
  Example: `Error6: 0.1314`, `Error4: -0.0087`, `Error5: -0.0065`

- **Epoch 1000:** Error values have significantly decreased.  
  Example: `Error6: 0.0005`, `Error4: -0.0000`, `Error5: -0.0001`

This reduction confirms the model's learning and increasing prediction accuracy.

### 3. Weight Updates
- **Epoch 1:** Weights are updated with large values.  
  Example: `ΔW46: 0.0393`, `ΔW56: 0.0650`

- **Epoch 1000:** Weight updates are minimal.  
  Example: `ΔW46: 0.0000`, `ΔW56: 0.0000`

This indicates that the model is approaching a minimum error point and has started to stabilize.

### 4. Bias Updates
- **Epoch 1:** Bias values are updated with relatively large amounts.  
  Example: `ΔQ6: 0.1183`, `Q6: 0.2183`

- **Epoch 1000:** Bias values have become more stable.  
  Example: `ΔQ6: 0.0004`, `Q6: 4.0450`

Smaller bias updates support the idea that the model has finished learning and is now producing accurate outputs.

### 5. Overall Evaluation
- **Epoch 1:** Model outputs are more random and error rates are high. Learning has not fully started yet.
- **Epoch 1000:** The model predicts with high accuracy. Weights and biases have stabilized, indicating a successfully completed training process.

The model has learned to recognize the nonlinear patterns necessary to solve the XOR problem and now produces stable, accurate results.
