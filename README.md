# ML-Based Adaptive BPSK Receiver using PyTorch

A Machine Learning implementation of a Binary Phase Shift Keying (BPSK) receiver using PyTorch.

Instead of relying solely on the conventional threshold detector, this project trains a Multi-Layer Perceptron (MLP) using the Adam Optimizer to classify noisy BPSK symbols transmitted through an Additive White Gaussian Noise (AWGN) channel.

The trained neural network is then compared against the classical detector using Bit Error Rate (BER).

---

## Features

- Simulates a complete BPSK communication system
- Generates random binary data
- Performs BPSK modulation
- Adds AWGN noise
- Trains a PyTorch neural network receiver
- Uses Adam optimizer
- Binary Cross Entropy loss
- Evaluates Bit Error Rate (BER)
- Compares:
  - Theoretical BER
  - Classical Threshold Detector
  - Machine Learning Receiver

---

## System Workflow

Random Bits
      ↓
BPSK Modulation
      ↓
AWGN Channel
      ↓
Received Signal
      ↓
MLP Receiver (PyTorch)
      ↓
Predicted Bits
      ↓
BER Evaluation

---

## Neural Network

Input Layer
↓
1 Feature (Received Signal)

↓

Hidden Layer
- 8 Neurons
- ReLU Activation

↓

Output Layer
- 1 Neuron
- Sigmoid Activation

Loss Function:
Binary Cross Entropy (BCELoss)

Optimizer:
Adam

---

## Hyperparameters

| Parameter | Value |
|------------|---------|
| SNR | 5 dB |
| Number of Bits | 20,000 |
| Hidden Neurons | 8 |
| Learning Rate | 0.01 |
| Epochs | 25 |
| Batch Size | 256 |
| Train/Test Split | 70/30 |

---

## Technologies Used

- Python
- PyTorch
- NumPy

---

## Dataset

The dataset consists of noisy BPSK symbols generated through simulation.

Each sample contains

- Received signal
- Corresponding transmitted bit

---

## Results

The project compares

- Theoretical BER
- Classical Threshold BER
- MLP Receiver BER

The trained MLP closely approaches the optimal detector performance under AWGN conditions.

---

## Installation

```bash
git clone https://github.com/yourusername/ML-Based-BPSK-Receiver-PyTorch.git

cd ML-Based-BPSK-Receiver-PyTorch

pip install -r requirements.txt
```

Run

```bash
python bpsk_receiver.py
```

---

## Future Improvements

- CNN Receiver
- LSTM Receiver
- Deep Neural Receiver
- QPSK Extension
- OFDM Receiver
- Rayleigh Fading Channel
- GPU Training
- BER vs SNR Plot

---

## Author

Sameer Gaonkar

Electronics and Communication Engineering

National Institute of Technology Goa
