"""
=============================================================================
  BPSK Communication System with MLP Adaptive Receiver — PyTorch Simulation
=============================================================================
Author  : ML Engineer / Comms Systems Scientist
Purpose : Simulate an AWGN-corrupted BPSK channel and train a Multi-Layer
          Perceptron (MLP) to act as an adaptive receiver/detector.  Results
          are compared against a classical hard-decision threshold detector.

Theory background
-----------------
Binary Phase Shift Keying (BPSK) maps binary bits to antipodal symbols:
    bit 1  →  s = +1
    bit 0  →  s = -1

The Additive White Gaussian Noise (AWGN) channel model is:
    r = s + n,   n ~ N(0, σ²)

where the noise variance is chosen from the user-specified SNR (dB):
    σ² = 1 / (2 · 10^(SNR_dB / 10))

The theoretical BER for BPSK over AWGN is:
    BER_theory = Q(√(2 · Eb/N0)) = Q(√(2 · 10^(SNR_dB/10)))

A classical detector simply decides:
    r > 0  →  bit 1
    r ≤ 0  →  bit 0

The MLP learns a soft, nonlinear decision boundary that approximates (and can
generalize beyond) the classical detector, demonstrating the power of a
neural-network-based receiver.
=============================================================================
"""

# ---------------------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------------------
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Reproducibility — fix seeds for numpy and PyTorch
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# 1. User-Configurable Hyperparameters
# ---------------------------------------------------------------------------
SNR_DB        = 5          # Signal-to-Noise Ratio in dB
NUM_BITS      = 20_000     # Total number of random bits to generate
TRAIN_RATIO   = 0.70       # Fraction of data used for training (70 / 30 split)
HIDDEN_NODES  = 8          # Number of neurons in the single hidden layer
LEARNING_RATE = 0.01       # Adam optimiser learning rate
NUM_EPOCHS    = 25         # Total training epochs
BATCH_SIZE    = 256        # Mini-batch size for the DataLoader
PRINT_EVERY   = 5          # Print training loss every N epochs

# ---------------------------------------------------------------------------
# 2. Data Generation
# ---------------------------------------------------------------------------
print("=" * 65)
print("  Step 1 — Generating BPSK dataset")
print("=" * 65)

# 2a. Random binary bits: 0 or 1 with equal probability
bits = np.random.randint(0, 2, size=NUM_BITS).astype(np.float32)
print(f"  Generated {NUM_BITS:,} random binary bits.")
print(f"  Bit distribution — 0s: {int((bits == 0).sum()):,}  "
      f"1s: {int((bits == 1).sum()):,}")

# 2b. BPSK modulation: bit 1 → +1.0,  bit 0 → -1.0
#     Using the mapping s = 2b - 1
bpsk_symbols = 2.0 * bits - 1.0          # shape: (N,)
print(f"\n  BPSK mapping applied  (bit 0 → -1, bit 1 → +1).")

# 2c. Compute noise variance from SNR (dB)
#     σ² = 1 / (2 · 10^(SNR_dB / 10))
snr_linear    = 10 ** (SNR_DB / 10.0)    # linear SNR (Eb/N0)
noise_variance = 1.0 / (2.0 * snr_linear)
noise_std      = math.sqrt(noise_variance)

print(f"\n  SNR           = {SNR_DB} dB")
print(f"  SNR (linear)  = {snr_linear:.4f}")
print(f"  Noise σ²      = {noise_variance:.6f}")
print(f"  Noise σ       = {noise_std:.6f}")

# 2d. AWGN channel: add Gaussian noise to BPSK symbols
noise           = np.random.normal(loc=0.0, scale=noise_std,
                                   size=NUM_BITS).astype(np.float32)
received_signals = bpsk_symbols + noise   # r = s + n,  shape: (N,)
print(f"\n  AWGN channel applied  — noisy received_signals generated.")

# ---------------------------------------------------------------------------
# 3. Train / Test Split  (70% / 30%)
# ---------------------------------------------------------------------------
split_idx = int(NUM_BITS * TRAIN_RATIO)

# --- Training set ---
X_train_np = received_signals[:split_idx].reshape(-1, 1)   # shape: (14000, 1)
y_train_np = bits[:split_idx].reshape(-1, 1)               # shape: (14000, 1)

# --- Test set ---
X_test_np  = received_signals[split_idx:].reshape(-1, 1)   # shape: (6000, 1)
y_test_np  = bits[split_idx:].reshape(-1, 1)               # shape: (6000, 1)

print("\n" + "=" * 65)
print("  Step 2 — Train/Test split")
print("=" * 65)
print(f"  Training samples : {X_train_np.shape[0]:,}")
print(f"  Test samples     : {X_test_np.shape[0]:,}")

# --- Convert numpy arrays to PyTorch Tensors ---
X_train = torch.from_numpy(X_train_np)   # dtype: torch.float32
y_train = torch.from_numpy(y_train_np)
X_test  = torch.from_numpy(X_test_np)
y_test  = torch.from_numpy(y_test_np)

# Wrap in TensorDataset & DataLoader for mini-batch iteration
train_dataset = TensorDataset(X_train, y_train)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"  Tensors created — X_train: {tuple(X_train.shape)}, "
      f"y_train: {tuple(y_train.shape)}")
print(f"  X_test : {tuple(X_test.shape)},  y_test : {tuple(y_test.shape)}")

# ---------------------------------------------------------------------------
# 4. MLP Model Definition
# ---------------------------------------------------------------------------
class BPSKReceiverMLP(nn.Module):
    """
    A lightweight Multi-Layer Perceptron acting as an adaptive BPSK detector.

    Architecture
    ────────────
    Input  (1)  →  Linear(1→8)  →  ReLU  →  Linear(8→1)  →  Sigmoid  →  Output (1)

    The Sigmoid output is interpreted as P(bit == 1 | received signal r).
    A hard decision is made at the 0.5 threshold during inference.

    Parameters
    ──────────
    hidden_nodes : int
        Number of neurons in the single hidden layer (default 8).
    """

    def __init__(self, hidden_nodes: int = HIDDEN_NODES):
        super(BPSKReceiverMLP, self).__init__()

        # --- Hidden layer: Linear transformation + ReLU ---
        self.hidden = nn.Linear(in_features=1, out_features=hidden_nodes)
        self.relu   = nn.ReLU()

        # --- Output layer: Linear transformation + Sigmoid ---
        self.output  = nn.Linear(in_features=hidden_nodes, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ──────────
        x : torch.Tensor, shape (batch_size, 1)
            Noisy received signal values.

        Returns
        ───────
        torch.Tensor, shape (batch_size, 1)
            Probability that the transmitted bit was 1.
        """
        x = self.hidden(x)      # Linear: (batch, 1) → (batch, 8)
        x = self.relu(x)        # ReLU activation
        x = self.output(x)      # Linear: (batch, 8) → (batch, 1)
        x = self.sigmoid(x)     # Sigmoid squashes to (0, 1)
        return x


# Instantiate the model
model = BPSKReceiverMLP(hidden_nodes=HIDDEN_NODES)

print("\n" + "=" * 65)
print("  Step 3 — Model Architecture")
print("=" * 65)
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Trainable parameters : {total_params}")

# ---------------------------------------------------------------------------
# 5. Loss Function & Optimiser
# ---------------------------------------------------------------------------
# BCELoss is appropriate for binary classification with Sigmoid output.
# It computes:  L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
criterion = nn.BCELoss()

# Adam combines adaptive learning rates with momentum.
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---------------------------------------------------------------------------
# 6. Training Loop
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print(f"  Step 4 — Training  ({NUM_EPOCHS} epochs, batch size {BATCH_SIZE})")
print("=" * 65)

model.train()   # Set model to training mode (enables Dropout/BN if present)

for epoch in range(1, NUM_EPOCHS + 1):

    epoch_loss  = 0.0
    num_batches = 0

    for X_batch, y_batch in train_loader:

        # ── 6a. Zero accumulated gradients from the previous step ──────────
        optimizer.zero_grad()

        # ── 6b. Forward pass: compute predicted probabilities ──────────────
        y_pred = model(X_batch)          # shape: (batch_size, 1)

        # ── 6c. Compute Binary Cross-Entropy loss ──────────────────────────
        loss = criterion(y_pred, y_batch)

        # ── 6d. Backward pass: compute gradients via auto-differentiation ──
        loss.backward()

        # ── 6e. Update model parameters using computed gradients ───────────
        optimizer.step()

        epoch_loss  += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches

    # Print progress every PRINT_EVERY epochs and on the final epoch
    if epoch % PRINT_EVERY == 0 or epoch == NUM_EPOCHS:
        print(f"  Epoch [{epoch:>3}/{NUM_EPOCHS}]  |  "
              f"Avg BCE Loss : {avg_loss:.6f}")

# ---------------------------------------------------------------------------
# 7. Evaluation
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  Step 5 — Evaluation on Test Set")
print("=" * 65)

model.eval()   # Disable training-specific behaviour (e.g., Dropout)

with torch.no_grad():   # No need to track gradients during inference
    # --- MLP predictions ---
    y_prob_test  = model(X_test)                       # shape: (N_test, 1), floats in (0,1)
    # Hard decision: ≥ 0.5 → bit 1,  < 0.5 → bit 0
    y_pred_test  = (y_prob_test >= 0.5).float()        # shape: (N_test, 1)

# --- MLP Bit Error Rate ---
mlp_errors = (y_pred_test != y_test).float().sum().item()
mlp_ber    = mlp_errors / X_test.shape[0]

# ---------------------------------------------------------------------------
# 8. Classical Threshold Detector (Baseline)
# ---------------------------------------------------------------------------
# The optimal AWGN detector for BPSK is:
#   r > 0  →  decide bit 1
#   r ≤ 0  →  decide bit 0
classical_pred = (X_test > 0.0).float()   # shape: (N_test, 1)
classical_errors = (classical_pred != y_test).float().sum().item()
classical_ber    = classical_errors / X_test.shape[0]

# ---------------------------------------------------------------------------
# 9. Theoretical BER (for reference)
# ---------------------------------------------------------------------------
# BER_theory = Q(sqrt(2 * Eb/N0))   where Q(x) = 0.5 * erfc(x / sqrt(2))
# In Python: scipy.special.erfc, or use math.erfc
import math
q_arg       = math.sqrt(2.0 * snr_linear)
ber_theory  = 0.5 * math.erfc(q_arg / math.sqrt(2.0))

# ---------------------------------------------------------------------------
# 10. Results Summary
# ---------------------------------------------------------------------------
print(f"\n  SNR                            : {SNR_DB} dB")
print(f"  Test samples                   : {X_test.shape[0]:,}")
print()
print(f"  ┌─────────────────────────────────────────────────────┐")
print(f"  │  Detector                     │   BER               │")
print(f"  ├─────────────────────────────────────────────────────┤")
print(f"  │  Theoretical BPSK (Q-func)    │  {ber_theory:.6f}           │")
print(f"  │  Classical Threshold (r > 0)  │  {classical_ber:.6f}           │")
print(f"  │  MLP Adaptive Receiver        │  {mlp_ber:.6f}           │")
print(f"  └─────────────────────────────────────────────────────┘")

improvement = classical_ber - mlp_ber
if improvement > 0:
    print(f"\n  ✔  MLP improves over classical detector by "
          f"{improvement:.6f} BER points.")
elif improvement == 0:
    print(f"\n  ✔  MLP matches the classical detector exactly.")
else:
    print(f"\n  ℹ  Classical detector outperforms MLP by "
          f"{-improvement:.6f} BER points at this SNR.")

print("\n  Note: At high SNR the classical threshold is already near-optimal,")
print("  so MLP and classical BER values converge toward the theoretical BER.")
print("=" * 65)
print("  Simulation complete.")
print("=" * 65)
