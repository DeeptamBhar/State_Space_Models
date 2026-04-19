# Mamba S6 — HiPPO vs Mamba Analysis & Regularisation

A research codebase for training Mamba SSM models, extracting their internal state-space matrices, comparing them against HiPPO bases, and training with HiPPO-regularised loss.

---

## Files

| File | Description |
|---|---|
| `mamba_clean.py` | Train Mamba on Selective Copying and Induction Heads tasks. Extracts and saves SSM matrices (A, B, C, delta) per layer after training. |
| `mamba_extraction_dna.py` | Train Mamba on DNA sequence data (LongSafari/open-genome). Extracts and saves SSM matrices to `.npz` files. |
| `mamba_analysis_clean.py` | Load saved `.npz` matrix files and run full geometric analysis: Frobenius distance, subspace angles, eigenvalue spectra, HiPPO projection, operator shift, and PCA clustering. Produces 8 plots. |
| `mamba_hippo_clean.py` | Train Mamba with a HiPPO-regularised loss `L = CrossEntropy + λ * HiPPO_Loss(A)` and sweep over lambda values. Produces an accuracy-vs-lambda plot. |

---

## Pipeline

```
mamba_clean.py          →  checkpoints/*.pt
mamba_extraction_dna.py →  extracted_matrices/*.npz
                               ↓
                        mamba_analysis_clean.py  →  mamba_analysis_results/*.png
                        mamba_hippo_clean.py     →  hippo_lambda_sweep.png
```

Run the training scripts first to generate checkpoints and `.npz` files, then run the analysis script.

---

## Installation

### Requirements

- Linux (recommended)
- Python 3.9+
- NVIDIA GPU with CUDA 12.1+ (for the fused mamba_ssm kernels)
- If no NVIDIA GPU is available, set `FORCE_PURE_PYTORCH = True` at the top of any script to use the CPU-compatible pure-PyTorch S6 fallback

### Step 1 — Install PyTorch

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only:

```bash
pip install torch
```

### Step 2 — Install Mamba CUDA kernels

```bash
pip install causal-conv1d>=1.4.0 --no-build-isolation
pip install mamba-ssm --no-build-isolation
```

> These require a working CUDA compiler (`nvcc`). Skip this step and set `FORCE_PURE_PYTORCH = True` if you do not have an NVIDIA GPU.

### Step 3 — Install remaining dependencies

```bash
pip install einops numpy scipy scikit-learn matplotlib tqdm
pip install datasets transformers soundfile
```

### Full one-liner

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121 && \
pip install causal-conv1d>=1.4.0 --no-build-isolation && \
pip install mamba-ssm --no-build-isolation && \
pip install einops numpy scipy scikit-learn matplotlib tqdm datasets transformers soundfile
```

---

## Usage

### 1. Train on Selective Copying and Induction Heads + extract matrices

```bash
python mamba_clean.py
```

Outputs:
- `checkpoints/selective_copying.pt`
- `checkpoints/induction_heads.pt`
- `extracted_matrices/selective_copying.npz`
- `extracted_matrices/induction_heads.npz`

### 2. Train on DNA sequences + extract matrices

```bash
python mamba_extraction_dna.py
```

Outputs:
- `checkpoints/audio_waveform.pt`
- `extracted_matrices/dna_sequence.npz`

> Note: streams data from HuggingFace. Requires an internet connection.

### 3. Run the HiPPO analysis

```bash
python mamba_analysis_clean.py
```

Requires all three `.npz` files to exist in `extracted_matrices/`. Outputs 8 plots and a `summary.txt` to `mamba_analysis_results/`.

| Plot | Description |
|---|---|
| `01_frobenius_heatmap.png` | Frobenius distance between collapsed A matrices and each HiPPO variant |
| `02_subspace_angles.png` | Principal angles between column spaces |
| `03_eigenvalue_spectra.png` | Eigenvalue spectra of Mamba A vs HiPPO matrices |
| `04a_projection_cosine_heatmap.png` | Cosine similarity of A matrices onto HiPPO bases |
| `04b_projection_coefficients.png` | Projection coefficients per layer and dataset |
| `04c_projection_residuals.png` | Residuals after projection onto HiPPO bases |
| `05a_operator_shift.png` | Shift magnitude relative to a reference operator |
| `05b_pairwise_shift.png` | Pairwise operator shift between datasets |
| `05c_shift_hippo_alignment.png` | Alignment of shift direction with HiPPO bases |
| `06_pca_clustering.png` | PCA of all A matrices including HiPPO matrices |
| `07_collapse_method_comparison.png` | Comparison of three collapse methods |
| `08_matrix_heatmaps.png` | Visual heatmaps of collapsed A matrices vs HiPPO |

### 4. Run the HiPPO lambda sweep

```bash
python mamba_hippo_clean.py
```

Outputs:
- `hippo_lambda_sweep.png` — accuracy vs lambda plot
- `hippo_lambda_results.json` — raw numbers

---

## Loading Saved Matrices

```python
import numpy as np

data = np.load("extracted_matrices/selective_copying.npz")

A_log  = data["layer_0__A_log"]    # shape: [d_inner, d_state]
A_cont = data["layer_0__A_cont"]   # shape: [d_inner, d_state]
D      = data["layer_0__D"]        # shape: [d_inner]
delta  = data["layer_0__delta"]    # shape: [batch, seq_len, d_inner]
B      = data["layer_0__B"]        # shape: [batch, seq_len, d_state]
C      = data["layer_0__C"]        # shape: [batch, seq_len, d_state]
A_bar  = data["layer_0__A_bar"]    # shape: [batch, seq_len, d_inner, d_state]
B_bar  = data["layer_0__B_bar"]    # shape: [batch, seq_len, d_inner, d_state]
```

---

## HiPPO Matrices

Three HiPPO variants are used in the analysis, all constructed at size 16x16:

| Name | Measure | Description |
|---|---|---|
| `LegS` | Shifted Legendre | Sliding window history |
| `LegT` | Translated Legendre | Fixed-length window |
| `LagT` | Translated Laguerre | Exponential decay memory |

---

## HiPPO Loss

The regularisation loss used in `mamba_hippo_clean.py` encourages Mamba's diagonal SSM matrix to stay close to the HiPPO-LagT diagonal:

```
hippo_target[n] = -1  for all n

L_hippo = mean over layers of MSE(A_cont, hippo_target)
L_total = CrossEntropy + lambda * L_hippo
```

Lambda values swept: `[0.00001,0.0001,0.01, 0.1, 1.0, 10.0]`

---

## Reference

Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752.  
Official repo: [https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)

Gu, A., Johnson, I., Goel, K., Saab, K., Dao, T., Rudra, A. & Ré, C. (2021). *HiPPO: Recurrent Memory with Optimal Polynomial Projections*. NeurIPS 2021. arXiv:2008.07669.  
Official repo: [https://github.com/HazyResearch/hippo](https://github.com/HazyResearch/hippo)
