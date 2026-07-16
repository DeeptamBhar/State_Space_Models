# State Space Models

A research repository dedicated to the systematic study of **State Space Models (SSMs)** — from the HiPPO framework for continuous-time memory to the selective-scan **Mamba (S6)** architecture. The project spans theoretical analysis (do trained Mamba models learn HiPPO-like dynamics?), controlled benchmarks (SSMs vs. convolutional models), chaotic dynamical systems, and real-world edge deployment on embedded hardware.

📑 **Project presentation:** [SSM_Presentation.pdf](./SSM_Presentation.pdf)

---

## 1. Introduction

State Space Models are a class of sequence models that evolve a hidden state through a linear dynamical system:

```
h'(t) = A h(t) + B x(t)
y(t)  = C h(t) + D x(t)
```

The **HiPPO** framework showed that specific structured choices of the `A` matrix allow the state to optimally memorize the history of the input. **Mamba (S6)** made the state-space parameters input-dependent (selective) and achieved linear-time sequence modeling competitive with Transformers.

This repository investigates SSMs from four complementary angles, each developed on its own branch. **All branches are preserved**; `main` aggregates every experiment in its own directory.

---

## 2. Repository Structure

| Directory on `main` | Branch | Focus |
|---|---|---|
| [`chaos/`](./chaos) | [`chaos`](../../tree/chaos) | Modeling the chaotic Lorenz-63 system with a HiPPO-based SSM |
| [`HiPPO-vs-Mamba/`](./HiPPO-vs-Mamba) | [`HiPPO-vs-Mamba`](../../tree/HiPPO-vs-Mamba) | Do trained Mamba models converge to HiPPO dynamics? Matrix extraction, geometric analysis & HiPPO-regularised training |
| [`SSM_vs_CNN/`](./SSM_vs_CNN) | [`SSM_vs_CNN`](../../tree/SSM_vs_CNN) | Mamba vs. TCN/CNN benchmarks: sCIFAR-10, Speech Commands, HG38 genomics, DPDD image deblurring |
| [`cnn-mamba/`](./cnn-mamba) | [`cnn-mamba`](../../tree/cnn-mamba) | Hybrid CNN→Mamba video action recognition deployed on Jetson Orin Nano |

---

## 3. Experiments

### 3.1 Chaos — SSMs on the Lorenz-63 Attractor (`chaos/`)

Can an SSM learn and reproduce chaotic continuous-time dynamics?

- Data: 100,000 steps of the Lorenz-63 system integrated with 4th-order Runge-Kutta
- A custom `HiPPOCell` (normalized HiPPO transition matrix) is trained on sliding windows (L = 50, 80/20 train/val split)
- Evaluation: stateful autoregressive rollout reconstructs the 3D chaotic attractor

Key artifacts: `chaos.ipynb` (full pipeline), `images/` (ground-truth vs. predicted attractors, learned A-matrix heatmap, gradient stability plots).

### 3.2 HiPPO vs. Mamba — Does Mamba Learn HiPPO? (`HiPPO-vs-Mamba/`)

A study of the internal state-space matrices of trained Mamba models.

- **Matrix extraction**: Mamba models are trained on Selective Copying, Induction Heads (synthetic) and DNA sequences (LongSafari/open-genome); per-layer `A`, `B`, `C`, `Δ`, `A_bar`, `B_bar` are extracted to `.npz`
- **Geometric analysis** (`Matrix_Analysis.py`): Frobenius distance, principal subspace angles, eigenvalue spectra, projection onto HiPPO bases (LegS / LegT / LagT), operator-shift analysis and PCA clustering — 12 diagnostic plots
- **HiPPO-regularised training** (`Mamba_Regularised.py`): `L = CrossEntropy + λ · MSE(A_cont, A_HiPPO)` with a λ sweep from 1e-5 to 10

All scripts run with official CUDA `mamba_ssm` kernels or fall back to a pure-PyTorch S6 implementation (`FORCE_PURE_PYTORCH = True`) on CPU. See the branch README inside the folder for the full pipeline and usage.

### 3.3 SSM vs. CNN — Controlled Benchmarks (`SSM_vs_CNN/`)

Head-to-head comparisons of PureMamba, PureTCN and hybrid parallel Mamba+TCN architectures across four domains:

- **Sequential CIFAR-10** (`CIFAR-10_task/`): pixel-by-pixel image classification
- **Speech Commands** (`SpeechCommand_task/`): raw-waveform keyword spotting
- **HG38 genomics** (`HG_38_task/`): next-token prediction on the human genome, sweeping sequence length × depth
- **DPDD image deblurring** (`DPDD_Deblurring_task/`): five architecture variants — baseline, advanced, two expanded, and SE-style gated fusion of global (Mamba) and local (depthwise conv) branches

Includes `SSM_ppt.pdf` with results and `check_mamba.py` for environment sanity checks.

### 3.4 CNN–Mamba Hybrid on the Edge (`cnn-mamba/`)

A deployable hybrid for video action recognition (UCF-101) on the **NVIDIA Jetson Orin Nano**:

- **`TruncatedMobileNetV3`**: frozen, pre-trained CNN sliced at a configurable *handoff index* extracts per-frame spatial features
- **`LightweightMambaHead`** (d_model = 128): models the temporal sequence of frame features and classifies
- **Handoff sweep**: depths 2–12 trade CNN spatial richness against Mamba temporal load
- **On-device measurements**: live thermal logs at 7 W and 15 W power modes, per-class metrics, confusion matrices, and macro-vs-micro accuracy analysis (`jetson_scripts/`, `results/`)

Trained checkpoints for every handoff depth are included in `checkpoints/`.

---

## 4. Key Takeaways

- SSMs can faithfully reconstruct chaotic attractors, capturing continuous-time dependencies where discrete models struggle.
- Trained Mamba A-matrices can be quantitatively compared against HiPPO bases, and HiPPO-regularisation offers a controllable prior on learned dynamics.
- Mamba is competitive with — and often complementary to — convolutional models; hybrid parallel Mamba+CNN blocks combine global context with local inductive bias.
- CNN→Mamba handoff architectures run efficiently within edge power/thermal envelopes, making SSMs practical on embedded hardware.

---

## 5. Getting Started

Each directory is self-contained with its own scripts (and README where applicable). Common setup:

```bash
# PyTorch (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Mamba CUDA kernels (requires nvcc; optional — pure-PyTorch fallback available)
pip install causal-conv1d>=1.4.0 --no-build-isolation
pip install mamba-ssm --no-build-isolation

# Common dependencies
pip install einops numpy scipy scikit-learn matplotlib tqdm datasets transformers soundfile torchvision torchaudio
```

To work on a specific experiment branch directly:

```bash
git clone https://github.com/DeeptamBhar/State_Space_Models.git
cd State_Space_Models
git checkout <branch-name>   # chaos | HiPPO-vs-Mamba | SSM_vs_CNN | cnn-mamba
```

## Datasets

- Lorenz-63 (synthetic, RK4-generated)
- Selective Copying & Induction Heads (synthetic)
- LongSafari/open-genome & HG38 (DNA)
- CIFAR-10, Speech Commands, DPDD, UCF-101

## References

1. Gu, A. & Dao, T. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023. https://arxiv.org/abs/2312.00752
2. Gu, A., Johnson, I., Goel, K., Saab, K., Dao, T., Rudra, A. & Ré, C. *HiPPO: Recurrent Memory with Optimal Polynomial Projections*, NeurIPS 2020. https://arxiv.org/abs/2008.07669
3. Gu, A., Goel, K. & Ré, C. *Efficiently Modeling Long Sequences with Structured State Spaces (S4)*, ICLR 2022. https://arxiv.org/abs/2111.00396
4. Lorenz, E. N. *Deterministic Nonperiodic Flow*, Journal of the Atmospheric Sciences, 1963.

## Author

- [Deeptam Bhar](https://github.com/DeeptamBhar)

## License

Released under the [MIT License](./LICENSE).
