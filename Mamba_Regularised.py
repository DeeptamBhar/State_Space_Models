import os
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from einops import rearrange

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

FORCE_PURE_PYTORCH: bool = False

def _check_kernels() -> bool:
    if FORCE_PURE_PYTORCH:
        return False
    try:
        from mamba_ssm import Mamba as _
        if not torch.cuda.is_available():
            warnings.warn("mamba_ssm installed but no CUDA — falling back to pure-PyTorch S6.")
            return False
        return True
    except ImportError:
        warnings.warn("mamba_ssm not found — falling back to pure-PyTorch S6. Install: pip install mamba-ssm --no-build-isolation")
        return False

USE_OFFICIAL_KERNEL: bool = _check_kernels()
print(f"[kernel] {'mamba_ssm CUDA fused kernels' if USE_OFFICIAL_KERNEL else 'pure-PyTorch S6 fallback'}")


class _PurePytorchS6(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_min=0.001, dt_max=0.1, conv_bias=True, bias=False, layer_idx=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.d_conv = d_conv
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, bias=conv_bias)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.dt_proj.bias._no_reinit = True
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, hidden_states, inference_params=None):
        B_sz, L, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        x_c = self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_c = F.silu(x_c)
        x_dbl = self.x_proj(rearrange(x_c, "b l d -> (b l) d"))
        dt_r, B_raw, C_raw = x_dbl.split([self.dt_rank, self.d_state, self.d_state], -1)
        dt = F.softplus(self.dt_proj(dt_r))
        dt = rearrange(dt, "(b l) d -> b l d", b=B_sz)
        B_raw = rearrange(B_raw, "(b l) n -> b l n", b=B_sz)
        C_raw = rearrange(C_raw, "(b l) n -> b l n", b=B_sz)
        A_cont = -torch.exp(self.A_log.float())
        dA = torch.einsum("b l d, d n -> b l d n", dt, A_cont)
        A_bar = torch.exp(dA)
        B_bar = torch.einsum("b l d, b l n -> b l d n", dt, B_raw)
        h = torch.zeros(B_sz, self.d_inner, self.d_state, device=hidden_states.device, dtype=hidden_states.dtype)
        ys = []
        for t in range(L):
            h = A_bar[:, t] * h + B_bar[:, t] * x_c[:, t, :, None]
            ys.append((h * C_raw[:, t, None, :]).sum(-1))
        y = torch.stack(ys, 1)
        y = y + x_c * self.D
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, layer_idx=None):
        super().__init__()
        if USE_OFFICIAL_KERNEL:
            from mamba_ssm import Mamba
            self.mixer = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, use_fast_path=True, layer_idx=layer_idx)
        else:
            self.mixer = _PurePytorchS6(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, layer_idx=layer_idx)

    def forward(self, hidden_states, inference_params=None):
        return self.mixer(hidden_states, inference_params)

    def get_A_cont(self) -> torch.Tensor:
        return -torch.exp(self.mixer.A_log.float())


def hippo_diagonal(d_state: int, device: torch.device) -> torch.Tensor:
    return -torch.ones(d_state, dtype=torch.float32, device=device)


def hippo_loss(model: "MambaLMModel") -> torch.Tensor:
    device = next(model.parameters()).device
    layer_losses = []
    for layer in model.layers:
        A_cont = layer.mixer.get_A_cont()
        d_state = A_cont.shape[1]
        hippo_tgt = hippo_diagonal(d_state, device)
        mse = ((A_cont - hippo_tgt.unsqueeze(0)) ** 2).mean()
        layer_losses.append(mse)
    return torch.stack(layer_losses).mean()


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (self.weight * x * torch.rsqrt(v + self.eps)).to(x.dtype)


class MambaResidualBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, layer_idx=None):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand, layer_idx=layer_idx)

    def forward(self, x, residual=None):
        res = (x + residual).to(torch.float32) if residual is not None else x.float()
        h = self.mixer(self.norm(res.to(self.norm.weight.dtype)))
        return h, res


class MambaLMModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if vocab_size % 8:
            vocab_size = (vocab_size // 8 + 1) * 8
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([MambaResidualBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand, layer_idx=i) for i in range(n_layers)])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        x, res = self.embedding(input_ids), None
        for layer in self.layers:
            x, res = layer(x, res)
        x = self.norm_f((x + res).to(self.norm_f.weight.dtype))
        return self.lm_head(x)


SCALE_FACTOR = 50

@dataclass
class ExperimentConfig:
    name: str
    task: str
    d_model: int = 64
    n_layers: int = 2
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    vocab_size: int = 20
    seq_len: int = 256
    batch_size: int = 32
    lr: float = 8e-3
    weight_decay: float = 0.01
    epochs: int = 20
    warmup_steps: int = 100
    n_train_samples: int = 4000
    n_val_samples: int = 800


EXPERIMENTS: List[ExperimentConfig] = [
    ExperimentConfig(name="dna_sequence", task="dna", d_model=128, n_layers=4, d_state=16, seq_len=512, vocab_size=5, batch_size=16, lr=5e-4, epochs=20, n_train_samples=2000 * 20, n_val_samples=400 * 20),
]

LAMBDAS: List[float] = [1e-5,1e-4,1e-2, 1e-1, 1.0, 10.0]


class SelectiveCopyDataset(Dataset):
    def __init__(self, n_samples: int, seq_len: int, vocab_size: int, n_selective: int = 10, seed: int = 0):
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(n_samples):
            positions = torch.randperm(seq_len - 1, generator=rng)[:n_selective].sort().values
            tokens = torch.randint(2, vocab_size, (n_selective,), generator=rng)
            src = torch.zeros(seq_len, dtype=torch.long)
            for pos, tok in zip(positions.tolist(), tokens.tolist()):
                src[pos] = tok
            tgt = src.clone()
            tgt[:-1] = src[1:]
            tgt[-1] = 0
            self.samples.append((src, tgt))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


class DNADataset(Dataset):
    DNA_VOCAB = {c: i for i, c in enumerate("ACGTN")}

    def __init__(self, seq_len: int, n_samples: int, seed: int = 0):
        from datasets import load_dataset
        from tqdm import tqdm
        print("  [DNA] Streaming LongSafari/open-genome (stage1) ...")
        ds = load_dataset("LongSafari/open-genome", "stage1", split="train", streaming=True)
        self.items: List[torch.Tensor] = []
        needed = seq_len + 1
        for item in tqdm(ds.take(n_samples * 4), total=n_samples * 4, desc="  DNA"):
            raw = item["text"]
            seq = raw.split("|", 1)[1].upper() if "|" in raw else raw.upper()
            ids = [self.DNA_VOCAB.get(c, 4) for c in seq]
            if len(ids) < needed // 2:
                continue
            if len(ids) < needed:
                ids += [4] * (needed - len(ids))
            ids = ids[:needed]
            self.items.append(torch.tensor(ids, dtype=torch.long))
            if len(self.items) >= n_samples:
                break
        if not self.items:
            raise RuntimeError("[DNA] No samples loaded — check internet.")
        print(f"  [DNA] Loaded {len(self.items)} samples.")

    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        t = self.items[i]
        return t[:-1], t[1:]


class AudioDataset(Dataset):
    def __init__(self, seq_len: int, n_samples: int, seed: int = 0):
        from datasets import load_dataset, Audio as HFAudio
        from tqdm import tqdm
        print("  [Audio] Streaming google/speech_commands v0.01 ...")
        ds = load_dataset("google/speech_commands", split="train", streaming=True, revision="refs/convert/parquet").cast_column("audio", HFAudio(sampling_rate=16000))
        self.items: List[torch.Tensor] = []
        needed = seq_len + 1
        for item in tqdm(ds.take(n_samples * 3), total=n_samples * 3, desc="  Audio"):
            tokens = self._mu_law(item["audio"]["array"])
            if len(tokens) < needed:
                tokens = F.pad(tokens, (0, needed - len(tokens)))
            tokens = tokens[:needed]
            self.items.append(tokens)
            if len(self.items) >= n_samples:
                break
        if not self.items:
            raise RuntimeError("[Audio] No samples — check internet / soundfile.")
        print(f"  [Audio] Loaded {len(self.items)} samples.")

    @staticmethod
    def _mu_law(audio, mu: int = 255) -> torch.Tensor:
        x = torch.tensor(audio, dtype=torch.float32).clamp(-1.0, 1.0)
        enc = torch.sign(x) * torch.log1p(mu * x.abs()) / math.log(1 + mu)
        return ((enc + 1.0) / 2.0 * mu + 0.5).long().clamp(0, mu)

    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        t = self.items[i]
        return t[:-1], t[1:]


_DATASET_CACHE: Dict[str, Tuple[Dataset, Dataset]] = {}

def get_train_val_datasets(cfg: ExperimentConfig) -> Tuple[Dataset, Dataset]:
    key = cfg.task
    if key in _DATASET_CACHE:
        return _DATASET_CACHE[key]
    if cfg.task == "selective_copy":
        train_ds = SelectiveCopyDataset(cfg.n_train_samples, cfg.seq_len, cfg.vocab_size, seed=0)
        val_ds = SelectiveCopyDataset(cfg.n_val_samples, cfg.seq_len, cfg.vocab_size, seed=42)
    elif cfg.task == "dna":
        full = DNADataset(cfg.seq_len, cfg.n_train_samples + cfg.n_val_samples)
        train_ds = torch.utils.data.Subset(full, range(cfg.n_train_samples))
        val_ds = torch.utils.data.Subset(full, range(cfg.n_train_samples, len(full)))
    elif cfg.task == "audio":
        full = AudioDataset(cfg.seq_len, cfg.n_train_samples + cfg.n_val_samples)
        train_ds = torch.utils.data.Subset(full, range(cfg.n_train_samples))
        val_ds = torch.utils.data.Subset(full, range(cfg.n_train_samples, len(full)))
    else:
        raise ValueError(f"Unknown task: {cfg.task}")
    _DATASET_CACHE[key] = (train_ds, val_ds)
    return train_ds, val_ds


def train_and_eval(cfg: ExperimentConfig, lam: float, device: torch.device, dtype: torch.dtype) -> float:
    print(f"  lam={lam:.0e}  ", end="", flush=True)
    model = MambaLMModel(vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_layers=cfg.n_layers, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand).to(device=device, dtype=dtype)
    train_ds, val_ds = get_train_val_datasets(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=(str(device) == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=2)
    decay = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
    no_decay = [p for p in model.parameters() if p.ndim < 2 and p.requires_grad]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr, betas=(0.9, 0.95))
    total_steps = cfg.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=max(total_steps, 1), pct_start=min(cfg.warmup_steps / max(total_steps, 1), 0.3))
    for epoch in range(cfg.epochs):
        model.train()
        running_ce = running_hippo = 0.0
        steps = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16, enabled=(str(device).startswith("cuda"))):
                logits = model(x)
                V = logits.size(-1)
                ce = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
                h_loss = hippo_loss(model)
                loss = ce + lam * h_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            scheduler.step()
            running_ce += ce.item()
            running_hippo += h_loss.item()
            steps += 1
        avg_ce = running_ce / steps
        avg_hippo = running_hippo / steps
        print(f"ep{epoch+1}: CE={avg_ce:.3f} H={avg_hippo:.3f}  ", end="", flush=True)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.numel()
    acc = correct / total * 100.0
    print(f"-> val_acc={acc:.2f}%")
    return acc


def plot_results(results: Dict[str, List[float]], lambdas: List[float], out_path: str = "hippo_lambda_sweep.png"):
    COLORS = {"selective_copy": "#F4A261", "dna_sequence": "#2EC4B6", "audio_waveform": "#E07BE0"}
    LABELS = {"selective_copy": "Selective Copy  (S4.1)", "dna_sequence": "DNA  (S4.3, hg38)", "audio_waveform": "Audio  (S4.4, SC09)"}
    MARKERS = {"selective_copy": "o", "dna_sequence": "s", "audio_waveform": "D"}
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")
    x_pos = np.arange(len(lambdas))
    x_labels = []
    for l in lambdas:
        exp = math.log10(l)
        if exp == int(exp):
            e = int(exp)
            if e == 0:
                x_labels.append("1")
            elif e == 1:
                x_labels.append("10")
            else:
                x_labels.append(f"$10^{{{e}}}$")
        else:
            x_labels.append(f"{l:.0e}")
    for name, accs in results.items():
        color = COLORS.get(name, "#FFFFFF")
        label = LABELS.get(name, name)
        marker = MARKERS.get(name, "o")
        ax.plot(x_pos, accs, color=color, marker=marker, markersize=8, linewidth=2.2, label=label, zorder=3)
        best_idx = int(np.argmax(accs))
        ax.annotate(f"{accs[best_idx]:.1f}%", xy=(x_pos[best_idx], accs[best_idx]), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8, color=color, fontweight="bold", arrowprops=dict(arrowstyle="-", color=color, lw=1))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=11, color="#C9D1D9")
    ax.set_xlabel("HiPPO Regularisation  lam", fontsize=13, color="#C9D1D9", labelpad=8)
    ax.set_ylabel("Next-Token Accuracy  (%)", fontsize=13, color="#C9D1D9", labelpad=8)
    ax.set_title("Mamba x HiPPO Loss  -  lam Sweep", fontsize=16, color="#F0F6FC", pad=14, fontweight="bold")
    ax.tick_params(colors="#8B949E", which="both")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.grid(which="major", color="#21262D", linewidth=1.0, zorder=0)
    ax.grid(which="minor", color="#161B22", linewidth=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")
    leg = ax.legend(loc="best", framealpha=0.25, facecolor="#0D1117", edgecolor="#30363D", fontsize=10, labelcolor="linecolor")
    for text in leg.get_texts():
        text.set_color("#C9D1D9")
    fig.tight_layout(pad=1.5)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n  Plot saved -> {out_path}")
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32)
    print(f"\n[device] {device}  dtype={dtype}\n")
    results: Dict[str, List[float]] = {cfg.name: [] for cfg in EXPERIMENTS}
    for cfg in EXPERIMENTS:
        print(f"\n{'='*68}")
        print(f"  DATASET: {cfg.name.upper()}  (d_model={cfg.d_model}, n_layers={cfg.n_layers}, d_state={cfg.d_state})")
        print(f"  seq_len={cfg.seq_len}  vocab={cfg.vocab_size}  train={cfg.n_train_samples}  val={cfg.n_val_samples}")
        print(f"{'='*68}")
        print("  Loading dataset ...")
        get_train_val_datasets(cfg)
        for lam in LAMBDAS:
            acc = train_and_eval(cfg, lam, device, dtype)
            results[cfg.name].append(acc)
    print(f"\n{'='*68}")
    print("  RESULTS  (next-token accuracy  %)")
    print(f"  {'lam':<12}", end="")
    for cfg in EXPERIMENTS:
        print(f"  {cfg.name:<22}", end="")
    print()
    print("  " + "-" * (12 + 24 * len(EXPERIMENTS)))
    for i, lam in enumerate(LAMBDAS):
        print(f"  {lam:<12.0e}", end="")
        for cfg in EXPERIMENTS:
            print(f"  {results[cfg.name][i]:<22.2f}", end="")
        print()
    print(f"{'='*68}")
    out_png = "hippo_lambda_sweep.png"
    plot_results(results, LAMBDAS, out_path=out_png)
    import json
    with open("hippo_lambda_results.json", "w") as f:
        json.dump({"lambdas": LAMBDAS, "results": results}, f, indent=2)
    print("  Raw numbers -> hippo_lambda_results.json")
    print(f"\n{'='*68}")
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()