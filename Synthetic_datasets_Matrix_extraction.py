import os
import math
import json
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from einops import rearrange

FORCE_PURE_PYTORCH: bool = False

def _check_kernels() -> bool:
    if FORCE_PURE_PYTORCH:
        return False
    try:
        from mamba_ssm import Mamba as _MambaCheck
        import torch
        if not torch.cuda.is_available():
            warnings.warn("mamba_ssm installed but no CUDA device found. Falling back to pure-PyTorch S6.", stacklevel=2)
            return False
        return True
    except ImportError:
        warnings.warn("mamba_ssm not installed. Falling back to pure-PyTorch S6. Install with: pip install mamba-ssm --no-build-isolation", stacklevel=2)
        return False

USE_OFFICIAL_KERNEL: bool = _check_kernels()
print(f"[kernel] Using {'official mamba_ssm CUDA kernels' if USE_OFFICIAL_KERNEL else 'pure-PyTorch S6 fallback'}")


class _PurePytorchS6(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1, conv_bias=True, bias=False, layer_idx=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, bias=conv_bias)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.last_A_bar: Optional[torch.Tensor] = None
        self.last_B_bar: Optional[torch.Tensor] = None
        self.last_C: Optional[torch.Tensor] = None
        self.last_delta: Optional[torch.Tensor] = None

    def forward(self, hidden_states, inference_params=None):
        B, L, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        x_dbl = self.x_proj(rearrange(x_conv, "b l d -> (b l) d"))
        dt_r, B_in, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], -1)
        dt = F.softplus(self.dt_proj(dt_r))
        dt = rearrange(dt, "(b l) d -> b l d", b=B)
        B_in = rearrange(B_in, "(b l) n -> b l n", b=B)
        C = rearrange(C, "(b l) n -> b l n", b=B)
        A_cont = -torch.exp(self.A_log.float())
        dA = torch.einsum("b l d, d n -> b l d n", dt, A_cont)
        A_bar = torch.exp(dA)
        B_bar = torch.einsum("b l d, b l n -> b l d n", dt, B_in)
        h = torch.zeros(B, self.d_inner, self.d_state, device=hidden_states.device, dtype=hidden_states.dtype)
        ys = []
        for t in range(L):
            h = A_bar[:, t] * h + B_bar[:, t] * x_conv[:, t, :, None]
            yt = (h * C[:, t, None, :]).sum(-1)
            ys.append(yt)
        y = torch.stack(ys, 1)
        y = y + x_conv * self.D
        y = y * F.silu(z)
        self.last_A_bar = A_bar.detach().float().cpu()
        self.last_B_bar = B_bar.detach().float().cpu()
        self.last_C = C.detach().float().cpu()
        self.last_delta = dt.detach().float().cpu()
        return self.out_proj(y)


class MambaBlockWithExtraction(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1, layer_idx=None):
        super().__init__()
        if USE_OFFICIAL_KERNEL:
            from mamba_ssm import Mamba
            self.mixer = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dt_rank=dt_rank, dt_min=dt_min, dt_max=dt_max, use_fast_path=True, layer_idx=layer_idx)
            self._official = True
            self._hook_delta: Optional[torch.Tensor] = None
            self._hook_B: Optional[torch.Tensor] = None
            self._hook_C: Optional[torch.Tensor] = None
            self._hook_A_bar: Optional[torch.Tensor] = None
            self._hook_B_bar: Optional[torch.Tensor] = None
            self._register_scan_hook()
        else:
            self.mixer = _PurePytorchS6(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dt_rank=dt_rank, dt_min=dt_min, dt_max=dt_max, layer_idx=layer_idx)
            self._official = False

    def _register_scan_hook(self):
        orig_forward = self.mixer.forward

        def patched_forward(hidden_states, inference_params=None):
            batch, seqlen, _ = hidden_states.shape
            conv_state, ssm_state = None, None
            A = -torch.exp(self.mixer.A_log.float())
            x_ = rearrange(hidden_states, "b l d -> b d l")
            y = orig_forward(hidden_states, inference_params)
            with torch.no_grad():
                xz = self.mixer.in_proj(hidden_states)
                x_part, _ = xz.chunk(2, dim=-1)
                from torch.nn.functional import pad
                x_c = x_part.transpose(1, 2)
                x_c = F.conv1d(F.pad(x_c, (self.mixer.d_conv - 1, 0)), self.mixer.conv1d.weight, self.mixer.conv1d.bias, groups=self.mixer.d_inner)[:, :, :seqlen].transpose(1, 2)
                x_c = F.silu(x_c)
                x_dbl = self.mixer.x_proj(rearrange(x_c, "b l d -> (b l) d"))
                dt_r, B_raw, C_raw = x_dbl.split([self.mixer.dt_rank, self.mixer.d_state, self.mixer.d_state], -1)
                dt_out = F.softplus(rearrange(self.mixer.dt_proj.weight @ dt_r.t(), "d (b l) -> b d l", l=seqlen) + self.mixer.dt_proj.bias[:, None])
                dt_out = rearrange(dt_out, "b d l -> b l d")
                B_out = rearrange(B_raw, "(b l) n -> b l n", b=batch)
                C_out = rearrange(C_raw, "(b l) n -> b l n", b=batch)
                dA = torch.einsum("b l d, d n -> b l d n", dt_out, A)
                A_bar = torch.exp(dA)
                B_bar = torch.einsum("b l d, b l n -> b l d n", dt_out, B_out)
            self._hook_delta = dt_out.detach().float().cpu()
            self._hook_B = B_out.detach().float().cpu()
            self._hook_C = C_out.detach().float().cpu()
            self._hook_A_bar = A_bar.detach().float().cpu()
            self._hook_B_bar = B_bar.detach().float().cpu()
            return y

        self.mixer.forward = patched_forward

    def forward(self, hidden_states, inference_params=None):
        return self.mixer(hidden_states, inference_params)

    def get_ssm_matrices(self) -> Dict[str, Optional[torch.Tensor]]:
        if self._official:
            return {
                "A_log": self.mixer.A_log.data.float().cpu(),
                "A_cont": -torch.exp(self.mixer.A_log.data.float()).cpu(),
                "D": self.mixer.D.data.float().cpu(),
                "dt_proj_bias": self.mixer.dt_proj.bias.data.float().cpu(),
                "delta": self._hook_delta,
                "B": self._hook_B,
                "C": self._hook_C,
                "A_bar": self._hook_A_bar,
                "B_bar": self._hook_B_bar,
            }
        else:
            m = self.mixer
            return {
                "A_log": m.A_log.data.float().cpu(),
                "A_cont": -torch.exp(m.A_log.data.float()).cpu(),
                "D": m.D.data.float().cpu(),
                "dt_proj_bias": m.dt_proj.bias.data.float().cpu(),
                "delta": m.last_delta,
                "B": m.last_C,
                "C": m.last_C,
                "A_bar": m.last_A_bar,
                "B_bar": m.last_B_bar,
            }


class MambaRMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x_norm).to(x.dtype)


class MambaResidualBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, layer_idx=None, rms_norm=True, residual_in_fp32=True):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.norm = MambaRMSNorm(d_model) if rms_norm else nn.LayerNorm(d_model)
        self.mixer = MambaBlockWithExtraction(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, layer_idx=layer_idx)

    def forward(self, hidden_states, residual=None, inference_params=None):
        residual = (hidden_states + residual).to(torch.float32 if self.residual_in_fp32 else hidden_states.dtype) if residual is not None else hidden_states.to(torch.float32 if self.residual_in_fp32 else hidden_states.dtype)
        hidden_states = self.norm(residual.to(self.norm.weight.dtype))
        hidden_states = self.mixer(hidden_states, inference_params)
        return hidden_states, residual


class MambaLMModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, d_state=16, d_conv=4, expand=2, rms_norm=True, residual_in_fp32=True, pad_vocab_size_multiple=8):
        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size = (vocab_size // pad_vocab_size_multiple + 1) * pad_vocab_size_multiple
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([MambaResidualBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand, layer_idx=i, rms_norm=rms_norm, residual_in_fp32=residual_in_fp32) for i in range(n_layers)])
        self.norm_f = MambaRMSNorm(d_model) if rms_norm else nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, inference_params=None):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, inference_params)
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.to(self.norm_f.weight.dtype))
        logits = self.lm_head(hidden_states)
        return logits

    def get_all_ssm_matrices(self) -> Dict[str, Dict]:
        return {f"layer_{i}": layer.mixer.get_ssm_matrices() for i, layer in enumerate(self.layers)}


@dataclass
class ExperimentConfig:
    name: str
    task: str
    d_model: int = 64
    n_layers: int = 2
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    vocab_size: int = 16
    seq_len: int = 256
    batch_size: int = 32
    lr: float = 8e-3
    weight_decay: float = 0.01
    epochs: int = 5
    warmup_steps: int = 100


PAPER_EXPERIMENTS: List[ExperimentConfig] = [
    ExperimentConfig(name="selective_copying", task="selective_copy", d_model=64, n_layers=2, d_state=16, seq_len=256, vocab_size=20, batch_size=32, lr=8e-3, epochs=5),
    ExperimentConfig(name="induction_heads", task="induction", d_model=64, n_layers=2, d_state=16, seq_len=256, vocab_size=16, batch_size=32, lr=8e-3, epochs=5),
]

USE_HF_DATASET: bool = False


class SelectiveCopyDataset(Dataset):
    def __init__(self, n_samples=8000, seq_len=256, vocab_size=20, n_selective=10):
        self.samples = []
        content_vocab = vocab_size - 2
        for _ in range(n_samples):
            positions = sorted(torch.randperm(seq_len - n_selective - 1)[:n_selective].tolist())
            tokens = torch.randint(2, vocab_size, (n_selective,))
            src = torch.zeros(seq_len, dtype=torch.long)
            for pos, tok in zip(positions, tokens):
                src[pos] = tok
            tgt = src.clone()
            tgt[:-1] = src[1:]
            tgt[-1] = 0
            self.samples.append((src, tgt))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


class InductionHeadsDataset(Dataset):
    def __init__(self, n_samples=8000, seq_len=256, vocab_size=16):
        self.samples = []
        half = seq_len // 2
        for _ in range(n_samples):
            prefix = torch.randint(1, vocab_size, (half,))
            seq = torch.cat([prefix, prefix])
            tgt = torch.cat([seq[1:], torch.zeros(1, dtype=torch.long)])
            self.samples.append((seq, tgt))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


class RandomTokenDataset(Dataset):
    def __init__(self, n_samples=2000, seq_len=128, vocab_size=256):
        data = torch.randint(0, vocab_size, (n_samples, seq_len + 1))
        self.x = data[:, :-1]
        self.y = data[:, 1:]

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]


def _load_hf_lm_dataset(cfg: ExperimentConfig):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    print("  [data] Loading OpenWebText via HuggingFace (streaming)...")
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    ds = load_dataset("openwebtext", split="train", streaming=True)

    class StreamDS(Dataset):
        def __init__(self, n=4000):
            self.items = []
            for item in ds.take(n):
                enc = tok(item["text"], truncation=True, max_length=cfg.seq_len + 1, padding="max_length")
                ids = torch.tensor(enc["input_ids"])
                self.items.append((ids[:-1], ids[1:]))

        def __len__(self): return len(self.items)
        def __getitem__(self, i): return self.items[i]

    return StreamDS()


def _load_hf_dna_dataset(cfg: ExperimentConfig):
    from datasets import load_dataset
    DNA_VOCAB = {c: i for i, c in enumerate("ACGTN")}
    print("  [data] Loading Human Reference Genome (hg38) via HuggingFace...")
    ds = load_dataset("InstaDeepAI/human_reference_genome", "6kbp", split="train", streaming=True)

    class DNADS(Dataset):
        def __init__(self, n=2000):
            self.items = []
            for item in ds.take(n):
                seq = item["sequence"].upper()
                ids = torch.tensor([DNA_VOCAB.get(c, 4) for c in seq[:cfg.seq_len + 1]], dtype=torch.long)
                if len(ids) < cfg.seq_len + 1:
                    ids = F.pad(ids, (0, cfg.seq_len + 1 - len(ids)))
                self.items.append((ids[:-1], ids[1:]))

        def __len__(self): return len(self.items)
        def __getitem__(self, i): return self.items[i]

    return DNADS()


def _load_hf_audio_dataset(cfg: ExperimentConfig):
    from datasets import load_dataset
    print("  [data] Loading SC09 (speech_commands) via HuggingFace...")
    ds = load_dataset("speech_commands", "v0.01", split="train", streaming=True)

    def mu_law_encode(audio, mu=255):
        audio = torch.tensor(audio, dtype=torch.float32).clamp(-1, 1)
        encoded = torch.sign(audio) * torch.log1p(mu * audio.abs()) / math.log(1 + mu)
        return ((encoded + 1) / 2 * mu + 0.5).long().clamp(0, mu)

    class AudioDS(Dataset):
        def __init__(self, n=2000):
            self.items = []
            for item in ds.take(n):
                tokens = mu_law_encode(item["audio"]["array"])
                L = cfg.seq_len + 1
                tokens = tokens[:L] if len(tokens) >= L else F.pad(tokens, (0, L - len(tokens)))
                self.items.append((tokens[:-1], tokens[1:]))

        def __len__(self): return len(self.items)
        def __getitem__(self, i): return self.items[i]

    return AudioDS()


def get_dataset(cfg: ExperimentConfig) -> Dataset:
    if cfg.task == "selective_copy":
        return SelectiveCopyDataset(seq_len=cfg.seq_len, vocab_size=cfg.vocab_size)
    if cfg.task == "induction":
        return InductionHeadsDataset(seq_len=cfg.seq_len, vocab_size=cfg.vocab_size)
    if USE_HF_DATASET:
        if cfg.task == "lm": return _load_hf_lm_dataset(cfg)
        if cfg.task == "dna": return _load_hf_dna_dataset(cfg)
        if cfg.task == "audio": return _load_hf_audio_dataset(cfg)
    print(f"  [data] Using synthetic random-token dataset for '{cfg.task}' (set USE_HF_DATASET=True to use the real dataset)")
    return RandomTokenDataset(seq_len=cfg.seq_len, vocab_size=cfg.vocab_size)


def train_experiment(cfg: ExperimentConfig, save_dir: str = "checkpoints") -> MambaLMModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    print(f"\n{'='*64}")
    print(f"  {cfg.name.upper()}  ({cfg.task})")
    print(f"  d_model={cfg.d_model}  n_layers={cfg.n_layers}  d_state={cfg.d_state}  device={device}  dtype={dtype}")
    print(f"{'='*64}")
    model = MambaLMModel(vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_layers=cfg.n_layers, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand).to(device=device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    ds = get_dataset(cfg)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=(device == "cuda"), num_workers=0)
    decay_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and p.requires_grad]
    no_decay_params = [p for n, p in model.named_parameters() if p.ndim < 2 and p.requires_grad]
    opt = torch.optim.AdamW([{"params": decay_params, "weight_decay": cfg.weight_decay}, {"params": no_decay_params, "weight_decay": 0.0}], lr=cfg.lr, betas=(0.9, 0.95))
    total_steps = cfg.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=total_steps, pct_start=cfg.warmup_steps / max(total_steps, cfg.warmup_steps + 1))
    scaler = torch.cuda.amp.GradScaler() if (dtype == torch.float32 and device == "cuda") else None
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss, n_steps = 0.0, 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits = model(x)
                V = logits.size(-1)
                loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            opt.zero_grad(set_to_none=True)
            scheduler.step()
            total_loss += loss.item()
            n_steps += 1
        avg_loss = total_loss / n_steps
        ppl = math.exp(min(avg_loss, 20))
        print(f"  epoch {epoch}/{cfg.epochs}  loss={avg_loss:.4f}  ppl={ppl:.2f}")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"{cfg.name}.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": cfg.__dict__, "kernel": "official_mamba_ssm" if USE_OFFICIAL_KERNEL else "pure_pytorch"}, ckpt_path)
    print(f"  Checkpoint -> {ckpt_path}")
    return model


class SSMMatrixExtractor:
    @staticmethod
    def from_model(model: MambaLMModel, input_ids: torch.Tensor) -> Dict[str, Dict]:
        model.eval()
        with torch.no_grad():
            _ = model(input_ids)
        return model.get_all_ssm_matrices()

    @staticmethod
    def from_checkpoint(ckpt_path: str, input_ids: Optional[torch.Tensor] = None) -> Dict[str, Dict]:
        print(f"\n  Loading: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ExperimentConfig(**ckpt["config"])
        model = MambaLMModel(vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_layers=cfg.n_layers, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand)
        model.load_state_dict(ckpt["model_state_dict"])
        if input_ids is not None:
            return SSMMatrixExtractor.from_model(model, input_ids)
        out = {}
        for i, layer in enumerate(model.layers):
            mixer = layer.mixer.mixer
            out[f"layer_{i}"] = {"A_log": mixer.A_log.data.float(), "A_cont": -torch.exp(mixer.A_log.data.float()), "D": mixer.D.data.float(), "dt_proj_bias": mixer.dt_proj.bias.data.float()}
        return out

    @staticmethod
    def from_hf(model_name: str = "state-spaces/mamba-130m", input_ids: Optional[torch.Tensor] = None) -> Dict[str, Dict]:
        try:
            from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        except ImportError:
            raise ImportError("pip install mamba-ssm transformers")
        print(f"\n  Loading HF model: {model_name}")
        model = MambaLMHeadModel.from_pretrained(model_name, dtype=torch.float32)
        model.eval()
        sd = model.state_dict()
        layers: Dict[str, Dict] = {}
        for key, val in sd.items():
            if "layers." not in key:
                continue
            parts = key.split(".")
            try:
                li = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                continue
            lk = f"layer_{li}"
            if lk not in layers:
                layers[lk] = {}
            if "A_log" in key:
                layers[lk]["A_log"] = val.float()
                layers[lk]["A_cont"] = -torch.exp(val.float())
            elif key.endswith(".D"):
                layers[lk]["D"] = val.float()
            elif "dt_proj.bias" in key:
                layers[lk]["dt_proj_bias"] = val.float()
            elif "dt_proj.weight" in key:
                layers[lk]["dt_proj_weight"] = val.float()
            elif "x_proj.weight" in key:
                layers[lk]["x_proj_weight"] = val.float()
            elif "in_proj.weight" in key:
                layers[lk]["in_proj_weight"] = val.float()
            elif "out_proj.weight" in key:
                layers[lk]["out_proj_weight"] = val.float()
        if input_ids is not None and USE_OFFICIAL_KERNEL:
            hook_data: Dict[str, Dict] = {}

            def make_hook(name):
                def hook(module, inp, out):
                    with torch.no_grad():
                        hidden = inp[0]
                        B_sz, L, _ = hidden.shape
                        xz = module.in_proj(hidden)
                        x_, _ = xz.chunk(2, -1)
                        x_c = module.conv1d(x_.transpose(1, 2))[:, :, :L].transpose(1, 2)
                        x_c = F.silu(x_c)
                        x_dbl = module.x_proj(rearrange(x_c, "b l d -> (b l) d"))
                        dr, B_raw, C_raw = x_dbl.split([module.dt_rank, module.d_state, module.d_state], -1)
                        dt = F.softplus(rearrange(module.dt_proj.weight @ dr.t(), "d (b l) -> b d l", l=L) + module.dt_proj.bias[:, None]).permute(0, 2, 1)
                        Bm = rearrange(B_raw, "(b l) n -> b l n", b=B_sz)
                        Cm = rearrange(C_raw, "(b l) n -> b l n", b=B_sz)
                        A = -torch.exp(module.A_log.float())
                        dA = torch.einsum("b l d, d n -> b l d n", dt, A)
                        hook_data[name] = {"delta": dt.detach().float().cpu(), "B": Bm.detach().float().cpu(), "C": Cm.detach().float().cpu(), "A_bar": torch.exp(dA).detach().float().cpu(), "B_bar": torch.einsum("b l d, b l n -> b l d n", dt, Bm).detach().float().cpu()}
                return hook

            handles = []
            for name, mod in model.named_modules():
                if type(mod).__name__ == "Mamba":
                    handles.append(mod.register_forward_hook(make_hook(name)))
            with torch.no_grad():
                model(input_ids)
            for h in handles:
                h.remove()
            for hkey, hval in hook_data.items():
                parts = hkey.split(".")
                try:
                    li = int(parts[parts.index("layers") + 1])
                    layers[f"layer_{li}"].update(hval)
                except (ValueError, IndexError):
                    pass
        return layers


def print_matrices(matrices: Dict[str, Dict], max_layers: int = 4):
    print("\n" + "="*64)
    print("  EXTRACTED SSM MATRICES")
    print("="*64)
    for lname, mats in list(matrices.items())[:max_layers]:
        print(f"\n  [{lname}]")
        for mname, t in mats.items():
            if t is None:
                print(f"    {mname:20s}  -")
                continue
            tf = t.float()
            print(f"    {mname:20s}  {str(tuple(tf.shape)):32s}  min={tf.min():.4f}  max={tf.max():.4f}  mean={tf.mean():.4f}")


def save_matrices(matrices: Dict[str, Dict], path: str):
    flat = {}
    for lname, mats in matrices.items():
        for mname, t in mats.items():
            if t is not None:
                flat[f"{lname}__{mname}"] = t.float().numpy()
    np.savez(path, **flat)
    print(f"  Matrices -> {path}")


def main():
    CKPT_DIR = "checkpoints"
    MATRIX_DIR = "extracted_matrices"
    os.makedirs(MATRIX_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for cfg in PAPER_EXPERIMENTS:
        model = train_experiment(cfg, save_dir=CKPT_DIR)
        model.eval()
        sample = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len)).to(device)
        print(f"\n  Extracting matrices for [{cfg.name}] ...")
        matrices = SSMMatrixExtractor.from_model(model, sample)
        print_matrices(matrices)
        npz_path = os.path.join(MATRIX_DIR, f"{cfg.name}.npz")
        save_matrices(matrices, npz_path)
        ckpt_path = os.path.join(CKPT_DIR, f"{cfg.name}.pt")
        static = SSMMatrixExtractor.from_checkpoint(ckpt_path)
        print(f"  Static A_cont (layer 0): shape={tuple(static['layer_0']['A_cont'].shape)}  mean={static['layer_0']['A_cont'].mean():.4f}")
    print("\n" + "="*64)
    print("  ALL EXPERIMENTS COMPLETE")
    print("="*64)


if __name__ == "__main__":
    main()