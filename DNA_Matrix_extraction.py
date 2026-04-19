
import os,math,warnings
from dataclasses import dataclass
from typing import Optional,Dict,List
import torch,torch.nn as nn,torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import numpy as np
from einops import rearrange
from tqdm import tqdm

FORCE_PURE_PYTORCH=False
def _check_kernels():
    if FORCE_PURE_PYTORCH:return False
    try:
        from mamba_ssm import Mamba as _
        if not torch.cuda.is_available():
            warnings.warn("mamba_ssm installed but no CUDA device — falling back to pure-PyTorch S6.")
            return False
        return True
    except ImportError:
        warnings.warn("mamba_ssm not found — falling back to pure-PyTorch S6.\nInstall: pip install mamba-ssm --no-build-isolation")
        return False

USE_OFFICIAL_KERNEL=_check_kernels()
print(f"[kernel] {'mamba_ssm CUDA fused kernels' if USE_OFFICIAL_KERNEL else 'pure-PyTorch S6 fallback'}")

class _PurePytorchS6(nn.Module):
    def __init__(self,d_model,d_state=16,d_conv=4,expand=2,dt_min=0.001,dt_max=0.1,conv_bias=True,bias=False,layer_idx=None):
        super().__init__()
        self.d_model=d_model;self.d_state=d_state;self.d_conv=d_conv
        self.d_inner=int(expand*d_model);self.dt_rank=math.ceil(d_model/16)
        self.layer_idx=layer_idx
        self.in_proj=nn.Linear(d_model,self.d_inner*2,bias=bias)
        self.conv1d=nn.Conv1d(self.d_inner,self.d_inner,kernel_size=d_conv,groups=self.d_inner,padding=d_conv-1,bias=conv_bias)
        self.x_proj=nn.Linear(self.d_inner,self.dt_rank+2*d_state,bias=False)
        self.dt_proj=nn.Linear(self.dt_rank,self.d_inner,bias=True)
        dt=torch.exp(torch.rand(self.d_inner)*(math.log(dt_max)-math.log(dt_min))+math.log(dt_min)).clamp(min=1e-4)
        with torch.no_grad():self.dt_proj.bias.copy_(dt+torch.log(-torch.expm1(-dt)))
        self.dt_proj.bias._no_reinit=True
        A=torch.arange(1,d_state+1,dtype=torch.float32).repeat(self.d_inner,1)
        self.A_log=nn.Parameter(torch.log(A));self.A_log._no_weight_decay=True
        self.D=nn.Parameter(torch.ones(self.d_inner));self.D._no_weight_decay=True
        self.out_proj=nn.Linear(self.d_inner,d_model,bias=bias)
        self.last_delta=self.last_B=self.last_C=self.last_A_bar=self.last_B_bar=None

    def forward(self,hidden_states,inference_params=None):
        B_sz,L,_=hidden_states.shape
        xz=self.in_proj(hidden_states);x,z=xz.chunk(2,dim=-1)
        x_c=self.conv1d(x.transpose(1,2))[:,:,:L].transpose(1,2)
        x_c=F.silu(x_c)
        x_dbl=self.x_proj(rearrange(x_c,"b l d -> (b l) d"))
        dt_r,B_raw,C_raw=x_dbl.split([self.dt_rank,self.d_state,self.d_state],-1)
        dt=F.softplus(self.dt_proj(dt_r))
        dt=rearrange(dt,"(b l) d -> b l d",b=B_sz)
        B_raw=rearrange(B_raw,"(b l) n -> b l n",b=B_sz)
        C_raw=rearrange(C_raw,"(b l) n -> b l n",b=B_sz)
        A_cont=-torch.exp(self.A_log.float())
        dA=torch.einsum("b l d, d n -> b l d n",dt,A_cont)
        A_bar=torch.exp(dA)
        B_bar=torch.einsum("b l d, b l n -> b l d n",dt,B_raw)
        h=torch.zeros(B_sz,self.d_inner,self.d_state,device=hidden_states.device,dtype=hidden_states.dtype)
        ys=[]
        for t in range(L):
            h=A_bar[:,t]*h+B_bar[:,t]*x_c[:,t,:,None]
            ys.append((h*C_raw[:,t,None,:]).sum(-1))
        y=torch.stack(ys,1)
        y=y+x_c*self.D
        y=y*F.silu(z)
        self.last_delta=dt.detach().float().cpu()
        self.last_B=B_raw.detach().float().cpu()
        self.last_C=C_raw.detach().float().cpu()
        self.last_A_bar=A_bar.detach().float().cpu()
        self.last_B_bar=B_bar.detach().float().cpu()
        return self.out_proj(y)

class MambaBlockWithExtraction(nn.Module):
    """
    Wraps the official mamba_ssm.Mamba (fused CUDA kernel) and patches its
    forward to cheaply re-derive B, C, delta for matrix extraction.
    Falls back to _PurePytorchS6 when no CUDA kernel is available.
    """
    def __init__(self,d_model,d_state=16,d_conv=4,expand=2,layer_idx=None):
        super().__init__()
        if USE_OFFICIAL_KERNEL:
            from mamba_ssm import Mamba
            self.mixer=Mamba(d_model=d_model,d_state=d_state,d_conv=d_conv,expand=expand,use_fast_path=True,layer_idx=layer_idx)
            self._official=True;self._delta=self._B=self._C=self._A_bar=self._B_bar=None
            self._patch_forward()
        else:
            self.mixer=_PurePytorchS6(d_model,d_state=d_state,d_conv=d_conv,expand=expand,layer_idx=layer_idx)
            self._official=False

    def _patch_forward(self):
        orig=self.mixer.forward;blk=self
        def patched(hidden_states,inference_params=None):
            y=orig(hidden_states,inference_params)
            with torch.no_grad():
                B_sz,L,_=hidden_states.shape
                xz=blk.mixer.in_proj(hidden_states);x_,_=xz.chunk(2,-1)
                x_c=F.conv1d(F.pad(x_.transpose(1,2),(blk.mixer.d_conv-1,0)),blk.mixer.conv1d.weight,blk.mixer.conv1d.bias,groups=blk.mixer.d_inner)[:,:,:L].transpose(1,2)
                x_c=F.silu(x_c)
                x_dbl=blk.mixer.x_proj(rearrange(x_c,"b l d -> (b l) d"))
                dt_r,B_raw,C_raw=x_dbl.split([blk.mixer.dt_rank,blk.mixer.d_state,blk.mixer.d_state],-1)
                dt=F.softplus(rearrange(blk.mixer.dt_proj.weight@dt_r.t(),"d (b l) -> b d l",l=L)+blk.mixer.dt_proj.bias[:,None]).permute(0,2,1)
                B_m=rearrange(B_raw,"(b l) n -> b l n",b=B_sz)
                C_m=rearrange(C_raw,"(b l) n -> b l n",b=B_sz)
                A=-torch.exp(blk.mixer.A_log.float())
                dA=torch.einsum("b l d, d n -> b l d n",dt,A)
                blk._delta=dt.detach().float().cpu()
                blk._B=B_m.detach().float().cpu()
                blk._C=C_m.detach().float().cpu()
                blk._A_bar=torch.exp(dA).detach().float().cpu()
                blk._B_bar=torch.einsum("b l d, b l n -> b l d n",dt,B_m).detach().float().cpu()
            return y
        self.mixer.forward=patched

    def forward(self,hidden_states,inference_params=None):return self.mixer(hidden_states,inference_params)

    def get_ssm_matrices(self):
        mx=self.mixer
        if self._official:
            return dict(A_log=mx.A_log.data.float().cpu(),A_cont=-torch.exp(mx.A_log.data.float()).cpu(),D=mx.D.data.float().cpu(),dt_proj_bias=mx.dt_proj.bias.data.float().cpu(),delta=self._delta,B=self._B,C=self._C,A_bar=self._A_bar,B_bar=self._B_bar)
        return dict(A_log=mx.A_log.data.float().cpu(),A_cont=-torch.exp(mx.A_log.data.float()).cpu(),D=mx.D.data.float().cpu(),dt_proj_bias=mx.dt_proj.bias.data.float().cpu(),delta=mx.last_delta,B=mx.last_B,C=mx.last_C,A_bar=mx.last_A_bar,B_bar=mx.last_B_bar)

class RMSNorm(nn.Module):
    def __init__(self,d,eps=1e-5):super().__init__();self.weight=nn.Parameter(torch.ones(d));self.eps=eps
    def forward(self,x):v=x.float().pow(2).mean(-1,keepdim=True);return (self.weight*x*torch.rsqrt(v+self.eps)).to(x.dtype)

class MambaResidualBlock(nn.Module):
    def __init__(self,d_model,d_state=16,d_conv=4,expand=2,layer_idx=None):
        super().__init__();self.norm=RMSNorm(d_model)
        self.mixer=MambaBlockWithExtraction(d_model,d_state=d_state,d_conv=d_conv,expand=expand,layer_idx=layer_idx)
    def forward(self,x,residual=None):
        res=(x+residual).to(torch.float32) if residual is not None else x.float()
        h=self.mixer(self.norm(res.to(self.norm.weight.dtype)));return h,res

class MambaLMModel(nn.Module):
    def __init__(self,vocab_size,d_model,n_layers,d_state=16,d_conv=4,expand=2):
        super().__init__()
        if vocab_size%8:vocab_size=(vocab_size//8+1)*8
        self.embedding=nn.Embedding(vocab_size,d_model)
        self.layers=nn.ModuleList([MambaResidualBlock(d_model,d_state=d_state,d_conv=d_conv,expand=expand,layer_idx=i) for i in range(n_layers)])
        self.norm_f=RMSNorm(d_model)
        self.lm_head=nn.Linear(d_model,vocab_size,bias=False)
        self.lm_head.weight=self.embedding.weight

    def forward(self,input_ids):
        x,res=self.embedding(input_ids),None
        for layer in self.layers:x,res=layer(x,res)
        x=self.norm_f((x+res).to(self.norm_f.weight.dtype))
        return self.lm_head(x)

    def get_all_ssm_matrices(self):return {f"layer_{i}":lyr.mixer.get_ssm_matrices() for i,lyr in enumerate(self.layers)}

@dataclass
class ExperimentConfig:
    name:str;task:str;d_model:int=128;n_layers:int=4;d_state:int=16;d_conv:int=4;expand:int=2
    vocab_size:int=256;seq_len:int=256;batch_size:int=16;lr:float=8e-4
    weight_decay:float=0.01;epochs:int=3;warmup_steps:int=200;n_train_samples:int=3000

EXPERIMENTS=[ExperimentConfig(name="audio_waveform",task="audio",d_model=64,n_layers=4,d_state=16,seq_len=4000,vocab_size=256,batch_size=4,lr=8e-4,epochs=2,n_train_samples=2000)]

class DNADataset(Dataset):
    """
    Human genome DNA sequences (hg38 / multi-species).
    HF repo  : LongSafari/open-genome  (Parquet-based, no loading script)
    Config   : 'sample'  — small validation split, instant to stream
    Field    : 'sequence'  (raw nucleotide string)
    Vocab    : A=0  C=1  G=2  T=3  N=4  (character-level, 5 tokens)

    Why switched from InstaDeepAI/human_reference_genome:
      That repo still uses an old Python loading script which is no longer
      supported by datasets >= 3.x  ('Dataset scripts are no longer supported').
      LongSafari/open-genome stores data as plain Parquet — works with any
      recent datasets version, no trust_remote_code needed.
    """
    DNA_VOCAB={c:i for i,c in enumerate("ACGTN")}
    def __init__(self,seq_len,n_samples):
        from datasets import load_dataset
        print("  [DNA] Downloading LongSafari/open-genome (stage1) from HuggingFace via streaming …")
        ds=load_dataset("LongSafari/open-genome","stage1",split="train",streaming=True)
        self.items=[];needed=seq_len+1
        print(f"  [DNA] Collecting {n_samples} samples (seq_len={seq_len}) …")
        for item in tqdm(ds.take(n_samples*3),total=n_samples*3,desc="  DNA"):
            raw=item["text"]
            seq=raw.split("|",1)[1].upper() if "|" in raw else raw.upper()
            ids=[self.DNA_VOCAB.get(c,4) for c in seq]
            if len(ids)<needed//2:continue
            if len(ids)<needed:ids+=[4]*(needed-len(ids))
            ids=ids[:needed]
            self.items.append(torch.tensor(ids,dtype=torch.long))
            if len(self.items)>=n_samples:break
        if len(self.items)==0:raise RuntimeError("[DNA] No samples loaded — check internet connection.")
        print(f"  [DNA] Loaded {len(self.items)} samples.")
    def __len__(self):return len(self.items)
    def __getitem__(self,i):t=self.items[i];return t[:-1],t[1:]

def get_dataset(cfg):
    if cfg.task=="dna":return DNADataset(seq_len=cfg.seq_len,n_samples=cfg.n_train_samples)
    raise ValueError(f"Unknown task: {cfg.task}")

def train(cfg,save_dir="checkpoints"):
    device="cuda" if torch.cuda.is_available() else "cpu"
    dtype=torch.bfloat16 if device=="cuda" and torch.cuda.is_bf16_supported() else torch.float32
    print(f"\n{'='*64}\n  TRAIN  {cfg.name}  |  d_model={cfg.d_model}  layers={cfg.n_layers}  d_state={cfg.d_state}\n  device={device}  dtype={dtype}  seq_len={cfg.seq_len}\n{'='*64}")
    model=MambaLMModel(cfg.vocab_size,cfg.d_model,cfg.n_layers,cfg.d_state,cfg.d_conv,cfg.expand).to(device=device,dtype=dtype)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    ds=get_dataset(cfg)
    loader=DataLoader(ds,batch_size=cfg.batch_size,shuffle=True,num_workers=2,pin_memory=(device=="cuda"))
    decay=[p for p in model.parameters() if p.ndim>=2 and p.requires_grad]
    no_decay=[p for p in model.parameters() if p.ndim<2 and p.requires_grad]
    opt=torch.optim.AdamW([{"params":decay,"weight_decay":cfg.weight_decay},{"params":no_decay,"weight_decay":0.0}],lr=cfg.lr,betas=(0.9,0.95))
    total_steps=cfg.epochs*len(loader)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=cfg.lr,total_steps=max(total_steps,1),pct_start=min(cfg.warmup_steps/max(total_steps,1),0.3))
    for epoch in range(1,cfg.epochs+1):
        model.train();total_loss=steps=0
        pbar=tqdm(loader,desc=f"  epoch {epoch}/{cfg.epochs}",leave=False)
        for x,y in pbar:
            x,y=x.to(device),y.to(device)
            with torch.amp.autocast(device_type=device,dtype=torch.bfloat16,enabled=(device=="cuda")):
                logits=model(x);loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1))
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step();opt.zero_grad(set_to_none=True);scheduler.step()
            total_loss+=loss.item();steps+=1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg=total_loss/steps
        print(f"  epoch {epoch}  loss={avg:.4f}  ppl={math.exp(min(avg,20)):.2f}")
    os.makedirs(save_dir,exist_ok=True)
    ckpt=os.path.join(save_dir,f"{cfg.name}.pt")
    torch.save({"model_state_dict":model.state_dict(),"config":cfg.__dict__},ckpt)
    print(f"  ✓ Checkpoint saved → {ckpt}")
    return model

def extract_and_save(model,cfg,out_dir="extracted_matrices"):
    device=next(model.parameters()).device
    model.eval()
    ds=get_dataset(cfg)
    loader=DataLoader(ds,batch_size=2,shuffle=True)
    x,_=next(iter(loader));x=x.to(device)
    print(f"\n  [extract] Forward pass on 2 real {cfg.task} samples …")
    with torch.no_grad():_=model(x)
    matrices=model.get_all_ssm_matrices()
    print(f"\n  SSM matrices for [{cfg.name}]")
    print(f"  {'layer':<10} {'matrix':<20} {'shape':<35} {'mean':>8}")
    print(f"  {'-'*75}")
    for lname,mats in matrices.items():
        for mname,t in mats.items():
            if t is None:continue
            tf=t.float()
            print(f"  {lname:<10} {mname:<20} {str(tuple(tf.shape)):<35} {tf.mean().item():>8.4f}")
    os.makedirs(out_dir,exist_ok=True)
    flat={}
    for lname,mats in matrices.items():
        for mname,t in mats.items():
            if t is not None:flat[f"{lname}__{mname}"]=t.float().numpy()
    npz_path=os.path.join(out_dir,f"{cfg.name}.npz")
    np.savez_compressed(npz_path,**flat)
    print(f"\n  ✓ Matrices saved → {npz_path}")
    print(f"    Keys in file: {list(flat.keys())[:6]} … ({len(flat)} total)")
    return npz_path

def main():
    CKPT_DIR="checkpoints";MATRIX_DIR="extracted_matrices";saved_npz={}
    for cfg in EXPERIMENTS:
        model=train(cfg,save_dir=CKPT_DIR)
        path=extract_and_save(model,cfg,out_dir=MATRIX_DIR)
        saved_npz[cfg.name]=path
    print(f"\n{'='*64}\n  DONE — output .npz files:")
    for name,path in saved_npz.items():
        size_mb=os.path.getsize(path)/1e6
        print(f"  {name:<25}  {path}  ({size_mb:.1f} MB)")
    print(f"{'='*64}")
    print("""
  HOW TO LOAD THE MATRICES LATER
  ────────────────────────────────────────────────────────────
  import numpy as np
  dna = np.load("extracted_matrices/dna_sequence.npz")
  print(list(dna.files))
  A_log=dna["layer_0__A_log"]
  A_cont=dna["layer_0__A_cont"]
  D=dna["layer_0__D"]
  delta=dna["layer_0__delta"]
  B=dna["layer_0__B"]
  C=dna["layer_0__C"]
  A_bar=dna["layer_0__A_bar"]
  B_bar=dna["layer_0__B_bar"]
  ────────────────────────────────────────────────────────────
""")

if __name__=="__main__":main()