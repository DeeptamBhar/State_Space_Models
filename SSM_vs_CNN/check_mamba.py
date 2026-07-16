import torch
import time

try:
    # 1. Check GPU and Torch
    print(f"--- 1. Hardware Check ---")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 2. Check Causal Conv1d (The first kernel we built)
    print(f"\n--- 2. Conv1d Kernel Check ---")
    from causal_conv1d import causal_conv1d_fn
    x = torch.randn(2, 16, 64, device="cuda", dtype=torch.float16)
    weight = torch.randn(16, 4, device="cuda", dtype=torch.float16)
    out_conv = causal_conv1d_fn(x, weight)
    print("Success: Causal Conv1d kernel is operational.")

    # 3. Check Mamba Selective Scan (The core architecture)
    print(f"\n--- 3. Mamba Core Check ---")
    from mamba_ssm import Mamba
    model = Mamba(d_model=128, d_state=16, d_conv=4, expand=2).to("cuda").half()
    input_tensor = torch.randn(2, 64, 128, device="cuda", dtype=torch.float16)
    
    # Warmup
    _ = model(input_tensor)
    
    # Timing a forward pass
    start = time.time()
    output = model(input_tensor)
    end = time.time()
    
    print(f"Success: Mamba forward pass completed in {(end-start)*1000:.2f}ms")
    print(f"Output shape: {output.shape}")

    print("\n✅ INSTALLATION VERIFIED: Everything is good to go.")

except ImportError as e:
    print(f"\n❌ IMPORT ERROR: {e}")
    print("Likely cause: The package was not installed in the current environment.")
except RuntimeError as e:
    print(f"\n❌ RUNTIME ERROR: {e}")
    print("Likely cause: CUDA/Torch version mismatch or kernel compilation failure.")
except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")