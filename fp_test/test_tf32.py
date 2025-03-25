import torch
from gospel.util import Timer
import gc
gc.collect()

# Tensor Core 활성화 (최적화된 커널을 강제 실행)
torch.set_float32_matmul_precision("high")

def warmup():
    print("Warmup ...")
    # A = torch.randn(128, 128, dtype=torch.float64).cuda()
    A = torch.randn(128, 128, dtype=torch.float32).cuda()
    for _ in range(10):
        A.T @ A

warmup()

# N, M = 128, 128
N, M = 5000, 5000
ntry = 10

torch.cuda.empty_cache()
torch.cuda.synchronize()
A_64 = torch.randn(N, M, dtype=torch.float64).cuda()
with Timer.track("matmul fp64", A_64.device, True, True):
    for _ in range(ntry):
        val_64 = A_64.T @ A_64

torch.cuda.empty_cache()
torch.cuda.synchronize()
A_16 = A_64.half()
with Timer.track("matmul fp16", A_64.device, True, True):
    for _ in range(ntry):
        val_16 = A_16.T @ A_16
print(f"Error (float64 - float16): {(val_64 - val_16).abs().max()}")

torch.cuda.empty_cache()
torch.cuda.synchronize()

torch.backends.cuda.matmul.allow_tf32 = False
A_32 = A_64.float()
with Timer.track("matmul fp32", A_64.device, True, True):
    for _ in range(ntry):
        val_32 = A_32.T @ A_32
print(f"Error (float64 - float32, TF32=False): {(val_64 - val_32).abs().max()}")

torch.cuda.empty_cache()
torch.cuda.synchronize()

torch.backends.cuda.matmul.allow_tf32 = True
A_32 = A_64.float()
with Timer.track("matmul tf32", A_64.device, True, True):
    for _ in range(ntry):
        val_32 = A_32.T @ A_32
print(f"Error (float64 - float32, TF32=True): {(val_64 - val_32).abs().max()}")
