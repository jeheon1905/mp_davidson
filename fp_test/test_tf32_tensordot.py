import torch
torch.manual_seed(42)
from gospel.util import Timer


def my_tensordot(rho, Fx, Fy, Fz):
    gpts = [Fx.shape[0], Fy.shape[0], Fz.shape[0]]
    nbands = rho.shape[-1]

    rho = rho.reshape(*gpts, nbands)
    tmp = torch.tensordot(rho, Fx, ([0], [0]))
    tmp = torch.tensordot(tmp, Fy, ([0], [0]))
    tmp = torch.tensordot(tmp, Fz, ([0], [0]))
    return tmp


def warmup(device):
    if device == "cpu":
        pass
    else:
        A = torch.randn(128, 128, dtype=torch.float64).to(device)
        for _ in range(10):
            A.T @ A


if __name__ == "__main__":
    # gpts, nbands = [128, 128, 128], 128
    gpts, nbands = [64, 64, 64], 10
    # gpts, nbands = [64, 64, 64], 300
    # gpts, nbands = [77, 77, 77], 77
    print(f"gpts: {gpts}, nbands: {nbands}")
    ntry = 10
    # device = "cuda"
    device = "cpu"

    rho = torch.randn(*gpts, nbands, dtype=torch.float64).to(device)
    Fx = torch.randn(gpts[0], gpts[0], dtype=torch.float64).to(device)
    Fy = torch.randn(gpts[1], gpts[1], dtype=torch.float64).to(device)
    Fz = torch.randn(gpts[2], gpts[2], dtype=torch.float64).to(device)

    # import torch.fx
    # my_tensordot = torch.fx.symbolic_trace(my_tensordot)
    # print(my_tensordot.graph)

    warmup(device)

    with Timer.track("tensordot fp64", rho.device, True, True):
        for i in range(ntry):
            val_64 = my_tensordot(rho, Fx, Fy, Fz)

    rho_16 = rho.half()
    Fx_16 = Fx.half()
    Fy_16 = Fy.half()
    Fz_16 = Fz.half()
    with Timer.track("tensordot fp16", rho.device, True, True):
        for i in range(ntry):
            val_16 = my_tensordot(rho_16, Fx_16, Fy_16, Fz_16)
    print(f"Error (float64 - float16): {(val_64 - val_16).abs().max()}")


    torch.backends.cuda.matmul.allow_tf32 = False
    rho_32 = rho.float()
    Fx_32 = Fx.float()
    Fy_32 = Fy.float()
    Fz_32 = Fz.float()
    with Timer.track("tensordot fp32", rho.device, True, True):
        for i in range(ntry):
            val_32 = my_tensordot(rho_32, Fx_32, Fy_32, Fz_32)
    print(f"Error (float64 - float32): {(val_64 - val_32).abs().max()}")

    torch.backends.cuda.matmul.allow_tf32 = True
    # torch.set_float32_matmul_precision("high")
    rho_32 = rho.float()
    Fx_32 = Fx.float()
    Fy_32 = Fy.float()
    Fz_32 = Fz.float()
    with Timer.track("tensordot tf32", rho.device, True, True):
        for i in range(ntry):
            val_32 = my_tensordot(rho_32, Fx_32, Fy_32, Fz_32)
    print(f"Error (float64 - tf32): {(val_64 - val_32).abs().max()}")
