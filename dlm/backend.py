"""Pick the reader backend: "mlx" (dlm/reads.py, Apple silicon, 4-bit) or "torch" (dlm/reads_torch.py, CUDA, bf16)."""

from __future__ import annotations


def get_reader(backend: str = "mlx", **kw):
    if backend == "torch":
        from dlm.reads_torch import Reader
        return Reader(**kw)
    if backend == "mlx":
        from dlm.reads import Reader
        return Reader(**kw)
    raise ValueError(backend)


def add_backend_arg(ap):
    ap.add_argument("--backend", choices=["mlx", "torch"], default="mlx")
    return ap
