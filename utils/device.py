"""Device selection with an up-front CUDA architecture check.

A PyTorch wheel only carries kernels for the GPU architectures it was built for.
Running on an older card than the wheel supports does not fail at startup - it
fails later, deep inside the first convolution, as

    RuntimeError: FIND was unable to find an engine to execute this computation

which says nothing about the real cause. `resolve_device` checks the compute
capability against `torch.cuda.get_arch_list()` before any work starts.
"""

from __future__ import annotations

from typing import List, Optional

import torch

# CUDA binary compatibility runs forwards across minor revisions only: a cubin
# built for sm_60 runs on sm_61, but not the other way round, and never across a
# major revision. PTX (`compute_XX`) can be JIT-compiled for anything newer.
def arch_supports(device_cc: tuple[int, int], arch_list: List[str]) -> bool:
    major, minor = device_cc
    for arch in arch_list:
        kind, _, version = arch.partition("_")
        if not version.isdigit():
            continue
        arch_major, arch_minor = int(version[:-1]), int(version[-1])
        if kind == "sm" and arch_major == major and arch_minor <= minor:
            return True
        if kind == "compute" and (arch_major, arch_minor) <= (major, minor):
            return True   # PTX is forward-compatible via JIT
    return False


def cuda_arch_problem(index: int = 0) -> Optional[str]:
    """Return an explanatory message if this GPU has no usable kernels, else None."""
    if not torch.cuda.is_available():
        return None
    try:
        capability = torch.cuda.get_device_capability(index)
        arch_list = torch.cuda.get_arch_list()
        name = torch.cuda.get_device_name(index)
    except (RuntimeError, AssertionError):
        return None
    if not arch_list or arch_supports(capability, arch_list):
        return None

    major, minor = capability
    version = torch.__version__
    return (
        f"{name} is compute capability sm_{major}{minor}, but the installed "
        f"torch {version} only has kernels for: {', '.join(arch_list)}.\n"
        "Convolutions will fail at runtime with 'FIND was unable to find an engine "
        "to execute this computation'.\n"
        "Reinstall a build that covers this GPU, for example:\n"
        f"    pip install torch=={version.split('+')[0]} "
        "--index-url https://download.pytorch.org/whl/cu126\n"
        "(cu126 and cu118 wheels include Pascal/Turing kernels; CUDA 13 wheels do "
        "not.) Verify with:\n"
        "    python -c \"import torch; print(torch.cuda.get_arch_list())\"\n"
        "Or pass --device cpu to run on the CPU instead."
    )


def resolve_device(requested: Optional[str] = None) -> torch.device:
    """Pick the device, refusing a GPU whose architecture this wheel cannot serve.

    `requested` of None means "cuda if available". An unusable GPU raises rather
    than silently falling back, because a silent fall-back to CPU can waste hours
    before anyone notices.
    """
    if requested is not None and requested.split(":")[0] == "cpu":
        return torch.device("cpu")

    if requested is None and not torch.cuda.is_available():
        return torch.device("cpu")

    device = torch.device(requested or "cuda")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Pass --device cpu to run on the CPU."
            )
        problem = cuda_arch_problem(device.index or 0)
        if problem:
            raise RuntimeError(problem)
    return device
