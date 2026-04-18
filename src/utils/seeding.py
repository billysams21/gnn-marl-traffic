import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Recommended for deterministic GEMM on CUDA when deterministic algorithms are enabled
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=False)

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(False)

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
