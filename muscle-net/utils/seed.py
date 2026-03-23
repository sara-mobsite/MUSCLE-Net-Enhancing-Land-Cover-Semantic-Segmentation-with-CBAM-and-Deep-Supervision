import os
import random
import numpy as np
import torch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def set_seed(seed: int) -> torch.Generator:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    return torch.Generator().manual_seed(seed)


def seed_worker_factory(seed: int):
    def seed_worker(worker_id: int):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)
    return seed_worker
