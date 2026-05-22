

def set_all_seeds(seed: int, deterministic: bool = False):
    """Seed all RNGs for reproducibility.

    Args:
        seed: The random seed.
        deterministic: If True, force bit-identical results across runs
            (slower — disables cuDNN autotuner). If False, results may
            vary slightly between runs but training is faster.
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic