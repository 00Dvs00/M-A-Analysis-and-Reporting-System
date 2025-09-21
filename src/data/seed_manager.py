"""Seed management for reproducible randomization."""
import os
import random
import numpy as np
import pandas as pd

def set_global_seed(seed: int = None):
    """Set global random seeds for reproducibility.
    
    Args:
        seed: Integer seed value. If None, uses SEED environment variable or defaults to 42.
    """
    if seed is None:
        seed = int(os.getenv('SEED', 42))
    
    # Set seeds for various random number generators
    random.seed(seed)
    np.random.seed(seed)
    
    # Set global pandas seed (affects operations like random sampling)
    pd.options.compute.use_numexpr = False  # Ensure deterministic behavior
    
    # Log seed value
    print(f"Set global random seed to: {seed}")
    return seed

def get_random_state():
    """Get the current random state for checkpointing."""
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state()
    }

def set_random_state(state):
    """Restore a previously captured random state."""
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])