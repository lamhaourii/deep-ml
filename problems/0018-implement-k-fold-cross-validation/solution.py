import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    indices= np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    extra= n_samples % k
    res=[]

    for i in range(0,n_samples,n_samples//k):
        fold=indices[i:i+n_samples//k]
        train= [ind for ind in indices if ind not in fold]
        res.append((train, fold))
    return res


