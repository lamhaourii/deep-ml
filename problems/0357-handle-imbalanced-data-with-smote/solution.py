import numpy as np

def smote(X_minority: np.ndarray, n_synthetic: int, k: int = 5) -> np.ndarray:
    """
    Generate synthetic samples using SMOTE algorithm.

    Note: the random seed is set by the grader before your function runs,
    so you do NOT need to set it. Just use numpy's global RNG directly
    (np.random.randint, np.random.random, ...).

    Args:
        X_minority: 2D array of minority class samples (n_samples, n_features)
        n_synthetic: Number of synthetic samples to generate
        k: Number of nearest neighbors to consider

    Returns:
        2D array of synthetic samples (n_synthetic, n_features)
    """
    #ur code here Yo
    n_s, n_f= X_minority.shape
    k_actual= min(k, n_s-1)
    if k_actual==0 or n_synthetic==0:
        return np.zeros((0,n_f))
    X_syn= np.zeros((n_synthetic,n_f))
    for _ in range(n_synthetic):
        i= np.random.randint(0, n_s)
        xi= X_minority[i]
        d= np.array([np.sqrt(np.sum((X_minority[j,:]-xi)**2)) for j in range(n_s) if j!=i]).argsort()
        d= np.where(d>=i, d+1, d)
        d= X_minority[d]
        j= np.random.randint(0, k_actual)
        xnn= d[j]
        gap= np.random.random()
        x_syn= xi+ gap*(xnn- xi)
        X_syn[_]=x_syn
    return X_syn
    







