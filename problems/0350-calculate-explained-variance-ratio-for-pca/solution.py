import numpy as np

def explained_variance_ratio(X):
    """
    Calculate the explained variance ratio for PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
    
    Returns:
        List of explained variance ratios sorted in descending order
    """
    # Your code here
    mean= np.mean(X, axis=0)

    X= X-mean
    cov= np.cov(X, rowvar=False,ddof=1)
    val,vec= np.linalg.eigh(cov)
    prop= np.abs(val)/np.sum(val)

    return np.flip(np.sort(prop))

    