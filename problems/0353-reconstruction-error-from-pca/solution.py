import numpy as np

def pca_reconstruction_error(X: np.ndarray, n_components: int) -> float:
    """
    Compute the mean squared reconstruction error from PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_components: Number of principal components to keep
        
    Returns:
        The mean squared reconstruction error (float)
    """
    mean= np.mean(X, axis=0)

    X= X-mean
    cov= np.cov(X, rowvar=False)
    val,vec= np.linalg.eigh(cov)
    indices= np.flip(np.argsort(val))
    comps= vec[:,indices][:,:n_components]
    proj= X @ comps
    const= proj @ comps.T
    return np.mean(
        ((X.flatten()-const.flatten())**2)
    )