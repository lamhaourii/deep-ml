import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Your code here
    mean= np.mean(data, axis=0)
    std= np.std(data, axis=0)
    data= (data-mean)/std
    cov= np.cov(data, rowvar=False)
    val,vec= np.linalg.eigh(cov)
    for i in range(vec.shape[1]):
        for e in vec[:,i]:
            if abs(e)>1e-10 and e<0:
                vec[:,i]*=-1
            break
    indices= np.flip(np.argsort(val))

    sort_vec= vec[:,indices]
    return np.round(sort_vec[:,:k],4)


