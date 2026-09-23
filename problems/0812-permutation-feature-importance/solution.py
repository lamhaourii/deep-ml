import numpy as np

def r_sq(yt, yp):
    m= np.mean(yt)
    return 1-(
        np.sum((yp-yt)**2)/np.sum((yt-m)**2)
    )
def permutation_importance(X, y, weights, bias, permutations):
    """
    Compute permutation feature importance for a linear regression model.

    Args:
        X: 2D array-like of shape (n_samples, n_features)
        y: 1D array-like of shape (n_samples,)
        weights: 1D array-like of shape (n_features,)
        bias: float, the intercept
        permutations: list of length n_features; permutations[j] is a list of
                      permutation index arrays to apply to column j

    Returns:
        List of feature importances (length n_features).
    """
    out= []
    X= np.array(X)
    y= np.array(y)
    weights= np.array(weights)
    bias= np.array(bias)
    for i,f in enumerate(permutations):
        rs=[]
        ra=r_sq(y, X@ weights+ bias)
        for p in f:
            X_n= X.copy()
            X_n[:,i]=X_n[:,i][p]
            yp= X_n@ weights+ bias
            r= r_sq(y, yp)
            rs.append(r)
        out.append(ra- np.mean(rs))
    return out



