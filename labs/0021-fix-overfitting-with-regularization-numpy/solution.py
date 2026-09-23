import numpy as np

from itertools import product


def elastic_net_grads(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
) -> tuple:
    y_pred= X@ w +b
    n = X.shape[0]
    grad_w=(1/n)* np.dot(X.T,y_pred-y) +alpha1*(np.sign(w))+ 2*alpha2*w
    grad_b= (1/n)*np.sum(y_pred-y)
    return grad_w, grad_b
    
def loss(y,yp, w,alpha1, alpha2):
    n= np.size(y)
    return (1/(2*n))*(
        np.sum((y-yp)**2)
    )+ alpha1* np.sum(
        np.abs(w)
        )+ alpha2*np.sum(
            w**2
        )


def explained_variance_ratio(X):
    mean= np.mean(X, axis=0)
    X= X-mean
    cov= np.cov(X, rowvar=False,ddof=1)
    val,vec= np.linalg.eigh(cov)
    prop= np.abs(val)/np.sum(val)
    return np.flip(np.sort(prop))

def pca(data: np.ndarray, k: int) -> np.ndarray:
    mean= np.mean(data, axis=0)
    data= (data-mean)
    cov= np.cov(data, rowvar=False)
    val,vec= np.linalg.eigh(cov)
    for i in range(vec.shape[1]):
        for e in vec[:,i]:
            if abs(e)>1e-10 and e<0:
                vec[:,i]*=-1
            break
    indices= np.flip(np.argsort(val))

    sort_vec= vec[:,indices]
    return sort_vec[:,:k]

def fit_pca(X,var):
    mean= np.mean(X, axis=0)
    _,f= X.shape
    k=f
    exp_var= explained_variance_ratio(X)
    for i in range(1, f):
        exp_var[i]+=exp_var[i-1]
        if exp_var[i]>=var:
            k=i+1
            break
    
    comps= pca(X,k)
    return comps, mean

def transform_pca(X, comps, mean):
    return (X-mean)@ comps



def validate(X, y, w, b, alpha1=0., alpha2=0.):
    yp= X@ w+ b
    return loss(y,yp, w, alpha1, alpha2)
def train(X_train, y_train, X_val, y_val):
    """
    Train a regression model that generalizes well despite having
    MORE features than training samples (many are noise or redundant).
    
    WARNING: An unregularized approach WILL overfit here.
    - Unregularized OLS: Train RÂ² â 1.0, Val RÂ² â -5.0
    - You need regularization to pass!
    
    Args:
        X_train: numpy array of shape (n_samples, n_features) -- standardized
                 (~250 samples, ~264 features -- more features than samples!)
        y_train: numpy array of shape (n_samples,) -- target values
        X_val:   numpy array of shape (n_val, n_features) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation targets
    
    Returns:
        predict: callable that takes X (n, n_features) and returns y_pred (n,)
    """
    # TODO: implement a training strategy that avoids overfitting
    # Some ideas:
    #   - Ridge regression (L2): add alpha * ||W||^2 to loss
    #   - Gradient descent with L2 penalty
    #   - Feature selection (remove low-variance or uncorrelated features)
    #   - Early stopping on gradient descent
    #   - Any combination of the above!

    #standarization
    
    #pca
    comps, mean= fit_pca(X_train, 0.95)
    X_train= transform_pca(X_train, comps, mean)
    X_val= transform_pca(X_val, comps, mean)
    
    y_train= np.asarray(y_train).reshape(-1)
    y_val= np.asarray(y_val).reshape(-1)
    

    param_grid={
        'lrs':[0.0001,0.001,0.01],
        'alphas1':[0.01, 0.1, 0.3, 0.5, 0.7, 0.9],
        'alphas2':[ 0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
    }
    combos= list(product(*[param_grid[key] for key in param_grid.keys()]))
    epochs= 20000
    best_w= np.zeros(X_train.shape[1])
    best_b= 0.0
    delta= 1e-4
    over_all_best_val= float('inf')
    for params in combos:
        w = np.zeros(X_train.shape[1])
        b = 0.0
        patience= 5
        no_imp=0
        best_val= float('inf')
        lr, alpha1, alpha2 = params
        for epoch in range(epochs): 
            grad_w,grad_b= elastic_net_grads(X_train, y_train, w, b,alpha1,alpha2)
            w-=lr* grad_w
            b-=lr*grad_b
            val_loss= validate(X_val, y_val, w,b)
            if val_loss< best_val- delta:
                best_val=val_loss
                w_=w.copy()
                b_= b
                no_imp=0
            else:
                no_imp+=1
            if no_imp==patience:
                break
        if best_val< over_all_best_val:
            best_w=w_.copy()
            best_b= b_
            best_params= params
            over_all_best_val= best_val
    def predict(X):
        X= transform_pca(X, comps, mean)
        return X@ best_w + best_b
    return predict

            


    






