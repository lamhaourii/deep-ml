import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    # Implement Elastic Net regression here
    y = np.asarray(y).reshape(-1)
    w = np.zeros(X.shape[1])
    b = 0.0
    n = X.shape[0]
    for i in range(max_iter):
        y_pred= X@ w +b
        n=np.size(y)
        grad_w=(1/n)* np.dot(X.T,y_pred-y) +alpha1*(np.sign(w))+ 2*alpha2*w
        if np.sum(np.abs(grad_w)) <tol:
            break
            
        grad_b= (1/n)*np.sum(y_pred-y)
        w-= learning_rate* grad_w
        b-=learning_rate*grad_b
    return w,b


    