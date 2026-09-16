import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:

	y_pred= X@ w 
	n= np.size(y_true)
	return (1/n)* np.sum((y_true-y_pred)**2) + alpha* np.sum(w**2)
