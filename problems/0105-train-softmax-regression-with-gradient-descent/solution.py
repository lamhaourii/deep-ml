import numpy as np

def train_softmaxreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for Softmax regression, optimizing parameters with Cross Entropy loss.
	"""
	c=max(y)+1
	cls= set(y)
	X= np.concatenate([np.ones((X.shape[0],1)),X], axis=1)
	W= np.zeros((c,X.shape[1]))
	losses=[]
	Y= np.array([np.eye(c)[yi] for yi in y])
	for i in range(iterations):
		P= (np.exp(X@ W.T))/np.sum(np.exp(X@ W.T), axis=1, keepdims=True)
		loss= -Y*np.log(P)
		loss=np.sum(loss.flatten())
		losses.append(loss)
		W-=(learning_rate*(X.T @(P -Y))).T

	return W.tolist(), losses
		
		



	