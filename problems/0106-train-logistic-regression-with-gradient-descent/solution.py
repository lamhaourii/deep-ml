import numpy as np

def sigmoid(z):
	return 1/(1+np.exp(-z))
def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	n,m= np.shape(X)
	w=np.zeros((m,1))
	b=np.zeros((1,1))
	y = y.reshape(-1, 1)
	losses=[]
	for i in range(iterations):
		z=np.dot(X,w)+b
		p=sigmoid(z)
		loss=0.
		for i in range(np.size(y)):
			loss+=y[i]*np.log(p[i]) + (1-y[i])*np.log(1-p[i])
		loss=-loss
		losses.append(loss[0])
		w-= learning_rate* np.dot((np.transpose(X)), (p-y))
		b-= learning_rate* np.sum(p-y, keepdims=True)
	return (list(np.round(np.concatenate([b,w], axis=0).flatten(),4)), losses)








