import numpy as np

def gaussian_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
	"""
	Implements Gaussian Naive Bayes classifier.
	
	Args:
		X_train: Training features (shape: N_train x D)
		y_train: Training labels (shape: N_train)
		X_test: Test features (shape: N_test x D)
	
	Returns:
		Predicted class labels for X_test (shape: N_test)
	"""
	# Your code here
	_,counts= np.unique(y_train, return_counts=True)
	prior= np.log(counts/(np.size(y_train)))
	likes=[]
	for d in np.unique(y_train):
		mean= np.mean(X_train[y_train==d],axis=0)
		var_= np.var(X_train[y_train==d],axis=0)+ 1e-9
		like=-(1/2) * np.log(2*np.pi*var_) - (
				(X_test-mean)**2/(2*var_)
			)
		likes.append(like)
	scores= prior+ np.array([np.sum(like,axis=1) for like in likes]).T
	return np.argmax(scores, axis=1)
	

		
