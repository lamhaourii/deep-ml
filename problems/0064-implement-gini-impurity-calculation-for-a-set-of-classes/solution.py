
import numpy as np

def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
	_,freqs= np.unique(y,return_counts=True)
	val= 1 - np.sum((freqs/np.size(y))**2)
	return round(val,3)