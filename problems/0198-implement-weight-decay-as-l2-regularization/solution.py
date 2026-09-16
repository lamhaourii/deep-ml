import numpy as np
def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
	"""
	Apply weight decay (L2 regularization) to parameters.
	
	Args:
		parameters: List of parameter arrays
		gradients: List of gradient arrays
		lr: Learning rate
		weight_decay: Weight decay factor
		apply_to_all: Boolean list indicating which parameter groups get weight decay
	
	Returns:
		Updated parameters
	"""
	return ((1-lr*weight_decay*np.array(apply_to_all))*np.array(parameters) - lr*np.array(gradients)).tolist()