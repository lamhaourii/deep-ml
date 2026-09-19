import numpy as np
from typing import List, Tuple

def k_fold(n_samples: int, k: int = 5) -> List[Tuple[List[int], List[int]]]:
   
    indices= np.arange(n_samples)
	base_size= n_samples//k
    extra= n_samples % k
    res=[]
	start=0
    for i in range(0,n_samples,n_samples//k):
		fold_size= base_size + (1 if i< extra else 0)
        fold=indices[start:start+ fold_size]
        train= [ind for ind in indices if ind not in fold]
        res.append([train, fold])
		start+=fold_size

    return res




def stratified_kfold_indices(y, n_splits):
	"""
	Generate train/test indices for stratified K-fold cross-validation.

	Args:
		y: 1D array-like of integer class labels
		n_splits: number of folds

	Returns:
		A list of [train_indices, test_indices] pairs, one per fold.
	"""
	cls, cnts= np.unique(y, return_counts=True)
	cls_flds= []
	for c in cls:
		ind = np.where(y == c)[0]
		flds= k_fold(np.size(ind), n_splits)

		class_folds = []

		for train, test in flds:
			train = ind[train]
			test = ind[test]
			class_folds.append([train, test])
		cls_flds.append(class_folds)

	res=[]
	for i in range(n_splits):
		test= np.concatenate([cls_flds[c][i][1] for c in range(len(cls))])
		test= np.sort(test)
		all_ind= np.arange(len(y))
		train=np.array([ind for ind in all_ind if ind not in test])
		res.append([train.tolist(), test.tolist()])
	return res