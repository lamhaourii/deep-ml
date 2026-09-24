import numpy as np


def train_clf(X,y,w):
	best_error= float('inf')
	best_th= float('inf')
	
	best_feat= 0
	for f in range(X.shape[1]):
		ths= np.unique(X[:,f])
		for th in ths:
			polarity= 1
			preds= X[:,f]>=th
			preds= np.where(preds==0, -1, preds)
			error= np.sum(w * (preds != y))
			if error > 0.5:
				error= 1- error
				polarity*= -1
				preds*= -1

			if error<best_error:
				best_error= error
				best_feat= f
				best_th= th
				best_preds= preds
				best_polarity= polarity
	
	alpha= 0.5* np.log(
		(1- best_error)/ (best_error+ 1e-10)
	)
	return best_preds,{
		'polarity': best_polarity,
		'threshold': best_th,
		'feature_index': best_feat,
		'alpha':alpha
	}
def adaboost_fit(X, y, n_clf):
	n_samples, n_features = np.shape(X)
	w = np.full(n_samples, (1 / n_samples))
	clfs = []
	for _ in range(n_clf):
		preds, clf= train_clf(X, y, w)
		clfs.append(clf)

		w*= np.exp(-clf['alpha']* (y*preds))
		w/= np.sum(w)


	return clfs
    