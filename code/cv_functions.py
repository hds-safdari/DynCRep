"""
	Functions used in the k-fold cross-validation procedure.
"""
 
import CRepDyn_w_DYN as crep
import numpy as np
from sklearn import metrics
import yaml
import sys


def Likelihood_conditional(M, beta,data,data_tm1,mask=None,EPS=1e-12):
	"""
		Compute the log-likelihood of the data conditioned in the previous time step

		Parameters
		----------
		data : sptensor/dtensor
			   Graph adjacency tensor.
		data_T : sptensor/dtensor
				 Graph adjacency tensor (transpose).
		mask : ndarray
			   Mask for selecting the held out set in the adjacency tensor in case of cross-validation.

		Returns
		-------
		l : float
			 log-likelihood value.
	"""  
	l = - M.sum()
	sub_nz_and = np.logical_and(data>0,(1-data_tm1)>0 )
	Alog = data[sub_nz_and] * (1-data_tm1)[sub_nz_and] * np.log(M[sub_nz_and]+EPS)  
	l += Alog.sum()  
	sub_nz_and = np.logical_and(data>0,data_tm1>0)
	l += np.log(1-beta+EPS) * (data[sub_nz_and] * data_tm1[sub_nz_and]).sum()
	sub_nz_and = np.logical_and(data_tm1>0,(1-data)>0) 
	l += np.log(beta+EPS) * ((1-data)[sub_nz_and] * data_tm1[sub_nz_and]).sum()
	if np.isnan(l):
		print("Likelihood is NaN!!!!")
		sys.exit(1)
	else:
		return l

def _lambda0_full(u, v, w):
	"""
		Compute the mean lambda0 for all entries.

		Parameters
		----------
		u : ndarray
			Out-going membership matrix.
		v : ndarray
			In-coming membership matrix.
		w : ndarray
			Affinity tensor.

		Returns
		-------
		M : ndarray
			Mean lambda0 for all entries.
	"""
 

	if w.ndim == 2:
		M = np.einsum('ik,jk->ijk', u, v)
		M = np.einsum('ijk,ak->ij', M, w)
	else:
		M = np.einsum('ik,jq->ijkq', u, v)
		M = np.einsum('ijkq,akq->ij', M, w)

	return M


def transpose_ij(M):
	"""
		Compute the transpose of a matrix.

		Parameters
		----------
		M : ndarray
			Numpy matrix.

		Returns
		-------
		Transpose of the matrix.
	"""

	return np.einsum('ij->ji', M)


def calculate_expectation(u, v, w, eta=0.0):
	"""
		Compute the expectations, e.g. the parameters of the marginal distribution m_{ij}.

		Parameters
		----------
		u : ndarray
			Out-going membership matrix.
		v : ndarray
			In-coming membership matrix.
		w : ndarray
			Affinity tensor.
		eta : float
			  Reciprocity coefficient.

		Returns
		-------
		M : ndarray
			Matrix whose elements are m_{ij}.
	"""

	lambda0 = _lambda0_full(u, v, w)
	lambda0T = transpose_ij(lambda0)
	M = (lambda0 + eta * lambda0T) / (1. - eta * eta)

	return M


def calculate_conditional_expectation(B,B_to_T, u, v, w, eta=0.0, beta=1.):
	"""
		Compute the conditional expectations, e.g. the parameters of the conditional distribution lambda_{ij}.

		Parameters
		----------
		B : ndarray
			Graph adjacency tensor.
		u : ndarray
			Out-going membership matrix.
		v : ndarray
			In-coming membership matrix.
		w : ndarray
			Affinity tensor.
		eta : float
			  Reciprocity coefficient.
		beta : float
			  rate of edge removal.
		mean : ndarray
			   Matrix with mean entries.

		Returns
		-------
		Matrix whose elements are lambda_{ij}.
	"""
	M = (beta * (_lambda0_full(u, v, w) + eta * transpose_ij(B_to_T)) ) / (1. + beta * (_lambda0_full(u, v, w) + eta * transpose_ij(B_to_T)) )
	return M

def calculate_AUC(pred, data0, mask=None):
	"""
		Return the AUC of the link prediction. It represents the probability that a randomly chosen missing connection
		(true positive) is given a higher score by our method than a randomly chosen pair of unconnected vertices
		(true negative).

		Parameters
		----------
		pred : ndarray
			   Inferred values.
		data0 : ndarray
				Given values.
		mask : ndarray
			   Mask for selecting a subset of the adjacency tensor.

		Returns
		-------
		AUC value.
	"""

	data = (data0 > 0).astype('int')
	if mask is None:
		fpr, tpr, thresholds = metrics.roc_curve(data.flatten(), pred.flatten())
	else:
		fpr, tpr, thresholds = metrics.roc_curve(data[mask > 0], pred[mask > 0])

	return metrics.auc(fpr, tpr)


def shuffle_indices_all_matrix(N, L, rseed=10):
	"""
		Shuffle the indices of the adjacency tensor.

		Parameters
		----------
		N : int
			Number of nodes.
		L : int
			Number of layers.
		rseed : int
				Random seed.

		Returns
		-------
		indices : ndarray
				  Indices in a shuffled order.
	"""

	n_samples = int(N * N)
	indices = [np.arange(n_samples) for _ in range(L)]
	rng = np.random.RandomState(rseed)
	for l in range(L):
		rng.shuffle(indices[l])

	return indices


def extract_mask_kfold(indices, N, fold=0, NFold=5):
	"""
		Extract a non-symmetric mask using KFold cross-validation. It contains pairs (i,j) but possibly not (j,i).
		KFold means no train/test sets intersect across the K folds.

		Parameters
		----------
		indices : ndarray
				  Indices of the adjacency tensor in a shuffled order.
		N : int
			Number of nodes.
		fold : int
			   Current fold.
		NFold : int
				Number of total folds.

		Returns
		-------
		mask : ndarray
			   Mask for selecting the held out set in the adjacency tensor.
	"""

	L = len(indices)
	mask = np.zeros((L, N, N), dtype=bool)
	for l in range(L):
		n_samples = len(indices[l])
		test = indices[l][fold * (n_samples // NFold):(fold + 1) * (n_samples // NFold)]
		mask0 = np.zeros(n_samples, dtype=bool)
		mask0[test] = 1
		mask[l] = mask0.reshape((N, N))

	return mask


def fit_model(data, T, nodes, K,algo = 'Crep_wtemp', **conf):
	"""
		Model directed networks by using a probabilistic generative model that assume community parameters and
		reciprocity coefficient. The inference is performed via EM algorithm.

		Parameters
		----------
		B : ndarray
			Graph adjacency tensor.
		B_T : None/sptensor
			  Graph adjacency tensor (transpose).
		data_T_vals : None/ndarray
					  Array with values of entries A[j, i] given non-zero entry (i, j).
		nodes : list
				List of nodes IDs.
		N : int
			Number of nodes.
		L : int
			Number of layers.
		algo : str
			   Configuration to use (CRep, CRepnc, CRep0).
		K : int
			Number of communities.

		Returns
		-------
		u_f : ndarray
			  Out-going membership matrix.
		v_f : ndarray
			  In-coming membership matrix.
		w_f : ndarray
			  Affinity tensor.
		eta_f : float
				Reciprocity coefficient.
		maxL : float
				 Maximum  log-likelihood.
		mod : obj
			  The CRep object.
	"""

	# setting to run the algorithm
	with open(conf['out_folder'] + '/setting_' + algo + '.yaml', 'w') as f:
		yaml.dump(conf, f) 

		if algo in  ['Crep_static', 'Crep_wtemp']: 
			model = crep.CRepDyn_w_temp(**conf) 
			uf, vf, wf, etaf,betaf, maxL = model.fit(T=T, data=data, K=K, nodes=nodes)
		else:
			raise ValueError('algo is invalid',algo)

	return uf, vf, wf, etaf, betaf, maxL,model


def CalculatePermuation(U_infer,U0):  
	"""
	Permuting the overlap matrix so that the groups from the two partitions correspond
	U0 has dimension NxK, reference memebership
	"""
	N,RANK=U0.shape
	M=np.dot(np.transpose(U_infer),U0)/float(N);   #  dim=RANKxRANK
	rows=np.zeros(RANK);
	columns=np.zeros(RANK);
	P=np.zeros((RANK,RANK));  # Permutation matrix
	for t in range(RANK):
	# Find the max element in the remaining submatrix,
	# the one with rows and columns removed from previous iterations
		max_entry=0.;c_index=1;r_index=1;
		for i in range(RANK):
			if columns[i]==0:
				for j in range(RANK):
					if rows[j]==0:
						if M[j,i]>max_entry:
							max_entry=M[j,i];
							c_index=i;
							r_index=j;
	 
		P[r_index,c_index]=1;
		columns[c_index]=1;
		rows[r_index]=1;

	return P

def cosine_similarity(U_infer,U0):
	"""
	It is assumed that matrices are row-normalized  
	"""
	P=CalculatePermuation(U_infer,U0) 
	U_infer=np.dot(U_infer,P);      # Permute infered matrix
	N,K=U0.shape
	U_infer0=U_infer.copy()
	U0tmp=U0.copy()
	cosine_sim=0.
	norm_inf=np.linalg.norm(U_infer,axis=1)
	norm0=np.linalg.norm(U0,axis=1  )
	for i in range(N):
		if(norm_inf[i]>0.):U_infer[i,:]=U_infer[i,:]/norm_inf[i]
		if(norm0[i]>0.): U0[i,:]=U0[i,:]/norm0[i]
	   
	for k in range(K):
		cosine_sim+=np.dot(np.transpose(U_infer[:,k]),U0[:,k])
	U0=U0tmp.copy()
	return U_infer0,cosine_sim/float(N) 


def normalize_nonzero_membership(U):
	"""
		Given a matrix, it returns the same matrix normalized by row.

		Parameters
		----------
		U: ndarray
		   Numpy Matrix.

		Returns
		-------
		The matrix normalized by row.
	"""

	den1 = U.sum(axis=1, keepdims=True)
	nzz = den1 == 0.
	den1[nzz] = 1.

	return U / den1

def evalu(U_infer, U0, metric='f1', com=False):
	"""
		Compute an evaluation metric.

		Compare a set of ground-truth communities to a set of detected communities. It matches every detected
		community with its most similar ground-truth community and given this matching, it computes the performance;
		then every ground-truth community is matched with a detected community and again computed the performance.
		The final performance is the average of these two metrics.

		Parameters
		----------
		U_infer : ndarray
				  Inferred membership matrix (detected communities).
		U0 : ndarray
			 Ground-truth membership matrix (ground-truth communities).
		metric : str
				 Similarity measure between the true community and the detected one. If 'f1', it used the F1-score,
				 if 'jaccard', it uses the Jaccard similarity.
		com : bool
			  Flag to indicate if U_infer contains the communities (True) or if they have to be inferred from the
			  membership matrix (False).

		Returns
		-------
		Evaluation metric.
	"""

	if metric not in {'f1', 'jaccard'}:
		raise ValueError('The similarity measure can be either "f1" to use the F1-score, or "jaccard" to use the '
						 'Jaccard similarity!')

	K = U0.shape[1]

	gt = {}
	d = {}
	threshold = 1 / U0.shape[1]
	for i in range(K):
		gt[i] = list(np.argwhere(U0[:, i] > threshold).flatten())
		if com:
			try:
				d[i] = U_infer[i]
			except:
				pass
		else:
			d[i] = list(np.argwhere(U_infer[:, i] > threshold).flatten())
	# First term
	R = 0
	for i in np.arange(K):
		ground_truth = set(gt[i])
		_max = -1
		M = 0
		for j in d.keys():
			detected = set(d[j])
			if len(ground_truth & detected) != 0:
				precision = len(ground_truth & detected) / len(detected)
				recall = len(ground_truth & detected) / len(ground_truth)
				if metric == 'f1':
					M = 2 * (precision * recall) / (precision + recall)
				elif metric == 'jaccard':
					M = len(ground_truth & detected) / len(ground_truth.union(detected))
			if M > _max:
				_max = M
		R += _max
	# Second term
	S = 0
	for j in d.keys():
		detected = set(d[j])
		_max = -1
		M = 0
		for i in np.arange(K):
			ground_truth = set(gt[i])
			if len(ground_truth & detected) != 0:
				precision = len(ground_truth & detected) / len(detected)
				recall = len(ground_truth & detected) / len(ground_truth)
				if metric == 'f1':
					M = 2 * (precision * recall) / (precision + recall)
				elif metric == 'jaccard':
					M = len(ground_truth & detected) / len(ground_truth.union(detected))
			if M > _max:
				_max = M
		S += _max

	return np.round(R / (2 * len(gt)) + S / (2 * len(d)), 4)
