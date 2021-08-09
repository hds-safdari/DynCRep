"""
	Class for generation and management of synthetic networks with anomalies
"""

import math 
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from scipy.stats import poisson

from scipy.optimize import brentq, root

EPS = 1e-12

class CRepDyn(object):

	def __init__(self, N, K, T=1,eta=0., L = 1, avg_degree = 5., ExpM=None, prng=0,verbose=0, 
					beta=0.2, ag = 0.1, bg = 0.1, eta_dir = 0.5, L1=True, corr=1.,over=0., label=None,
					 end_file=".dat", undirected= False,folder='',  structure = 'assortative',
					 output_parameters = False,output_adj=False,outfile_adj=None
					 ):
		self.N = N 
		self.K = K
		self.T = T
		self.L = L 
		self.avg_degree = avg_degree
		self.prng = prng
		self.end_file = end_file
		self.undirected = undirected
		self.folder = folder
		self.output_parameters = output_parameters
		self.output_adj = output_adj
		self.outfile_adj = outfile_adj

		if label is not None:
			self.label = label
		else:
			self.label = ('_').join([str(N),str(K),str(avg_degree),str(T),str(eta),str(beta)])
		self.structure = structure
		print('='*30)
		print('self.structure:', self.structure)
		
		if ExpM is None: self.ExpM = self.avg_degree * self.N * 0.5
		else: self.ExpM = float(ExpM)

		# Set verbosity flag
		if verbose > 2 and not isinstance(verbose, int):
			raise ValueError('The verbosity parameter can only assume values in {0,1,2}!')
		self.verbose = verbose

		if eta < 0 :
			raise ValueError('The parameter eta has to be positive!')
		self.eta = eta

		if beta < 0 or beta > 1:
			raise ValueError('The parameter beta has to be in [0, 1]!')
		if beta == 1: beta = 1 - EPS
		if beta == 0: beta = EPS
		self.beta = beta
		

		### Set MT inputs
		# Set the affinity matrix structure
		if structure not in ['assortative', 'disassortative', 'core-periphery', 'directed-biased']:
			raise ValueError('The available structures for the affinity matrix w '
							 'are: assortative, disassortative, core-periphery '
							 'and directed-biased!')
   

		# Set alpha parameter of the Gamma distribution
		if ag <= 0 and not L1:
			raise ValueError('The Gamma parameter alpha has to be positive!')
		self.ag = ag
		# Set beta parameter of the Gamma distribution
		if bg <= 0 and not L1:
			raise ValueError('The Gamma parameter beta has to be positive!')
		self.bg = bg
		self.eta_dir = eta_dir
		# Set u,v generation preference
		self.L1 = L1
		# Set correlation between u and v synthetically generated
		if (corr < 0) or (corr > 1):
			raise ValueError('The correlation parameter has to be in [0, 1]!')
		self.corr = corr
		# Set fraction of nodes with mixed membership
		if (over < 0) or (over > 1):
				raise ValueError('The overlapping parameter has to be in [0, 1]!')
		self.over = over

	def Exp_ija_matrix(self,u,v,w):
		Exp_ija=np.einsum('ik,kq->iq',u,w)
		Exp_ija=np.einsum('iq,jq->ij',Exp_ija,v) 
		return Exp_ija

	def CRepDyn_network(self, parameters = None):
		"""
			Generate a directed, possibly weighted network by using CRep Dyn
			Steps:
				1. Generate a network A[0]
				2. Extract A[t] entries (network edges) using transition probabilities
			INPUT
			----------
			parameters : object
						 Latent variables eta, beta, u, v and w.
			OUTPUT
			----------
			G : Digraph
				DiGraph NetworkX object. Self-loops allowed.
		"""

		# Set seed random number generator
		prng = np.random.RandomState(self.prng)

		### Latent variables
		if parameters is None:
			# Generate latent variables
			self.u, self.v, self.w = self._generate_lv(prng)
		else:
			# Set latent variables
			self.u, self.v, self.w = parameters


		### Network generation
		G = [nx.DiGraph() for t in range(self.T+1)]
		for t in range(self.T+1):
			for i in range(self.N):
				G[t].add_node(i)

		# Compute M_ij
		M = self.Exp_ija_matrix(self.u, self.v,self.w)
		np.fill_diagonal(M, 0)

		# Set c sparsity parameter
		Exp_M_inferred = M.sum()
		c = self.ExpM / Exp_M_inferred 
		# t == 0
		for i in range(self.N):
			for j in range(self.N):
				if i != j:
					A_ij = prng.poisson(c * M[i,j], 1)[0]
					if A_ij > 0:
						G[0].add_edge(i, j, weight = 1) # binarized

		# t > 0
		for t in range(self.T):
			for i in range(self.N):
				for j in range(self.N):
					if i != j:

						if G[t].has_edge(j,i): # reciprocal edge determines Poisson rate
							lambda_ij = c * M[i,j] + self.eta
						else:
							lambda_ij = c * M[i,j] 

						if G[t].has_edge(i,j): # the edge at previous time step: determines the transition rate
							q = 1 - self.beta
						else:
							q = self.beta * lambda_ij
						r = prng.rand()
						if r <= q:
							G[t+1].add_edge(i, j, weight = 1) # binarized


		### Network post-processing
		nodes = list(G[0].nodes())
		assert len(nodes) == self.N
		A = [nx.to_scipy_sparse_matrix(G[t], nodelist=nodes, weight='weight') for t in range(len(G))]

		# Keep largest connected component
		A_sum = A[0].copy()
		for t in range(1,len(A)): A_sum += A[t]
		G_sum = nx.from_scipy_sparse_matrix(A_sum,create_using=nx.DiGraph)
		Gc = max(nx.weakly_connected_components(G_sum), key=len)
		nodes_to_remove = set(G_sum.nodes()).difference(Gc)
		G_sum.remove_nodes_from(list(nodes_to_remove))

		if self.output_adj:
			self._output_adjacency(nodes,A_sum,A,nodes_to_keep=list(G_sum.nodes()), outfile = self.outfile_adj)

		nodes = list(G_sum.nodes())

		for t in range(len(G)):
			G[t].remove_nodes_from(list(nodes_to_remove))

		if self.u is not None:
			self.u = self.u[nodes]
			self.v = self.v[nodes]
		self.N = len(nodes)

		if self.verbose > 0:print(f'Removed {len(nodes_to_remove)} nodes, because not part of the largest connected component')

		if self.verbose > 0:
			for t in range(len(G)):
				print('-'*30)
				print('t=',t)
				ave_w_deg = np.round(2 * G[t].number_of_edges() / float(G[t].number_of_nodes()), 3)   
				print(f'Number of nodes: {G[t].number_of_nodes()} \n'
					  f'Number of edges: {G[t].number_of_edges()}')
				print(f'Average degree (2E/N): {ave_w_deg}')
				print(f'Reciprocity at t: {nx.reciprocity(G[t])}')
				print('-'*30)

			self.check_reciprocity_tm1(A,A_sum)

		if self.output_parameters:
			self._output_results(nodes)

		if self.verbose == 2:
			self._plot_A(A)
			if M is not None: self._plot_M(M)

		return G

	def _generate_lv(self, prng = 42):
		"""
			Generate z, u, v, w latent variables.
			INPUT
			----------
			prng : int
				   Seed for the random number generator.
			OUTPUT
			----------
			u : Numpy array
				Matrix NxK of out-going membership vectors, positive element-wise.
				With unitary L1 norm computed row-wise.

			v : Numpy array
				Matrix NxK of in-coming membership vectors, positive element-wise.
				With unitary L1 norm computed row-wise.

			w : Numpy array
				Affinity matrix KxK. Possibly None if in pure SpringRank.
				Element (k,h) gives the density of edges going from the nodes
				of group k to nodes of group h.
		"""

		# Generate u, v for overlapping communities
		u, v = membership_vectors(prng, self.L1, self.eta_dir, self.ag, self.bg, self.K,
								 self.N, self.corr, self.over)
		# Generate w
		w = affinity_matrix(self.structure, self.N, self.K, self.avg_degree)
		return u, v, w

	def _build_multilayer_edgelist(self,nodes,A_tot,A,nodes_to_keep=None):
		A_coo = A_tot.tocoo()
		data_dict = {'source':A_coo.row,'target':A_coo.col}
		for t in range(len(A)):
			data_dict['weight_t'+str(t)] = np.squeeze(np.asarray(A[t][A_tot.nonzero()]))
		  
		df_res = pd.DataFrame(data_dict)
		print(len(df_res))
		if nodes_to_keep is not None:
			df_res = df_res[df_res.source.isin(nodes_to_keep) & df_res.target.isin(nodes_to_keep)]

		nodes = list(set(df_res.source).union(set(df_res.target)))
		id2node = {}
		for i,n in enumerate(nodes):id2node[i] = n

		df_res['source'] = df_res.source.map(id2node)
		df_res['target'] = df_res.target.map(id2node)
	
		return df_res

	def _output_results(self, nodes):
		"""
			Output results in a compressed file.
			INPUT
			----------
			nodes : list
					List of nodes IDs.
		""" 
		output_parameters = self.folder + 'theta_' + self.label + '_' + str(self.prng)
		np.savez_compressed(output_parameters + '.npz',  u=self.u, v=self.v,
							w=self.w, eta=self.eta, beta=self.beta, nodes=nodes)
		if self.verbose:
			print()
			print(f'Parameters saved in: {output_parameters}.npz')
			print('To load: theta=np.load(filename), then e.g. theta["u"]')

	def _output_adjacency(self,nodes,A_tot,A,nodes_to_keep=None, outfile = None):
		"""
			Output the adjacency matrix. Default format is space-separated .csv
			with 3 columns: node1 node2 weight
			INPUT
			----------
			G: Digraph
			   DiGraph NetworkX object.
			outfile: str
					 Name of the adjacency matrix.
		"""
		if outfile is None:
			outfile = 'syn_' + self.label + '_' + str(self.prng)  + '.dat'

		df = self._build_multilayer_edgelist(nodes,A_tot,A,nodes_to_keep=nodes_to_keep)
		df.to_csv(self.folder + outfile, index=False, sep=' ')
		if self.verbose:
			print(f'Adjacency matrix saved in: {self.folder + outfile}')

	def _plot_A(self, A, cmap = 'PuBuGn'):
		"""
			Plot the adjacency matrix produced by the generative algorithm.
			INPUT
			----------
			A : Scipy array
				Sparse version of the NxN adjacency matrix associated to the graph.
			cmap : Matplotlib object
				   Colormap used for the plot.
		"""
		for i in range(len(A)):
			Ad = A[i].todense()
			fig, ax = plt.subplots(figsize=(7, 7))
			ax.matshow(Ad, cmap = plt.get_cmap(cmap))
			ax.set_title('Adjacency matrix', fontsize = 15)
			for PCM in ax.get_children():
				if isinstance(PCM, plt.cm.ScalarMappable):
					break
			plt.colorbar(PCM, ax=ax)
			plt.show()

	def _plot_M(self, M, cmap = 'PuBuGn'):
		"""
			Plot the M matrix produced by the generative algorithm. Each entry is the
			poisson mean associated to each couple of nodes of the graph.
			INPUT
			----------
			M : Numpy array
				NxN M matrix associated to the graph. Contains all the means used
				for generating edges.
			cmap : Matplotlib object
				   Colormap used for the plot.
		"""

		fig, ax = plt.subplots(figsize=(7, 7))
		ax.matshow(M, cmap = plt.get_cmap(cmap))
		ax.set_title('MT means matrix', fontsize = 15)
		for PCM in ax.get_children():
			if isinstance(PCM, plt.cm.ScalarMappable):
				break
		plt.colorbar(PCM, ax=ax)
		plt.show()

	def check_reciprocity_tm1(self,A,A_sum):
		for t in range(1,len(A)):
			ref_subs = A[t].nonzero()
			M_t_T = A[t].transpose()[ref_subs]
			M_tm1_T = A[t-1].transpose()[ref_subs]
			nnz = float(A[t].count_nonzero())
			print(nnz,M_t_T.nonzero()[0].shape[0]/nnz,M_tm1_T.nonzero()[0].shape[0]/nnz)

def membership_vectors(prng = 10, L1 = False, eta_dir = 0.5, alpha = 0.6, beta = 1, K = 2, N = 100, corr = 0., over = 0.):
	"""
		Compute the NxK membership vectors u, v using a Dirichlet or a Gamma distribution.
		INPUT
		----------
		prng: Numpy Random object
			  Random number generator container.
		L1 : bool
			 Flag for parameter generation method. True for Dirichlet, False for Gamma.
		eta : float
			  Parameter for Dirichlet.
		alpha : float
			Parameter (alpha) for Gamma.
		beta : float
			Parameter (beta) for Gamma.
		N : int
			Number of nodes.
		K : int
			Number of communities.
		corr : float
			   Correlation between u and v synthetically generated.
		over : float
			   Fraction of nodes with mixed membership.
		OUTPUT
		-------
		u : Numpy array
			Matrix NxK of out-going membership vectors, positive element-wise.
			Possibly None if in pure SpringRank or pure Multitensor.
			With unitary L1 norm computed row-wise.

		v : Numpy array
			Matrix NxK of in-coming membership vectors, positive element-wise.
			Possibly None if in pure SpringRank or pure Multitensor.
			With unitary L1 norm computed row-wise.
	"""
	# Generate equal-size unmixed group membership
	size = int(N / K)
	u = np.zeros((N, K))
	v = np.zeros((N, K))
	for i in range(N):
		q = int(math.floor(float(i) / float(size)))
		if q == K:
			u[i:, K - 1] = 1.
			v[i:, K - 1] = 1.
		else:
			for j in range(q * size, q * size + size):
				u[j, q] = 1.
				v[j, q] = 1.
	# Generate mixed communities if requested
	if over != 0.:
		overlapping = int(N * over)  # number of nodes belonging to more communities
		ind_over = np.random.randint(len(u), size=overlapping)
		if L1:
			u[ind_over] = prng.dirichlet(eta * np.ones(K), overlapping)
			v[ind_over] = corr * u[ind_over] + (1. - corr) * prng.dirichlet(eta * np.ones(K), overlapping)
			if corr == 1.:
				assert np.allclose(u, v)
			if corr > 0:
				v = normalize_nonzero_membership(v)
		else:
			u[ind_over] = prng.gamma(alpha, 1. / beta, size=(N, K))
			v[ind_over] = corr * u[ind_over] + (1. - corr) * prng.gamma(alpha, 1. / beta, size=(overlapping, K))
			u = normalize_nonzero_membership(u)
			v = normalize_nonzero_membership(v)
	return u, v

def affinity_matrix(structure = 'assortative', N = 100, K = 2, avg_degree = 4., a = 0.1, b = 0.3):
	"""
		Compute the KxK affinity matrix w with probabilities between and within groups.
		INPUT
		----------
		structure : string
					Structure of the network.
		N : int
			Number of nodes.
		K : int
			Number of communities.
		a : float
			Parameter for secondary probabilities.
		OUTPUT
		-------
		p : Numpy array
			Array with probabilities between and within groups. Element (k,h)
			gives the density of edges going from the nodes of group k to nodes of group h.
	"""

	b *= a
	p1 = avg_degree * K / N

	if structure == 'assortative':
		p = p1 * a * np.ones((K,K))  # secondary-probabilities
		np.fill_diagonal(p, p1 * np.ones(K))  # primary-probabilities

	elif structure == 'disassortative':
		p = p1 * np.ones((K,K))   # primary-probabilities
		np.fill_diagonal(p, a * p1 * np.ones(K))  # secondary-probabilities

	elif structure == 'core-periphery':
		p = p1 * np.ones((K,K))
		np.fill_diagonal(np.fliplr(p), a * p1)
		p[1, 1] = b * p1

	elif structure == 'directed-biased':
		p = a * p1 * np.ones((K,K))
		p[0, 1] = p1
		p[1, 0] = b * p1 

	return p

def normalize_nonzero_membership(u):
	"""
		Given a matrix, it returns the same matrix normalized by row.
		INPUT
		----------
		u: Numpy array
		   Numpy Matrix.
		OUTPUT
		-------
		The matrix normalized by row.
	"""

	den1 = u.sum(axis=1, keepdims=True)
	nzz = den1 == 0.
	den1[nzz] = 1.

	return u / den1

def eq_c(c,M, N,E,rho_a,mu):

	return np.sum(np.exp(-c*M)) - (N**2 -N) + E * (1-rho_a) / (1-mu)






