"""
	Main function to implement link prediction task  given  a number of communities.

	- Hold-out the data at the latest time snapshot (at time T);
	- Infer parameters on the observed data (data up to time T-1);
	- Calculate performance measures in the hidden set (AUC).
"""

import csv
import os
import pickle
from argparse import ArgumentParser
import cv_functions as cvfun 
import numpy as np
import tools as tl
import yaml
import sktensor as skt
import time

def main(): 
	p = ArgumentParser()
	p.add_argument('-a', '--algorithm', type=str, choices=['Crep_static','Crep_wtemp'], default='Crep_wtemp')  # configuration
	p.add_argument('-K', '--K', type=int, default=4)  # number of communities
	p.add_argument('-T', '--T', type=int, default=5)  # number of time snapshots 
	p.add_argument('-l', '--label', type=str,default='email-Eu-core')
	p.add_argument('-f', '--in_folder', type=str, default='../data/input/')  # path of the input network
	p.add_argument('-o', '--out_folder', type=str,default='../data/output/5-fold_cv/wtemp/')   # path to store outputs, will be generated automatically
	p.add_argument('-e', '--ego', type=str, default='source')  # name of the source of the edge
	p.add_argument('-t', '--alter', type=str, default='target')  # name of the target of the edge
	p.add_argument('-r', '--out_results', type=int, default=True)  # flag to output the results in a csv file
	p.add_argument('-i', '--out_inference', type=int, default=True)  # flag to output the inferred parameters
	p.add_argument('-A', '--assortative', type=int, default=0)  #flag to change the structure of the affinity matrix
	p.add_argument('-eta', '--fix_eta', type=int, default=True)  # flag to fix reciprocity coefficient 
	p.add_argument('-s', '--sep', type=str, default='\s+')  # flag to output the results in a csv file  
	p.add_argument('-na','--et0',type=float,default= 0.0) #initial value for the reciprocity coefficient
	p.add_argument('-nb','--bt0',type=float,default= 0.1) #initial value for the rate of edge removal 
	p.add_argument('-z', '--rseed', type=int, default=100)#seed to generate random number 
	p.add_argument('-E', '--end_file', type=str, default='.csv')   #the format of the output file
	

	args = p.parse_args()
	rseed = args.rseed
	end_file=args.end_file 

	'''
	Inference parameters and set up output directory
	'''
	out_results = bool(args.out_results)
	out_folder = args.out_folder
	if not os.path.exists(out_folder):
		os.makedirs(out_folder)

	algorithm = args.algorithm  # algorithm to use to generate the samples 
	with open('setting_' + algorithm + '.yaml') as f:
		conf = yaml.load(f, Loader=yaml.FullLoader)
	K = args.K

	conf['out_folder'] = out_folder
	conf['out_inference'] = bool(args.out_inference)
	conf['fix_eta'] = bool(args.fix_eta)
	conf['rseed'] = rseed

	if args.assortative == 0: conf['assortative'] = False
	if args.assortative == 1: 
		conf['assortative'] = True
	'''
	Model parameters
	'''
	flag_data_T = conf['flag_data_T']

	'''
	Import data
	'''
	label = args.label  
	print('='*30)
	print(label)
	network = args.in_folder+label + args.end_file
	
	A, B, B_T, data_T_vals = tl.import_data(network, ego=args.ego, alter=args.alter, force_dense=True, header=0,sep=args.sep,binary=True)  
	nodes = A[0].nodes() 
	valid_types = [np.ndarray, skt.dtensor, skt.sptensor]
	assert any(isinstance(B, vt) for vt in valid_types) 
 
	T = max(0, min(args.T, B.shape[0]-1))   

	print('\n### CV procedure ###', T)
	comparison = [0 for _ in range(16)]
	comparison[0] = algorithm if conf['fix_eta'] == 0 else algorithm+'0'
	comparison[1] = conf['constrained']
	comparison[2] = flag_data_T 

	comparison[3] = rseed
	comparison[4] = K 
	comparison[5] = args.et0
	comparison[6] = args.bt0

	conf['eta0'] = theta['eta'] if conf['fix_eta'] == False else 0. 
	'''
	output results
	'''
	cols = ['algo','constrained','flag_data_T', 'rseed','K', 'eta0','beta0','T']#7
	cols.extend(['eta','eta_aggr', 'beta', 'beta_aggr', 'auc', 'auc_aggr','loglik', 'loglik_aggr'])#15

	if out_results:
		out_file = out_folder + label + '_cv.csv'
		if not os.path.isfile(out_file):  # write header
			with open(out_file, 'w') as outfile:
				wrtr = csv.writer(outfile, delimiter=',', quotechar='"')
				wrtr.writerow(cols)
		outfile = open(out_file, 'a')
		wrtr = csv.writer(outfile, delimiter=',', quotechar='"')
		print(f'Results will be saved in: {out_file}')

	for t in range(1,T+1):  # skip first and last time step (last is hidden)
		if t == 1:conf['fix_beta'] = True # for the first time step beta cannot be inferred
		else:conf['fix_beta'] = False

		comparison[7] = t
		
		B_train = B[:t] # use data up to time t-1 for training
		print(B_train.shape)

		time_start = time.time()
		N = B_train.shape[-1]

		conf['end_file'] = label +'_'+ str(t) +'_'+ str(K) # needed to plot inferred theta

		'''
		Run CRep on the training 
		'''                         
		s = time.process_time() 
		u, v, w, eta, beta, maxL, algo_obj = cvfun.fit_model(B_train, t, nodes=nodes, K=K, algo=algorithm, **conf) 
		e = time.process_time() 

		'''
		Calculate performance results
		'''
		comparison[8] = eta
		comparison[10] = beta
		if flag_data_T == 1: # if 0: previous time step, 1: same time step
			M = cvfun.calculate_conditional_expectation(B[t-1],B[t], u, v, w, eta=eta, beta=beta) # use data_T at time t to predict t
		elif flag_data_T == 0:
			M = cvfun.calculate_conditional_expectation(B[t-1],B[t-1], u, v, w, eta=eta, beta=beta)# use data_T at time t-1 to predict t

		s = time.process_time() 
		loglik_test = cvfun.Likelihood_conditional(M,beta,B[t],B[t-1])
		e = time.process_time() 
		if t > 1:
			M[B[t-1].nonzero()] = 1 - beta # to calculate AUC
		
		comparison[12] = cvfun.calculate_AUC(M, B[t])
		comparison[14] = loglik_test
		'''
		Inference using aggregated data
		'''

		conf['end_file'] = label +'_'+ str(t) +'_'+ str(K) + '_aggre' # needed to plot inferred theta

		conf['fix_beta'] = True
		B_aggr = B_train.sum(axis=0)
		B_aggr[B_aggr>1] = 1 # binarize 

		u, v, w, eta, beta,maxL, algo_obj = cvfun.fit_model(B_aggr[np.newaxis,:,:], 0, nodes=nodes, algo=algorithm,K=K, **conf)
		comparison[9] = eta
		comparison[11] = beta

		M = cvfun.calculate_conditional_expectation(B_aggr,B_aggr, u, v, w, eta=eta, beta=1)

		loglik_test = cvfun.Likelihood_conditional(M, 1,B[t],B_aggr)
		# M[B[t-1].nonzero()] = 0 # to calculate AUC
		
		comparison[13] = cvfun.calculate_AUC(M, B[t])
		comparison[15] = loglik_test

		print(t,comparison)

		if out_results:
			wrtr.writerow(comparison)
			outfile.flush()

	if out_results:
		outfile.close()
		print(f'Results saved in: {out_file}')
		

if __name__ == '__main__':
	main()
