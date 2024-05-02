"""
Script to test the algorithms.
"""
import numpy as np

import tools as tl
import CRepDyn_w_STATIC as crep
import CRepDyn_w_DYN as crepw

import networkx as nx

import cv_functions as cvfun

def flt(x, d=3):
    return round(x, d)
# +

def expected_Aija(U, V, W):
    if W.ndim == 1:
        M = np.einsum('ik,jk->ijk', U, V)
        M = np.einsum('ijk,k->ij', M, W)
    else:
        M = np.einsum('ik,jq->ijkq', U, V)
        M = np.einsum('ijkq,kq->ij', M, W)
    return M

def check_shape_and_sum(matrix, expected_shape, expected_sum, matrix_name):
    assert matrix.shape == expected_shape, f"Expected {matrix_name} to have shape {expected_shape}, but got {matrix.shape}"
    assert np.isclose(np.sum(matrix), expected_sum, atol=TOLERANCE_1), f"Expected sum of {matrix_name} to be {expected_sum}, but got {np.sum(matrix)}"


# Constants
TOLERANCE_1 = 1e-3
TOLERANCE_2 = 1e-3

label = '100_2_5.0_4_0.2_0.2_0'

# Define the path to the data file
data_path = '../data/input/synthetic/'

# Then use this variable in your code
theta = np.load(data_path + 'theta_' + label + '.npz', allow_pickle=True)

K = theta['u'].shape[1]

# We load the synthetic network
network = data_path + 'syn_' + label + '.dat'
# We load the data
A, B, B_T, data_T_vals = tl.import_data(network, header=0)
# We get the nodes
nodes = A[0].nodes()
# We get the position of the nodes and how many there are
pos = nx.spring_layout(A[0])
N = len(nodes)

# ### Initializing near ground truth

T = B.shape[0] - 1

algo = 1  # 1: w_temp

print('='*65)
print('*****Test of the temporal version')

model = crepw.CRepDyn_w_temp(
    plot_loglik=True,
    verbose=1,
    N_real=1,
    beta0=theta['beta'],
    undirected=False,
    flag_data_T=0,
    fix_beta=False,
    initialization=2,
    in_parameters=data_path + 'theta_' + label,
    max_iter=800,
    end_file=label,
    eta0=0.2,
    constrained=False,
    ag=1.1,
    bg=0.5,
    fix_eta=False)


u, v, w, eta, beta, Loglikelihood = model.fit(data=B, T=T, nodes=nodes, K=K)

# For u
expected_u_shape = (100, 2)
expected_u_sum = 40.00025694829692
check_shape_and_sum(u, expected_u_shape, expected_u_sum, 'u')

# For v
expected_v_shape = (100, 2)
expected_v_sum = 40.001933007145794
check_shape_and_sum(v, expected_v_shape, expected_v_sum, 'v')

# For w
expected_w_shape = (5, 2, 2)
expected_w_sum = 3.0039155951245258
check_shape_and_sum(w, expected_w_shape, expected_w_sum, 'w')

# ### Eta and beta
expected_eta = 0.21687084165382248
expected_beta = 0.20967743180393628


assert np.isclose(eta, expected_eta,
                  atol=TOLERANCE_1), f"Expected eta to be close to {expected_eta}, but got {eta}"
assert np.isclose(beta, expected_beta,
                  atol=TOLERANCE_1), f"Expected beta to be close to {expected_beta}, but got {beta}"

# ### Loglikelihood
expected_loglikelihood = -2872.6923935067616


assert np.isclose(Loglikelihood, expected_loglikelihood,
                  atol=TOLERANCE_1), (f"Expected Loglikelihood to be close to {expected_loglikelihood},"
                                    f" but got {Loglikelihood}")

# ### AUC

expected_aucs = [0.811, 0.829, 0.841, 0.842, 0.843]


lambda_inf = expected_Aija(u, v, w[0])
M_inf = lambda_inf + eta * crep.transpose_tensor(B)

for l in range(model.T + 1):
    auc = flt(cvfun.calculate_AUC(M_inf[l], B[l].astype('int')))
    assert np.isclose(
        auc, expected_aucs[l], atol=TOLERANCE_2), (f"Expected AUC for index {l} to be close to "
                                                 f"{expected_aucs[l]}, but got {auc}")


# Testing the static one

algo = 0  # 0: static
print('='*65)

print('*****Test of the static version')

model = crep.CRepDyn(
    plot_loglik=True,
    verbose=1,
    N_real=5,
    beta0=0.25,
    undirected=False,
    flag_data_T=1,
    fix_beta=False,
    initialization=2,
    in_parameters=data_path + 'theta_' + label,
    max_iter=800,
    end_file=label,
    eta0=0.2,
    constrained=True,
    ag=1.1,
    bg=0.5,
    fix_eta=False)

u, v, w, eta, beta, Loglikelihood = model.fit(data=B, T=T, nodes=nodes, K=K)

# For u
expected_u_shape = (100, 2)
expected_u_sum = 100.0
check_shape_and_sum(u, expected_u_shape, expected_u_sum, 'u')

# For v
expected_v_shape = (100, 2)
expected_v_sum = 99.92123973890051
check_shape_and_sum(v, expected_v_shape, expected_v_sum, 'v')

# For w
expected_w_shape = (1, 2)
expected_w_sum = 0.03792499007908572
check_shape_and_sum(w, expected_w_shape, expected_w_sum, 'w')

# ### Eta and beta
expected_eta = 0.06141942760787744
expected_beta = 0.35602236108533

assert np.isclose(eta, expected_eta,
                  atol=TOLERANCE_1), f"Expected eta to be close to {expected_eta}, but got {eta}"
assert np.isclose(beta, expected_beta,
                  atol=TOLERANCE_1), f"Expected beta to be close to {expected_beta}, but got {beta}"

# ### Loglikelihood
expected_loglikelihood = -3174.04938026765

assert np.isclose(Loglikelihood, expected_loglikelihood,
                  atol=TOLERANCE_1), (f"Expected Loglikelihood to be close to "
                                    f"{expected_loglikelihood}, but got {Loglikelihood}")

# ### AUC

expected_aucs = [0.785, 0.806, 0.812, 0.816, 0.817]

lambda_inf = expected_Aija(u, v, w[0])
M_inf = lambda_inf + eta * crep.transpose_tensor(B)

for l in range(model.T + 1):
    auc = flt(cvfun.calculate_AUC(M_inf[l], B[l].astype('int')))
    assert np.isclose(
        auc, expected_aucs[l], atol=TOLERANCE_2), (f"Expected AUC for index {l} to be close to "
                                                 f"{expected_aucs[l]}, but got {auc}")

print('*****Test passed successfully!')