#!/usr/bin/env python
# encoding: utf-8
# File Name: eigen.py
# Author: Jiezhong Qiu
# Create Time: 2017/07/13 16:05
# TODO:

import scipy.io
import scipy.sparse as sparse
from scipy.sparse import csgraph
import numpy as np
import argparse
import logging
import theano
from theano import tensor as T

logger = logging.getLogger(__name__)
theano.config.exception_verbosity='high'


def load_adjacency_matrix(file, variable_name="network"):
    data = scipy.io.loadmat(file)
    logger.info("loading mat file %s", file)
    return data[variable_name]

def deepwalk_filter(evals, window):
    for i in range(len(evals)):
        x = evals[i]
        evals[i] = 1. if x >= 1 else x*(1-x**window) / (1-x) / window
    evals = np.maximum(evals, 0)
    logger.info("After filtering, max eigenvalue=%f, min eigenvalue=%f",
            np.max(evals), np.min(evals))
    return evals

def approximate_normalized_graph_laplacian(A, rank, which="LA"):
    n = A.shape[0]
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    # X = D^{-1/2} W D^{-1/2}
    X = sparse.identity(n) - L
    logger.info("Eigen decomposition...")
    #evals, evecs = sparse.linalg.eigsh(X, rank,
    #        which=which, tol=1e-3, maxiter=300)
    evals, evecs = sparse.linalg.eigsh(X, rank, which=which)
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(evals), np.min(evals))
    logger.info("Computing D^{-1/2}U..")
    D_rt_inv = sparse.diags(d_rt ** -1)
    D_rt_invU = D_rt_inv.dot(evecs)
    return evals, D_rt_invU

def approximate_deepwalk_matrix(evals, D_rt_invU, window, vol, b):
    evals = deepwalk_filter(evals, window=window)
    X = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    m = T.matrix()
    mmT = T.dot(m, m.T) * (vol/b)
    f = theano.function([m], T.log(T.maximum(mmT, 1)))
    Y = f(X.astype(theano.config.floatX))
    logger.info("Computed DeepWalk matrix with %d non-zero elements",
            np.count_nonzero(Y))
    return sparse.csr_matrix(Y)

def svd_deepwalk_matrix(X, dim):
    u, s, v = sparse.linalg.svds(X, dim, return_singular_vectors="u")
    # return U \Sigma^{1/2}
    return sparse.diags(np.sqrt(s)).dot(u.T).T


def netmf_large(args):
    logger.info("Running NetMF for a large window size...")
    logger.info("Window size is set to be %d", args.window)
    # load adjacency matrix
    A = load_adjacency_matrix(args.input,
            variable_name=args.matfile_variable_name)
    vol = float(A.sum())
    # perform eigen-decomposition of D^{-1/2} A D^{-1/2}
    # keep top #rank eigenpairs
    evals, D_rt_invU = approximate_normalized_graph_laplacian(A, rank=args.rank, which="LA")

    # approximate deepwalk matrix
    deepwalk_matrix = approximate_deepwalk_matrix(evals, D_rt_invU,
            window=args.window,
            vol=vol, b=args.negative)

    # factorize deepwalk matrix with SVD
    deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=args.dim)

    logger.info("Save embedding to %s", args.output)
    np.save(args.output, deepwalk_embedding, allow_pickle=False)


def direct_compute_deepwalk_matrix(A, window, b):
    n = A.shape[0]
    vol = float(A.sum())
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    # X = D^{-1/2} A D^{-1/2}
    X = sparse.identity(n) - L
    S = np.zeros_like(X)
    X_power = sparse.identity(n)
    for i in range(window):
        logger.info("Compute matrix %d-th power", i+1)
        X_power = X_power.dot(X)
        S += X_power
    S *= vol / window / b
    D_rt_inv = sparse.diags(d_rt ** -1)
    M = D_rt_inv.dot(D_rt_inv.dot(S).T)
    m = T.matrix()
    f = theano.function([m], T.log(T.maximum(m, 1)))
    Y = f(M.todense().astype(theano.config.floatX))
    return sparse.csr_matrix(Y)

def netmf_small(args):
    logger.info("Running NetMF for a small window size...")
    logger.info("Window size is set to be %d", args.window)
    # load adjacency matrix
    A = load_adjacency_matrix(args.input,
            variable_name=args.matfile_variable_name)
    # directly compute deepwalk matrix
    deepwalk_matrix = direct_compute_deepwalk_matrix(A,
            window=args.window, b=args.negative)

    # factorize deepwalk matrix with SVD
    deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=args.dim)
    logger.info("Save embedding to %s", args.output)
    np.save(args.output, deepwalk_embedding, allow_pickle=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
            help=".mat input file path")
    parser.add_argument('--matfile-variable-name', default='network',
            help='variable name of adjacency matrix inside a .mat file.')
    parser.add_argument("--output", type=str, required=True,
            help="embedding output file path")

    parser.add_argument("--rank", default=256, type=int,
            help="#eigenpairs used to approximate normalized graph laplacian.")
    parser.add_argument("--dim", default=128, type=int,
            help="dimension of embedding")
    parser.add_argument("--window", default=10,
            type=int, help="context window size")
    parser.add_argument("--negative", default=1.0, type=float,
            help="negative sampling")

    parser.add_argument('--large', dest="large", action="store_true",
            help="using netmf for large window size")
    parser.add_argument('--small', dest="large", action="store_false",
            help="using netmf for small window size")
    parser.set_defaults(large=True)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(message)s') # include timestamp

    if args.large:
        netmf_large(args)
    else:
        netmf_small(args)

