#!/usr/bin/env python
# encoding: utf-8
# File Name: sparse.py
# Author: Jiezhong Qiu
# Create Time: 2018/06/11 14:32
# TODO:



"""
https://arxiv.org/abs/1502.03496
"""

import os
import pickle as pkl
import igraph
import itertools
import numpy as np
import argparse
import logging
import utils
import scipy.sparse as sp
import multiprocessing as mp
from collections import Counter

logger = logging.getLogger(__name__)

def random_walk(graph, start):
    current = start
    stop = False
    while not stop:
        stop = yield current
        current = np.random.choice(graph.neighbors(current))

def nth(iterable, n, default=None):
    return next(itertools.islice(iterable, n, None), default)

# https://stackoverflow.com/questions/32598422/why-my-python-multiprocessing-code-return-the-same-result-in-randomized-number
# path_sampling can run in parallel
def path_sampling(graph, length):
    """
        We want to sample a path p w.p. propotional to \tau(p)=w(p)Z(p),
        where w(p) and Z(p) are defined in Lemma 3.3 in
        https://arxiv.org/abs/1502.03496
    """
    e_idx = np.random.randint(graph.ecount())
    k = np.random.randint(length)+1
    u, v = graph.es[e_idx].tuple
    if np.random.randint(2) == 1:
        u, v = v, u

    x = nth(random_walk(graph, u), k-1)
    y = nth(random_walk(graph, v), length-k)

    return x, y, 1.


def path_sampling_mp(graph, seed, sample, T):
    np.random.seed(seed)
    rd = sample // graph.ecount()
    logger.info("pid %d, will process %d rounds", os.getpid(), rd)
    cnt = Counter()
    for i in range(rd):
        for e in graph.es:
            u, v = e.tuple
            length = np.random.randint(T)+1
            k = np.random.randint(length)+1
            x = nth(random_walk(graph, u), k-1)
            y = nth(random_walk(graph, v), length-k)
            cnt[(max(x, y), min(x, y))] += 1
        logger.info("pid %d, %d rounds done", os.getpid(), i+1)
    return cnt

def dispatch_jobs(G, num_job, subsample, args):
    pool = mp.Pool()
    res_async = []
    for i in range(num_job):
        seed = args.seed + i
        res = pool.apply_async(path_sampling_mp, args=(G, seed, subsample, args.window))
        res_async.append(res)

    pool.close()
    pool.join()
    cnt = Counter()
    for res in res_async:
        cnt += res.get()
    return cnt

def netmf_sparse_sampling(args):
    A = utils.load_adjacency_matrix(args.input,
            variable_name=args.matfile_variable_name)
    A.eliminate_zeros()
    A = A.astype(int)
    A = sp.tril(A).todok()

    n, m = A.shape[0], A.nnz
    edge_list = list()
    for item in A.items():
        (row, col), data = item
        for i in range(data):
            edge_list.append((row, col))

    graph = igraph.Graph(n)
    graph.add_edges(edge_list)
    graph.to_undirected(mode='each')

    # sample Tmlog(n)/eps^2
    #logger.info("n=%d, m=%d", n, m)
    sample = args.window * m * np.log(n) / args.eps / args.eps
    sample = int(sample / args.job / graph.ecount() + 1) * args.job * graph.ecount()
    #sample = args.job * graph.ecount()
    logger.info("total number of samples required = %d", sample)

    cnt = dispatch_jobs(graph, args.job, sample//args.job, args)
    with open("path.pkl", "wb") as f:
        pkl.dump(cnt, f)
    return cnt


def netmf_sparse_factorization(args, cnt):
    A = utils.load_adjacency_matrix(args.input,
            variable_name=args.matfile_variable_name)
    A.eliminate_zeros()
    vol = float(A.sum())
    _, d_root = sp.csgraph.laplacian(A, normed=True, return_diag=True)
    D_inv = sp.diags(d_root ** -2)
    A = A.astype(int)
    A = sp.tril(A).todok()

    n, m = A.shape[0], A.nnz

    sample = sum(cnt.values())
    logger.info("total number of samples = %d", sample)
    items = list()
    for k, v in cnt.items():
        weight = float(v) * m / sample
        items.append((k[0], k[1], weight))
        if k[0] != k[1]:
            items.append((k[1], k[0], weight))
    rows, cols, data = zip(*items)

    sparsifier = sp.coo_matrix((data, (rows, cols)), shape=(n,n), dtype=np.float64)
    logger.info("initial sparsifier done, with %d nnz, and vol=%.2f", sparsifier.nnz, sparsifier.sum())
    L_sp, d_sp = sp.csgraph.laplacian(sparsifier, normed=False, return_diag=True)
    M = sp.diags(d_sp) - L_sp

    M = D_inv.dot(D_inv.dot(M).T)
    print(M.min(), M.max())
    M = M.maximum(0)
    print(M.min(), M.max())
    M = M.multiply(vol/args.negative)
    print(M.min(), M.max())
    M = M.log1p()
    print(M.min(), M.max())

    deepwalk_embedding = utils.svd_deepwalk_matrix(M, dim=args.dim)

    logger.info("Save embedding to %s", args.output)
    np.save(args.output, deepwalk_embedding, allow_pickle=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='blogcatalog.mat',
            help=".mat input file path")
    parser.add_argument("--path", type=str, default='path.pkl',
            help="sampled paths (.pkl)")
    parser.add_argument("--output", type=str, required=True,
            help="embedding output file path")
    parser.add_argument('--matfile-variable-name', default='network',
            help='variable name of adjacency matrix inside a .mat file.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument("--rank", default=256, type=int,
            help="#eigenpairs used to approximate normalized graph laplacian.")
    parser.add_argument("--dim", default=128, type=int,
            help="dimension of embedding")
    parser.add_argument("--window", default=10,
            type=int, help="context window size")
    parser.add_argument("--job", default=24,
            type=int, help="number of jobs")
    parser.add_argument("--negative", default=1.0, type=float,
            help="negative sampling")
    parser.add_argument("--eps", default=0.1, type=float,
            help="approximation error")

    parser.add_argument('--large', dest="large", action="store_true",
            help="using netmf for large window size")
    parser.add_argument('--small', dest="large", action="store_false",
            help="using netmf for small window size")
    parser.set_defaults(large=True)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(message)s') # include timestamp

    np.random.seed(args.seed)
    if len(args.path) > 0:
        logger.info("path samples exists at %s, loading...", args.path)
        with open(args.path, "rb") as f:
            cnt = pkl.load(f)
        logger.info("loaded!")
    else:
        cnt = netmf_sparse_sampling(args)

    np.random.seed(args.seed)
    netmf_sparse_factorization(args, cnt)
