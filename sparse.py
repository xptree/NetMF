#!/usr/bin/env python
# encoding: utf-8
# File Name: sparse.py
# Author: Jiezhong Qiu
# Create Time: 2018/06/11 14:32
# TODO:



"""
https://arxiv.org/abs/1502.03496 """

import igraph
import itertools
import numpy as np
import argparse
import logging
import utils
import pygsp
import scipy.sparse as sp
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def random_walk(graph, start):
    current = start
    stop = False
    while not stop:
        stop = yield current
        current = np.random.choice(graph.neighbors(current))

def path_sampling(graph, length, A):
    """
        We want to sample a path p w.p. propotional to \tau(p)=w(p)Z(p),
        where w(p) and Z(p) are defined in Lemma 3.3 in
        https://arxiv.org/abs/1502.03496
    """
    e_idx = np.random.randint(graph.ecount())
    k = np.random.randint(length)+1
    u, v = graph.es[e_idx].tuple

    walk_u = list(itertools.islice(random_walk(graph, u), k))
    walk_v = list(itertools.islice(random_walk(graph, v), length-k+1))

    # a path from u to v
    path = walk_u[::-1] + walk_v

    # keep track of Z(path)
    Z = 0.
    for i in range(length):
        Z += 2. / A[path[i], path[i+1]]
    return path[0], path[-1], 2.*length/Z



def netmf_sparse(args):
    A = utils.load_adjacency_matrix(args.input,
            variable_name=args.matfile_variable_name)
    A.eliminate_zeros()
    A = A.astype(int)
    A = A.todok()

    n, m = A.shape[0], A.nnz
    edge_list = list()
    for item in A.items():
        (row, col), data = item
        assert A[col, row] == data
        for i in range(data):
            edge_list.append((row, col))

    graph = igraph.Graph(n)
    graph.add_edges(edge_list)

    """
    graph = pygsp.graphs.BarabasiAlbert(N=200, m0=2, m=2, seed=args.seed)
    graph.set_coordinates(kind='spring', seed=42)
    fig, axes = plt.subplots(1, 2)
    axes[0].spy(graph.W, markersize=2)
    graph.plot(ax=axes[1])
    plt.show()
    """

    # sample Tmlog(n)/eps^2
    logger.info("n=%d, m=%d", n, m)
    M = int(args.window * m * np.log(n) / args.eps / args.eps)+1
    M = 100000
    logger.info("total number of samples required = %d", M)


    sparsifier = sp.dok_matrix((n, n), dtype=np.float64)
    for i in range(M):
        if (i+1) % 100000 == 0:
            logger.info("%d paths sampled", i+1)
        length = np.random.randint(args.window) + 1
        u, v, weight = path_sampling(graph, length, A)
        weight = weight * m / M
        sparsifier[u, v] += weight
    logger.info("initial sparsifier done..")
    print(sparsifier.nnz)
    print(sparsifier.sum())

    #sparsifier = pygsp.graphs.Graph(sparsifier)
    #sparsifier = pygsp.reduction.graph_sparsify(sparsifier, epsilon=args.eps)
    #print(sparsifier.W)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='./data/ppi/Homo_sapiens.mat',
            help=".mat input file path")
    parser.add_argument('--matfile-variable-name', default='network',
            help='variable name of adjacency matrix inside a .mat file.')
    #parser.add_argument("--output", type=str, required=True,
    #        help="embedding output file path")

    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument("--rank", default=256, type=int,
            help="#eigenpairs used to approximate normalized graph laplacian.")
    parser.add_argument("--dim", default=128, type=int,
            help="dimension of embedding")
    parser.add_argument("--window", default=2,
            type=int, help="context window size")
    parser.add_argument("--negative", default=1.0, type=float,
            help="negative sampling")
    parser.add_argument("--eps", default=0.8, type=float,
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
    netmf_sparse(args)
