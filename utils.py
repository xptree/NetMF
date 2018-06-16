#!/usr/bin/env python
# encoding: utf-8
# File Name: utils.py
# Author: Jiezhong Qiu
# Create Time: 2018/06/12 14:08
# TODO:

import scipy.io
import scipy.sparse as sp
import numpy as np

def load_adjacency_matrix(file, variable_name="network"):
    data = scipy.io.loadmat(file)
    return data[variable_name]

def svd_deepwalk_matrix(X, dim):
    u, s, v = sp.linalg.svds(X, dim, return_singular_vectors="u")
    # return U \Sigma^{1/2}
    return sp.diags(np.sqrt(s)).dot(u.T).T

