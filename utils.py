#!/usr/bin/env python
# encoding: utf-8
# File Name: utils.py
# Author: Jiezhong Qiu
# Create Time: 2018/06/12 14:08
# TODO:

import scipy.io

def load_adjacency_matrix(file, variable_name="network"):
    data = scipy.io.loadmat(file)
    return data[variable_name]

