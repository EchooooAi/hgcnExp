"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from annoy import AnnoyIndex


def load_data(args, datapath):
    if args.task == 'nc':
        data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed)
    elif args.task == 'dr':
        data = load_swiss_data(dataset="swissRoll")
        # data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed)
    else:
        data = load_data_lp(args.dataset, args.use_feats, datapath)
        adj = data['adj_train']
        if args.task == 'lp':
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                    adj, args.val_prop, args.test_prop, args.split_seed
            )
            data['adj_train'] = adj_train
            data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
            data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
            data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
    
    data['adj_train_norm'], data['features'] = process(
            data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
    )
    print("data['features'] ", data['features'][0,:])
    if args.dataset == 'airport':
        data['features'] = augment(data['adj_train'], data['features'])
    return data


# ############### FEATURES PROCESSING ####################################


def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ############### DATA SPLITS #####################################################


def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


# ############### LINK PREDICTION DATA LOADERS ####################################


def load_data_lp(dataset, use_feats, data_path):
    if dataset in ['cora', 'pubmed']:
        adj, features = load_citation_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'disease_lp':
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'airport':
        adj, features = load_data_airport(dataset, data_path, return_label=False)
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    data = {'adj_train': adj, 'features': features}
    return data


# ############### NODE CLASSIFICATION DATA LOADERS ####################################


def load_data_nc(dataset, use_feats, data_path, split_seed):
    if dataset in ['cora', 'pubmed']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            dataset, use_feats, data_path, split_seed
        )
    else:
        if dataset == 'disease_nc':
            adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.60
        elif dataset == 'airport':
            adj, features, labels = load_data_airport(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        else:
            raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)

    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    return data


# ############### DATASETS ####################################


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + 500)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(adj.shape[0])
    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels


def load_data_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph.node[u]['feat'] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features




# ############### kNNGraph Construction ####################################
import networkx as nx
from annoy import AnnoyIndex
import numpy as np
import scipy.sparse as sp
import torch

# def kGraph(x, features, distance):
#     if sp.isspmatrix(features):
#         features = np.array(features.todense())
#     if normalize_feats:
#         features = normalize(features)
#     features = torch.Tensor(features)
#     if normalize_adj:
#         adj = normalize(adj + sp.eye(adj.shape[0]))
#     adj = sparse_mx_to_torch_sparse_tensor(adj)
#     return adj, features

def construct_adjacent(dataset, train_data, distance='angular', nbr=5):
    from annoy import AnnoyIndex
    import random
    import csv
    import networkx as nx

    f = train_data.shape[1]
    t = AnnoyIndex(f, distance)  # Length of item vector that will be indexed
    for i in range(train_data.shape[0]):
        v = train_data[i,:]
        t.add_item(i, v)

    t.build(10) # 10 trees
    t.save('test.ann')

    # ...

    u = AnnoyIndex(f, distance)
    u.load('test.ann') # super fast, will just mmap the file
    # print(u.get_nns_by_item(0, 5)) # will find the 1000 nearest neighbors
    kadj = np.zeros((train_data.shape[0],train_data.shape[0]))
    with open(dataset +'_' + str(nbr) + distance +"_adjlist.txt", 'w') as x:
        for i in range(train_data.shape[0]):
            # outlist = u.get_nns_by_item(i, nbr)
            # print(outlist)
            # csv.writer(x,delimiter=" ").writerows(outlist)
            x.writelines("%s " % node for node in u.get_nns_by_item(i, nbr))
            x.write('\n')

def load_swiss_data(dataset="swissRoll", nbr=10):
    print('Loading {} dataset...'.format(dataset))
    distance = 'euclidean'
    remove = None
    if remove is None:
        train_data, train_label = make_swiss_roll(n_samples=800, noise=0, random_state=1)
    else:
        train_data, train_label = make_swiss_roll(n_samples=800, noise=0, random_state=1+1, remove=remove, center=[10, 10], r=8)
        train_data = train_data / 20
    try:
        print('Loading from existing file')
        G = nx.read_adjlist(dataset +'_' + str(nbr) + distance  +"_adjlist.txt", nodetype=int)
    except:
        print('Constructing new adjlist by ', distance, 'distance with nbrs', str(nbr))

        construct_adjacent(dataset, train_data, distance = distance, nbr=nbr)
        G = nx.read_adjlist(dataset +'_' + distance +"_adjlist.txt", nodetype=int)
    node_order = list(range( train_data.shape[0] ))
    adj_dense = nx.to_numpy_array(G, nodelist=node_order)
    adj = sp.csr_matrix(adj_dense)
    train_data = sp.csr_matrix(train_data)

    
    # non_adj = 1 - adj_dense
    # non_adj = torch.FloatTensor(non_adj)
    adj_dense = torch.Tensor(adj_dense)


    # features = normalize(train_data)
    # adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(800)
    idx_val = range(800,800)
    idx_test = range(800,800)

    # features = torch.FloatTensor(train_data)
    # labels = torch.tensor(train_label)

    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    data = {'adj_train': adj, 'adj_dense': adj_dense,'features': train_data, 'labels': train_label, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    return data
    # return adj, non_adj, adj_dense, features, labels, idx_train, idx_val, idx_test









# ############### Manifold making ####################################

import numbers
from collections.abc import Iterable

import math
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np

import scipy.sparse as sp
from scipy import integrate, linalg
from sklearn.datasets import *
from sklearn.utils import check_array, check_random_state
def int_map(t, step=0.001, ):

    t_reshape = t.reshape(-1)
    start = t.min()-0.01
    end = t.max()+0.01
    dic, d_list, angle_list = curve_param_2d(start, end)

    t_reshape = np.around(t_reshape, 3)
    mapput = []
    for i in range(t_reshape.shape[0]):
        mapput.append(dic[t_reshape[i]])
    mapput = np.array(mapput)

    return mapput.reshape((1, -1))
def curve_param_2d(a, b, dt=0.001, plot=True):

    dt = dt  # 变化率
    t = np.arange(a, b, dt)
    x = t*np.cos(t)
    y = t*np.sin(t)

    area_list = []

    area_list = [np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)
                 for i in range(1, len(t))]

    sum_a = []
    for i in range(len(area_list)):
        if i == 0:
            sum_a.append(0)
        else:

            sum_a.append(sum(area_list[:i]))

    t = np.around(t, 3)
    dictionary = dict(zip(t[1:], sum_a))

    return dictionary, sum_a, list(t[1:])
def make_swiss_roll(n_samples=100, noise=0.0, random_state=None, remove=False, scale=1.0, center=None, r=None):
    """Generate a swiss roll dataset.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of sample points on the S curve.

    noise : float, optional (default=0.0)
        The standard deviation of the gaussian noise.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : array of shape [n_samples, 3]
        The points.

    t : array of shape [n_samples]
        The univariate position of the sample according to the main dimension
        of the points in the manifold.

    Notes
    -----
    The algorithm is from Marsland [1].

    References
    ----------
    .. [1] S. Marsland, "Machine Learning: An Algorithmic Perspective",
           Chapter 10, 2009.
           http://seat.massey.ac.nz/personal/s.r.marsland/Code/10/lle.py
    """
    generator = check_random_state(random_state)

    t = 1.5 * np.pi * (1 + 2 * generator.rand(1, n_samples))
    x = t * np.cos(t)
    y = 21 * generator.rand(1, n_samples)
    z = t * np.sin(t)

    XX = []
    tt = []

    x_p = int_map(t)
    y_p = y
    z_p = 0

    X = np.concatenate((x, y, z)).transpose()
    XX = np.concatenate((x_p, y_p)).transpose()
    tt = x_p.reshape(-1)

    mask = np.zeros((XX.shape[0]))

    if remove == 'circle':
        for i in range(XX.shape[0]):
            if np.linalg.norm(XX[i]-center) < r:
                mask[i] = 1
            else:
                mask[i] = 0

        XX = XX[mask == 0]
        X = X[mask == 0]
        tt = tt[mask == 0]

    if remove == 'square':
        for i in range(XX.shape[0]):
            if (np.abs(XX[i, 0]-center[0]) < r) and (np.abs(XX[i, 1]-center[1])) < r:
                mask[i] = 1
            else:
                mask[i] = 0

        XX = XX[mask == 0]
        X = X[mask == 0]
        tt = tt[mask == 0]

    if remove == 'square2':
        for i in range(XX.shape[0]):
            if np.sum(np.abs(XX[i]-center)) < r:
                mask[i] = 1
            else:
                mask[i] = 0

        XX = XX[mask == 0]
        X = X[mask == 0]
        tt = tt[mask == 0]

    if remove == 'star':

        def area(x1, y1, x2, y2, x3, y3):

            return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                        + x3 * (y1 - y2)) / 2.0)

        # A function to check whether point P(x, y) lies inside the triangle formed by A(x1, y1), B(x2, y2) and C(x3, y3)
        def isInside(p1, p2, p3, p):

            x1, y1 = list(p1)
            x2, y2 = list(p2)
            x3, y3 = list(p3)
            x, y = list(p)

            # # Calculate area of triangle ABC, PBC, PAC, PAB, A3
            A = area(x1, y1, x2, y2, x3, y3)
            A1 = area(x, y, x2, y2, x3, y3)
            A2 = area(x1, y1, x, y, x3, y3)
            A3 = area(x1, y1, x2, y2, x, y)

            # Check if sum of A1, A2 and A3 is same as A
            if abs(A - (A1 + A2 + A3)) < 0.1:
                return True
            else:
                return False

        def isInstar(center, r, XX):

            list_zuobiao_out = [(r*math.cos((90+72*x)/180*3.14), r*math.sin((90+72*x)/180*3.14)) for x in [0,1,2,3,4]]
            r = (r*math.sin((18)/180*3.14)) / math.sin((126)/180*3.14)
            list_zuobiao_in = [(r*math.cos((126+72*x)/180*3.14), r*math.sin((126+72*x)/180*3.14)) for x in [0,1,2,3,4]]

            list_zuobiao_out = np.array(list_zuobiao_out) + np.array(center)
            list_zuobiao_in = np.array(list_zuobiao_in)+ np.array(center)

            in_list = [
                isInside(list_zuobiao_out[0], list_zuobiao_in[1], list_zuobiao_in[3], XX),
                isInside(list_zuobiao_out[1], list_zuobiao_in[2], list_zuobiao_in[4], XX),
                isInside(list_zuobiao_out[2], list_zuobiao_in[3], list_zuobiao_in[0], XX),
                isInside(list_zuobiao_out[3], list_zuobiao_in[4], list_zuobiao_in[1], XX),
                isInside(list_zuobiao_out[4], list_zuobiao_in[0], list_zuobiao_in[2], XX),
                
            ]
            return any(in_list)

        for i in range(XX.shape[0]):
            if isInstar(
                center, r, XX[i]
            ):
                mask[i] = 1
            else:
                mask[i] = 0

        XX = XX[mask == 0]
        X = X[mask == 0]
        tt = tt[mask == 0]


    if remove == 'fivecircle':
        center = np.zeros((5, 2))
        length, _ = integrate.quad(f, 1.5 * np.pi, 4.5 * np.pi)
        center = np.array([[length / 16 * 2, 10.5],
                           [length / 16 * 5, 10.5],
                           [length / 16 * 8, 10.5],
                           [length / 16 * 11, 10.5],
                           [length / 16 * 14, 10.5]])
        points = []
        T = []
        num = 0

        while num < n_samples:
            t = 1.5 * np.pi * (1 + 2 * np.random.rand())
            x = t * np.cos(t)
            y = 21 * np.random.rand()
            z = t * np.sin(t)
            v, _ = integrate.quad(f, 1.5 * np.pi, t)

            # If in five rings, the point will be preserved
            Flag = 0
            for i in range(5):
                if ((y - center[i, 1]) ** 2 + (v - center[i, 0]) ** 2) ** 0.5 < 21 * 0.5 and ((y - center[i, 1]) ** 2 + (v - center[i, 0]) ** 2) ** 0.5 > 21 * 0.15:
                    Flag = 1

            if Flag == 1:
                points.append(np.array([x, y, z]))
                T.append(t)
                num += 1

        points = np.array(points)
        T = np.array(T)
        X = points
        tt = T

    return X, tt