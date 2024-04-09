import numpy as np
import pickle
import networkx as nx
import scipy.sparse as sp
import sys
import datetime
import scipy.io as sio
import sklearn.metrics as sk_metrics
import gzip
import math
import tensorflow.compat.v1 as tf

# import pyscipopt as scip
import time
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # features = features/features.shape[1]
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def construct_feed_dict(features, support, labels, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: features})
    if support is not None:
        feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    return feed_dict

def construct_feed_dict4pred(features, support, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    if support is not None:
        feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    
    return feed_dict


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # adj_normalized = normalize_adj(adj)
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # adj_normalized = sp.coo_matrix(adj)
    return sparse_to_tuple(adj_normalized)
    
def simple_polynomials(adj, k):
    """Calculate polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    # print("Calculating polynomials up to order {}...".format(k))
    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(laplacian)

    for i in range(2, k+1):
        t_new = t_k[-1]*laplacian
        t_k.append(t_new)
    
    return sparse_to_tuple(t_k)

def simple_polynomials_to_dense(adj, k):
    """Calculate polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(laplacian)

    for i in range(2, k+1):
        t_new = t_k[-1]*laplacian
        t_k.append(t_new)
    for i in range(len(t_k)):
        t_k[i] = t_k[i].toarray()

    return t_k



def log(line, logfile=None):        
    line = f'[{datetime.datetime.now()}] {line}' if line is not None else "\n\n\n\n"
    print(line)
    if logfile is not None:
        print(line, file=logfile)
    sys.stdout.flush()


def read_single_gcn(dual_path, data_path):

    nb_node = -1; nb_edge = -1; adj = None

    with open(data_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] == 'p':
                _, _, nb_node, nb_edge = line.split(' ')
                nb_node, nb_edge = int(nb_node), int(nb_edge)
                adj = np.zeros((nb_node, nb_node))
            elif line [0] == 'e':
                try:
                    _, node1, node2 = line.split(' ')
                    node1, node2 = int(node1)-1, int(node2)-1
                    adj[node1, node2] = 1
                    adj[node2, node1] = 1
                except:
                    print(data_path)
                    
    xs = np.ones((nb_node, 32))
    ys = np.zeros((nb_node, 1))
    # if there are multiple optimal duals, use average value
    
    if dual_path is None:
        return xs, None, adj
        
    with open(dual_path, 'r') as f:
        lines = f.readlines()
        nb_optimal_dual = len(lines)
        for line in lines:
            tmp = line.strip().split(' ')
            assert(len(tmp) == nb_node)
            tokens = np.expand_dims(np.array(tmp, dtype=float), axis=1)
            ys += tokens
        ys /= nb_optimal_dual

    return xs, ys, adj

def read_batch_ml(dual_paths, feat_paths, data_paths, forwhat='nn'):
    
    if dual_paths is None:
        xs = []
        for feat_path, data_path in zip(feat_paths, data_paths):
            x, _ = read_single_ml(None, feat_path, data_path)
            xs.append(x)
        if forwhat=='nn':
            return list(zip(xs,[None for i in range(len(xs))]))
        else:
            return xs
    else:
        xs, ys = [], []
        for dual_path,feat_path, data_path in zip(dual_paths, feat_paths, data_paths):
            x, y = read_single_ml(dual_path, feat_path, data_path)
            xs.append(x)
            ys.append(y)
        if forwhat=='nn':
            return list(zip(xs, ys))
        else:
            return xs, ys

def read_single_ml(dual_path, feat_path, data_path):


    nb_node = -1; nb_edge = -1
    with open(data_path, 'r') as f:
        lines = f.readlines()
        # print(lines[25:35])

        for line in lines:
            try:
                line = line.strip()
                line = line.replace('  ', ' ')
                if len(line) == 0:
                    continue
                if line[0] == 'p':
                    _, _, nb_node, nb_edge = line.split(' ')
                    nb_node, nb_edge = int(nb_node.strip()), int(nb_edge.strip())
                    break
            except:
                print(data_path)
                
            
    xs = []
    with open(feat_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            features = line.strip().split(' ')
            features = [float(feat) for feat in features]
            xs.append(features)
    if len(xs)!= nb_node:
        print(len(xs), nb_node, feat_path)
    assert(len(xs) == nb_node)
    xs = np.array(xs)
    if dual_path is None:
        return xs, None

    ys = np.zeros((nb_node, 1))
    with open(dual_path, 'r') as f:
        lines = f.readlines()
        nb_optimal_dual = len(lines)
        for line in lines:
            tmp = line.strip().split(' ')
            assert(len(tmp) == nb_node)
            tokens = np.expand_dims(np.array(tmp, dtype=float), axis=1)
            ys += tokens
        ys /= nb_optimal_dual

    return xs, ys


def set_params(loss_type='mse', num_layer=0, out_act='identity'):
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden', 32, 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probaNUmbility).')
    flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 500, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 1, 'Maximum Chebyshev polynomial degree.')
    flags.DEFINE_integer('num_supports', 2, 'number of supports')
    flags.DEFINE_integer('num_layer', num_layer, 'number of layers gcn.')
    flags.DEFINE_string('loss_type', loss_type, 'loss function') 
    flags.DEFINE_string('out_act', out_act, 'identity/sigmoid/01cut') 
    flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'

    # dense matrix can enjoey tf parallelism
    # but if the problem have a graph that is too large to fit into memory, we need to use sparse matrix
    flags.DEFINE_string('matrix_type', 'dense', 'Model string.')  # 'sparse', 'dense'
    return FLAGS