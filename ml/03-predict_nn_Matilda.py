from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append( f'{os.path.dirname(os.path.realpath(__file__))}/models')
import time
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from copy import deepcopy
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from utils import *
from models import GCN, MLP
import time


def predict_gcn(loss_type, nb_layer, out_act):
    
    feat_dim = 32
    FLAGS = set_params(loss_type=loss_type, num_layer=nb_layer, out_act=out_act)

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(FLAGS.num_supports)] 
                    if FLAGS.matrix_type == 'sparse' else [tf.placeholder(tf.float32) for _ in range(FLAGS.num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=(None, feat_dim)) 
                    if FLAGS.matrix_type == 'sparse' else  tf.placeholder(tf.float32, shape=(None, feat_dim)), # featureless: #points
        'labels': tf.placeholder(tf.float32, shape=(None, 1)), # 0: not linked, 1:linked
        'dropout': tf.placeholder_with_default(0., shape=()),
    }

    model = GCN(placeholders, input_dim=feat_dim, logging=True)

    ####### session #######
    model_dir = f'trained_models/gcn_{loss_type}_{nb_layer}_{out_act}/'
    config = tf.ConfigProto()
    tf.device('/cpu:0')
    sess = tf.Session(config=config)
    saver=tf.train.Saver(max_to_keep=1000)
    sess.run(tf.global_variables_initializer())
    ckpt=tf.train.get_checkpoint_state(model_dir)
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

    ## data ##
    data_dir = f'../Matilda/instances/'
    with open('../Matilda/lists/test.txt', 'r') as f:
        fnames = [line.strip() for line in f.readlines()]
    inst_fpaths = [(fname, f'{data_dir}/{fname}.col') for fname in fnames]
    nsample = len(inst_fpaths)

    t1 = time.time()
    for idd in range(nsample):
        data_fname, data_fpath = inst_fpaths[idd]
        # print('processing data file: ' + data_fname)
        xs, _, adj = read_single_gcn(None, data_fpath)
    
        if FLAGS.matrix_type == 'sparse':
            xs = sparse_to_tuple(sp.lil_matrix(xs))
            support = simple_polynomials(adj, FLAGS.max_degree)  if FLAGS.model == 'gcn_cheby' else [preprocess_adj_sparse(adj)]
        else:
            support = simple_polynomials_to_dense(adj, FLAGS.max_degree)  if FLAGS.model == 'gcn_cheby' else [preprocess_adj(adj)]

        t2 = time.time()
        feed_dict_val = construct_feed_dict4pred(xs, support, placeholders)
        outs_val = sess.run([model.outputs], feed_dict=feed_dict_val)[0]
        pred_dual = outs_val[:, 0].tolist()
        # write probability map to file
        write_dir = f'../Matilda/dual_pred_gcn_{loss_type}_{nb_layer}_{out_act}/'
        os.makedirs(write_dir, exist_ok=True)
        with open(f'{write_dir}/{data_fname}.pred', 'w+') as f:
            for d in pred_dual:
                f.write(f'{d}\n')
        # print(f'time used: {time.time() - t2}')

    print(f'average time used: {(time.time() - t1)/nsample}')




def predict_mlp(loss_type, nb_layer, out_act):

    ####### model #######
    feat_dim = 9
    FLAGS = set_params(loss_type=loss_type, num_layer=nb_layer, out_act=out_act)
    placeholders = {
        'features': tf.placeholder(tf.float32, shape=(None, feat_dim)), # featureless: #points
        'labels': tf.placeholder(tf.float32, shape=(None, 1)), # 0: not linked, 1:linked
        'dropout': tf.placeholder_with_default(0., shape=()),
    }
    model = MLP(placeholders, input_dim=feat_dim, logging=True)

    ####### session #######
    model_dir = f'trained_models/mlp_{loss_type}_{nb_layer}_{out_act}/'
    config = tf.ConfigProto()
    tf.device('/cpu:0')
    sess = tf.Session(config=config)
    saver=tf.train.Saver(max_to_keep=1000)
    sess.run(tf.global_variables_initializer())
    ckpt=tf.train.get_checkpoint_state(model_dir)
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

    ####### data #######
    feat_dir = f'../Matilda/features_5/'
    data_dir = f'../Matilda/instances/'

    with open('../Matilda/lists/test.txt', 'r') as f:
        fnames = [line.strip() for line in f.readlines()]
    feat_fpaths = [f'{feat_dir}/{fname}.feat' for fname in fnames]
    inst_fpaths = [f'{data_dir}/{fname}.col' for fname in fnames]
    cache_data = read_batch_ml(None, feat_fpaths, inst_fpaths)
    nsample = len(inst_fpaths)

    ####### Train model #######

    print(f'dataset size: {nsample}')

    t1 = time.time()
    all_loss = []
    for idd in range(nsample):

        t2 = time.time()
        xs, _ = cache_data[idd]
        # Construct feed dictionary
        feed_dict = construct_feed_dict4pred(xs, None, placeholders)
        outs_val = sess.run([model.outputs], feed_dict=feed_dict)[0]
        pred_dual = outs_val[:, 0].tolist()
        # write probability map to file
        write_dir = f'../Matilda/dual_pred_mlp_{loss_type}_{nb_layer}_{out_act}/'
        os.makedirs(write_dir, exist_ok=True)
        with open(f'{write_dir}/{fnames[idd]}.pred', 'w+') as f:
            for d in pred_dual:
                f.write(f'{d}\n')

    print(f'total time used: {(time.time() - t1)}')



if __name__ == '__main__':

    _, model_type, nb_layer = sys.argv 

    if model_type == 'gcn':
        predict_gcn(loss_type='mse', nb_layer=int(nb_layer), out_act='identity')
    elif model_type == 'mlp':
        predict_mlp(loss_type='mse', nb_layer=int(nb_layer), out_act='identity')