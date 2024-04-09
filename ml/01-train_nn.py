import sys
import os
sys.path.append( f'{os.path.dirname(os.path.realpath(__file__))}/models')
import time
import scipy.io as sio
import numpy as np
from copy import deepcopy
import scipy.sparse as sp
import sklearn.metrics as metrics
from utils import *
from models import GCN, MLP
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import pathlib
import random; random.seed(1)
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from utils import *
import xgboost as xgb
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression

import threading


def train_gcn(loss_type, nb_layer, out_act):
    # train gcn
    save_model_to = f'trained_models/gcn_{loss_type}_{nb_layer}_{out_act}/'
    os.makedirs(save_model_to, exist_ok=True)

    ####### data #######
    data_dir = f'../Matilda/instances/'
    optimal_dual_dir = f"../Matilda/dual_opt_processed/"    

    with open('../Matilda/lists/train.txt', 'r') as f:
        fnames = [line.strip() for line in f.readlines()]
    dual_fpaths = [f'{optimal_dual_dir}/{fname}.dual' for fname in fnames]
    data_fpaths = [f'{data_dir}/{fname}.col' for fname in fnames]


    ####### model #######
    feat_dim = 32
    FLAGS = set_params(loss_type=loss_type, num_layer=nb_layer, out_act=out_act)
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
    config = tf.ConfigProto()
    tf.device('/cpu:0')
    sess = tf.Session(config=config)
    saver=tf.train.Saver(max_to_keep=1000)
    sess.run(tf.global_variables_initializer())

    ####### Train model #######
    log_file=open(f"{save_model_to}/gcn.log",'w+')

    samples_per_epoch = len(data_fpaths)
    print(f'dataset size: {len(data_fpaths)}')
    best_loss = 1e9

    cache_data = [None for _ in range(len(data_fpaths))]

    for epoch in range(FLAGS.epochs):
        t1 = time.time()
        all_loss = []
        for idd in range(samples_per_epoch):

            t2 = time.time()
            if cache_data[idd] is None:
                xs, ys, adj = read_single_gcn(dual_fpaths[idd], data_fpaths[idd])
                cache_data[idd] = (xs, ys, adj)
            else:
                xs, ys, adj = cache_data[idd]

            if FLAGS.matrix_type == 'sparse':
                xs = sparse_to_tuple(sp.lil_matrix(xs))
                support = simple_polynomials(adj, FLAGS.max_degree)  if FLAGS.model == 'gcn_cheby' else [preprocess_adj_sparse(adj)]
            else:
                support = simple_polynomials_to_dense(adj, FLAGS.max_degree)  if FLAGS.model == 'gcn_cheby' else [preprocess_adj(adj)]
                

            # Construct feed dictionary
            feed_dict = construct_feed_dict(xs, support, ys, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            _, loss, output = sess.run([model.opt_op, model.loss, model.outputs], feed_dict=feed_dict)
            all_loss.append(loss)
            
        line = 'epoch={} loss={:.4f} time_sample={:.1f}'.format(
            epoch + 1, 
            np.mean(all_loss[-samples_per_epoch:]),
            time.time() - t2)
        log(line, log_file)

        loss_cur_epoch = np.mean(all_loss)
        line = '[{} finished!] loss={:.4f} time_epoch={:.1f}'.format(
            epoch + 1, loss_cur_epoch, time.time() - t1)
        log(line, log_file)

        if loss_cur_epoch < best_loss:
            log(f'best model currently, save to {save_model_to}', log_file)
            saver.save(sess,f"{save_model_to}/model.ckpt")
            best_loss = loss_cur_epoch
        sys.stdout.flush()
    log_file.flush(); log_file.close()
    print("Optimization Finished!")



def train_mlp(loss_type, nb_layer, out_act):
    # train gcn
    save_model_to = f'trained_models/mlp_{loss_type}_{nb_layer}_{out_act}/'
    os.makedirs(save_model_to, exist_ok=True)

    ####### data #######
    feat_dir = f'../Matilda/features_5/'
    optimal_dual_dir = f"../Matilda/dual_opt_processed/"    
    data_dir = f'../Matilda/instances/'

    with open('../Matilda/lists/train.txt', 'r') as f:
        fnames = [line.strip() for line in f.readlines()]
    dual_fpaths = [f'{optimal_dual_dir}/{fname}.dual' for fname in fnames]
    feat_fpaths = [f'{feat_dir}/{fname}.feat' for fname in fnames]
    data_fpaths = [f'{data_dir}/{fname}.col' for fname in fnames]

    cache_data = read_batch_ml(dual_fpaths, feat_fpaths, data_fpaths)


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
    config = tf.ConfigProto()
    tf.device('/cpu:0')
    sess = tf.Session(config=config)
    saver=tf.train.Saver(max_to_keep=1000)
    sess.run(tf.global_variables_initializer())

    ####### Train model #######
    log_file=open(f"{save_model_to}/mlp_{nb_layer}.log",'w+')

    samples_per_epoch = len(data_fpaths)
    print(f'dataset size: {len(data_fpaths)}')
    best_loss = 1e9

    for epoch in range(FLAGS.epochs):
        t1 = time.time()
        all_loss = []
        for idd in range(samples_per_epoch):

            t2 = time.time()
            xs, ys = cache_data[idd]

            # Construct feed dictionary
            feed_dict = construct_feed_dict(xs, None, ys, placeholders)

            # Training step
            _, loss, output = sess.run([model.opt_op, model.loss, model.outputs], feed_dict=feed_dict)
            all_loss.append(loss)
            
        loss_cur_epoch = np.mean(all_loss)
        line = '[{} finished!] loss={:.4f} time_epoch={:.1f}'.format(
            epoch + 1, loss_cur_epoch, time.time() - t1)
        log(line, log_file)

        if loss_cur_epoch < best_loss:
            log(f'best model currently, save to {save_model_to}', log_file)
            saver.save(sess,f"{save_model_to}/model.ckpt")
            best_loss = loss_cur_epoch
        sys.stdout.flush()
    log_file.flush(); log_file.close()
    print("Optimization Finished!")





if __name__ == '__main__':


    model_type, nb_layer, loss_type, out_act = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    nb_layer = int(nb_layer)

    if model_type == 'gcn':
        train_gcn(loss_type = loss_type, nb_layer=nb_layer, out_act=out_act)
    elif model_type == 'mlp':
        train_mlp(loss_type = loss_type, nb_layer=nb_layer, out_act=out_act)
