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
import random; random.seed(1)


def get_inst_in_density(lb,ub, test_names):
    names = []
    for test_name in test_names:
        with open(f'../Matilda/instances/{test_name}.col', 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line[0] == 'p':
                break
        tmp = line.split(' ')
        nb_node, nb_edge = float(tmp[2]), float(tmp[3])
        density = 2 * nb_edge / (nb_node * (nb_node-1))
        if density >= lb and density <= ub:
            names.append(test_name)
    
    return names
    
def test_gcn(data_type, loss_type, nb_layer, out_act, fnames):

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
    model_dir = f'trained_models/gcn_{loss_type}_{nb_layer}_{out_act}/'
    config = tf.ConfigProto()
    tf.device('/cpu:0')
    sess = tf.Session(config=config)
    saver=tf.train.Saver(max_to_keep=1000)
    sess.run(tf.global_variables_initializer())
    ckpt=tf.train.get_checkpoint_state(model_dir)
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

    ####### data #######
    data_dir = f'../Matilda/instances/'
    optimal_dual_dir = f"../Matilda/dual_opt_processed/"    

    dual_fpaths = [f'{optimal_dual_dir}/{fname}.dual' for fname in fnames]
    inst_fpaths = [f'{data_dir}/{fname}.col' for fname in fnames]
    nsample = len(inst_fpaths)

    t1 = time.time()
    all_loss = []; all_mse = []; all_max_err = []; all_std = []
    for idd in range(nsample):

        xs, ys, adj = read_single_gcn(dual_fpaths[idd], inst_fpaths[idd])

        if FLAGS.matrix_type == 'sparse':
            xs = sparse_to_tuple(sp.lil_matrix(xs))
            support = simple_polynomials(adj, FLAGS.max_degree)  if FLAGS.model == 'gcn_cheby' else [preprocess_adj_sparse(adj)]
        else:
            support = simple_polynomials_to_dense(adj, FLAGS.max_degree)  if FLAGS.model == 'gcn_cheby' else [preprocess_adj(adj)]

        # testing step
        feed_dict_val = construct_feed_dict(xs, support, ys, placeholders)
        loss, mse, max_err, std = sess.run([model.loss, model.mse, model.max_err, model.std], feed_dict=feed_dict_val)
        all_loss.append(loss)
        all_mse.append(mse)
        all_max_err.append(max_err)
        all_std.append(std)
    t2 = time.time()

    if data_type == 'test':
        with open(f'gcn{nb_layer}.mse', 'w') as f:
            for mse in all_mse:
                f.write(f'{mse}\n')
        with open(f'gcn{nb_layer}.max', 'w') as f:
            for err in all_max_err:
                f.write(f'{err}\n')
        with open(f'gcn{nb_layer}.std', 'w') as f:
            for std in all_std:
                f.write(f'{std}\n')
    return (t2-t1), np.mean(all_mse), np.std(all_mse)



def test_mlp(data_type, loss_type, nb_layer, out_act, fnames):

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
    optimal_dual_dir = f"../Matilda/dual_opt_processed/"    
    data_dir = f'../Matilda/instances/'

    dual_fpaths = [f'{optimal_dual_dir}/{fname}.dual' for fname in fnames]
    feat_fpaths = [f'{feat_dir}/{fname}.feat' for fname in fnames]
    data_fpaths = [f'{data_dir}/{fname}.col' for fname in fnames]
    cache_data = read_batch_ml(dual_fpaths, feat_fpaths, data_fpaths)
    nsample = len(dual_fpaths)

    ####### Train model #######

    print(f'dataset size: {nsample}')

    t1 = time.time()
    all_loss = []; all_mse = []; all_max_err = [];all_std=[]
    for idd in range(nsample):

        t2 = time.time()
        xs, ys = cache_data[idd]
        # Construct feed dictionary
        feed_dict = construct_feed_dict(xs, None, ys, placeholders)
        loss, mse, max_err, std = sess.run([model.loss, model.mse, model.max_err, model.std], feed_dict=feed_dict)
        all_loss.append(loss)
        all_mse.append(mse)
        all_max_err.append(max_err)
        all_std.append(std)
    t2 = time.time()

    # if data_type == 'test':
    #     with open(f'ffnn{nb_layer}.mse', 'w') as f:
    #         for mse in all_mse:
    #             f.write(f'{mse}\n')
    #     with open(f'ffnn{nb_layer}.max', 'w') as f:
    #         for err in all_max_err:
    #             f.write(f'{err}\n')
    #     with open(f'ffnn{nb_layer}.std', 'w') as f:
    #         for std in all_std:
    #             f.write(f'{std}\n')
    return (t2-t1), np.mean(all_mse), np.std(all_mse)





if __name__ == '__main__':


    data_type, model_type, nb_layer, loss_type, out_act, density_type = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
    nb_layer = int(nb_layer)

    with open(f'../Matilda/lists/{data_type}.txt', 'r') as f:
        fnames = [line.strip() for line in f.readlines()]
    
    if density_type == 'sparse':
        fnames = get_inst_in_density(0., 0.3, fnames)
    elif density_type == 'dense':
        fnames = get_inst_in_density(0.3, 1.0, fnames)

    if model_type == 'gcn':
        time, mean_mse, std_mse =  test_gcn(data_type=data_type, loss_type = loss_type, nb_layer=nb_layer, out_act=out_act, fnames=fnames)
    elif model_type == 'mlp':
        time, mean_mse, std_mse =  test_mlp(data_type=data_type,loss_type = loss_type, nb_layer=nb_layer, out_act=out_act, fnames=fnames)

    print(f'{model_type},{nb_layer},{mean_mse},{time}')
