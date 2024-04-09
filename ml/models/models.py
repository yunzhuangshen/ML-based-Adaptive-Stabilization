from layers import *
from layers import _LAYER_UIDS

flags = tf.app.flags
FLAGS = flags.FLAGS

def lrelu(x):
    return tf.maximum(x*0.2,x)

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None
        self.pred = None

        self.loss = 0
        self.optimizer = None
        self.opt_op = None
        self.mse = 0
        self.max_err=0
        self.std=0
        self.act_lastlayer = None
        if FLAGS.out_act == 'identity':
            self.act_lastlayer = lambda x: x
        elif FLAGS.out_act == 'sigmoid':
            self.act_lastlayer = tf.nn.sigmoid
        elif FLAGS.out_act == '01cut':
            self.act_lastlayer = lambda x: tf.math.minimum(tf.math.maximum(x, 0), 1)
        else:
            raise Exception('error')


    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        layer_id = 0
        self.activations.append(self.inputs)
        for layer in self.layers:
            if layer_id < len(self.layers)-1:
                hidden = tf.nn.relu(layer(self.activations[-1]))
                self.activations.append(hidden)
                layer_id = layer_id + 1
            else:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
                layer_id = layer_id + 1
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self.metric_mse()
        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        return self.outputs

    def metric_mse(self):
        self.mse = tf.reduce_mean(tf.square(self.outputs-self.placeholders['labels']))
        self.max_err = tf.reduce_max(tf.square(self.outputs-self.placeholders['labels']))
        self.std = tf.sqrt(tf.reduce_mean(tf.square(self.outputs - tf.reduce_mean(self.outputs))))
    def _loss(self):
        # Weight decay loss
        for layer in self.layers:
            for val in layer.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(val)

        if FLAGS.loss_type == 'mae':
            self.loss += tf.reduce_mean(tf.abs(self.outputs-self.placeholders['labels']))
        elif FLAGS.loss_type == 'mse':
            self.loss += tf.nn.l2_loss(self.outputs-self.placeholders['labels'])


    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)



class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.placeholders = placeholders
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()

    def _build(self):
        _LAYER_UIDS['linear'] = 0

        # linear
        if FLAGS.num_layer == 1:
            self.layers.append(Dense(input_dim=self.input_dim,
                            output_dim=1,
                            placeholders=self.placeholders,
                            act=self.act_lastlayer,
                            dropout=True,
                            logging=self.logging))
        else:
            assert(FLAGS.num_layer > 1)

            self.layers.append(Dense(input_dim=self.input_dim,
                                                output_dim=FLAGS.hidden,
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=True,
                                                logging=self.logging))

            for i in range(FLAGS.num_layer-2):
                self.layers.append(Dense(input_dim= FLAGS.hidden,
                                                    output_dim=FLAGS.hidden,
                                                    placeholders=self.placeholders,
                                                    act=tf.nn.relu,
                                                    dropout=True,
                                                    logging=self.logging))
            self.layers.append(Dense(input_dim= FLAGS.hidden,
                                                output_dim=1,
                                                placeholders=self.placeholders,
                                                act=self.act_lastlayer,
                                                dropout=True,
                                                logging=self.logging))


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.placeholders = placeholders
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()


    def _build(self):
        _LAYER_UIDS['graphconvolution'] = 0
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs= FLAGS.matrix_type == 'sparse' ,
                                            logging=self.logging))
        for i in range(FLAGS.num_layer-2):
            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                                output_dim=FLAGS.hidden,
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=True,
                                                logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                            output_dim=1,
                                            placeholders=self.placeholders,
                                            act=self.act_lastlayer,
                                            dropout=True,
                                            logging=self.logging))