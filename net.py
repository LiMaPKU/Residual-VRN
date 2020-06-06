import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow import image

from tensorflow.contrib.framework.python.ops import add_arg_scope


@add_arg_scope
def mc_relu(_x, trainable=True):
    betas = tf.get_variable('betas', _x.get_shape()[-1], initializer=tf.constant_initializer(0.5), dtype=tf.float32,
                            trainable=trainable)
    bias_1 = tf.get_variable('bias_1', _x.get_shape()[-1], initializer=tf.constant_initializer(0.), dtype=tf.float32,
                             trainable=trainable)
    bias_2 = tf.get_variable('bias_2', _x.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32,
                             trainable=trainable)
    return tf.concat((tf.nn.relu(_x - bias_1), (betas * (_x - bias_2 - abs(betas * _x)))), axis=-1)


@add_arg_scope
def parametric_relu(_x):
    alphas = tf.get_variable('alpha', _x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg


@add_arg_scope
def residual_block_3(tensor, training=True, scope='blk_n', reuse=None, trainable=True):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=mc_relu,
                            normalizer_fn=None,
                            normalizer_params={'scale': True, 'is_training': training},
                            trainable=trainable):
            conv1 = slim.conv2d(tensor, 64, (3, 3), scope='conv1')
            conv2 = slim.conv2d(conv1, 64, (1, 1), activation_fn=None, scope='conv2')
            sum = tensor + conv2
            return sum


class residual_net(object):
    def __init__(self):
        self.scopes = {}

    def rvrn_net(self, rec_frame, pred_frame, training=True, reuse=None, nb_layer=6, trainable=True):
        with tf.variable_scope('r-vrn', reuse=reuse) as scope:
            self.scopes['r-vrn'] = scope.name
            with slim.arg_scope([slim.conv2d], padding='SAME', stride=1, activation_fn=None,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'scale': True, 'is_training': training},
                                trainable=trainable,
                                biases_initializer=None):
                with slim.arg_scope([mc_relu, slim.batch_norm], trainable=trainable):
                    _, h, w, _ = rec_frame.get_shape().as_list()
                    residual_frame = tf.subtract(rec_frame, pred_frame)
                    input_frame = tf.concat([residual_frame, pred_frame], axis=-1)
                    conv1 = slim.conv2d(input_frame, 64, (5, 5), scope='conv1')
                    blocks = []
                    for i in range(nb_layer):
                        blocks.append(
                            residual_block_3(blocks[-1] if i else conv1, training=training, scope='blk_1%d' % i,
                                             trainable=trainable))

                    conv2 = slim.conv2d(blocks[-1], 64, (3, 3), scope='conv2', activation_fn=tf.nn.relu)
                    conv3 = slim.conv2d(conv2, 1, (3, 3), activation_fn=None, normalizer_fn=None, scope='conv3')
                    pred = tf.add(conv3, rec_frame)
                    pred = pred + tf.stop_gradient(tf.clip_by_value(tf.round(pred), 0., 255.) - pred)
                    return pred

    def drvrn_net(self, rec_frame, pred_frame, training=True, reuse=None, nb_layer=6, trainable=True):
        with tf.variable_scope('dr-vrn', reuse=reuse) as scope:
            self.scopes['dr-vrn'] = scope.name
            with slim.arg_scope([slim.conv2d], padding='SAME', stride=1, activation_fn=None,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'scale': True, 'is_training': training},
                                trainable=trainable,
                                biases_initializer=None):
                with slim.arg_scope([mc_relu, slim.batch_norm], trainable=trainable):
                    _, h, w, _ = rec_frame.get_shape().as_list()
                    residual_frame = tf.subtract(rec_frame, pred_frame)
                    input_frame = tf.concat([residual_frame, pred_frame], axis=-1)
                    conv1 = slim.conv2d(input_frame, 64, (5, 5), scope='conv1')
                    blocks = []
                    for i in range(nb_layer):
                        blocks.append(
                            residual_block_1(blocks[-1] if i else conv1, training=training, scope='blk_1%d' % i,
                                             trainable=trainable))

                    conv2 = slim.conv2d(blocks[-1], 64, (3, 3), scope='conv2', activation_fn=tf.nn.relu)
                    conv3 = slim.conv2d(conv2, 1, (3, 3), activation_fn=None, normalizer_fn=None, scope='conv3')
                    pred = tf.add(conv3, rec_frame)
                    pred = pred + tf.stop_gradient(tf.clip_by_value(tf.round(pred), 0., 255.) - pred)
                    return pred
