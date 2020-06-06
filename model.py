import tensorflow as tf
import logging
import os
import sys
import numpy as np

from tensorflow.python.framework import graph_util

sys.path.append('../')
from dataflow import nn_metric, metric_tf
from dataflow import dataset
import net

logger = logging.getLogger()

PRINT_INTERVAL = 15


class RELU_NN(object):
    def __init__(self, qp, block_size, batch_size, model_version=0, sao=0, dbk=0):
        self.batch_size = batch_size
        self.block_size = block_size
        self.model_version = model_version
        self.net = net.test_relu_net()
        self.qp = qp
        self.sao = sao
        self.dbk = dbk

    def _build_common(self):
        # set up global step
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        # flag tensor for print info
        self.logger = tf.constant(0.)
        self.logger = tf.Print(self.logger, [self.global_step], "Step(Relu nn): ")

    def build_input(self, dataset_name):
        print('entering build input...')
        # set input info
        if  dataset_name == 'mscoco':
            print('-----------using mscoco-------------')
            data_reader = dataset.mscoco_10k_residue()
            dir_path = '/datasets/MLG/mali/from_megatron/data_coco_residue'
        else:
            raise NotImplementedError
        batched_input = data_reader.get_iterator(dir_path, block_size=self.block_size, qp=self.qp,
                                                 batch_size=self.batch_size, sao=self.sao, dbk=self.dbk)
        self.initializer = batched_input.initializer
        self.ori_frame = batched_input.image_Y
        self.rec_frame = batched_input.image_Y_code
        self.pred_frame = batched_input.image_Y_pred

        _, height, width, channels = self.ori_frame.get_shape().as_list()
        sigma = [0.5, 1., 2., 4., 8.]
        gaussian = np.exp(-1. * np.arange(-(width / 2), width / 2) ** 2 / (2 * sigma[-1] ** 2))
        gaussian = np.outer(gaussian, gaussian.reshape((width, 1)))  # extend to 2D
        gaussian = gaussian / np.sum(gaussian)  # normailization
        gaussian = np.reshape(gaussian, (1, 1, width, width))  # reshape to 4D
        self.gaussian = np.tile(gaussian, (self.batch_size, channels, 1, 1))

        print(self.ori_frame.shape)
        print('done...')

    def build_deploy(self):
        rec_frame = tf.placeholder(tf.uint8, shape=[None, None, None, 1], name='rec_frame')
        rec_frame = tf.to_float(rec_frame)
        pred_frame = tf.placeholder(tf.uint8, shape=[None, None, None, 1], name='pred_frame')
        pred_frame = tf.to_float(pred_frame)

        if self.model_version == 0:
            predict_frame = self.net.rvrn_net(rec_frame=rec_frame, pred_frame=pred_frame, training=False,
                                                  reuse=True)
        elif self.model_version == 1:
            predict_frame = self.net.drvrn_net(rec_frame=rec_frame, pred_frame=pred_frame, training=False,
                                                           reuse=True, nb_layer=64, trainable=True)
        elif self.model_version == 2:
            predict_frame = self.net.drvrn_net(rec_frame=rec_frame, pred_frame=pred_frame, training=False,
                                                           reuse=True, nb_layer=16, trainable=True)
        elif self.model_version == 3:
            predict_frame = self.net.drvrn_net(rec_frame=rec_frame, pred_frame=pred_frame, training=False,
                                                           reuse=True, nb_layer=32, trainable=True)
        else:
            raise NotImplementedError

        predict_frame = tf.identity(predict_frame, name='predict_frame')

    def _export_freeze_graph(self, sess):
        output_node_names = "predict_frame"
        output_graph_path = os.path.join(self.logdir, 'relu_nn_version{}_qp{}_sao{}_dbk{}_freeze_graph.pb'.format(self.model_version, self.qp, self.sao, self.dbk))
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, sess.graph_def, output_node_names.split(","))
        with tf.gfile.GFile(output_graph_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    def build(self, optimizer, lr):
        self._build_common()
        self.preds = {}

        if self.model_version == 0:
            self.preds['frame'] = self.net.rvrn_net(rec_frame=self.rec_frame, pred_frame=self.pred_frame,
                                                        training=True, reuse=None)
        elif self.model_version == 1:
            self.preds['frame'] = self.net.drvrn_net(rec_frame=self.rec_frame, pred_frame=self.pred_frame,
                                                                 training=True, reuse=None, trainable=True, nb_layer=64)
        elif self.model_version == 2:
            self.preds['frame'] = self.net.drvrn_net(rec_frame=self.rec_frame, pred_frame=self.pred_frame,
                                                         training=True, reuse=None, trainable=True, nb_layer=16)
        elif self.model_version == 3:
            self.preds['frame'] = self.net.drvrn_net(rec_frame=self.rec_frame, pred_frame=self.pred_frame,
                                                                 training=True, reuse=None, trainable=True, nb_layer=32)
        else:
            raise NotImplementedError

        self.build_deploy()
        self._build_loss()
        self.logger = tf.Print(self.logger, [self.err], "Error Total: ")
        self._build_train_op(optimizer, lr)
        self._build_summary()
        self.summary_op = tf.summary.merge_all()

    def _build_loss(self):
        with tf.name_scope('loss'):
            self.err = tf.constant(0.)
            # Reconstruction Loss
            with tf.name_scope('rec_loss'):
                err_msssim = tf.reduce_mean(
                    metric_tf.ssim_multiscale(self.preds['frame'], self.ori_frame, max_val=255))

                tf.summary.scalar('err_msssim', err_msssim)
                self.logger = tf.Print(self.logger, [err_msssim], "err_msssim :")
                self.err += 0.84 * (1. - err_msssim)

                err_gsl1 = metric_tf.tf_l1Gass_loss(self.preds['frame'], self.ori_frame, batch=self.batch_size,
                                                    gaussian=self.gaussian)
                tf.summary.scalar('err_gsl1', err_gsl1)
                self.logger = tf.Print(self.logger, [err_gsl1], "err_gsl1 :")
                self.err += 0.16 * err_gsl1

        with tf.name_scope('metric'):
            psnr_Y = nn_metric.psnr(self.preds['frame'], self.ori_frame, maxval=255.)
            tf.summary.scalar('psnr_Y', psnr_Y)
            self.logger = tf.Print(self.logger, [psnr_Y], "errG PSNR_Y: ")

            psnr_Y_anchor = nn_metric.psnr(self.rec_frame, self.ori_frame, maxval=255.)
            tf.summary.scalar('psnr_Y_anchor', psnr_Y_anchor)
            self.logger = tf.Print(self.logger, [psnr_Y_anchor], "errG PSNR_Y_anchor: ")

            psnr_diff = psnr_Y - psnr_Y_anchor
            tf.summary.scalar('psnr(diff)', psnr_diff)
            self.logger = tf.Print(self.logger, [psnr_diff], "errG PSNR(diff): ")

    def _build_train_op(self, optimizer, lr):
        with tf.name_scope('train_op'):
            if optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(lr)
            elif optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(lr)
            elif optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(lr)
            else:
                raise NotImplementedError
            print(optim)
            print('----------- Grads ------------')
            grads_and_vars = optim.compute_gradients(self.err)
            for grad, var in grads_and_vars:
                print(var.op.name)
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/grad', grad)
                if var is not None:
                    tf.summary.histogram(var.op.name + '/value', var)

            print('----------- BN updates ------------')
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            for op in update_ops:
                print(op.op.name)
            with tf.control_dependencies(update_ops):
                self.train_op = optim.apply_gradients(grads_and_vars, global_step=self.global_step)

    def _build_summary(self):
        def tf_summary_image(name, tensor):
            if tensor is None:
                return
            tensor = tf.clip_by_value(tensor, 0., 255.)
            tf.summary.image(name, tensor)

        tf_summary_image('frame_ori', self.ori_frame)
        tf_summary_image('frame_rec', self.rec_frame)
        tf_summary_image('frame_pred', self.preds['frame'])
        tf_summary_image('residual_frame', tf.abs(self.rec_frame - self.pred_frame))

        tmp_scalar1 = self.ori_frame - self.preds['frame']
        tmp_scalar2 = self.ori_frame - self.rec_frame
        tf.summary.scalar('E_1', tf.reduce_mean(tmp_scalar1))
        tf.summary.scalar('sigma_1', tf.reduce_mean(tmp_scalar1 ** 2) - tf.reduce_mean(tmp_scalar1) ** 2)
        tf.summary.scalar('E_2', tf.reduce_mean(tmp_scalar2))
        tf.summary.scalar('sigma_2', tf.reduce_mean(tmp_scalar2 ** 2) - tf.reduce_mean(tmp_scalar2) ** 2)

    def build_supervisor(self, logdir):
        self.logdir = logdir
        self.sv = tf.train.Supervisor(logdir=self.logdir, summary_op=None, global_step=self.global_step)

    def fit(self, n_iters, save_summaries_steps, gpu_fraction=1.0):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        with self.sv.managed_session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run([self.initializer])
            while not self.sv.should_stop():
                [gs] = sess.run([self.global_step])
                try:
                    if gs % PRINT_INTERVAL == 0:
                        _, _, summaries = sess.run([self.train_op, self.logger, self.summary_op])
                    else:
                        _, summaries = sess.run([self.train_op, self.summary_op])

                    # write summaries
                    if gs % save_summaries_steps == 0:
                        self.sv.summary_computed(sess, summaries)
                        self._export_freeze_graph(sess)
                    if n_iters and gs > n_iters:
                        break
                except tf.errors.InvalidArgumentError:
                    pass
