import tensorflow as tf
import numpy as np
import sys
import os
import time

sys.path.append('../')
from dataflow import ios


def predict(pred_path, rec_path, predict_path, model_path, batch_size, n_frames):
    """
    :param pred_path: the path of predicted frames
    :param rec_path:  the path of reconstructed frames
    :param predict_path: the path of result
    :param model_path: model path
    :param batch_size: batch size
    :param n_frames: nums of the frames
    :return: None
    """
    with tf.Graph().as_default() as g:
        with tf.gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        ## input
        pred_frame = g.get_tensor_by_name('pred_frame:0')
        rec_frame = g.get_tensor_by_name('rec_frame:0')

        ## output
        predict_frame = g.get_tensor_by_name('predict_frame:0')

        pred_reader = ios.YuvReader(pred_path, v_format='yuv420')
        rec_reader = ios.YuvReader(rec_path, v_format='yuv420')
        predict_writer = ios.YuvWriter(predict_path)

        frame_count = 0

        with tf.Session() as sess:
            while True:

                batch_pred, _, _ = pred_reader.read(batch_size, yuv_is_tuple=True)
                if batch_pred is None:
                    break
                batch_rec, _, _ = rec_reader.read(batch_pred.shape[0], yuv_is_tuple=True)

                frame_count += batch_pred.shape[0]

                ## decoded time
                if frame_count == 1:
                    time_start = time.time()
                ## run session to get pred_frames

                predict_y = sess.run(predict_frame, feed_dict={pred_frame: batch_pred, rec_frame: batch_rec})
                if frame_count == 1:
                    time_end = time.time()
                    print('decoded time: {} s'.format((time_end - time_start)))

                predict_writer.write(predict_y, yuv_is_tuple=False)

                if frame_count % 10 == 0:
                    print('%d frames has been writen ...' % (frame_count))
                if frame_count == n_frames:
                    break
            predict_writer.done(predict_path + 'has been writen...')
            rec_reader.done()
            pred_reader.done()


#

def main():
    # qps = [22, 27, 32, 37]
    qps = [22]

    models = ['ResidualVRN']
    batchsize = 1
    model_versions = [0]
    class_names = ['A']
    # class_names = ['A', 'B', 'C', 'D', 'E']
    for qp in qps:
        print(qp)
        for class_name in class_names:
            for model in models:
                for model_version in model_versions:
                    rec_dir = './rec/Class{}/qp{}'.format(class_name, qp)
                    pred_dir = './pred/Class{}/qp{}'.format(class_name, qp)
                    predict_dir = './results/Class{}/{}_qp{}_version{}'.format(
                        class_name, model, qp, model_version)

                    model_path = './models/{}_qp{}_version{}/freeze_graph.pb'.format(
                        model, qp, model_version)

                    if not os.path.exists(predict_dir):
                        os.makedirs(predict_dir)

                    print('using model: %s' % model_path)
                    rec_list = os.listdir(rec_dir)
                    pred_list = os.listdir(pred_dir)
                    for rec_name, pred_name in zip(rec_list, pred_list):
                        if rec_name[-6:-4] == 'op':
                            continue
                        n_frames = int(rec_name[-6:-4])
                        rec_path = os.path.join(rec_dir, rec_name)
                        pred_path = os.path.join(pred_dir, pred_name)
                        predict(pred_path, rec_path, os.path.join(predict_dir, rec_name[0:-4] + '.yuv'),
                                model_path=model_path,
                                batch_size=batchsize,
                                n_frames=n_frames)


if __name__ == '__main__':
    main()
