from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import collections
import tensorflow as tf

# from tensorflow.keras.preprocessing.image import random.rotation

__all__ = ["BatchedInput", "get_iterator"]

BatchedInput = collections.namedtuple("BatchedInput",
                                      ("initializer", "Image_Y", "Image_Y_code"))


class Feature_filter_data(object):
    def get_iterator(self, output_dir, batch_size, CU_size, qp):
        def get_filename_list(dir, qp):
            image_Y_dir = os.path.join(dir, 'ori')
            image_Y_code_dir = os.path.join(dir, 'rec_' + str(qp) + '_nofilter')

            image_Y_folder = os.listdir(image_Y_dir)
            image_Y_code_folder = os.listdir(image_Y_code_dir)

            image_Y_list = map(lambda x: os.path.join(image_Y_dir, x), sorted(image_Y_folder))
            image_Y_code_list = map(lambda x: os.path.join(image_Y_code_dir, x), sorted(image_Y_code_folder))

            return image_Y_list, image_Y_code_list

        def get_one_CU(path, channels, flip_flag):
            height = 256
            width = 256
            ## get one CU from a yuv path
            image_string = tf.read_file(path)
            image_decoded = tf.image.decode_image(image_string, channels=channels)
            img = tf.to_float(image_decoded)
            img = tf.reshape(img, (height, width, channels))
            flip_img = tf.cond(flip_flag, lambda: tf.image.flip_left_right(img),
                                          lambda: img)

            # augmentation_img = random.rotation(img, rg=0.5, row_axis=0, col_axis=1, channel_axis=2)
            return flip_img

        def get_two_CU(path1, path2, channels):
            flag = tf.less(tf.random_uniform([], minval=0, maxval=1), 0.5)
            img1 = get_one_CU(path1, channels, flag)
            img2 = get_one_CU(path2, channels, flag)
            return img1, img2

        image_Y_list, image_Y_code_list = get_filename_list(output_dir, qp)
        dataset1 = tf.data.Dataset.from_tensor_slices(image_Y_list)
        dataset2 = tf.data.Dataset.from_tensor_slices(image_Y_code_list)
        dataset = tf.data.Dataset.zip((dataset1, dataset2))

        dataset = dataset.map(lambda x, y: (get_two_CU(x, y, channels=1))).prefetch(1000)
        dataset = dataset.shuffle(buffer_size=1000)
        batched_dataset = dataset.batch(batch_size=batch_size)
        batched_dataset = batched_dataset.repeat()  # ??????
        batched_iter = batched_dataset.make_initializable_iterator()
        (batch_image_Y, batch_image_Y_code) = batched_iter.get_next()

        return BatchedInput(initializer=batched_iter.initializer,
                            Image_Y=batch_image_Y,
                            Image_Y_code=batch_image_Y_code)
