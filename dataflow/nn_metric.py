import numpy as np
import tensorflow as tf


def psnr(true, pred, maxval, merge=True):
    """Image quality metric based on maximal signal power vs. power of the noise.

    Args:
      true: the ground truth image.
      pred: the predicted image.
    Returns:
      scalar, peak signal to noise ratio (PSNR)
    """
    if true.get_shape().ndims != 4 or pred.get_shape().ndims != 4:
        raise ValueError("Input tensor true and pred must have 4 dims")
    mse = tf.reduce_mean(tf.square(true - pred), axis=[1, 2, 3])
    val = tf.clip_by_value(10.0 * tf.log(maxval ** 2 / mse) / tf.log(10.0), 0., 999.)  # [N, 1]
    if merge:
        val = tf.reduce_mean(val)  # scalar
    return val

# def psnr(true, pred, maxval, modality='RGB', merge=True):
#     """Image quality metric based on maximal signal power vs. power of the noise.
#
#     Args:
#       true: the ground truth image.
#       pred: the predicted image.
#       modality: RGB or YUV
#     Returns:
#       scalar, peak signal to noise ratio (PSNR)
#     """
#     if modality == 'YUV':
#         mse_yuv = tf.reduce_mean(tf.square(true - pred), axis=[1, 2])
#         weights = tf.constant([2.0, 0.5, 0.5])
#         mse = tf.reduce_mean(mse_yuv * weights, axis=[1])
#
#     else:
#         mse = tf.reduce_mean(tf.square(true - pred), axis=[1, 2, 3])
#     val = tf.clip_by_value(10.0 * tf.log(maxval ** 2 / mse) / tf.log(10.0), 0., 999.)  # [N, 1]
#     if merge:
#         val = tf.reduce_mean(val)  # scalar
#     return val


def mean_squared_error(true, pred):
    """L2 distance between tensors true and pred.

    Args:
      true: the ground truth image.
      pred: the predicted image.
    Returns:
      mean squared error between ground truth and predicted image.
    """
    return tf.reduce_mean(tf.square(true - pred))


def gradient_difference(true, pred, alpha=1, merge=True):
    """
    Calculates the GDL losses between the predicted and ground truth frames.

    @param true: The ground truth frames
    @param pred: The predicted frames
    @param alpha: The power to which each gradient term is raised.

    @return: The GDL loss.
    """
    # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
    ndims = pred.get_shape().ndims
    if ndims == 3:
        gen_d1 = pred[1:, :, :] - pred[:-1, :, :]
        gen_d2 = pred[:, 1:, :] - pred[:, :-1, :]
        gt_d1 = true[1:, :, :] - true[:-1, :, :]
        gt_d2 = true[:, 1:, :] - true[:, :-1, :]
        axis = None
    elif ndims == 4:
        gen_d1 = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        gen_d2 = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        gt_d1 = true[:, 1:, :, :] - true[:, :-1, :, :]
        gt_d2 = true[:, :, 1:, :] - true[:, :, :-1, :]
        axis = [1, 2, 3]
    else:
        raise ValueError('\'inputs\' must be either 3 or 4-dimensional.')

    gd1 = tf.reduce_mean(tf.abs(gt_d1 - gen_d1) ** alpha, axis=axis)
    gd2 = tf.reduce_mean(tf.abs(gt_d2 - gen_d2) ** alpha, axis=axis)

    tot_gdl = gd1 + gd2
    if merge:
        tot_gdl = tf.reduce_mean(tot_gdl)

    return tot_gdl


def total_variation(images, alpha=1, merge=True):
    """
    Calculates Total Variation Loss

    @param pred: The predicted frames

    @return: The TV loss.
    """

    ndims = images.get_shape().ndims

    if ndims == 3:
        # The input is a single image with shape [height, width, channels].

        # Calculate the difference of neighboring pixel-values.
        # The images are shifted one pixel along the height and width by slicing.
        pixel_dif1 = images[1:, :, :] - images[:-1, :, :]
        pixel_dif2 = images[:, 1:, :] - images[:, :-1, :]
        axis = None
    elif ndims == 4:
        # The input is a batch of images with shape:
        # [batch, height, width, channels].

        # Calculate the difference of neighboring pixel-values.
        # The images are shifted one pixel along the height and width by slicing.
        pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
        pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]
        axis = [1, 2, 3]
    else:
        raise ValueError('\'images\' must be either 3 or 4-dimensional.')

    # Calculate the total variation by taking the absolute value of the
    # pixel-differences and summing over the appropriate axis.
    tv1 = tf.reduce_mean(tf.abs(pixel_dif1) ** alpha, axis=axis)
    tv2 = tf.reduce_mean(tf.abs(pixel_dif2) ** alpha, axis=axis)

    tot_var = tv1 + tv2
    if merge:
        tot_var = tf.reduce_mean(tot_var)

    return tot_var


def grad_huber_loss(pred, delta=1.0):
    # get conv template of conv
    n_channels = pred.get_shape()[-1].value
    eye = tf.constant(np.identity(n_channels), dtype=tf.float32)
    w_x = [0., 0., 0., -0.5, 0., 0.5, 0., 0., 0.]
    w_y = [0., -0.5, 0., 0., 0., 0., 0., 0.5, 0.]
    filter_x = tf.reshape(tf.stack([w * eye for w in w_x], axis=0), (3, 3, n_channels, n_channels))
    filter_y = tf.reshape(tf.stack([w * eye for w in w_y], axis=0), (3, 3, n_channels, n_channels))
    strides = [1, 1, 1, 1]
    padding = 'VALID'
    # cal grad
    grad_x = tf.nn.conv2d(pred, filter_x, strides, padding=padding)
    grad_y = tf.nn.conv2d(pred, filter_y, strides, padding=padding)
    grad = tf.sqrt(grad_x ** 2 + grad_y ** 2)
    # cal loss
    loss = tf.losses.huber_loss(labels=tf.zeros_like(grad), predictions=grad, delta=delta)
    return loss


def sharpness(true, pred, maxval):
    n_channels = true.get_shape()[-1].value
    pos = tf.constant(np.identity(n_channels), dtype=tf.float32)
    neg = -1 * pos
    filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
    filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
    strides = [1, 1, 1, 1]  # stride of (1, 1)
    padding = 'SAME'

    gen_dx = tf.abs(tf.nn.conv2d(pred, filter_x, strides, padding=padding))
    gen_dy = tf.abs(tf.nn.conv2d(pred, filter_y, strides, padding=padding))
    gt_dx = tf.abs(tf.nn.conv2d(true, filter_x, strides, padding=padding))
    gt_dy = tf.abs(tf.nn.conv2d(true, filter_y, strides, padding=padding))

    grad_sum_gt = gt_dx + gt_dy
    grad_sum_gen = gen_dx + gen_dy

    grad_abs_diff = tf.reduce_mean(tf.abs(grad_sum_gt - grad_sum_gen))

    return 10.0 * tf.log(maxval / grad_abs_diff) / tf.log(10.0)
