import numpy as np
import tensorflow as tf
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function


from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops


def _verify_compatible_image_shapes(img1, img2):
    """Checks if two image tensors are compatible for applying SSIM or PSNR.
  
    This function checks if two sets of images have ranks at least 3, and if the
    last three dimensions match.
  
    Args:
      img1: Tensor containing the first image batch.
      img2: Tensor containing the second image batch.
  
    Returns:
      A tuple containing: the first tensor shape, the second tensor shape, and a
      list of control_flow_ops.Assert() ops implementing the checks.
  
    Raises:
      ValueError: When static shape check fails.
    """
    shape1 = img1.get_shape().with_rank_at_least(3)
    shape2 = img2.get_shape().with_rank_at_least(3)
    shape1[-3:].assert_is_compatible_with(shape2[-3:])

    if shape1.ndims is not None and shape2.ndims is not None:
        for dim1, dim2 in zip(reversed(shape1[:-3]), reversed(shape2[:-3])):
            if not (dim1 == 1 or dim2 == 1 or dim1.is_compatible_with(dim2)):
                raise ValueError(
                    'Two images are not compatible: %s and %s' % (shape1, shape2))

    # Now assign shape tensors.
    shape1, shape2 = array_ops.shape_n([img1, img2])

    # TODO(sjhwang): Check if shape1[:-3] and shape2[:-3] are broadcastable.
    checks = []
    checks.append(control_flow_ops.Assert(
        math_ops.greater_equal(array_ops.size(shape1), 3),
        [shape1, shape2], summarize=10))
    checks.append(control_flow_ops.Assert(
        math_ops.reduce_all(math_ops.equal(shape1[-3:], shape2[-3:])),
        [shape1, shape2], summarize=10))
    return shape1, shape2, checks


_SSIM_K1 = 0.01
_SSIM_K2 = 0.03


def _ssim_helper(x, y, reducer, max_val, compensation=1.0):
    r"""Helper function for computing SSIM.
  
    SSIM estimates covariances with weighted sums.  The default parameters
    use a biased estimate of the covariance:
    Suppose `reducer` is a weighted sum, then the mean estimators are
      \mu_x = \sum_i w_i x_i,
      \mu_y = \sum_i w_i y_i,
    where w_i's are the weighted-sum weights, and covariance estimator is
      cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
    with assumption \sum_i w_i = 1. This covariance estimator is biased, since
      E[cov_{xy}] = (1 - \sum_i w_i ^ 2) Cov(X, Y).
    For SSIM measure with unbiased covariance estimators, pass as `compensation`
    argument (1 - \sum_i w_i ^ 2).
  
    Arguments:
      x: First set of images.
      y: Second set of images.
      reducer: Function that computes 'local' averages from set of images.
        For non-covolutional version, this is usually tf.reduce_mean(x, [1, 2]),
        and for convolutional version, this is usually tf.nn.avg_pool or
        tf.nn.conv2d with weighted-sum kernel.
      max_val: The dynamic range (i.e., the difference between the maximum
        possible allowed value and the minimum allowed value).
      compensation: Compensation factor. See above.
  
    Returns:
      A pair containing the luminance measure, and the contrast-structure measure.
    """
    c1 = (_SSIM_K1 * max_val) ** 2
    c2 = (_SSIM_K2 * max_val) ** 2

    # SSIM luminance measure is
    # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
    mean0 = reducer(x)
    mean1 = reducer(y)
    num0 = mean0 * mean1 * 2.0
    den0 = math_ops.square(mean0) + math_ops.square(mean1)
    luminance = (num0 + c1) / (den0 + c1)

    # SSIM contrast-structure measure is
    #   (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2).
    # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
    #   cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
    #          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
    num1 = reducer(x * y) * 2.0
    den1 = reducer(math_ops.square(x) + math_ops.square(y))
    c2 *= compensation
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)

    # SSIM score is the product of the luminance and contrast-structure measures.
    return luminance, cs


def _fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    size = ops.convert_to_tensor(size, dtypes.int32)
    sigma = ops.convert_to_tensor(sigma)

    coords = math_ops.cast(math_ops.range(size), sigma.dtype)
    coords -= math_ops.cast(size - 1, sigma.dtype) / 2.0

    g = math_ops.square(coords)
    g *= -0.5 / math_ops.square(sigma)

    g = array_ops.reshape(g, shape=[1, -1]) + array_ops.reshape(g, shape=[-1, 1])
    g = array_ops.reshape(g, shape=[1, -1])  # For tf.nn.softmax().
    g = nn_ops.softmax(g)
    return array_ops.reshape(g, shape=[size, size, 1, 1])


def _ssim_per_channel(img1, img2, max_val=1.0):
    """Computes SSIM index between img1 and img2 per color channel.
  
    This function matches the standard SSIM implementation from:
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
    quality assessment: from error visibility to structural similarity. IEEE
    transactions on image processing.
  
    Details:
      - 11x11 Gaussian filter of width 1.5 is used.
      - k1 = 0.01, k2 = 0.03 as in the original paper.
  
    Args:
      img1: First image batch.
      img2: Second image batch.
      max_val: The dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
  
    Returns:
      A pair of tensors containing and channel-wise SSIM and contrast-structure
      values. The shape is [..., channels].
    """
    filter_size = constant_op.constant(11, dtype=dtypes.int32)
    filter_sigma = constant_op.constant(1.5, dtype=img1.dtype)

    shape1, shape2 = array_ops.shape_n([img1, img2])
    checks = [
        control_flow_ops.Assert(math_ops.reduce_all(math_ops.greater_equal(
            shape1[-3:-1], filter_size)), [shape1, filter_size], summarize=8),
        control_flow_ops.Assert(math_ops.reduce_all(math_ops.greater_equal(
            shape2[-3:-1], filter_size)), [shape2, filter_size], summarize=8)]

    # Enforce the check to run before computation.
    with ops.control_dependencies(checks):
        img1 = array_ops.identity(img1)

    # TODO(sjhwang): Try to cache kernels and compensation factor.
    kernel = _fspecial_gauss(filter_size, filter_sigma)
    kernel = array_ops.tile(kernel, multiples=[1, 1, shape1[-1], 1])

    # The correct compensation factor is `1.0 - tf.reduce_sum(tf.square(kernel))`,
    # but to match MATLAB implementation of MS-SSIM, we use 1.0 instead.
    compensation = 1.0

    # TODO(sjhwang): Try FFT.
    # TODO(sjhwang): Gaussian kernel is separable in space. Consider applying
    #   1-by-n and n-by-1 Gaussain filters instead of an n-by-n filter.
    def reducer(x):
        shape = array_ops.shape(x)
        x = array_ops.reshape(x, shape=array_ops.concat([[-1], shape[-3:]], 0))
        y = nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
        return array_ops.reshape(y, array_ops.concat([shape[:-3],
                                                      array_ops.shape(y)[1:]], 0))

    luminance, cs = _ssim_helper(img1, img2, reducer, max_val, compensation)

    # Average over the second and the third from the last: height, width.
    axes = constant_op.constant([-3, -2], dtype=dtypes.int32)
    ssim_val = math_ops.reduce_mean(luminance * cs, axes)
    cs = math_ops.reduce_mean(cs, axes)
    return ssim_val, cs


def ssim(img1, img2, max_val):
    """Computes SSIM index between img1 and img2.
  
    This function is based on the standard SSIM implementation from:
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
    quality assessment: from error visibility to structural similarity. IEEE
    transactions on image processing.
  
    Note: The true SSIM is only defined on grayscale.  This function does not
    perform any colorspace transform.  (If input is already YUV, then it will
    compute YUV SSIM average.)
  
    Details:
      - 11x11 Gaussian filter of width 1.5 is used.
      - k1 = 0.01, k2 = 0.03 as in the original paper.
  
    The image sizes must be at least 11x11 because of the filter size.
  
    Example:
  
    ```python
        # Read images from file.
        im1 = tf.decode_png('path/to/im1.png')
        im2 = tf.decode_png('path/to/im2.png')
        # Compute SSIM over tf.uint8 Tensors.
        ssim1 = tf.image.ssim(im1, im2, max_val=255)
  
        # Compute SSIM over tf.float32 Tensors.
        im1 = tf.image.convert_image_dtype(im1, tf.float32)
        im2 = tf.image.convert_image_dtype(im2, tf.float32)
        ssim2 = tf.image.ssim(im1, im2, max_val=1.0)
        # ssim1 and ssim2 both have type tf.float32 and are almost equal.
    ```
  
    Args:
      img1: First image batch.
      img2: Second image batch.
      max_val: The dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
  
    Returns:
      A tensor containing an SSIM value for each image in batch.  Returned SSIM
      values are in range (-1, 1], when pixel values are non-negative. Returns
      a tensor with shape: broadcast(img1.shape[:-3], img2.shape[:-3]).
    """
    _, _, checks = _verify_compatible_image_shapes(img1, img2)
    with ops.control_dependencies(checks):
        img1 = array_ops.identity(img1)

    # Need to convert the images to float32.  Scale max_val accordingly so that
    # SSIM is computed correctly.
    max_val = math_ops.cast(max_val, img1.dtype)
    max_val = tf.image.convert_image_dtype(max_val, dtypes.float32)
    img1 = tf.image.convert_image_dtype(img1, dtypes.float32)
    img2 = tf.image.convert_image_dtype(img2, dtypes.float32)
    ssim_per_channel, _ = _ssim_per_channel(img1, img2, max_val)
    # Compute average over color channels.
    return math_ops.reduce_mean(ssim_per_channel, [-1])


# Default values obtained by Wang et al.
_MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def ssim_multiscale(img1, img2, max_val, power_factors=_MSSSIM_WEIGHTS):
    """Computes the MS-SSIM between img1 and img2.
  
    This function assumes that `img1` and `img2` are image batches, i.e. the last
    three dimensions are [height, width, channels].
  
    Note: The true SSIM is only defined on grayscale.  This function does not
    perform any colorspace transform.  (If input is already YUV, then it will
    compute YUV SSIM average.)
  
    Original paper: Wang, Zhou, Eero P. Simoncelli, and Alan C. Bovik. "Multiscale
    structural similarity for image quality assessment." Signals, Systems and
    Computers, 2004.
  
    Arguments:
      img1: First image batch.
      img2: Second image batch. Must have the same rank as img1.
      max_val: The dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      power_factors: Iterable of weights for each of the scales. The number of
        scales used is the length of the list. Index 0 is the unscaled
        resolution's weight and each increasing scale corresponds to the image
        being downsampled by 2.  Defaults to (0.0448, 0.2856, 0.3001, 0.2363,
        0.1333), which are the values obtained in the original paper.
  
    Returns:
      A tensor containing an MS-SSIM value for each image in batch.  The values
      are in range [0, 1].  Returns a tensor with shape:
      broadcast(img1.shape[:-3], img2.shape[:-3]).
    """
    # Shape checking.
    shape1 = img1.get_shape().with_rank_at_least(3)
    shape2 = img2.get_shape().with_rank_at_least(3)
    shape1[-3:].merge_with(shape2[-3:])

    with ops.name_scope(None, 'MS-SSIM', [img1, img2]):
        shape1, shape2, checks = _verify_compatible_image_shapes(img1, img2)
        with ops.control_dependencies(checks):
            img1 = array_ops.identity(img1)

        # Need to convert the images to float32.  Scale max_val accordingly so that
        # SSIM is computed correctly.
        max_val = math_ops.cast(max_val, img1.dtype)
        max_val = tf.image.convert_image_dtype(max_val, dtypes.float32)
        img1 = tf.image.convert_image_dtype(img1, dtypes.float32)
        img2 = tf.image.convert_image_dtype(img2, dtypes.float32)

        imgs = [img1, img2]
        shapes = [shape1, shape2]

        # img1 and img2 are assumed to be a (multi-dimensional) batch of
        # 3-dimensional images (height, width, channels). `heads` contain the batch
        # dimensions, and `tails` contain the image dimensions.
        heads = [s[:-3] for s in shapes]
        tails = [s[-3:] for s in shapes]

        divisor = [1, 2, 2, 1]
        divisor_tensor = constant_op.constant(divisor[1:], dtype=dtypes.int32)

        def do_pad(images, remainder):
            padding = array_ops.expand_dims(remainder, -1)
            padding = array_ops.pad(padding, [[1, 0], [1, 0]])
            return [array_ops.pad(x, padding, mode='SYMMETRIC') for x in images]

        mcs = []
        for k in range(len(power_factors)):
            with ops.name_scope(None, 'Scale%d' % k, imgs):
                if k > 0:
                    # Avg pool takes rank 4 tensors. Flatten leading dimensions.
                    flat_imgs = [
                        array_ops.reshape(x, array_ops.concat([[-1], t], 0))
                        for x, t in zip(imgs, tails)
                    ]

                    remainder = tails[0] % divisor_tensor
                    need_padding = math_ops.reduce_any(math_ops.not_equal(remainder, 0))
                    # pylint: disable=cell-var-from-loop
                    padded = control_flow_ops.cond(need_padding,
                                                   lambda: do_pad(flat_imgs, remainder),
                                                   lambda: flat_imgs)
                    # pylint: enable=cell-var-from-loop

                    downscaled = [nn_ops.avg_pool(x, ksize=divisor, strides=divisor,
                                                  padding='VALID')
                                  for x in padded]
                    tails = [x[1:] for x in array_ops.shape_n(downscaled)]
                    imgs = [
                        array_ops.reshape(x, array_ops.concat([h, t], 0))
                        for x, h, t in zip(downscaled, heads, tails)
                    ]

                # Overwrite previous ssim value since we only need the last one.
                ssim_per_channel, cs = _ssim_per_channel(*imgs, max_val=max_val)
                mcs.append(nn_ops.relu(cs))

        # Remove the cs score for the last scale. In the MS-SSIM calculation,
        # we use the l(p) at the highest scale. l(p) * cs(p) is ssim(p).
        mcs.pop()  # Remove the cs score for the last scale.
        mcs_and_ssim = array_ops.stack(mcs + [nn_ops.relu(ssim_per_channel)],
                                       axis=-1)
        # Take weighted geometric mean across the scale axis.
        ms_ssim = math_ops.reduce_prod(math_ops.pow(mcs_and_ssim, power_factors),
                                       [-1])

        return math_ops.reduce_mean(ms_ssim, [-1])  # Avg over color channels.


#
# def _tf_fspecial_gauss(size, sigma):
#     """Function to mimic the 'fspecial' gaussian MATLAB function
#     """
#     x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
#
#     x_data = np.expand_dims(x_data, axis=-1)
#     x_data = np.expand_dims(x_data, axis=-1)
#
#     y_data = np.expand_dims(y_data, axis=-1)
#     y_data = np.expand_dims(y_data, axis=-1)
#
#     x = tf.constant(x_data, dtype=tf.float32)
#     y = tf.constant(y_data, dtype=tf.float32)
#
#     g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
#     return g / tf.reduce_sum(g)
#
#
# def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
#     window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
#     K1 = 0.01
#     K2 = 0.03
#     L = 1  # depth of image (255 in case the image has a differnt scale)
#     C1 = (K1 * L) ** 2
#     C2 = (K2 * L) ** 2
#     print(img1)
#     mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
#     mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
#     mu1_sq = mu1 * mu1
#     mu2_sq = mu2 * mu2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
#     sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
#     sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2
#     if cs_map:
#         value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                               (sigma1_sq + sigma2_sq + C2)),
#                  (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
#     else:
#         value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                              (sigma1_sq + sigma2_sq + C2))
#
#     if mean_metric:
#         value = tf.reduce_mean(value)
#     return value
#
#
# def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
#     weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
#     mssim = []
#     mcs = []
#     for l in range(level):
#         ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
#         mssim.append(tf.reduce_mean(ssim_map))
#         mcs.append(tf.reduce_mean(cs_map))
#         filtered_im1 = tf.nn.avg_pool(img1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
#         filtered_im2 = tf.nn.avg_pool(img2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
#         img1 = filtered_im1
#         img2 = filtered_im2
#
#     # list to tensor of dim D+1
#     mssim = tf.stack(mssim, axis=0)
#     mcs = tf.stack(mcs, axis=0)
#
#     value = (tf.reduce_prod(mcs[0:level - 1] ** weight[0:level - 1]) *
#              (mssim[level - 1] ** weight[level - 1]))
#
#     if mean_metric:
#         value = tf.reduce_mean(value)
#     return value
#
#
# # def tf_l1Gass_loss(img1, img2, batch, mean_metric=True, level=5):
# #     _, height, width, channels = img1.get_shape().as_list()
# #     sigma = [0.5, 1., 2., 4., 8.]
# #     # w = np.empty([level, batch, height, width, channels])
# #     for i in range(level):
# #         gaussian = np.exp(-1. * np.arange(-(width / 2), width / 2) ** 2 / (2 * sigma[i] ** 2))
# #         gaussian = np.outer(gaussian, gaussian.reshape((width, 1)))  # extend to 2D
# #         gaussian = gaussian / np.sum(gaussian)  # normailization
# #         gaussian = np.reshape(gaussian, (1, 1, width, width))  # reshape to 4D
# #         gaussian = np.tile(gaussian, (batch, channels, 1, 1))
# #         # w[i, :, :, :, :] = gaussian
# #   ##loss_l1 = tf.reduce_sum(tf.abs(img1 - img2) * w[-1]) / batch * channel
# #     loss_l1 = tf.reduce_sum(tf.abs(img1 - img2) * gaussian) / batch
# #     return loss_l1


def tf_l1Gass_loss(img1, img2, batch, gaussian, image_depth=8):
    loss_l1 = tf.reduce_sum(tf.abs(img1 - img2) * gaussian) / (batch * (2 ** image_depth))
    return loss_l1
