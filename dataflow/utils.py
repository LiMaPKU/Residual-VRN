import cv2
import glob
import os
import re
import numpy as np
import scipy.misc


def get_filenames_from_dir(data_dir, file_fmts):
    assert os.path.isdir(data_dir)
    if isinstance(file_fmts, str):
        file_fmts = [file_fmts]

    data_path = [glob.glob(os.path.abspath(os.path.join(data_dir, file_fmt))) for file_fmt in file_fmts]
    return data_path


def get_width_height_from_name(paths_or_names):
    """paths is transform_matrix path string or list of path strings"""
    if paths_or_names is None:
        return None

    paths_or_names = [paths_or_names] if not isinstance(paths_or_names, (list, tuple)) else paths_or_names

    pattern = re.compile('[0-9]+x[0-9]+')
    shapes = []
    for path in paths_or_names:
        size = pattern.findall(path)
        # assert len(size) == 1, 'Shape must be inferred from name'
        shapes.append(map(int, size[-1].split('x')))
    return shapes[0] if len(paths_or_names) == 1 else shapes


def upsample_chroma(chroma_nhwc):
    """upsample from (H, W, C) to (2H, 2W, C), or from (N, H, W, C) to (N, 2H, 2W, C)"""
    ndim = chroma_nhwc.ndim
    dim_h = ndim - 3
    dim_w = ndim - 2
    upsampled_chroma = np.repeat(chroma_nhwc, 2, axis=dim_h)
    upsampled_chroma = np.repeat(upsampled_chroma, 2, axis=dim_w)
    return upsampled_chroma


def downsample_chroma(chroma_nhwc):
    if chroma_nhwc.ndim == 3:
        return chroma_nhwc[::2, ::2, :]
    elif chroma_nhwc.ndim > 3:
        return chroma_nhwc[..., ::2, ::2, :]
    else:
        raise NotImplementedError('Currently not support 2-dim or 1-dim np.ndarray')


def random_crop_image(image, crop_size, only_center_crop=False, more_fix_crop=True):
    h, w, c = image.shape
    crop_h, crop_w = crop_size
    if h < crop_h or w < crop_w:
        scale_ratio_h = float(crop_h) / h
        scale_ratio_w = float(crop_w) / w
        scale_ratio = max(scale_ratio_h, scale_ratio_w)
        h = int(scale_ratio * h + 1.)
        w = int(scale_ratio * w + 1.)
        image = imresize(image, (h, w))

    height_off = (h - crop_h) // 4
    width_off = (w - crop_w) // 4

    crop_offsets = []
    crop_offsets.append([2 * height_off, 2 * width_off])  # center-center
    if not only_center_crop:
        crop_offsets.append([0, 0])  # left-top
        crop_offsets.append([0, 4 * width_off])  # right-top
        crop_offsets.append([4 * height_off, 0])  # down-left
        crop_offsets.append([4 * height_off, 4 * width_off])  # down-right

        if more_fix_crop:
            crop_offsets.append([0, 2 * width_off])
            crop_offsets.append([4 * height_off, 2 * width_off])
            crop_offsets.append([2 * height_off, 0])
            crop_offsets.append([2 * height_off, 4 * width_off])
            crop_offsets.append([height_off, width_off])
            crop_offsets.append([height_off, 3 * width_off])
            crop_offsets.append([3 * height_off, width_off])
            crop_offsets.append([3 * height_off, 3 * width_off])

    out_images = []
    for crop_offset in crop_offsets:
        ho, wo = crop_offset
        out_images.append(image[ho: ho + crop_h, wo: wo + crop_w, :])

    stacked_images = np.stack(out_images, axis=0)
    return stacked_images


def split_images_to_cus(frames, cu_shape):
    """split frame from shape:(N,H,W,C) to shape:(N, H_in_cu, W_in_cu, cu_size, cu_size, C)  with padding !!!  """
    assert frames.ndim == 4, 'DataZoo must have shape:(N, H, W, C)'

    n, h, w, c = frames.shape
    if isinstance(cu_shape, int):
        cu_h = cu_w = cu_shape
    elif isinstance(cu_shape, (list, tuple, np.array)):
        cu_h = cu_shape[0]
        cu_w = cu_shape[1]
    else:
        raise ValueError

    height_in_cu = (h + cu_h - 1) // cu_h
    width_in_cu = (w + cu_w - 1) // cu_w

    frames_padding = np.ones([n, height_in_cu * cu_h, width_in_cu * cu_w, c], dtype=frames.dtype) * 128
    frames_padding[:, 0:h, 0:w, 0:c] = frames
    # split height into `height_in_cu` slices
    cu_rows = np.split(frames_padding, height_in_cu, axis=1)
    cu_rows_cols = []
    for row in cu_rows:
        cu_rows_cols.append(np.split(row, width_in_cu, axis=2))
    cu_rows_cols = np.array(cu_rows_cols)  # shape: (H_in_cu, W_in_cu, N, cu_size, cu_size, C)
    cu_batch = np.swapaxes(cu_rows_cols, 0, 2)
    cu_batch = np.swapaxes(cu_batch, 1, 2)  # shape: (N, H_in_cu, W_in_cu, cu_size, cu_size, C)
    return cu_batch


def split_images_to_cus_without_padding(frames, cu_shape):
    """split frame from shape:(N,H,W,C) to shape:(N, H_in_cu, W_in_cu, cu_size, cu_size, C)  without padding !!!  """
    assert frames.ndim == 4, 'DataZoo must have shape:(N, H, W, C)'

    n, h, w, c = frames.shape
    if isinstance(cu_shape, int):
        cu_h = cu_w = cu_shape
    elif isinstance(cu_shape, (list, tuple, np.array)):
        cu_h = cu_shape[0]
        cu_w = cu_shape[1]
    else:
        raise ValueError

    height_in_cu = h // cu_h
    width_in_cu = w // cu_w

    frames_withoutpadding = frames[:, 0:height_in_cu * cu_h, 0:width_in_cu * cu_w, :]
    # split height into `height_in_cu` slices
    cu_rows = np.split(frames_withoutpadding, height_in_cu, axis=1)
    cu_rows_cols = []
    for row in cu_rows:
        cu_rows_cols.append(np.split(row, width_in_cu, axis=2))
    cu_rows_cols = np.array(cu_rows_cols)  # shape: (H_in_cu, W_in_cu, N, cu_size, cu_size, C)
    cu_batch = np.swapaxes(cu_rows_cols, 0, 2)
    cu_batch = np.swapaxes(cu_batch, 1, 2)  # shape: (N, H_in_cu, W_in_cu, cu_size, cu_size, C)
    return cu_batch


def merge_cus_to_images(cu_batch):
    assert cu_batch.ndim == 6, 'cu_batch must have shape:(N, H_in_cu, W_in_cu, cu_size, cu_size, C)'
    cu_batch = np.swapaxes(cu_batch, 0, 1)
    cu_batch = np.swapaxes(cu_batch, 1, 2)  # (H_in_cu, W_in_cu, N, cu_size, cu_size, C)
    cu_batch = np.concatenate(list(cu_batch), axis=2)  # (W_in_cu, N, H_in_cu*cu_size, cu_size, C)
    cu_batch = np.concatenate(list(cu_batch), axis=2)  # (N, H_in_cu*cu_size, W_in_cu*cu_size, C)
    return cu_batch


def imresize(img, size, interp='bilinear', mode=None):
    img = img.astype(np.uint8)
    resized_img = scipy.misc.imresize(img, size, interp=interp, mode=mode)
    return resized_img


def rgb2yuv(img, version):
    if version == 'bt601':
        return _rgb2yuv_bt601(img)
    elif version == 'jpeg':
        return _rgb2yuv_jpeg(img)
    elif version == 'hdtv':
        return _rgb2yuv_hdtv(img)
    else:
        raise NotImplementedError('The specified version %s is not implemented' % version)


def yuv2rgb(img, version):
    if version == 'bt601':
        return _yuv2rgb_bt601(img)
    elif version == 'jpeg':
        return _yuv2rgb_jpeg(img)
    elif version == 'hdtv':
        return _yuv2rgb_hdtv(img)
    else:
        raise NotImplementedError('The specified version %s is not implemented' % version)


def _rgb2yuv_bt601(img):
    assert img.dtype == 'uint8', 'rgb2yuv only implemented for uint8 arrays'
    rgb = np.ndarray(img.shape, dtype=np.float64)  # float64
    rgb[:, :, :] = img[:, :, :]

    # transform matrix : ITU-R BT.601 version (SDTV)
    transform_matrix = np.array([[0.257, 0.504, 0.098],
                                 [-0.148, -0.291, 0.439],
                                 [0.439, -0.368, -0.071]])

    # transform and clip
    yuv = np.dot(rgb, transform_matrix.T)
    yuv[:, :, 0] = (yuv[:, :, 0] + 16.).clip(16, 235)
    yuv[:, :, 1:] = (yuv[:, :, 1:] + 128.).clip(16, 240)

    return yuv.astype('uint8')


def _yuv2rgb_bt601(img):
    assert img.dtype == 'uint8', 'yuv2rgb only implemented for uint8 arrays'
    # better clip input to the valid range just to be on the safe side
    yuv = np.ndarray(img.shape)  # float64
    yuv[:, :, 0] = img[:, :, 0].clip(16, 235).astype(yuv.dtype) - 16
    yuv[:, :, 1:] = img[:, :, 1:].clip(16, 240).astype(yuv.dtype) - 128

    # transform matrix : ITU-R BT.601 version (SDTV)
    transform_matrix = np.array([[1.164, 0., 1.596],
                                 [1.164, -0.392, -0.813],
                                 [1.164, 2.017, 0.]])

    ## ITU-R BT.709 version (HDTV)
    #  transform_matrix = array([[1.164,     0.,  1.793],
    #             [1.164, -0.213, -0.533],
    #             [1.164,  2.112,     0.]])

    rgb = np.dot(yuv, transform_matrix.T)
    return rgb.clip(0, 255).astype('uint8')


def _rgb2yuv_jpeg(img):
    assert img.dtype == 'uint8', 'rgb2yuv only implemented for uint8 arrays'
    rgb = np.ndarray(img.shape, dtype=np.float64)  # float64
    rgb[:, :, :] = img[:, :, :]

    # transform matrix : JPEG
    transform_matrix = np.array([[0.299, 0.587, 0.114],
                                 [-0.169, -0.331, 0.500],
                                 [0.500, -0.419, -0.081]])

    # transform and clip
    yuv = np.dot(rgb, transform_matrix.T)
    yuv[:, :, 0] = (yuv[:, :, 0]).clip(0, 255)
    yuv[:, :, 1:] = (yuv[:, :, 1:] + 128.).clip(0, 255)

    return yuv.astype('uint8')


def _yuv2rgb_jpeg(img):
    assert img.dtype == 'uint8', 'yuv2rgb only implemented for uint8 arrays'
    # better clip input to the valid range just to be on the safe side
    yuv = np.ndarray(img.shape)  # float64
    yuv[:, :, 0] = img[:, :, 0].clip(16, 235).astype(yuv.dtype)
    yuv[:, :, 1:] = img[:, :, 1:].clip(16, 240).astype(yuv.dtype) - 128

    # transform matrix : JPEG
    transform_matrix = np.array([[1.0, 0., 1.400],
                                 [1.0, -0.343, -0.711],
                                 [1.0, 1.765, 0.0]])

    rgb = np.dot(yuv, transform_matrix.T)
    return rgb.clip(0, 255).astype('uint8')


def _rgb2yuv_hdtv(img):
    assert img.dtype == 'uint8', 'rgb2yuv only implemented for uint8 arrays'
    rgb = np.ndarray(img.shape, dtype=np.float64)  # float64
    rgb[:, :, :] = img[:, :, :]

    # transform matrix : ITU-R BT.709 version (HDTV)
    transform_matrix = np.array([[0.183, 0.614, 0.062],
                                 [-0.101, -0.339, 0.439],
                                 [0.439, -0.399, -0.040]])

    # transform and clip
    yuv = np.dot(rgb, transform_matrix.T)
    yuv[:, :, 0] = (yuv[:, :, 0] + 16.).clip(16, 235)
    yuv[:, :, 1:] = (yuv[:, :, 1:] + 128.).clip(16, 240)

    return yuv.astype('uint8')


def _yuv2rgb_hdtv(img):
    assert img.dtype == 'uint8', 'yuv2rgb only implemented for uint8 arrays'
    # better clip input to the valid range just to be on the safe side
    yuv = np.ndarray(img.shape)  # float64
    yuv[:, :, 0] = img[:, :, 0].clip(16, 235).astype(yuv.dtype) - 16
    yuv[:, :, 1:] = img[:, :, 1:].clip(16, 240).astype(yuv.dtype) - 128

    # transform matrix : ITU-R BT.709 version (HDTV)
    transform_matrix = np.array([[1.164, 0., 1.793],
                                 [1.164, -0.213, -0.533],
                                 [1.164, 2.112, 0.]])

    rgb = np.dot(yuv, transform_matrix.T)
    return rgb.clip(0, 255).astype('uint8')


""" Image and Coding Units transform for Intra Mode Decision """


def get_cu_with_ref_cus(inputs, cu_size):
    # 1. pad the left, top and right of the image
    padded_inputs = np.pad(inputs,
                           [[0, 0], [cu_size, 0], [cu_size, cu_size], [0, 0]],
                           mode='constant',
                           constant_values=128)

    # 2. split image to coding units
    cu_tiles = split_images_to_cus(padded_inputs, cu_size)  # [N, height_in_cu, width_in_cu, cu_h, cu_w, C]

    height_in_cu = cu_tiles.shape[1]
    width_in_cu = cu_tiles.shape[2]
    # 3. get current cu and its reference cu
    total_ref_cus = []
    total_cur_cu = []
    for i in range(1, height_in_cu):
        for j in range(1, width_in_cu - 1):
            top_cu = cu_tiles[:, i - 1, j, ...]
            left_cu = cu_tiles[:, i, j - 1, ...]
            top_left_cu = cu_tiles[:, i - 1, j - 1, ...]
            top_right_cu = cu_tiles[:, i - 1, j + 1, ...]

            ref_cus = np.concatenate((top_cu, left_cu, top_left_cu, top_right_cu), axis=3)
            cur_cu = cu_tiles[:, i, j, ...]
            total_ref_cus.append(ref_cus)
            total_cur_cu.append(cur_cu)
    stacked_ref_cus = np.stack(total_ref_cus, 1)
    stacked_cur_cu = np.stack(total_cur_cu, 1)
    return stacked_cur_cu, stacked_ref_cus


def imread(image_path, is_grayscale=False):
    """read image: (H, W, C), in which the order of Channel is RGB"""
    # if image is yuv, then call yuvread
    if is_grayscale:
        img = cv2.imread(image_path, 0)
        img = np.expand_dims(img, axis=-1)
    else:
        img = cv2.imread(image_path)
    img[:, :, :] = img[:, :, [2, 1, 0]]  # change from BGR to RGB
    return img


def imwrite(img, image_path):
    """write img to image_path"""
    img[:, :, :] = img[:, :, [2, 1, 0]]  # change from RGB to BGR
    cv2.imwrite(image_path, img)


class YuvReader(object):
    def __init__(self, yuv_path, y_h=None, y_w=None, v_format='yuv420'):
        self.yuv_path = yuv_path
        self.v_format = v_format
        self.yuv_handle = open(yuv_path, 'rb')

        if y_h is None or y_w is None:
            y_w, y_h = get_width_height_from_name(yuv_path)
        self.y_w, self.y_h = y_w, y_h

        self.y_size = self.y_w * self.y_h

    def read(self, n_frames, yuv_is_tuple=False):
        """ read n_frames, which format is
        (N, H, W, C) if yuv_is_tuple is False,and C is 3 when yuv444,yuv420, C is 1 when yuv400
        else ((N,H,W,1),(N,H,W,1),(N,H,W,1))"""
        yuv = None
        if self.v_format == 'yuv444':
            yuv = self._read_yuv444(n_frames, yuv_is_tuple)
        elif self.v_format == 'yuv420':
            yuv = self._read_yuv420(n_frames, yuv_is_tuple)
        elif self.v_format == 'yuv400':
            yuv = self._read_yuv400(n_frames, yuv_is_tuple)
        return yuv

    def done(self):
        self.yuv_handle.close()

    def _read_with_exception(self, size):
        raw_data = self.yuv_handle.read(size)
        if len(raw_data) < size:
            raise _NoMoreDataException('no more data')
        data = list(map(ord, raw_data))
        return data

    def _read_yuv444(self, n_frames, yuv_is_tuple):
        self.u_w = self.y_w
        self.u_h = self.y_h
        self.u_size = self.u_w * self.u_h

        data_y, data_u, data_v = [], [], []
        for idx in range(n_frames):
            try:
                single_y = self._read_with_exception(self.y_size)
                single_u = self._read_with_exception(self.u_size)
                single_v = self._read_with_exception(self.u_size)
            except _NoMoreDataException:
                if len(data_y) == 0:
                    return [None, None, None] if yuv_is_tuple else None
                else:
                    break
            else:
                single_y = np.reshape(single_y, (1, self.y_h, self.y_w, 1))
                single_u = np.reshape(single_u, (1, self.u_h, self.u_w, 1))
                single_v = np.reshape(single_v, (1, self.u_h, self.u_w, 1))
                data_y.append(single_y)
                data_u.append(single_u)
                data_v.append(single_v)
        data_y = np.concatenate(data_y, axis=0)
        data_u = np.concatenate(data_u, axis=0)
        data_v = np.concatenate(data_v, axis=0)
        rets = [data_y, data_u, data_v]
        if yuv_is_tuple:
            return rets
        else:
            return np.concatenate(rets, axis=-1)

    def _read_yuv420(self, n_frames, yuv_is_tuple):
        self.u_w = self.y_w // 2
        self.u_h = self.y_h // 2
        self.u_size = self.u_w * self.u_h

        data_y, data_u, data_v = [], [], []
        for idx in range(n_frames):
            try:
                single_y = self._read_with_exception(self.y_size)
                single_u = self._read_with_exception(self.u_size)
                single_v = self._read_with_exception(self.u_size)
            except _NoMoreDataException:
                if len(data_y) == 0:
                    return [None, None, None] if yuv_is_tuple else None
                else:
                    break
            else:
                single_y = np.reshape(single_y, (1, self.y_h, self.y_w, 1))
                single_u = np.reshape(single_u, (1, self.u_h, self.u_w, 1))
                single_v = np.reshape(single_v, (1, self.u_h, self.u_w, 1))
                data_y.append(single_y)
                data_u.append(single_u)
                data_v.append(single_v)
        data_y = np.concatenate(data_y, axis=0)
        data_u = np.concatenate(data_u, axis=0)
        data_v = np.concatenate(data_v, axis=0)

        if yuv_is_tuple:
            return [data_y, data_u, data_v]
        else:
            data_u = upsample_chroma(data_u)
            data_v = upsample_chroma(data_v)
            return np.concatenate([data_y, data_u, data_v], axis=-1)

    def _read_yuv400(self, n_frames, yuv_is_tuple):
        self.u_w = self.u_h = self.u_size = 0

        data_y = []
        for idx in range(n_frames):
            try:
                single_y = self._read_with_exception(self.y_size)
            except _NoMoreDataException:
                if len(data_y) == 0:
                    return [None, None, None] if yuv_is_tuple else None
                else:
                    break
            else:
                single_y = np.reshape(single_y, (1, self.y_h, self.y_w, 1))
                data_y.append(single_y)
        data_y = np.concatenate(data_y, axis=0)
        rets = [data_y, None, None]
        if yuv_is_tuple:
            return rets
        else:
            return data_y


class YuvWriter(object):
    def __init__(self, yuv_path):
        self.yuv_path = yuv_path
        self.yuv_handle = open(yuv_path, 'wb')

    def done(self):
        self.yuv_handle.close()

    def write(self, yuv_data, yuv_is_tuple):
        if yuv_is_tuple:
            if yuv_data[1] is None or yuv_data[2] is None:
                # yuv400
                flatten_data = yuv_data[0].astype('uint8').ravel()
                stream_data = ''.join(map(chr, flatten_data))
                self.yuv_handle.write(stream_data)
            else:
                if yuv_data[0].ndim == 3:
                    yuv_data = list(map(lambda x: x[None, ...], yuv_data))

                assert yuv_data[0].shape[0] == yuv_data[1].shape[0] and \
                       yuv_data[1].shape[0] == yuv_data[2].shape[0]

                for frame_i in range(yuv_data[0].shape[0]):
                    for yuv_j in range(3):
                        data_i_j = yuv_data[yuv_j][frame_i]
                        flatten_data = data_i_j.astype('uint8').ravel()
                        stream_data = ''.join(map(chr, flatten_data))
                        self.yuv_handle.write(stream_data)
        else:
            self._write_nhwc_hwc(yuv_data)

    def _write_nhwc_hwc(self, data):
        dim_h = data.ndim - 3
        dim_w = data.ndim - 2
        dim_c = data.ndim - 1
        transposed_data = np.swapaxes(data, dim_h, dim_c)
        transposed_data = np.swapaxes(transposed_data, dim_c, dim_w)
        stream_data = transposed_data.astype('uint8').tobytes()
        self.yuv_handle.write(stream_data)


class _NoMoreDataException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)


def tiny_yuvread(yuv_path, y_h=None, y_w=None, n_frames=None, v_format='yuv420', yuv_is_tuple=False):
    """read one yuv file"""
    reader = YuvReader(yuv_path, y_h, y_w, v_format=v_format)
    yuv = reader.read(n_frames, yuv_is_tuple=yuv_is_tuple)
    reader.done()
    return yuv


def tiny_yuvwrite(yuv_path, yuv, yuv_is_tuple=False):
    writer = YuvWriter(yuv_path)
    writer.write(yuv, yuv_is_tuple)
    writer.done()
