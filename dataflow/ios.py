from __future__ import absolute_import
import numpy as np
import cv2
import sys

sys.path.append('../')
from dataflow import utils


def yuvread(yuv_path, y_h=None, y_w=None, n_frames=None, v_format='yuv420', yuv_is_tuple=False):
    """read one yuv file"""
    reader = YuvReader(yuv_path, y_h, y_w, v_format=v_format)
    yuv = reader.read(n_frames, yuv_is_tuple=yuv_is_tuple)
    reader.done()
    return yuv


def yuvwrite(yuv_path, yuv, yuv_is_tuple=False):
    writer = YuvWriter(yuv_path)
    writer.write(yuv, yuv_is_tuple)
    writer.done()


class YuvReader(object):
    def __init__(self, yuv_path, y_h=None, y_w=None, v_format='yuv420', bit=8):
        self.yuv_path = yuv_path
        self.v_format = v_format
        self.yuv_handle = open(yuv_path, 'rb')
        # self.bit = bit

        if y_h is None or y_w is None:
            y_w, y_h = utils.get_width_height_from_name(yuv_path)
        self.y_w, self.y_h = y_w, y_h

        self.y_size = self.y_w * self.y_h

    def read(self, n_frames, yuv_is_tuple=False, dtype=None):
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
        elif self.v_format == 'yuy2':
            yuv = self._read_yuy2(n_frames, yuv_is_tuple)
        elif self.v_format == 'y4':
            yuv = self._read_y4(n_frames, yuv_is_tuple)
        elif self.v_format == 'y2':
            yuv = self._read_y2(n_frames, yuv_is_tuple)
        elif self.v_format == 'y16':
            yuv = self._read_y16(n_frames, yuv_is_tuple)
        if dtype and yuv:
            if isinstance(yuv, (list, tuple)):
                yuv = [comp.astype(dtype) if comp is not None else None for comp in yuv]
            else:
                yuv = yuv.astype(dtype)
        return yuv

    def done(self):
        self.yuv_handle.close()

    def _read_with_exception(self, size):
        raw_data = self.yuv_handle.read(size)
        if len(raw_data) < size:
            raise _NoMoreDataException('no more data')

        data = np.fromstring(raw_data, dtype='uint8')
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
            data_u = self._upsample_chroma(data_u)
            data_v = self._upsample_chroma(data_v)
            return np.concatenate([data_y, data_u, data_v], axis=-1)

    def _read_yuy2(self, n_frames, yuv_is_tuple):
        self.u_w = self.y_w
        self.u_h = self.y_h // 2
        self.u_size = self.u_w * self.u_h
        print('u_size')
        print(self.u_size)

        data_y, data_u, data_v = [], [], []
        for idx in range(n_frames):
            try:
                single_frame = self._read_with_exception(self.y_size + 2 * self.u_size)
                print(self.y_size + 2 * self.u_size)
                single_y = single_frame[:-1:2]
                single_u = single_frame[1:-1:4]
                single_v = single_frame[3::4]
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
            data_u = self._upsample_chroma(data_u)
            data_v = self._upsample_chroma(data_v)
            return np.concatenate([data_y, data_u, data_v], axis=-1)

    def _read_y4(self, n_frames, yuv_is_tuple):

        data_y = []
        for idx in range(n_frames):
            try:
                single_frame = self._read_with_exception(4 * self.y_size)
            except _NoMoreDataException:
                if len(data_y) == 0:
                    return [None, None, None] if yuv_is_tuple else None
                else:
                    break
            else:
                single_y = np.reshape(single_frame, (1, self.y_h, self.y_w, 4))
                data_y.append(single_y)
        data_y = np.concatenate(data_y, axis=0)

        if yuv_is_tuple:
            return data_y
        else:
            return data_y

    def _read_y2(self, n_frames, yuv_is_tuple):

        data_y = []
        for idx in range(n_frames):
            try:
                single_frame = self._read_with_exception(2 * self.y_size)
            except _NoMoreDataException:
                if len(data_y) == 0:
                    return [None, None, None] if yuv_is_tuple else None
                else:
                    break
            else:
                single_y = np.reshape(single_frame, (1, self.y_h, self.y_w, 2))
                data_y.append(single_y)
        data_y = np.concatenate(data_y, axis=0)

        if yuv_is_tuple:
            return data_y
        else:
            return data_y

    def _read_y16(self, n_frames, yuv_is_tuple):

        data_y = []
        for idx in range(n_frames):
            try:
                single_frame = self._read_with_exception(16 * self.y_size)
            except _NoMoreDataException:
                if len(data_y) == 0:
                    return [None, None, None] if yuv_is_tuple else None
                else:
                    break
            else:
                single_y = np.reshape(single_frame, (1, self.y_h, self.y_w, 16))
                data_y.append(single_y)
        data_y = np.concatenate(data_y, axis=0)

        if yuv_is_tuple:
            return data_y
        else:
            return data_y


    def _upsample_chroma(self, data_u):
        ndim = data_u.ndim
        dim_h = ndim - 3
        dim_w = ndim - 2
        upsampled_chroma = np.repeat(data_u, 2, axis=dim_h)
        upsampled_chroma = np.repeat(upsampled_chroma, 2, axis=dim_w)
        return upsampled_chroma

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

    def done(self, msg=False):
        self.yuv_handle.close()
        if msg:
            print('Write %s done.' % self.yuv_path)

    def write(self, yuv_data, yuv_is_tuple):
        if yuv_is_tuple:
            if yuv_data[1] is None or yuv_data[2] is None:
                # yuv400
                stream_data = yuv_data[0].astype('uint8').tobytes()
                self.yuv_handle.write(stream_data)
            else:
                if yuv_data[0].ndim == 3:
                    yuv_data = list(map(lambda x: x[None, ...], yuv_data))

                assert yuv_data[0].shape[0] == yuv_data[1].shape[0] and \
                       yuv_data[1].shape[0] == yuv_data[2].shape[0]
                for frame_yuv in zip(*yuv_data):
                    for data in frame_yuv:
                        stream_data = data.astype('uint8').tobytes()
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


def reshape_yuvs(input_path, output_path, h, w, frames):
    Y, U, V = yuvread(input_path, yuv_is_tuple=True, n_frames=frames)
    y = []
    u = []
    v = []
    for i in range(frames):
        luma = cv2.resize(Y[i,], (w, h))
        luma = np.expand_dims(luma, 2)
        y.append(luma)
        chroma1 = cv2.resize(U[i,], (w / 2, h / 2))
        chroma1 = np.expand_dims(chroma1, 2)
        u.append(chroma1)
        chroma2 = cv2.resize(V[i,], (w / 2, h / 2))
        chroma2 = np.expand_dims(chroma2, 2)
        v.append(chroma2)
    Y = np.array(y)
    U = np.array(u)
    V = np.array(v)
    yuvwrite(output_path, [Y, U, V], yuv_is_tuple=True)
