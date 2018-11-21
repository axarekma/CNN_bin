import numpy as np
from math import ceil
from .utils import window


def split(image, block_size, sampling=1.0, ind_div=1):
    n_div = [
        int(ceil(sampling * d / b)) for d, b in zip(image.shape, block_size)
    ]

    limits = [
        divide(l, bs, n, ind_div=ind_div)
        for l, bs, n in zip(image.shape, block_size, n_div)
    ]

    image_blocks = []
    for i in range(n_div[0]):
        for j in range(n_div[1]):
            image_blocks.append(image[limits[0][0][i]:limits[0][1][i], limits[
                1][0][j]:limits[1][1][j]])

    return np.array(image_blocks)


def combine(blocklist, block_size, shape, sampling=1.0, windowfunc=None):
    n_div = [int(ceil(sampling * d / b)) for d, b in zip(shape, block_size)]

    limits = [divide(l, bs, n) for l, bs, n in zip(shape, block_size, n_div)]

    image = np.zeros(shape)
    image_n = np.zeros(shape)

    if windowfunc is None:
        index = 0
        for i in range(n_div[0]):
            for j in range(n_div[1]):
                image[limits[0][0][i]:limits[0][1][i], limits[1][0][j]:limits[
                    1][1][j]] += blocklist[index]
                image_n[limits[0][0][i]:limits[0][1][i], limits[1][0][j]:
                        limits[1][1][j]] += 1
                index += 1
    else:
        window_block = 0.1 + window(block_size, windowfunc)
        index = 0
        for i in range(n_div[0]):
            for j in range(n_div[1]):
                image[limits[0][0][i]:limits[0][1][i], limits[1][0][j]:limits[
                    1][1][j]] += window_block * blocklist[index]
                image_n[limits[0][0][i]:limits[0][1][i], limits[1][0][j]:
                        limits[1][1][j]] += window_block
                index += 1

    return image / image_n


def combine_rgb(blocklist, block_size, shape, sampling=1.0, windowfunc=None):
    n_div = [int(ceil(sampling * d / b)) for d, b in zip(shape, block_size)]

    limits = [divide(l, bs, n) for l, bs, n in zip(shape, block_size, n_div)]

    image = np.zeros(shape)
    image_n = np.zeros(shape)

    if windowfunc is None:
        index = 0
        for i in range(n_div[0]):
            for j in range(n_div[1]):

                image[limits[0][0][i]:limits[0][1][i], limits[1][0][j]:limits[
                    1][1][j], :] += blocklist[index]
                image_n[limits[0][0][i]:limits[0][1][i], limits[1][0][j]:
                        limits[1][1][j]] += 1
                index += 1
    else:
        window_block = 0.01 + window(block_size, windowfunc)
        # print(np.min(window_block), np.max(window_block))
        window_block = np.transpose([window_block, window_block, window_block],
                                    (1, 2, 0))
        index = 0
        for i in range(n_div[0]):
            for j in range(n_div[1]):

                image[limits[0][0][i]:limits[0][1][i], limits[1][0][j]:limits[
                    1][1][j]] += window_block * blocklist[index]
                image_n[limits[0][0][i]:limits[0][1][i], limits[1][0][j]:
                        limits[1][1][j]] += window_block
                index += 1

    # return 255 * image_n
    return image / image_n


def divide(length, block_size, n_blocks, ind_div=1):
    l_total = block_size * n_blocks
    overlap = (l_total - length) / (n_blocks - 1) / 2

    l_minus_edges = length - 2 * overlap
    center_points = [
        overlap + l_minus_edges * (n + 0.5) / (n_blocks)
        for n in range(n_blocks)
    ]

    lim_a = [(c - block_size / 2) for c in center_points]
    lim_b = [(c + block_size / 2) for c in center_points]

    return [int(round(a / ind_div) * ind_div) for a in
            lim_a], [int(round(b / ind_div) * ind_div) for b in lim_b]


class BlockDivide():
    def __init__(self, block_shape, sampling):
        self.block_shape = block_shape
        self.sampling = sampling

        self.data_shape = None

    def blocks(self, data):
        self.data_shape = data.shape
        return split(data, self.block_shape, self.sampling)

    def fuse(self, blocks):
        return combine(blocks, self.block_shape, self.data_shape,
                       self.sampling)
