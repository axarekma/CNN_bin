'''
Misc utility functions for CNNbin
'''
from functools import reduce
from math import ceil

import numpy as np


def window(shape, func, **kwargs):
    '''[summary]

    Args:
        shape (Tuple): Shape of image
        func (Function): Function returning 1d window function

    Returns:
        [ndarray]: windowing function along all dimesnions of shape
    '''

    windowfuncs = [func(l, **kwargs) for l in shape]
    return reduce(np.multiply, np.ix_(*windowfuncs))


def pad2bin(image, n_div=16, mode='reflect'):
    '''Pad image to be divisible by n_div

    Args:
        image (ndarray): input image
        n_div (int, optional): Defaults to 16.
        mode (string, optional): Defaults to 'reflect'. Mode for numpy.pad

    Returns:
        ndarray: Padded image
    '''

    shape = image.shape
    newshape = [n_div * ceil(i / n_div) for i in image.shape]

    if len(newshape) > 2:
        newshape[2] = shape[2]

    return np.pad(image, [(0, n_div - o)
                          for n_div, o in zip(newshape, shape)], mode)


def random_patches(image, block_size, max_patches, random_state=None):
    '''Generate random pathces drawn from image

    Args:
        image (ndarray): input image
        block_size (tuple): 2D shape of patches
        max_patches (int): number of patches
        random_state ({None, int, array_like}, optional):
            Random seed used to initialize the pseudo-random number generator.

    Returns:
        [ndarray]: array of patches, patches along dim 0.
    '''

    random_generator = np.random.RandomState(random_state)
    d_0, d_1 = image.shape[0], image.shape[1]

    ind0 = np.arange(1 + d_0 - block_size[0])
    ind1 = np.arange(1 + d_1 - block_size[1])

    sample0 = random_generator.choice(ind0, size=max_patches)
    sample1 = random_generator.choice(ind1, size=max_patches)

    if image.ndim == 3:
        retval = np.zeros((max_patches, *block_size, 3))
    else:
        retval = np.zeros((max_patches, *block_size))

    for i in range(max_patches):
        retval[i] = image[sample0[i]:sample0[i] +
                          block_size[0], sample1[i]:sample1[i] + block_size[1]]

    return retval.astype(image.dtype)


def split_stack(stack):
    new_shape = [i // 2 for i in stack.shape]
    new_shape[0] = stack.shape[0]

    stack1 = np.zeros(new_shape)
    stack2 = np.zeros(new_shape)

    for i, image in enumerate(stack):
        im1, im2 = split_diagonal(image)
        stack1[i] = im1
        stack2[i] = im2

    return stack1, stack2


def split_diagonal(image):
    n0, n1 = image.shape
    new_shape = (n0 // 2, 2, n1 // 2, 2)
    ndarray = image.reshape(new_shape)
    image1 = ndarray[:, 0, :, 0] + ndarray[:, 1, :, 1]
    image2 = ndarray[:, 0, :, 1] + ndarray[:, 1, :, 0]

    return image1 / 2, image2 / 2


def crop2bin(image, n=16):
    newshape = [n * (i // n) for i in image.shape]
    return image[:newshape[0], :newshape[1]]


def data_gaussian(image, sigma):
    image = crop2bin(image)
    noisy = image + sigma * np.random.standard_normal(image.shape)
    noisy = np.clip(noisy, 0, 1e9)
    return image, noisy


def bin_ndarray(ndarray, operation='mean'):
    new_shape = [i // 2 for i in ndarray.shape]

    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(
            ndarray.shape, new_shape))

    compression_pairs = [(d, c // d) for d, c in zip(new_shape, ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1 * (i + 1))
    return ndarray


def split_bins(image):
    n0, n1 = image.shape
    new_shape = (n0 // 2, 2, n1 // 2, 2)
    ndarray = image.reshape(new_shape)
    im1 = ndarray[:, 0, :, 0]
    im2 = ndarray[:, 0, :, 1]
    im3 = ndarray[:, 1, :, 0]
    im4 = ndarray[:, 1, :, 1]
    return im1, im2, im3, im4


def split_rgb(image, mode='diag'):
    new_shape = [i // 2 for i in image.shape]
    retval1 = np.zeros((new_shape[0], new_shape[1], 3))
    retval2 = np.zeros((new_shape[0], new_shape[1], 3))
    retval3 = np.zeros((new_shape[0], new_shape[1], 3))
    retval4 = np.zeros((new_shape[0], new_shape[1], 3))
    for i in range(3):
        im1, im2, im3, im4 = split_bins(image[:, :, i])
        retval1[:, :, i] = im1
        retval2[:, :, i] = im2
        retval3[:, :, i] = im3
        retval4[:, :, i] = im4
    if mode == 'all':
        return retval1, retval2, retval3, retval4
    if mode == 'diag':
        return retval1 + retval3, retval2 + retval4


def split_diagonal_rgb(image):
    new_shape = [i // 2 for i in image.shape]
    retval1 = np.zeros((new_shape[0], new_shape[1], 3))
    retval2 = np.zeros((new_shape[0], new_shape[1], 3))
    retval3 = np.zeros((new_shape[0], new_shape[1], 3))
    retval4 = np.zeros((new_shape[0], new_shape[1], 3))
    for i in range(3):
        im1, im2, im3, im4 = split_bins(image[:, :, i])
        retval1[:, :, i] = im1 / 2
        retval2[:, :, i] = im2 / 2
        retval3[:, :, i] = im3 / 2
        retval4[:, :, i] = im4 / 2
    return retval1 + retval3, retval2 + retval4


def split_stack_rgb(stack):
    new_shape = [i // 2 for i in stack.shape]
    new_shape[0] = stack.shape[0]
    new_shape[3] = stack.shape[3]

    stack1 = np.zeros(new_shape)
    stack2 = np.zeros(new_shape)

    for i, image in enumerate(stack):
        im1, im2 = split_rgb(image)
        stack1[i] = im1 / 2
        stack2[i] = im2 / 2

    return stack1, stack2


def psnr(f, x, psnr_range=None):
    if psnr_range is None:
        psnr_range = np.max(f) - np.min(f)
    MSE = np.mean((x - f) * (x - f))
    return 20 * np.log10(psnr_range / np.sqrt(MSE))
