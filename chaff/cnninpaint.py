import torch
import torch.nn as nn
from tqdm import tqdm
from skimage.measure import compare_psnr
from .unetn2n_relu import UNet  # relu works better fore some reason
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d


def checkerboard_level_set(image_shape, square_size=5):
    """Create a checkerboard level set with binary values.
    Parameters
    ----------
    image_shape : tuple of positive integers
        Shape of the image.
    square_size : int, optional
        Size of the squares of the checkerboard. It defaults to 5.
    Returns
    -------
    out : array with shape `image_shape`
        Binary level set of the checkerboard.
    See also
    --------
    circle_level_set
    """

    grid = np.mgrid[[slice(i) for i in image_shape]]
    grid = (grid // square_size)

    # Alternate 0/1 for even/odd numbers.
    grid = grid & 1

    checkerboard = np.bitwise_xor.reduce(grid, axis=0)
    res = np.int8(checkerboard)
    return res


def pad2bin(image, n=16):
    shape = image.shape
    newshape = [n * ceil(i / n) for i in image.shape]

    if len(newshape) > 2:
        newshape[2] = shape[2]

    return np.pad(image, [(0, n - o)
                          for n, o in zip(newshape, shape)], 'reflect')


def split_image(image):
    if len(image.shape) == 2:
        image_gv = image
        image = np.zeros((*image_gv.shape, 3))
        for i in range(3):
            image[:, :, i] = image_gv
        # print(f'Gray to RGB, {image.shape}')
    mask1 = checkerboard_level_set(image.shape, 1)
    mask1[:, :, 1] = mask1[:, :, 0]
    mask2 = 1 - mask1

    kernel = np.zeros((3, 3))
    kernel[0, 1] = 1
    kernel[1, 0] = 1
    kernel[1, 2] = 1
    kernel[2, 1] = 1
    kernel /= np.sum(kernel)
    image_interp = image * 0
    for i in range(3):
        image_interp[:, :, i] = convolve2d(
            image[:, :, i], kernel, boundary='symm', mode='same')

    im1 = mask1 * image + mask2 * image_interp
    im2 = mask2 * image + mask1 * image_interp
    return im1, im2


def split_image_gray(image):
    mask1 = checkerboard_level_set(image.shape, 1)
    mask2 = 1 - mask1

    kernel = np.zeros((3, 3))
    kernel[0, 1] = 1
    kernel[1, 0] = 1
    kernel[1, 2] = 1
    kernel[2, 1] = 1
    kernel /= np.sum(kernel)
    image_interp = convolve2d(image, kernel, boundary='symm', mode='same')

    im1 = mask1 * image + mask2 * image_interp
    im2 = mask2 * image + mask1 * image_interp
    return im1, im2


def split_image_noise(image):
    if len(image.shape) == 2:
        image_gv = image
        image = np.zeros((*image_gv.shape, 3))
        for i in range(3):
            image[:, :, i] = image_gv
        # print(f'Gray to RGB, {image.shape}')
    mask1 = checkerboard_level_set(image.shape, 1)
    mask1[:, :, 1] = mask1[:, :, 0]
    mask2 = 1 - mask1

    kernel = np.zeros((3, 3))
    kernel[0, 1] = 1
    kernel[1, 0] = 1
    kernel[1, 2] = 1
    kernel[2, 1] = 1
    kernel /= np.sum(kernel)

    image_interp = image * 0
    for i in range(3):
        image_interp[:, :, i] = convolve2d(
            image[:, :, i], kernel, boundary='symm', mode='same')

    im1 = mask1 * image + mask2 * image_interp
    im2 = mask2 * image + mask1 * image_interp

    noise = (np.std(im1 - im2))

    image_interp = image_interp + noise * np.random.standard_normal(
        image_interp.shape)

    im1 = mask1 * image + mask2 * image_interp
    im2 = mask2 * image + mask1 * image_interp

    return im1, im2


def combine_split(im1, im2):
    mask1 = checkerboard_level_set(im1.shape, 1)
    mask1[:, :, 1] = mask1[:, :, 0]
    mask2 = 1 - mask1

    return (mask1 * im1 + mask2 * im2)


class CNN_inpaint():
    def __init__(self, in_channels=1, depth=5, start_filts=48,
                 input_skip=True):

        self.model = UNet(
            in_channels=in_channels,
            depth=depth,
            start_filts=start_filts,
            input_skip=input_skip)
        self.criterion = nn.MSELoss()
        self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.res_psnr = None
        self.res_loss = None

        self.epoch = 0

    def sgd_step(self, im1, im2, alpha=0.95):
        b_colorimage = len(im1.shape) == 3
        if b_colorimage:
            # color image
            im_t = np.transpose(im1.astype('float32'), (2, 0, 1))
            torch_im1 = torch.from_numpy(im_t).view(1, *im_t.shape).cuda()
            im_t = np.transpose(im2.astype('float32'), (2, 0, 1))
            torch_im2 = torch.from_numpy(im_t).view(1, *im_t.shape).cuda()
        else:
            torch_im1 = torch.from_numpy(im1.astype('float32')).view(
                1, 1, *im1.shape).cuda()
            torch_im2 = torch.from_numpy(im2.astype('float32')).view(
                1, 1, *im2.shape).cuda()

        # ===================forward=====================
        output1 = self.model(torch_im1)
        estimate = torch_im1 * (1 - alpha) + torch_im2 * alpha
        loss = self.criterion(output1, estimate)
        # ===================backward====================
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        out1 = output1.cpu().detach().numpy()

        if b_colorimage:
            return np.transpose(out1[0, :], (1, 2, 0)), loss.item()
        else:
            return out1[0, 0, :], loss.item()

    def train(self, imagelist, num_epochs=10, lr=1e-3, alpha=0.95,
              mode='mean'):

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.res_loss = []
        self.res_psnr = []

        self.model.train()
        pbar = tqdm(
            range(num_epochs), ncols=60, desc='PSNR {}/{}'.format(0, 0))

        for epoch in pbar:
            self.epoch += 1
            for image in imagelist:
                if mode == 'mean':
                    im1_pad, im2_pad = split_image_gray(pad2bin(image))
                elif mode == 'noise':
                    im1_pad, im2_pad = split_image_gray(pad2bin(image))

                out_cnn1, loss1 = self.sgd_step(im1_pad, im2_pad, alpha)
                out_cnn2, loss2 = self.sgd_step(im2_pad, im1_pad, alpha)

                self.res_loss.append(loss1)
                self.res_loss.append(loss2)
                out_cnn = (out_cnn1 + out_cnn2) / 2
                # out_cnn = out_cnn1 + out_cnn2

                psnr_range = np.max(image) - np.min(image)
                psnr_cnn = compare_psnr(out_cnn1, out_cnn2, psnr_range)
                psnr_input = compare_psnr(im1_pad, im2_pad, psnr_range)

                self.res_psnr.append(psnr_cnn)
                pbar.set_description('PSNR ({:.3}/{:.3})'.format(
                    psnr_cnn, psnr_input))

    def filter_mean(self, image):
        b_colorimage = len(image.shape) == 3

        self.model.eval()
        shape = image.shape
        if b_colorimage:
            im1, im2 = split_image(pad2bin(image))
        else:
            im1, im2 = split_image_gray(pad2bin(image))

        with torch.no_grad():
            im_t = np.transpose(im1.astype('float32'), (2, 0, 1))
            torch_im1 = torch.from_numpy(im_t).view(1, *im_t.shape).cuda()
            im_t = np.transpose(im2.astype('float32'), (2, 0, 1))
            torch_im2 = torch.from_numpy(im_t).view(1, *im_t.shape).cuda()

            output1 = self.model(torch_im1)
            output2 = self.model(torch_im2)

        out1 = output1.cpu().detach().numpy()
        out2 = output2.cpu().detach().numpy()
        out1 = np.transpose(out1[0, :], (1, 2, 0))
        out2 = np.transpose(out2[0, :], (1, 2, 0))
        out_cnn = (out1 + out2) / 2
        retval = out_cnn[:shape[0], :shape[1]]

        if not b_colorimage:
            retval = np.mean(retval, 2)

        return retval

    def filter(self, image):
        b_colorimage = len(image.shape) == 3

        # if not b_colorimage:
        #     image_gv = image
        #     image = np.zeros((*image_gv.shape, 3))
        #     for i in range(3):
        #         image[:, :, i] = image_gv

        self.model.eval()
        shape = image.shape
        im1 = pad2bin(image)

        with torch.no_grad():
            if b_colorimage:
                # color image
                im_t = np.transpose(im1.astype('float32'), (2, 0, 1))
                torch_im1 = torch.from_numpy(im_t).view(1, *im_t.shape).cuda()
            else:
                torch_im1 = torch.from_numpy(im1.astype('float32')).view(
                    1, 1, *im1.shape).cuda()

            # print(torch_im1.shape)
            output1 = self.model(torch_im1)

        out1 = output1.cpu().detach().numpy()
        out1 = np.transpose(out1[0, :], (1, 2, 0))
        retval = out1[:shape[0], :shape[1]]

        if not b_colorimage:
            retval = np.mean(retval, 2)

        return retval

    def plot_train(self):
        plt.figure(figsize=(8, 3))
        plt.subplot(121)
        plt.plot(self.res_loss)
        plt.ylabel('L2 Loss')
        plt.subplot(122)
        plt.plot(self.res_psnr)
        plt.ylabel('PSNR')
        plt.show()
