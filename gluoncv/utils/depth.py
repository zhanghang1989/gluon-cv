# pylint: disable=arguments-differ
"""Custom Utils for Depth.
"""
from __future__ import absolute_import
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import nd
from mxnet.gluon.loss import Loss, _apply_weighting, _reshape_like
import mxnet.ndarray as F
import mxnet.gluon.nn as nn

# adapted from https://github.com/ClubAI/MonoDepth-PyTorch/blob/master/utils.py
class MonodepthLoss(gluon.loss.Loss):
    def __init__(self, nscales=3, ssim_weight=0.85, smooth_weigth=1.0, lr_weight=1.0,
                 weight=None, batch_axis=0, **kwargs):
        super(MonodepthLoss, self).__init__(weight, batch_axis, **kwargs)
        self.ssim_weight = ssim_weight
        self.smooth_weigth = smooth_weigth
        self.lr_weight = lr_weight
        self.nscales = nscales

    def forward(self, *inputs):
        """
        Args:
            inputs: [disp2, disp2, disp3... left, right]
        Return:
            The loss
        """
        # prediction, target
        prediction = inputs[:-2]
        left, right = inputs[-2], inputs[-1]
        assert len(prediction)==self.nscales
        left_pyramid = scale_pyramid(left, self.nscales)
        right_pyramid = scale_pyramid(right, self.nscales)

        # Prepare disparities
        disp_left_est = [d[:, 0, :, :].expand_dims(1) for d in prediction]
        disp_right_est = [d[:, 1, :, :].expand_dims(1) for d in prediction]

        # Generate images
        left_est = [generate_image_left(right_pyramid[i],
                    disp_left_est[i]) for i in range(self.nscales)]
        right_est = [generate_image_right(left_pyramid[i],
                     disp_right_est[i]) for i in range(self.nscales)]
        self.left_est = left_est
        self.right_est = right_est

        # L1
        l1_left = [F.mean(F.abs(left_est[i] - left_pyramid[i]))
                   for i in range(self.nscales)]
        l1_right = [F.mean(F.abs(right_est[i]
                    - right_pyramid[i])) for i in range(self.nscales)]

        # ssim
        ssim_left = [F.mean(ssim(left_est[i],
                     left_pyramid[i])) for i in range(self.nscales)]
        ssim_right = [F.mean(ssim(right_est[i],
                      right_pyramid[i])) for i in range(self.nscales)]

        image_loss_left = [self.ssim_weight * ssim_left[i]
                           + (1 - self.ssim_weight) * l1_left[i]
                           for i in range(self.nscales)]
        image_loss_right = [self.ssim_weight * ssim_right[i]
                            + (1 - self.ssim_weight) * l1_right[i]
                            for i in range(self.nscales)]
        image_loss = sum(image_loss_left + image_loss_right)

        # L-R Consistency
        right_left_disp = [generate_image_left(disp_right_est[i],
                           disp_left_est[i]) for i in range(self.nscales)]
        left_right_disp = [generate_image_right(disp_left_est[i],
                           disp_right_est[i]) for i in range(self.nscales)]

        lr_left_loss = [F.mean(F.abs(right_left_disp[i]
                        - disp_left_est[i])) for i in range(self.nscales)]
        lr_right_loss = [F.mean(F.abs(left_right_disp[i]
                         - disp_right_est[i])) for i in range(self.nscales)]
        lr_loss = sum(lr_left_loss + lr_right_loss)

        # Disparities smoothness
        disp_left_smoothness = disp_smoothness(disp_left_est, left_pyramid, self.nscales)
        disp_right_smoothness = disp_smoothness(disp_right_est, right_pyramid, self.nscales)
        disp_left_loss = [F.mean(F.abs(
                          disp_left_smoothness[i])) / 2 ** i
                          for i in range(self.nscales)]
        disp_right_loss = [F.mean(F.abs(
                           disp_right_smoothness[i])) / 2 ** i
                           for i in range(self.nscales)]
        disp_gradient_loss = sum(disp_left_loss + disp_right_loss)

        # total loss
        loss = image_loss + self.smooth_weigth * disp_gradient_loss \
               + self.lr_weight * lr_loss
        self.image_loss = image_loss
        self.disp_gradient_loss = disp_gradient_loss
        self.lr_loss = lr_loss
        return loss

def scale_pyramid(img, num_scales):
    scaled_imgs = [img]
    s = img.shape
    h = s[2]
    w = s[3]
    for i in range(num_scales - 1):
        ratio = 2 ** (i + 1)
        nh = h // ratio
        nw = w // ratio
        scaled_imgs.append(F.contrib.BilinearResize2D(img, nh, nw))
    return scaled_imgs

def gradient_x(img):
    # Pad input to keep output size consistent
    img = F.pad(img, pad_width=(0, 0, 0, 0, 0, 0, 0, 1), mode="edge")
    # NCHW
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]
    return gx

def gradient_y(img):
    # Pad input to keep output size consistent
    img = F.pad(img, pad_width=(0, 0, 0, 0, 0, 1, 0, 0), mode="edge")
    # NCHW
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gy

def generate_image_left(img, disp):
    return apply_disparity(img, -disp)

def generate_image_right(img, disp):
    return apply_disparity(img, disp)

def apply_disparity(img, disp):
    batch_size, _, height, width = img.shape

    # Original coordinates of pixels
    x_base = mx.nd.array(np.linspace(0, 1, height), ctx=img.context).repeat(batch_size*width). \
            reshape(batch_size, width, height).swapaxes(1, 2)
    y_base = mx.nd.array(np.linspace(0, 1, width), ctx=img.context).repeat(batch_size*height). \
            reshape(batch_size, height, width)
    # Apply shift in X direction
    # Disparity is passed in NCHW format with 1 channel
    x_shifts = disp[:, 0, :, :]
    flow_field = F.stack(x_base+x_shifts, y_base, axis=1)
    # BilinearSampler assumes the coordinates between -1 and 1
    output = F.BilinearSampler(img, 2*flow_field - 1)
    return output

def ssim(x, y):
    """ Unsupervised Monocular Depth Estimation with Left-Right Consistency
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2D(3, 1)(x)
    mu_y = nn.AvgPool2D(3, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2

    sigma_x = nn.AvgPool2D(3, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2D(3, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2D(3, 1)(x * y) - mu_x_mu_y

    ssim_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    ssim = ssim_n / ssim_d

    return F.clip((1 - ssim) / 2, 0, 1)

def disp_smoothness(disp, pyramid, nscales):
    disp_gradients_x = [gradient_x(d) for d in disp]
    disp_gradients_y = [gradient_y(d) for d in disp]

    image_gradients_x = [gradient_x(img) for img in pyramid]
    image_gradients_y = [gradient_y(img) for img in pyramid]

    weights_x = [F.exp(-F.mean(F.abs(g), 1,
                 keepdims=True)) for g in image_gradients_x]
    weights_y = [F.exp(-F.mean(F.abs(g), 1,
                 keepdims=True)) for g in image_gradients_y]

    smoothness_x = [disp_gradients_x[i] * weights_x[i]
                    for i in range(nscales)]
    smoothness_y = [disp_gradients_y[i] * weights_y[i]
                    for i in range(nscales)]

    return [F.abs(smoothness_x[i]) + F.abs(smoothness_y[i])
            for i in range(nscales)]
