"""Fully Convolutional Network with Strdie of 8"""
from __future__ import division
import numpy as np
from mxnet.gluon import nn
from mxnet.context import cpu
from mxnet.gluon.nn import HybridBlock, Block
from .segbase import SegBaseModel
# pylint: disable=unused-argument,abstract-method,missing-docstring,dangerous-default-value
# pylint: disable=redefined-outer-name, disable=arguments-differ

class MonoDepth(SegBaseModel):
    r"""MonoDepth Model

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
    pretrained_base : bool or str
        Refers to if the FCN backbone or the encoder is pretrained or not. If `True`,
        model weights of a model that was trained on ImageNet is loaded.

    Examples
    --------
    >>> TODO
    >>> print(model)
    """
    # pylint: disable=arguments-differ
    def __init__(self, backbone='resnet50', ctx=cpu(), pretrained_base=True,
                 height=256, width=512, dilated=False, **kwargs):
        super(MonoDepth, self).__init__(0, False, backbone, ctx=ctx,
                                  height=height, width=width, pretrained_base=pretrained_base,
                                  dilated=dilated, **kwargs)
        self.dilated = dilated
        with self.name_scope():
            # fake sequential block, used as a list
            self.upconvs = nn.HybridSequential()
            self.connects = nn.HybridSequential()
            self.disps = nn.HybridSequential()
            up_channels = [0, 128, 256, 512, 1024, 2048]
            inter_channels = [64, 64, 128, 256, 512]
            #h, w = height // 32, width // 32
            carry_channels = up_channels[-1]
            for i in range(4, -1, -1):
                #h, w = h * 2, w * 2
                depth_chs = 2 if i < 3 else 0
                self.upconvs.add(_UpConv(carry_channels, inter_channels[i], 3, **kwargs))
                self.connects.add(_ConvBnAct(up_channels[i]+inter_channels[i]+depth_chs, inter_channels[i], **kwargs))
                carry_channels = inter_channels[i]
                if i < 4:
                    self.disps.add(_DepthHead(inter_channels[i], 2, **kwargs))

            self.upconvs.initialize(ctx=ctx)
            self.connects.initialize(ctx=ctx)
            self.disps.initialize(ctx=ctx)

    def hybrid_forward(self, F, x):
        x_conv = self.conv1(x)
        x_bn = self.bn1(x_conv)
        x_relu = self.relu(x_bn)
        x_pool = self.maxpool(x_relu)
        c1 = self.layer1(x_pool)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        skips = [c3, c2, c1, x_conv, None] #, x_pool

        feat = c4
        disps = []
        #upfeats = []
        for i in range(len(skips)):
            skip = skips[i]
            upfeat = self.upconvs[i](feat)
            featcat = [upfeat, skip] if skip is not None else [upfeat]
            if len(disps) > 0:
                featcat.append(F.contrib.BilinearResize2D(disps[-1], scale_height=2, scale_width=2))
            cated = F.concat(*featcat, dim=1)
            feat = self.connects[i](cated)
            #upfeats.append(feat)
            if i > 0:
                disps.append(self.disps[i-1](feat))
        disps.reverse()
        #upfeats.reverse()
        #return tuple(upfeats)
        return tuple(disps)
        #return c1, c2, c3, c4

class _ConvBnAct(HybridBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=None, norm_layer=nn.BatchNorm, norm_kwargs={}):
        super(_ConvBnAct, self).__init__()
        padding = padding if padding else int(np.floor((kernel_size - 1) / 2))
        self.conv = nn.Conv2D(out_channels, kernel_size, stride, padding, in_channels=in_channels)
        self.bn = norm_layer(in_channels=out_channels, **norm_kwargs)

    def hybrid_forward(self, F, x):
        return F.Activation(self.bn(self.conv(x)), 'relu') 

class _UpConv(HybridBlock):
    def __init__(self, in_channels, out_channels, kernel_size,# height, width,
                 norm_layer=nn.BatchNorm, norm_kwargs={}):
        super(_UpConv, self).__init__()
        self.conv1 = _ConvBnAct(in_channels, out_channels, kernel_size, 1,
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x):
        x = F.contrib.BilinearResize2D(self.conv1(x), scale_height=2, scale_width=2)
        return x

class _DepthHead(HybridBlock):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm, norm_kwargs={}, **kwargs):
        super(_DepthHead, self).__init__()
        with self.name_scope():
            self.block = nn.HybridSequential()
            inter_channels = in_channels // 4
            with self.block.name_scope():
                self.block.add(nn.Conv2D(in_channels=in_channels, channels=inter_channels,
                                         kernel_size=3, padding=1, use_bias=False))
                self.block.add(norm_layer(in_channels=inter_channels, **norm_kwargs))
                self.block.add(nn.Activation('sigmoid'))
                #self.block.add(nn.Activation('relu'))
                self.block.add(nn.Dropout(0.1))
                self.block.add(nn.Conv2D(in_channels=inter_channels, channels=channels,
                                         kernel_size=1))

    def hybrid_forward(self, F, x):
        #print('mxnet x:', x)
        return self.block(x)

def get_mono_depth(dataset='kitti', backbone='resnet50', pretrained=False,
            root='~/.mxnet/models', ctx=cpu(0), pretrained_base=True, **kwargs):
    r"""FCN model from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf>`_

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    pretrained_base : bool or str, default True
        This will load pretrained backbone network, that was trained on ImageNet.

    Examples
    --------
    >>> model = get_mono_depth(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    # infer number of classes
    model = MonoDepth(backbone=backbone, ctx=ctx, pretrained_base=pretrained_base, **kwargs)
    return model

def get_mono_depth_resnet50_kitti(**kwargs):
    return get_mono_depth('kitti', 'resnet50', **kwargs)
