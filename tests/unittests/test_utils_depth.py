import mxnet as mx
from gluoncv.utils.depth import *
from gluoncv.model_zoo import get_model

from common import try_gpu, with_cpu

@try_gpu(0)
def test_ssim():
    ctx = mx.context.current_context()
    x = mx.nd.random.uniform(shape=(1, 3, 224, 224), ctx=ctx)
    y = mx.nd.random.uniform(shape=(1, 3, 224, 224), ctx=ctx)
    res = ssim(x, y)
    print(res.shape)

@try_gpu(0)
def test_apply_disparity():
    ctx = mx.context.current_context()
    x = mx.nd.ones(shape=(1, 1, 256, 512), ctx=ctx)
    disp = mx.nd.ones((1, 1, 256, 512), ctx=ctx) / 5
    y = apply_disparity(x, disp)
    print(x, y)

@try_gpu(0)
def test_mono_depth_loss():
    ctx = mx.context.current_context()
    # network and input, outputs
    net = get_model('mono_depth_resnet50_kitti', ctx=ctx)
    left = mx.nd.random.uniform(shape=(1, 3, 256, 512), ctx=ctx)
    right = mx.nd.random.uniform(shape=(1, 3, 256, 512), ctx=ctx)
    disps = net(left)
    # loss and criterion
    criterion = MonodepthLoss(3)
    loss = criterion(*disps, left, right)
    print(loss)

if __name__ == '__main__':
    import nose
    nose.runmodule()
