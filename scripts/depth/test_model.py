import mxnet as mx
import gluoncv
from gluoncv.model_zoo import get_model

from train import parse_args

if __name__ == "__main__":
    args = parse_args()
    print('Creating the model:')
    ctx=mx.gpu(0)
    model = get_model(args.model, pretrained=args.pretrained, ctx=ctx)
    print(model)
    x = mx.nd.ones(shape=(4, 3, 256, 512), ctx=ctx)
    out = model(x)
    for y in out:
        print(y.shape)
