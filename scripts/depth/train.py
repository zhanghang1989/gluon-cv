import os
import argparse
from tqdm import tqdm
import mxnet as mx
from mxnet.gluon.data.vision import transforms
from gluoncv.utils.depth import MonodepthLoss
from gluoncv.utils import LRScheduler
from gluoncv.data import KittiDepth
from gluoncv.utils.parallel import *
from gluoncv.model_zoo import get_model

def parse_args():
    parser = argparse.ArgumentParser(description='GluonCV Depth')

    # model and criterion
    parser.add_argument('--model', type=str, default='mono_depth_resnet50_kitti',
                        help='model name (default: mono_depth_resnet50_kitti)')
    parser.add_argument('--pretrained', action='store_true', default=
                        False, help='pretrained model')
    parser.add_argument('--ssim-weight', type=float, default=0.85,
                        help='SSIM loss weight')
    parser.add_argument('--smooth-weight', type=float, default=0.1,
                        help='smooth loss weight')
    parser.add_argument('--lr-weight', type=float, default=1,
                        help='left right consistancy loss weight')
    # synchronized Batch Normalization
    parser.add_argument('--syncbn', action='store_true', default= False,
                        help='using Synchronized Cross-GPU BatchNorm')

    # training
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of total epochs to run')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--eval', action='store_true', default= False,
                        help='evaluation only')
    # data
    parser.add_argument('--height', type=int, default=256,
                        help='image size')
    parser.add_argument('--width', type=int, default=512,
                        help='image size')
    parser.add_argument('--workers', type=int, default=16,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    parser.add_argument('--train-split', type=str, default='train',
                        help='dataset train split (default: train)')
    #parser.add_argument('--dataset', type=str, default='kitti',
    #                    help='dataset name (default: kitti)')
    # cuda and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--ngpus', type=int,
                        default=len(mx.test_utils.list_gpus()),
                        help='number of GPUs (default: 4)')
    parser.add_argument('--kvstore', type=str, default='device',
                        help='kvstore to use for trainer/module.')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='default',
                        help='set the checkpoint name')

    args = parser.parse_args()
    if args.no_cuda:
        print('Using CPU')
        args.kvstore = 'local'
        args.ctx = [mx.cpu(0)]
    else:
        print('Number of GPUs:', args.ngpus)
        args.ctx = [mx.gpu(i) for i in range(args.ngpus)]
    # Synchronized BatchNorm
    args.norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if args.syncbn \
        else mx.gluon.nn.BatchNorm
    args.norm_kwargs = {'num_devices': args.ngpus} if args.syncbn else {}
    return args

class Trainer(object):
    def __init__(self, args):
        self.args = args
        # prepare the model
        model = get_model(args.model, pretrained=args.pretrained, norm_layer=args.norm_layer,
                          norm_kwargs=args.norm_kwargs)
        if args.resume is not None:
            if os.path.isfile(args.resume):
                model.load_parameters(args.resume, ctx=args.ctx)
            else:
                raise RuntimeError("=> no checkpoint found at '{}'" \
                    .format(args.resume))
        print(model)
        model.cast(args.dtype)
        self.net = DataParallelModel(model, args.ctx, args.syncbn)

        # prepare the dataset
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        data_kwargs = {'transform': input_transform, 'height': args.height,
                       'width': args.width}
        trainset = KittiDepth(split=args.train_split, mode='train', **data_kwargs)
        #valset = KittiDepth(split='val', mode='val', **data_kwargs)
        print('len(trainset):', len(trainset))
        self.train_data = mx.gluon.data.DataLoader(
            trainset, args.batch_size, shuffle=True, last_batch='rollover',
            num_workers=args.workers)
        #self.val_data = mx.gluon.data.DataLoader(valset, args.test_batch_size,
        #    last_batch='rollover', num_workers=args.workers)

        # criterion
        criterion = MonodepthLoss(4, args.ssim_weight, args.smooth_weight, args.lr_weight)
        self.criterion = DataParallelCriterion(criterion, args.ctx, args.syncbn)

        # optimizer and lr scheduling
        self.lr_scheduler = LRScheduler(mode='poly', baselr=args.lr,
                                        niters=len(self.train_data), 
                                        nepochs=args.epochs)
        kv = mx.kv.create(args.kvstore)
        optimizer_params = {'lr_scheduler': self.lr_scheduler}
        if args.dtype == 'float16':
            optimizer_params['multi_precision'] = True

        self.optimizer = mx.gluon.Trainer(self.net.module.collect_params(), 'adam',
                                       optimizer_params, kvstore = kv)

    def train(self, epoch):
        tbar = tqdm(self.train_data)
        train_loss = 0.0
        alpha = 0.2
        for i, (left, right) in enumerate(tbar):
            self.lr_scheduler.update(i, epoch)
            with mx.autograd.record(True):
                outputs = self.net(left.astype(args.dtype, copy=False))
                losses = self.criterion(outputs, left, right)
                mx.nd.waitall()
                mx.autograd.backward(losses)
            self.optimizer.step(self.args.batch_size)
            for loss in losses:
                train_loss += loss.asnumpy()[0] / len(losses)
            tbar.set_description('Epoch %d, training loss %.3f'%\
                (epoch, train_loss/(i+1)))
            mx.nd.waitall()

        # save every epoch
        save_checkpoint(self.net.module, self.args)

def save_checkpoint(net, args):
    """Save Checkpoint"""
    directory = "runs/%s/%s/" % (args.model, args.checkname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename='checkpoint.params'
    filename = directory + filename
    net.save_parameters(filename)


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    if args.eval:
        print('Evaluating model: ', args.resume)
        #trainer.validation(args.start_epoch)
    else:
        print('Starting Epoch:', args.start_epoch)
        print('Total Epoches:', args.epochs)
        for epoch in range(args.start_epoch, args.epochs):
            trainer.train(epoch)
            #if not trainer.args.no_val:
            #    trainer.validation(epoch)
