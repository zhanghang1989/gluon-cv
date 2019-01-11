"""Kitti Dataloader"""
import os
import numpy as np
from PIL import Image

import mxnet as mx

from mxnet.gluon.data import dataset

class KittiDepth(dataset.Dataset):
    r"""
    """
    def __init__(self, root=os.path.expanduser('~/.mxnet/datasets/kitti'), split='train',
                 mode=None, transform=None, height=256, width=512, **kwargs):
        # TODO FIXME, seperate train/val sets
        self.left_paths, self.right_paths = find_all_pairs(root)
        assert len(self.left_paths) == len(self.right_paths)
        self.transform = transform
        self.height = height
        self.width = width
        self.mode = mode

    def __getitem__(self, idx):
        left_image = Image.open(self.left_paths[idx])
        right_image = Image.open(self.right_paths[idx])
        if self.mode == 'train':
            left_image, right_image = self._sync_transform(left_image, right_image)
        elif self.mode == 'val':
            left_image, right_image = self._val_sync_transform(left_image, right_image)
        else:
            assert self.mode == 'testval'
        if self.transform:
            left_image, right_image = self.transform(left_image, right_image)
        return left_image, right_image

    def __len__(self):
        return len(self.left_paths)

    def _val_sync_transform(self, left_image, right_image):
        # resize
        ow, oh = self.width, self.height
        left_image = left_image.resize((ow, oh), Image.BILINEAR)
        right_image = right_image.resize((ow, oh), Image.BILINEAR)
        left_image = self.image_transform(left_image)
        right_image =  self.image_transform(right_image)
        return left_image, right_image

    def _sync_transform(self, left_image, right_image):
        # resize
        ow, oh = self.width, self.height
        left_image = left_image.resize((ow, oh), Image.BILINEAR)
        right_image = right_image.resize((ow, oh), Image.BILINEAR)
        # random mirror
        if random.random() < 0.5:
            left_image = left_image.transpose(Image.FLIP_LEFT_RIGHT)
            right_image = right_image.transpose(Image.FLIP_LEFT_RIGHT)
        # final transform
        left_image = self.image_transform(left_image)
        right_image =  self.image_transform(right_image)
        return left_image, right_image

    def image_transform(self, left_image):
        return F.array(np.array(left_image), cpu(0))

def find_all_pairs(folder):
    asubfolders = get_direct_subfolders(folder)
    left_paths = []
    right_paths = []
    for a_folder in asubfolders:
        bsubfolders = get_direct_subfolders(a_folder)
        for b_folder in bsubfolders:
            csubfolders = [name for name in os.listdir(b_folder)
                           if os.path.isdir(os.path.join(b_folder, name))]
            if 'image_02' in csubfolders:
                left_path, right_path = find_lr_pairs(b_folder)
                left_paths += left_path
                right_paths += right_path
    assert len(left_paths) == len(right_paths)
    return left_paths, right_paths

def find_lr_pairs(folder):
    left_paths = []
    right_paths = []
    left_dir = os.path.join(folder, 'image_02/data/')
    right_dir = os.path.join(folder, 'image_03/data/')
    for filename in os.listdir(left_dir):
        left_path = os.path.join(left_dir, filename)
        right_path = os.path.join(right_dir, filename)
        if os.path.isfile(right_path):
            left_paths.append(left_path)
            right_paths.append(right_path)
        else:
            print('cannot find the right image:', right_path)
    #print('find ', len(left_paths), len(right_paths), ' images in ', folder)
    assert len(left_paths) == len(right_paths)
    return left_paths, right_paths

def get_direct_subfolders(cur_dir):
    return [os.path.join(cur_dir, name) for name in os.listdir(cur_dir)
            if os.path.isdir(os.path.join(cur_dir, name))]
