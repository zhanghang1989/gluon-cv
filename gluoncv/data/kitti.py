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
                 mode=None, transform=None, **kwargs):
        self.left_paths, self.right_paths = find_all_pairs(root)
        self.transform = transform
        self.mode = mode

    def __getitem__(self, idx):
        left_image = Image.open(self.left_paths[idx])
        right_image = Image.open(self.right_paths[idx])
        if self.transform:
            left_image, right_image = self.transform(left_image, right_image)
        return left_image, right_image

    def __len__(self):
        return len(self.left_paths)

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
    return left_paths, right_path

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
    print('find ', len(left_paths), len(right_paths), ' images in ', folder)
    assert len(left_paths) == len(right_paths)
    return left_paths, right_paths

def get_direct_subfolders(cur_dir):
    return [os.path.join(cur_dir, name) for name in os.listdir(cur_dir)
            if os.path.isdir(os.path.join(cur_dir, name))]
