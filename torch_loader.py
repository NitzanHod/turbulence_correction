from __future__ import print_function, division

import numbers
import random
from glob import glob
from math import ceil

import torch
import torchvision.transforms.functional as F
from PIL import Image
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms

import config


class TurbulenceDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.gt_paths, self.in_paths = [], []

        # update the list by getting all the images path
        scenes_path_list = glob(root_dir + '/*')
        for current_scene_path in scenes_path_list:
            curr_gt_path = glob(current_scene_path + '/GT/*.jpg')[0]
            curr_in_paths = glob(current_scene_path + '/distorted/*.jpg')

            for _ in range(config.batch_size):
                self.gt_paths.append(curr_gt_path)
                self.in_paths.append(curr_in_paths)

        # caching images for reduction of disk operations
        self.gt_cache = dict()
        self.in_cache = dict()

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        gt_path, in_paths = self.gt_paths[idx], self.in_paths[idx]

        # cache lookups and updates
        if gt_path in self.gt_cache.keys():
            gt_image = self.gt_cache[gt_path]
        else:
            gt_image = io.imread(gt_path)
            self.gt_cache[gt_path] = gt_image

        random.shuffle(in_paths)  # shuffle sequence images

        distorted_tensor_list = []
        for in_path in in_paths[:config.n_frames]:  # load up to n_frames images
            if in_path in self.in_cache.keys():
                distorted_tensor_list.append(self.in_cache[gt_path])
            else:
                distorted_tensor_list.append(io.imread(in_path))
                self.gt_cache[gt_path] = gt_image

        sample = dict()

        if self.transform:
            sample['gt_image'] = gt_image
            sample['distorted_tensor'] = distorted_tensor_list
            sample = self.transform(sample)

        return sample


class PowerPad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """

        sample['gt_image'] = self.pad(sample['gt_image'])
        sample['distorted_tensor'] = [self.pad(x) for x in sample['distorted_tensor']]
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f'(fill={self.fill}, padding_mode={self.padding_mode})'

    @staticmethod
    def get_padding(image):
        h, w = image.shape[:2]
        div = 2 ** config.n_down_sampling  # image dimensions must be divisible by this size
        out_h, out_w = ceil(h / div) * div, ceil(w / div) * div  # minimal divisible sizes
        # centering the patch by padding left-right and bottom-top equally

        w_padding = (out_w - w) / 2
        h_padding = (out_h - h) / 2

        l_pad = w_padding if w_padding % 1 == 0 else w_padding + 0.5
        t_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
        r_pad = w_padding if w_padding % 1 == 0 else w_padding - 0.5
        b_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5

        padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))

        return padding

    def pad(self, img):
        padding = self.get_padding(img)
        img = Image.fromarray(img)
        return F.pad(img, padding, self.fill, self.padding_mode)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

        Args:
            p (float): probability of the image being flipped. Default value is 0.5
        """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """

        if random.random() < self.p:
            sample['gt_image'] = F.hflip(sample['gt_image'])
            sample['distorted_tensor'] = [F.hflip(x) for x in sample['distorted_tensor']]
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ToTensor(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

        Args:
            p (float): probability of the image being flipped. Default value is 0.5
        """

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """

        sample['gt_image'] = transforms.ToTensor()(sample['gt_image'])
        sample['distorted_tensor'] = torch.stack([transforms.ToTensor()(x) for x in sample['distorted_tensor']])
        return sample

    def __repr__(self):
        return self.__class__.__name__
