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


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform):
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

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        gt_path, in_paths = self.gt_paths[idx], self.in_paths[idx]
        gt_image = io.imread(gt_path)

        # random.shuffle(in_paths)  # shuffle random sequence TODO: uncomment

        distorted_tensor_list = []
        for i in range(config.n_frames):
            distorted_tensor_list.append(io.imread(in_paths[i]))

        sample = dict()
        sample['gt_image'] = self.transform(gt_image)
        sample['distorted_tensor'] = torch.stack([self.transform(x) for x in distorted_tensor_list])

        return sample


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


class NewPad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        padding = get_padding(img)
        img = Image.fromarray(img)
        return F.pad(img, padding, self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + f'(fill={self.fill}, padding_mode={self.padding_mode})'


