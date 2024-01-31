import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity




def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class guazai(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    cmap = voc_cmap()
    def __init__(self,
                 root,
                 image_set='train',
                 download=False,
                 transform=None):

        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.image_set = image_set

        if image_set=='train':

            image_dir = os.path.join(self.root, 'train','images')
            label_dir = os.path.join(self.root, 'train','labels')
        if image_set=='val':
            image_dir = os.path.join(self.root, 'valid','images')
            label_dir = os.path.join(self.root, 'valid','labels')
        

        names = os.listdir(image_dir)
        
        self.images = [os.path.join(image_dir,name[:-3]+'jpg') for name in names]
        self.masks = [os.path.join(label_dir,name[:-3]+'png') for name in names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        # print(self.images[index])
        img = Image.open(self.images[index]).convert('RGB')
        print(self.images[index])
        print("img.size",img.size)
        target = Image.open(self.masks[index])
        print("target.size",img.size)
        if self.transform is not None:
            
            img, target = self.transform(img, target)
            
        return img, target


    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]
