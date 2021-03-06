from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from skimage import io

from color_normalization import *

van_prefix = lambda x:'Slide' + str(x).zfill(3)

VANCOUVER_CLASS_INDEXES_MAPPING = {'1': 0, '3': 1, '4': 2, '5': 3}

class VanDataset(Dataset):

    def __init__(self, root_folder, slide_indexs=[], classification_type='full', transform=None, stain_augmentation=False):
        """
        Input:
        - root_folder: root folder to the dataset
        - slide_indexes: select which slices are included in this dataset
        - classification_type: full, cancer, grade
        - augment: True if use classical augmentation
        """
        super().__init__()
        self.root_folder = root_folder
        # load images
        self.image_names = []
        self.classification_type = classification_type
        self.ratio = []

        if self.classification_type == 'full' or self.classification_type == 'cancer' or self.classification_type == 'three_class':
            # 4 class classification, or classify benign vs cancer
            for idx in slide_indexs:
                slide_name = van_prefix(idx)
                img_names = os.listdir(os.path.join(self.root_folder, slide_name))
                img_names = list(filter(lambda x:x[-5] != ')', img_names))
                
                img_names = list(filter(lambda x:int(x[-5]) != 0 and int(x[-5]) != 6, img_names))
                image_files = [os.path.join(slide_name, img) for img in img_names]
                self.image_names += image_files

        elif self.classification_type == 'grade':
            # classify low grade vs high grade
            for idx in slide_indexs:
                slide_name = van_prefix(idx)
                img_names = os.listdir(os.path.join(self.root_folder, slide_name))
                img_names = list(filter(lambda x:x[-5] != ')', img_names))
                
                img_names = list(filter(lambda x:int(x[-5]) != 0 and int(x[-5]) != 6 and int(x[-5]) != 1, img_names))
                image_files = [os.path.join(slide_name, img) for img in img_names]
                self.image_names += image_files

        # get ratio of the class
        if self.classification_type == 'full':
            self.ratio = np.zeros(4)
            for img_file in self.image_names:
                label = int(VANCOUVER_CLASS_INDEXES_MAPPING[img_file[-5]])
                self.ratio[label] += 1
        elif self.classification_type == 'cancer':
            self.ratio = np.zeros(2)
            for idx, _ in enumerate(self.image_names):
                label = int(VANCOUVER_CLASS_INDEXES_MAPPING[self.image_names[idx][-5]])
                label = 0 if label == 0 else 1
                self.ratio[label] += 1
        elif self.classification_type == 'grade':
            self.ratio = np.zeros(2)
            for idx, _ in enumerate(self.image_names):
                label = int(VANCOUVER_CLASS_INDEXES_MAPPING[self.image_names[idx][-5]])
                label = 0 if (label == 1) else 1
                self.ratio[label] += 1
        elif self.classification_type == 'three_class':
            self.ratio = np.zeros(3)
            for idx, _ in enumerate(self.image_names):
                label = int(VANCOUVER_CLASS_INDEXES_MAPPING[self.image_names[idx][-5]])
                label = 2 if (label == 3) else label
                self.ratio[label] += 1
        self.ratio = 1 / (self.ratio + 1) ## avoid divided by zero 
        self.ratio /= np.sum(self.ratio)

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

        # stain augmentation
        self.stain_augmentation = stain_augmentation
        if self.stain_augmentation:
            print("use stain augmentation, generating normalizers")
            self.color_normalizer = color_normalizers()
            # print(self.color_normalizer)
        else:
            self.color_normalizer = None

    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, idx):
        """
        load image and label (if label exists)
        Input:
        - idx: index of the images
        Output:
        - image and label
        """
        if self.stain_augmentation:
            image = staintools.read_image(os.path.join(self.root_folder, self.image_names[idx]))
            image = color_augmentation(image, self.color_normalizer)
            image = image / 255
        else:
            image = io.imread(os.path.join(self.root_folder, self.image_names[idx]))
        image = self.transform(image)

        # get class
        label = int(VANCOUVER_CLASS_INDEXES_MAPPING[self.image_names[idx][-5]])

        if self.classification_type == 'cancer':
            label = 0 if label == 0 else 1
        elif self.classification_type == 'grade':
            label = 0 if (label == 1) else 1
        elif self.classification_type == 'three_class':
            label = 2 if (label == 3) else label

        return {'image': image.float(), 'label':label}  


if __name__ == '__main__':
    # test opencv reading

    import matplotlib.pyplot as plt

    root_folder = '../data/VPC-10X'
    dataset = VanDataset(root_folder, slide_indexs=[1,3,2,5,6,7], classification_type='three_class')

    print(dataset.__len__())

    # dataset = VanDataset(root_folder, slide_indexs=[3], classification_type='grade')

    # print(dataset.__len__())

    # dataset = VanDataset(root_folder, slide_indexs=[1], classification_type='grade')

    # print(dataset.__len__())