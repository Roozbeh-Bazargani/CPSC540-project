from __future__ import print_function, division
import os
from torch.utils.data import Dataset
from torchvision import transforms

import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import torch

from color_normalization import *

COLORADO_CLASS_INDEXES_MAPPING = {'A': 0, 'AI': 0, 'B': 1, 'C': 2, 'F': 2, 'H': 0, 'I': 3, 'P': 2, 'S': 1, 'T': 2, 'M': 1, 'SC': 2}

class CODataset(Dataset):

    def __init__(self, root_folder, slide_indexs=[], classification_type='full', transform=None, stain_augmentation=False):
        """
        Input:
        - root_folder: root folder to the dataset
        - slide_indexes: select which slices are included in this dataset
        - classification_type: full, cancer, grade
        - transform: custom transformation for data augmentation
        """
        super().__init__()
        self.root_folder = root_folder
        # load images
        self.image_names = []
        self.label_idxes = os.listdir(root_folder)
        self.labels = []
        self.classification_type = classification_type
        if self.classification_type == 'full' or self.classification_type == 'cancer':
            # 4 class classification or benign vs cancer
            for idx in self.label_idxes:
                slide_folders = os.listdir(os.path.join(self.root_folder, str(idx)))
                slide_folders = list(filter(lambda x:int(x[1:3]) in slide_indexs, slide_folders))
                image_files = []
                for s in slide_folders:
                    for image_size in ['512']:
                        for core in ['10']:
                            
                            image_files = [os.path.join(idx, s, image_size, core, f) for f in os.listdir(os.path.join(root_folder, idx, s, image_size, core))]
                            self.image_names += image_files
                            self.labels += [COLORADO_CLASS_INDEXES_MAPPING[idx]] * len(image_files)

        elif self.classification_type == 'grade':
            # classify low grade vs high grade
            cancer_label_idxes = list(COLORADO_CLASS_INDEXES_MAPPING.keys())
            cancer_label_idxes = list(filter(lambda x: COLORADO_CLASS_INDEXES_MAPPING[x] != 0, cancer_label_idxes))
            print(cancer_label_idxes)
            for idx in cancer_label_idxes:
                slide_folders = os.listdir(os.path.join(self.root_folder, str(idx)))
                slide_folders = list(filter(lambda x:int(x[1:3]) in slide_indexs, slide_folders))
                image_files = []
                for s in slide_folders:
                    for image_size in ['512']:
                        for core in ['10']:
                            image_files = [os.path.join(idx, s, image_size, core, f) for f in os.listdir(os.path.join(root_folder, idx, s, image_size, core))]
                            self.image_names += image_files
                            self.labels += [COLORADO_CLASS_INDEXES_MAPPING[idx]] * len(image_files)
        elif self.classification_type == 'three_class':
            # classify benign vs low grade vs high grade
            for idx in self.label_idxes:
                slide_folders = os.listdir(os.path.join(self.root_folder, str(idx)))
                slide_folders = list(filter(lambda x:int(x[1:3]) in slide_indexs, slide_folders))
                image_files = []
                for s in slide_folders:
                    for image_size in ['512']:
                        for core in ['10']:
                            image_files = [os.path.join(idx, s, image_size, core, f) for f in os.listdir(os.path.join(root_folder, idx, s, image_size, core))]
                            self.image_names += image_files
                            self.labels += [COLORADO_CLASS_INDEXES_MAPPING[idx]] * len(image_files)

        # get ratio of the class
        if self.classification_type == 'full':
            self.ratio = np.zeros(4)
            for label in self.labels:
                self.ratio[label] += 1
        elif self.classification_type == 'cancer':
            self.ratio = np.zeros(2)
            for label in self.labels:
                l = 0 if label == 0 else 1
                self.ratio[l] += 1
        elif self.classification_type == 'grade':
            self.ratio = np.zeros(2)
            for label in self.labels:
                l = 0 if (label == 1) else 1
                self.ratio[l] += 1
        elif self.classification_type == 'three_class':
            self.ratio = np.zeros(3)
            for label in self.labels:
                l = 2 if (label == 3) else label
                self.ratio[l] += 1

        self.ratio = 1 / (self.ratio + 1) ## avoid divided by zero 
        self.ratio /= np.sum(self.ratio)
        
        # assert len(self.labels) == len(self.image_names)
        # for i in self.image_names:
        #     print(i)

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

        # stain augmentation
        self.stain_augmentation = stain_augmentation
        if self.stain_augmentation:
            print("use stain augmentation, generating normalizers")
            self.color_normalizer = color_normalizers()
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
        label = self.labels[idx]

        if self.classification_type == 'cancer':
            label = 0 if label == 0 else 1
        elif self.classification_type == 'grade':
            label = 0 if (label == 1) else 1
        elif self.classification_type == 'three_class':
            label = 2 if (label == 3) else label
        

        return {'image': image.float(), 'label':label}  


if __name__ == '__main__':
    # test image reading

    colorado = [0,1,2,3,4,5,6,94,96,97,98,99]

    import matplotlib.pyplot as plt

    root_folder = '../data/Colorado-10X'
    for i in colorado:
        dataset = CODataset(root_folder, slide_indexs=[i], classification_type='three_class')

        print(dataset.__len__())

    # dataset = CODataset(root_folder, slide_indexs=[96, 98], classification_type='grade')

    # print(dataset.__len__())

    # dataset = CODataset(root_folder, slide_indexs= [1, 97, 99],  classification_type='grade')

    # print(dataset.__len__())