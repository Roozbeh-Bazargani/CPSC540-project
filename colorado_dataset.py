from __future__ import print_function, division
import os
from torch.utils.data import Dataset
from torchvision import transforms

import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import torch

COLORADO_CLASS_INDEXES_MAPPING = {'A': 0, 'AI': 0, 'B': 1, 'C': 2, 'F': 2, 'H': 0, 'I': 3, 'P': 2, 'S': 1, 'T': 2}

class CODataset(Dataset):

    def __init__(self, root_folder, slide_indexs=[], classification_type='full', transform=None):
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
            for label in self.labels:
                l = 0 if (label == 1) else 1
                self.ratio[l] += 1
        
        self.ratio = 1 / (self.ratio + 1) ## avoid divided by zero 
        self.ratio /= np.sum(self.ratio)
        
        assert len(self.labels) == len(self.image_names)

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

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
        image = io.imread(os.path.join(self.root_folder, self.image_names[idx]))
        image = self.transform(image)
        label = self.labels[idx]

        if self.classification_type == 'cancer':
            label = 0 if label == 0 else 1
        elif self.classification_type == 'grade':
            label = 0 if (label == 1) else 1
        
        return {'image': image.float(), 'label':label}  


if __name__ == '__main__':
    # test image reading

    import matplotlib.pyplot as plt

    root_folder = '../data/Colorado-10X'
    dataset = CODataset(root_folder, slide_indexs=[96], classification_type='grade')

    data = dataset.__getitem__(1)

    fig, axes = plt.subplots()

    image = data['image']
    print(image.shape)
    plt.imshow(image.permute(1, 2, 0))
    label = data['label']
    print(label)

    plt.show()