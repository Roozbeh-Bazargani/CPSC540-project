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

van_prefix = lambda x:'Slide' + str(x).zfill(3)

VANCOUVER_CLASS_INDEXES_MAPPING = {'1': 0, '3': 1, '4': 2, '5': 3}

class VanDataset(Dataset):

    def __init__(self, root_folder, slide_indexs=[], classification_type='full', transform=None):
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

        if self.classification_type == 'full' or self.classification_type == 'cancer':
            # 4 class classification, or classify benign vs cancer
            for idx in slide_indexs:
                slide_name = van_prefix(idx)
                img_names = os.listdir(os.path.join(self.root_folder, slide_name))
                
                img_names = list(filter(lambda x:int(x[-5]) != 0 and int(x[-5]) != 6, img_names))
                image_files = [os.path.join(slide_name, img) for img in img_names]
                self.image_names += image_files

        elif self.classification_type == 'grade':
            # classify low grade vs high grade
            for idx in slide_indexs:
                slide_name = van_prefix(idx)
                img_names = os.listdir(os.path.join(self.root_folder, slide_name))
                
                img_names = list(filter(lambda x:int(x[-5]) != 0 and int(x[-5]) != 6 and int(x[-5]) != 1, img_names))
                image_files = [os.path.join(slide_name, img) for img in img_names]
                self.image_names += image_files

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

        # get class
        label = int(VANCOUVER_CLASS_INDEXES_MAPPING[self.image_names[idx][-5]])

        if self.classification_type == 'cancer':
            label = 0 if label == 0 else 1
        elif self.classification_type == 'grade':
            label = 0 if (label == 1) else 1

        return {'image': image.float(), 'label':label}  


if __name__ == '__main__':
    # test opencv reading

    import matplotlib.pyplot as plt

    root_folder = '../data/VPC-10X'
    dataset = VanDataset(root_folder, slide_indexs=[1,2,3], classification_type='grade')

    data = dataset.__getitem__(10)

    fig, axes = plt.subplots()

    image = data['image']
    print(image.shape)
    plt.imshow(image.permute(1, 2, 0))
    label = data['label']
    print(label)

    plt.show()