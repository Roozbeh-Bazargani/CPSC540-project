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


CLASS_INDEXES = [0,1,2,3]

class CODataset(Dataset):

    def __init__(self, root_folder, slide_indexs=[], augment=False, grayscale=True, crop=True, flip=True, rotate=True, noise=True, max_crop_pixel=20, max_rotate_angle=45, gaussian_noise_sigma=5):
        """
        Input:
        - root_folder: root folder to the dataset
        - slide_indexes: select which slices are included in this dataset
        - augment: True if use classical augmentation
        """
        super().__init__()
        self.root_folder = root_folder
        # load images
        self.image_names = []
        for idx in CLASS_INDEXES:
            img_names = os.listdir(os.path.join(self.root_folder, str(idx)))
            img_names = list(filter(lambda x:int(x[1:3]) in slide_indexs, img_names))
            image_files = [os.path.join(str(idx), img) for img in img_names]
            self.image_names += image_files

        self.image_names.sort()

        self.augment = augment
        self.grayscale = grayscale

        # flag for data augmentation
        self.flip = False
        self.rotate = False
        self.noise = False
        self.crop = False
        self.max_crop_pixel = None
        self.max_rotate_angle = None
        self.gaussian_noise_sigma = None
        self.use_mixup = False
        
        # select data augmentation option here
        if self.augment:
            self.flip = flip
            self.rotate = rotate
            self.noise = noise
            self.crop = crop
            self.max_crop_pixel = max_crop_pixel
            self.max_rotate_angle = max_rotate_angle
            self.gaussian_noise_sigma = gaussian_noise_sigma
        # can define other augmentation here


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

        # use opencv for data augmentation
        if self.augment:
            image = cv2.imread(os.path.join(self.root_folder, self.image_names[idx]))

            # random cropping
            if self.crop:
                h, w, _ = image.shape
                # generate random cropping
                left = np.random.randint(0, self.max_crop_pixel)
                right = np.random.randint(0, self.max_crop_pixel)
                top = np.random.randint(0, self.max_crop_pixel)
                bottom = np.random.randint(0, self.max_crop_pixel)

                image = image[top:h-bottom, left:w-right, :]

                image = cv2.resize(image, (w, h))
            # random left-right flipping
            if self.flip:
                if np.random.rand() > 0.5:
                    image = np.fliplr(image).copy()

            # random rotationt
            if self.rotate:
                h, w, _ = image.shape
                angle = np.random.randint(-self.max_rotate_angle, self.max_rotate_angle)
                center = (w / 2, h / 2)
                rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)

            # add random gaussian noise
            if self.noise:
                gauss = np.random.normal(0, self.gaussian_noise_sigma, image.shape)
                image = image + gauss
                image = np.where(image > 0, image, 0)
                image = np.where(image < 255, image, 255)
                image = image.astype(np.int)      


        else:
            image = cv2.imread(os.path.join(self.root_folder, self.image_names[idx]))
        

        # visualization helper
        # convert to tensor    
        # only read grayscale
        image = image / 255.
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image.copy())

        # get class
        label = int(self.image_names[idx][0])

        return {'image': image.float(), 'label':label}  


if __name__ == '__main__':
    # test opencv reading

    import matplotlib.pyplot as plt

    root_folder = 'Colorado-10X'
    dataset = CODataset(root_folder, slide_indexs=[96], augment='classic')

    data = dataset.__getitem__(-1)

    fig, axes = plt.subplots()

    image = data['image']
    print(image.shape)
    image = image.numpy()
    axes.imshow(image[0])
    label = data['label']
    print(label)

    plt.show()