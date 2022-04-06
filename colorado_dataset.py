from __future__ import print_function, division
import os
from torch.utils.data import Dataset
from torchvision import transforms

import matplotlib.pyplot as plt
from skimage import io


COLORADO_CLASS_INDEXES_MAPPING = {'A': 0, 'AI': 0, 'B': 1, 'C': 2, 'F': 2, 'H': 0, 'I': 3, 'P': 2, 'S': 1, 'T': 2}

class CODataset(Dataset):

    def __init__(self, root_folder, slide_indexs=[], transform=None):
        """
        Input:
        - root_folder: root folder to the dataset
        - slide_indexes: select which slices are included in this dataset
        - transform: custom transformation for data augmentation
        """
        super().__init__()
        self.root_folder = root_folder
        # load images
        self.image_names = []
        self.label_idxes = os.listdir(root_folder)
        self.labels = []
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
        print(self.image_names[idx])
        image = io.imread(os.path.join(self.root_folder, self.image_names[idx]))
        image = self.transform(image)
        label = self.labels[idx]

        return {'image': image.float(), 'label':label}  


if __name__ == '__main__':
    # test image reading

    import matplotlib.pyplot as plt

    root_folder = '../data/Colorado-10X'
    dataset = CODataset(root_folder, slide_indexs=[96])

    data = dataset.__getitem__(1)

    fig, axes = plt.subplots()

    image = data['image']
    print(image.shape)
    plt.imshow(image.permute(1, 2, 0))
    label = data['label']
    print(label)

    plt.show()