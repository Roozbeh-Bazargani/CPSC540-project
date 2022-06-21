import numpy as np
import staintools
import random
from torchvision import transforms


path_VPC = '/workspace/CPSC540/data/VPC-10X/'
path_colorado = '/workspace/CPSC540/data/Colorado-10X/'


#### Augmentation ####
REFERENCE_IMAGES = [path_VPC + 'Slide001/slide001_core100_blk035_stain001_scale001_class003.jpg',
                   path_VPC + 'Slide005/slide005_core097_blk060_stain001_scale001_class001.jpg',
                   path_VPC + 'Slide007/slide007_core006_blk034_stain001_scale001_class005.jpg',
                   path_colorado + 'A/S05-12166E/512/10/16384_55296.png',
                   path_colorado + 'T/S00-2937C/512/10/4096_24576.png',
                   path_colorado + 'F/S01-324B/512/10/18432_22528.png',
                   path_colorado + 'B/S99-13323B/512/10/26624_22528.png']
                   
COLOR_NORMALIZATION_METHODS = ['macenko', 'vahadane']


def color_normalizers(refs=REFERENCE_IMAGES, methods=COLOR_NORMALIZATION_METHODS):
    if (refs is None) or (methods is None):
        return None
    normalizers = []
    for method in methods:
        for ref in refs:
            target = staintools.read_image(ref)
            normalizer = staintools.StainNormalizer(method=method)
            normalizer.fit(target)
            normalizers.append(normalizer)
    normalizers.append('raw')
    return normalizers

def color_augmentation(img, normalizers):
    if normalizers is None:
        return img
    
    normalizer = random.choice(normalizers)
    if normalizer != 'raw':
        new_img = normalizer.transform(img)
        # print("augment")
        # print(np.mean(new_img - img))
        
        # import matplotlib.pyplot as plt
        # _, axes = plt.subplots(1, 2)
        # axes[0].imshow(new_img)
        # axes[1].imshow(img)
        # plt.show()
        img = new_img
    return img
    
# AUGMENTED_TRANSFORM = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
# #     transforms.GaussianBlur(20, sigma=(0,0.1)),
#     transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)]
# )

