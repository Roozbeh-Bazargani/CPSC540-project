from van_dataset import *
from colorado_dataset import *
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
import json
import pickle
import copy

## TODO: can change the data augmentation here
AUGMENTED_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomResizedCrop(512),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()]
)



def get_dataloader(dataset_path, slides, batch_size, classification_type, augment=False, shuffle=False, num_workers=1, stain_augment=False):
    
    if dataset_path.find('VPC') != -1:
        if augment:
            dataset = VanDataset(root_folder=dataset_path, slide_indexs=slides, classification_type=classification_type, transform=AUGMENTED_TRANSFORM, stain_augmentation=stain_augment)
        else:
            dataset = VanDataset(root_folder=dataset_path, slide_indexs=slides, classification_type=classification_type, stain_augmentation=stain_augment)
        # print("length of dataset %d\n" % dataset.__len__())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)
    elif dataset_path.find('Col') != -1:
        if augment:
            dataset = CODataset(root_folder=dataset_path, slide_indexs=slides, classification_type=classification_type, transform=AUGMENTED_TRANSFORM, stain_augmentation=stain_augment)
        else:
            dataset = CODataset(root_folder=dataset_path, slide_indexs=slides, classification_type=classification_type, stain_augmentation=stain_augment)
        # print("length of dataset %d\n" % dataset.__len__())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False) 
    return dataloader


def get_source_dataloader(dataset_path, slides, batch_size, classification_type, augment=False, shuffle=False, num_workers=1, stain_augment=False):
    if augment:
        dataset = VanDataset(root_folder=dataset_path, slide_indexs=slides, classification_type=classification_type, transform=AUGMENTED_TRANSFORM, stain_augmentation=stain_augment)
    else:
        dataset = VanDataset(root_folder=dataset_path, slide_indexs=slides, classification_type=classification_type, stain_augmentation=stain_augment)
    # print("length of dataset %d\n" % dataset.__len__())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False) 
    return dataloader

def get_target_dataloader(dataset_path, slides, batch_size, classification_type, augment=False, shuffle=False, num_workers=1, stain_augment=False):
    if augment:
        dataset = CODataset(root_folder=dataset_path, slide_indexs=slides, classification_type=classification_type, transform=AUGMENTED_TRANSFORM, stain_augmentation=stain_augment)
    else:
        dataset = CODataset(root_folder=dataset_path, slide_indexs=slides, classification_type=classification_type, stain_augmentation=stain_augment)
    # print("length of dataset %d\n" % dataset.__len__())

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False) 
    return dataloader

def save_json(dict, path):
    save_dict = copy.deepcopy(dict)
    for k in save_dict.keys():
        if isinstance(save_dict[k], np.ndarray):
            save_dict[k] = save_dict[k].flatten().tolist()
    with open(path, 'w') as handle:
        json.dump(save_dict, handle, indent=4)


def save_dict(dict, path):
    with open(path, 'wb') as handle:
        pickle.dump(dict, handle)


def patch_metrics(patch_info, num_classes):
    overall_auc = 0

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    gt_label = patch_info['gt_label']
    label_one_hot = np.zeros((len(gt_label), num_classes))
    label_one_hot[np.arange(len(gt_label)), gt_label] = 1
    y_score = patch_info['probability']
    for i in range(num_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(label_one_hot[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    overall_auc = np.mean([v for v in roc_auc.values()])
    overall_acc = metrics.accuracy_score(patch_info['gt_label'], patch_info['prediction'])
    overall_f1 = metrics.f1_score(patch_info['gt_label'], patch_info['prediction'], average='macro')
    conf_mat = metrics.confusion_matrix(patch_info['gt_label'], patch_info['prediction'], labels=list(range(num_classes)))
    acc_per_subtype = conf_mat.diagonal() / conf_mat.sum(axis=1)
    patch_perf = {'overall_auc': overall_auc, 
    'overall_acc': overall_acc, 
    'overall_f1': overall_f1,
    'conf_mat': conf_mat,
    'acc_per_subtype': acc_per_subtype}
    # TODO: slides patch
    return patch_perf


if __name__ == '__main__':
    dataloader = get_source_dataloader('../data/VPC-10X', [1, 2, 3], 8, 'full')
    print(dataloader.dataset.ratio)

    dataloader = get_target_dataloader('../data/Colorado-10X', [1], 8, 'full')
    print(dataloader.dataset.ratio)