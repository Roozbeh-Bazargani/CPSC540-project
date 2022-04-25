import os
from pkgutil import get_data
import sys
import random

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils import *
# from models import Model
from models import DANN
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torchvision

NUMBER_OF_CLASSES = {'full': 4, 'cancer': 2, 'grade': 2}


class NN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # self.num_classes = num_classes
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Sequential(nn.Linear(in_features=512, out_features=num_classes, bias=True, ))#,
                        #  nn.ReLU(),
                        #  nn.Linear(in_features=1000, out_features=num_classes, bias=True))
        # print(self.model)

    def forward(self, dictionary):
      return {'label': self.model(dictionary['img'])}

    
    def get_feature_extractor(self):
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
    
    def get_feature(self, image):
        return self.feature_extractor(image)

    def prediction(self, dictionary):
        return {'label': torch.argmax(self.forward(dictionary)['label'], dim=1)}


class SimpleNNTest(object):
    def __init__(self, cfg):
        """
        Initialize the configuration
        """
        self.cfg = cfg
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # build model
        self.model = NN(num_classes=self.cfg['num_classes'])
        checkpoint = torch.load(self.cfg['checkpoints'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.get_feature_extractor()
        self.model.to(self.device)

        # multiple GPU
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)


    # def forward(self, data, alpha):
    #     class_pred, domain_pred = self.model.forward(data, alpha)
    #     class_prob   = torch.softmax(class_pred, dim=1)
    #     domain_prob  = torch.softmax(domain_pred, dim=1)
    #     return class_pred, class_prob, domain_pred, domain_prob

    
    def get_testset_feature(self, dataset):
        """test the model by printing all the metrics for each saved model
        """
        print(f"\nStart SimpleTrain testing on {dataset} dataset.")
        if dataset == 'source':
            test_dataloader = get_source_dataloader(self.cfg['source_dataset'], slides=self.cfg['source_test_idx'], batch_size=self.cfg['batch_size'], classification_type=self.cfg['classification_type'], augment=False, num_workers=self.cfg['num_workers'])
        elif dataset == 'target':
            test_dataloader = get_target_dataloader(self.cfg['target_dataset'], slides=self.cfg['target_test_idx'], batch_size=self.cfg['batch_size'], classification_type=self.cfg['classification_type'], augment=False, num_workers=self.cfg['num_workers'])
        
        # get the last layer feature
        self.model.eval()
        save_feature = []
        save_label = []
        txt = 'Test : '
        with torch.no_grad():
            prefix = txt
            for batch_data in tqdm(test_dataloader, desc=prefix,
                    dynamic_ncols=True, leave=True, position=0):
                # load data
                data = batch_data['image']
                data  = data.cuda() if torch.cuda.is_available() else data
                feature_C = self.model.get_feature(data)
                save_feature.append(feature_C.to('cpu').detach().numpy())
                save_label.append(batch_data['label'].detach().numpy())

                del data
                del feature_C
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # concatenate
        save_feature = np.concatenate(save_feature)
        save_feature = np.squeeze(save_feature, 2)
        save_feature = np.squeeze(save_feature, 2)
        save_label = np.concatenate(save_label)
        print(save_feature.shape)
        return save_feature, save_label

    def plot_tSNE(self):
        source_feature, source_label = self.get_testset_feature('source')
        target_feature, target_label = self.get_testset_feature('target')

        # do tsne
        tsne = TSNE(n_components=2, perplexity=30)
        tsne_results = tsne.fit_transform(np.vstack((source_feature, target_feature)))
        print("t-SNE done!")

        # plot tsne
        tsne_source = tsne_results[:source_feature.shape[0], :]
        tsne_target = tsne_results[source_feature.shape[0]:, :]
        plt.figure()
        color = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        for i in range(self.cfg['num_classes']):
            plt.scatter(tsne_source[source_label == i, 0], tsne_source[source_label == i, 1], c=color[i], marker='+')
            plt.scatter(tsne_target[target_label == i, 0], tsne_target[target_label == i, 1], c=color[i], marker='x')
        
        model_name = self.cfg['checkpoints']
        model_name = os.path.split(model_name)
        model_name = model_name[1]
        plt.savefig(os.path.join('/workspace/CPSC540/resnet18/test_results/images', model_name + '.png'))

    def run(self):
        self.test('source')
        self.test('target')


def get_colorado_slice(path):
    """
    get the slide indexes for colorado dataset
    """
    slices = []
    for folders in os.listdir(path):
        imgs = os.listdir(os.path.join(path, folders))
        for img in imgs:
            if int(img[1:3]) not in slices:
                slices.append(int(img[1:3]))
    return slices


if __name__ == '__main__':
    # sample config file for the training class
    cfg = {
    'model_name': 'resnet18', # resnet34 or resnet18
    'classification_type': 'full', # classify all, or benign vs cancer, or low grade vs high grade. String: 'full', 'cancer' or 'grade'
    'use_weighted_loss': True,
    'optimizer': 'Adam',  # name of optimizer
    'lr': 0.00001,  # learning rate
    'momentum': 0, # momentum for SGD
    'wd': 0.000,  # weight decay
    'use_schedular': True,  # bool to select whether to use scheduler
    'use_earlystopping': True,
    'earlystopping_epoch': 10, # early stop the training if no improvement on validation for this number of epochs
    'epochs': 500,  # number of training epochs
    'source_dataset': '/workspace/CPSC540/data/VPC-10X',   # path to vancouver dataset
    'source_train_idx': [2,5,6,7],   # indexes of the slides used for training (van dataset)
    'source_val_idx': [3],   # indexes of the slides used for validation (van dataset)
    'source_test_idx': [1],   # indexes of the slides used for testing (van dataset)
    'batch_size': 16,  # batch size
    'augment': False,  # whether use classical cv augmentation
    'target_dataset': '/workspace/CPSC540/data/Colorado-10X',  # path to Colorado dataset
    'target_train_idx': [0, 2, 3, 4, 5, 6, 94],  # indexes of the slides used for training (CO dataset)
    'target_val_idx': [96, 98],
    'target_test_idx': [1, 97, 99],  # indexes of the slides used for testing (CO dataset)
    'num_workers': 1, # number of workers

    'val_criteria': 'val_loss', 
    'checkpoints': '/workspace/CPSC540/resnet18/baseline_model/baseline_model_original_balanced',  # dir to save the best model, training configurations and results

    }

    cfg['num_classes'] = NUMBER_OF_CLASSES[cfg['classification_type']]

    # init trainer
    trainer = SimpleNNTest(cfg)
    # run trainer
    trainer.plot_tSNE()