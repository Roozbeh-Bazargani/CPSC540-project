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

NUMBER_OF_CLASSES = {'full': 4, 'cancer': 2, 'grade': 2}


class SimpleDANNTest(object):
    def __init__(self, cfg):
        """
        Initialize the configuration
        """
        self.cfg = cfg
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if not os.path.exists(self.cfg["checkpoints"]):
            os.mkdir(self.cfg["checkpoints"])

        # build model
        self.model = DANN(model_name=cfg['model_name'], num_classes=cfg['num_classes'], feature_block=cfg['feature_block'])
        self.model.to(self.device)

        # multiple GPU
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)


    def forward(self, data, alpha):
        class_pred, domain_pred = self.model.forward(data, alpha)
        class_prob   = torch.softmax(class_pred, dim=1)
        domain_prob  = torch.softmax(domain_pred, dim=1)
        return class_pred, class_prob, domain_pred, domain_prob


    def test(self, dataset='source'):
        """test the model by printing all the metrics for each saved model
        """
        print(f"\nStart SimpleTrain testing on {dataset} dataset.")
        if dataset == 'source':
            test_dataloader = get_source_dataloader(self.cfg['source_dataset'], slides=self.cfg['source_test_idx'], batch_size=self.cfg['batch_size'], classification_type=self.cfg['classification_type'], augment=False, num_workers=self.cfg['num_workers'])
        elif dataset == 'target':
            test_dataloader = get_target_dataloader(self.cfg['target_dataset'], slides=self.cfg['target_test_idx'], batch_size=self.cfg['batch_size'], classification_type=self.cfg['classification_type'], augment=False, num_workers=self.cfg['num_workers'])
        
        # load best model
        path = os.path.join(self.cfg["checkpoints"], f"model_{self.cfg['val_criteria']}.pth")
        state = torch.load(path, map_location=self.device)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(state["model"], strict=True)
        else:
            self.model.load_state_dict(state["model"], strict=True)


        # run test on dataset
        self.model.eval()
        save_labels = []
        save_predict = []
        txt = 'Test : '
        with torch.no_grad():
            prefix = txt
            for batch_data in tqdm(test_dataloader, desc=prefix,
                    dynamic_ncols=True, leave=True, position=0):
                # load data
                data = batch_data['image']
                label = batch_data['label']
                data  = data.cuda() if torch.cuda.is_available() else data
                label = label.cuda() if torch.cuda.is_available() else label
                predicted, prob, _, _ = self.forward(data, 0)
                save_labels.append(label.to('cpu').detach().numpy())

                prediction = prob.to('cpu').detach().numpy()
                # prediction = np.argmax(prediction, axis=1)
                save_predict.append(prediction)
                # clean cache
                del data
                del label
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


        # concatenate
        save_labels = np.concatenate(save_labels)
        save_predict = np.vstack(save_predict)
        print(save_labels.shape)
        print(save_predict.shape)

        np.save(os.path.join('/workspace/CPSC540/resnet18/test_results', dataset + '_label_' + 'DANN_block' + str(self.cfg['feature_block']) + '_balanced_without_aug.npy'), save_labels)
        np.save(os.path.join('/workspace/CPSC540/resnet18/test_results', dataset + '_pred_' + 'DANN_block' + str(self.cfg['feature_block']) + '_balanced_without_aug.npy'), save_predict)

        print(roc_auc_score(save_labels, save_predict,multi_class='ovr'))

        return save_labels, save_predict
    
    def get_testset_feature(self, dataset):
        """test the model by printing all the metrics for each saved model
        """
        print(f"\nStart SimpleTrain testing on {dataset} dataset.")
        if dataset == 'source':
            test_dataloader = get_source_dataloader(self.cfg['source_dataset'], slides=self.cfg['source_test_idx'], batch_size=self.cfg['batch_size'], classification_type=self.cfg['classification_type'], augment=False, num_workers=self.cfg['num_workers'])
        elif dataset == 'target':
            test_dataloader = get_target_dataloader(self.cfg['target_dataset'], slides=self.cfg['target_test_idx'], batch_size=self.cfg['batch_size'], classification_type=self.cfg['classification_type'], augment=False, num_workers=self.cfg['num_workers'])
        
        # load best model
        path = os.path.join(self.cfg["checkpoints"], f"model_{self.cfg['val_criteria']}.pth")
        state = torch.load(path, map_location=self.device)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(state["model"], strict=True)
        else:
            self.model.load_state_dict(state["model"], strict=True)


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
                feature_C, _ = self.model.get_feature(data, 0)
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
        tsne = TSNE(n_components=2, perplexity=50)
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
        plt.savefig(os.path.join('/workspace/CPSC540/resnet18/test_results/images', 'pred_' + 'DANN_block' + str(self.cfg['feature_block']) + '_balanced_without_aug.png'))

    def run(self):
        self.test('source')
        self.test('target')

    def discriminator_acc(self, dataset):
        print(f"\nStart SimpleTrain testing on {dataset} dataset.")
        if dataset == 'source':
            test_dataloader = get_source_dataloader(self.cfg['source_dataset'], slides=self.cfg['source_train_idx'], batch_size=self.cfg['batch_size'], classification_type=self.cfg['classification_type'], augment=False, num_workers=self.cfg['num_workers'])
        elif dataset == 'target':
            test_dataloader = get_target_dataloader(self.cfg['target_dataset'], slides=self.cfg['target_train_idx'], batch_size=self.cfg['batch_size'], classification_type=self.cfg['classification_type'], augment=False, num_workers=self.cfg['num_workers'])
        # load best model
        path = os.path.join(self.cfg["checkpoints"], f"model_{self.cfg['val_criteria']}.pth")
        state = torch.load(path, map_location=self.device)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(state["model"], strict=True)
        else:
            self.model.load_state_dict(state["model"], strict=True)

        predict_domain = []
        self.model.eval()
        txt = 'Test : '
        with torch.no_grad():
            prefix = txt
            for batch_data in tqdm(test_dataloader, desc=prefix,
                    dynamic_ncols=True, leave=True, position=0):
                # load data
                data = batch_data['image']
                data  = data.cuda() if torch.cuda.is_available() else data
                _, _, domain_pred, domain_prob = self.forward(data, 0)
                domain_pred = domain_pred.to('cpu').detach().numpy()
                domain_pred = np.argmax(domain_pred, axis=1)
                predict_domain.append(domain_pred)

                del data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # concatenate
        predict_domain = np.concatenate(predict_domain)
        true_domain = np.zeros(predict_domain.shape)
        if dataset == 'source':
            true_domain = np.zeros(predict_domain.shape)
        else:
            true_domain = np.ones(predict_domain.shape)
        return predict_domain, true_domain



    def get_discriminator_acc(self):
        """test the model by printing all the metrics for each saved model
        """

        s_pred, s_true = self.discriminator_acc('source')
        t_pred, t_true = self.discriminator_acc('target')

        pred = np.concatenate((s_pred, t_pred))
        label = np.concatenate((s_true, t_true))
        acc = np.sum(pred == label) / label.shape[0]
        print(acc)
        


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
    # example config file
    # slices = get_colorado_slice('../data/Colorado-10X')
    # print("slice numbers for Colorado dataset:")
    # print(slices)
    # slice = [0, 1, 2, 3, 4, 5, 6, 94, 96, 97, 98, 99]

    # sample config file for the training class
    cfg = {
    'model_name': 'resnet18', # resnet34 or resnet18
    'feature_block': 4, # select the feature layer that is sent to the domain discriminator, int from 1 ~ 4
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
    'checkpoints': '/workspace/CPSC540/resnet18/DANN_full_resnet18_block4_ws',  # dir to save the best model, training configurations and results

    }

    cfg['num_classes'] = NUMBER_OF_CLASSES[cfg['classification_type']]

    # init trainer
    trainer = SimpleDANNTest(cfg)
    # run trainer
    # trainer.plot_tSNE()
    trainer.get_discriminator_acc()
