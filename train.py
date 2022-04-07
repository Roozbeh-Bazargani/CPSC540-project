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
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


NUMBER_OF_CLASSES = {'full': 4, 'cancer': 2, 'grade': 2}


class SimpleDANNTrain(object):
    def __init__(self, cfg):
        """
        Initialize the configuration
        """
        self.cfg = cfg
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.writer = SummaryWriter(log_dir=cfg['tensorboard_dir'])

        if not os.path.exists(self.cfg["checkpoints"]):
            os.mkdir(self.cfg["checkpoints"])

        # build model
        self.model = DANN(model_name=cfg['model_name'], num_classes=cfg['num_classes'], feature_block=cfg['feature_block'])
        # load previous model is provided with the path to model.pkl
        if self.cfg['saved_model_path'] != None:
            saved_weight = torch.load( self.cfg['saved_model_path'])
            self.model.load_state_dict(saved_weight['model'], map_location=self.device)
        self.model.to(self.device)

        # multiple GPU
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)

        # setup training loss
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion_domain = torch.nn.CrossEntropyLoss()

        # setup optimizer
        optimizer = getattr(torch.optim, self.cfg['optimizer'])
        if self.cfg['optimizer'] == 'Adam':
            self.optimizer = optimizer(self.model.parameters(), lr=self.cfg['lr'],
                                        weight_decay=self.cfg['wd'])
        elif self.cfg['optimizer'] == 'SGD':
            self.optimizer = optimizer(self.model.parameters(), lr=self.cfg['lr'], momentum=self.cfg['momentum'],
                                        weight_decay=self.cfg['wd'])
        # TODO: setup early stopping

        # setup scheduler
        if self.cfg['use_schedular']:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.8)


    def forward(self, data, alpha):
        class_pred, domain_pred = self.model.forward(data, alpha)
        class_prob   = torch.softmax(class_pred, dim=1)
        domain_prob  = torch.softmax(domain_pred, dim=1)
        return class_pred, class_prob, domain_pred, domain_prob

    def train_one_epoch(self, train_dataloader, target_train_dataloader, epoch, all_epoch):
        """Train the model for a epoch
        Patch accuract will be shown at the end of the epoch

        Parameters
        ----------
        epoch: int
        train_dataloader : torch.utils.data.DataLoader
        """
        loss_ = 0
        gt_labels   = []
        pred_labels = []
        self.model.train()

        len_sourceloader = len(train_dataloader)

        i = 0
        prefix = f'Training Epoch {epoch}: '
        
        for batch_data in tqdm(train_dataloader, desc=prefix,
                dynamic_ncols=True, leave=True, position=0):

            p = float(i + epoch * len_sourceloader) / all_epoch / len_sourceloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            if i%len(target_train_dataloader)==0:
                data_target_iter = iter(target_train_dataloader)

            i += 1

            # forward for source data
            data = batch_data['image']
            label = batch_data['label']
            s_domain_label = torch.zeros(len(label)).long()
            data  = data.cuda() if torch.cuda.is_available() else data
            label = label.cuda() if torch.cuda.is_available() else label
            s_domain_label = s_domain_label.cuda() if torch.cuda.is_available() else s_domain_label
            sc_pred, sc_prob, sd_pred, _ = self.forward(data, alpha)
            del data

            # forward for unlabeled target data
            batch_data_target = data_target_iter.next()
            t_data = batch_data_target['image']
            t_label = batch_data_target['label']
            t_domain_label = torch.ones(len(t_label)).long()

            t_data  = t_data.cuda() if torch.cuda.is_available() else t_data
            t_label = t_label.cuda() if torch.cuda.is_available() else t_label
            t_domain_label = t_domain_label.cuda() if torch.cuda.is_available() else t_domain_label

            _, _, td_pred, _ = self.forward(t_data, alpha)

            del t_data

            # get loss
            loss_s_label = self.criterion(sc_pred.type(torch.float), label.type(torch.long))
            loss_s_domain = self.criterion_domain(sd_pred.type(torch.float), s_domain_label.type(torch.long))
            loss_t_domain = self.criterion_domain(td_pred.type(torch.float), t_domain_label.type(torch.long))
            loss = loss_s_label + loss_s_domain + loss_t_domain

            # backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # get performance metrics
            loss_ += loss.item() * label.shape[0]
            gt_labels   += label.cpu().numpy().tolist()
            pred_labels += torch.argmax(sc_prob, dim=1).cpu().numpy().tolist()

            # clean cache
            del label
            del s_domain_label
            del t_label
            del t_domain_label
            del loss_s_label 
            del loss_s_domain
            del loss_t_domain
            del loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # get metrics for this epoch
        train_acc  = accuracy_score(gt_labels, pred_labels)
        train_loss = loss_ / len(gt_labels)
        self.writer.add_scalar('train/train_loss', train_loss, global_step=epoch)
        self.writer.add_scalar('train/train_acc', train_acc, global_step=epoch)
        print(f"\nTraining loss is {train_loss:.4f}, Training (patch) accuracy is {train_acc:.4f} at epoch {epoch}.")

    def validate(self, dataloader, epoch=None, test=False):
        """Validate the model for a epoch by finding AUC of patches
        Parameters
        ----------
        epoch: int
        dataloader : torch.utils.data.DataLoader
        test : bool
            Whether it is in the test mode (true) or validation one (false)
        """
        loss_ = 0
        patch_info = {'gt_label': [], 'prediction': [],
                      'probability': np.array([]).reshape(0, self.cfg['num_classes'])}

        self.model.eval()
        txt = f'Validation Epoch {epoch}: ' if not test else 'Test : '
        with torch.no_grad():
            prefix = txt
            for batch_data in tqdm(dataloader, desc=prefix,
                    dynamic_ncols=True, leave=True, position=0):
                # load data
                data = batch_data['image']
                label = batch_data['label']
                data  = data.cuda() if torch.cuda.is_available() else data
                label = label.cuda() if torch.cuda.is_available() else label
                predicted, prob, _, _ = self.forward(data, 0)
                if not test:
                    loss = self.criterion(predicted.type(torch.float), label.type(torch.long))
                    loss_ += loss.item() * label.shape[0]
                # save patch level results
                patch_info['gt_label']   += label.cpu().numpy().tolist()
                patch_info['prediction'] += torch.argmax(prob, dim=1).cpu().numpy().tolist()
                patch_info['probability'] = np.vstack((patch_info['probability'],
                                                       prob.cpu().numpy()))
                # clean cache
                del data
                del label
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not test:
            val_loss = loss_ / len(patch_info['gt_label'])
        # run patch evaluation
        perf = patch_metrics(patch_info, self.cfg['num_classes'])
        # write in tensorflow
        if not test:
            val_auc  = perf['overall_auc']
            val_acc  = perf['overall_acc']
            perf['val_loss'] = val_loss
            self.writer.add_scalar(f'validation/_val_acc', val_acc, global_step=epoch)
            self.writer.add_scalar(f'validation/_val_auc', val_auc, global_step=epoch)
            self.writer.add_scalar(f'validation/_val_loss', val_loss, global_step=epoch)
            print(f"Validation (patch) loss is {val_loss:.4f}, AUC is {val_auc:.4f}, ACC is {val_acc:.4f}.")
        else:
            val_auc  = perf['overall_auc']
            val_acc  = perf['overall_acc']
            print(f"Test (patch) AUC is {val_auc:.4f}, ACC is {val_acc:.4f}.")
        return {'patch': patch_info, 'performance':perf}


    def validate_with_target(self, dataloader, target_dataloader, epoch=None):
        """Validate the model for a epoch by finding AUC of patches
        Parameters
        ----------
        epoch: int
        dataloader : torch.utils.data.DataLoader
        test : bool
            Whether it is in the test mode (true) or validation one (false)
        """
        loss_ = 0
        patch_info = {'gt_label': [], 'prediction': [],
                      'probability': np.array([]).reshape(0, self.cfg['num_classes'])}

        self.model.eval()
        txt = f'Validation Epoch {epoch}: '
        with torch.no_grad():
            prefix = txt

            len_sourceloader = len(dataloader)

            i = 0
            prefix = f'Training Epoch {epoch}: '
            
            for batch_data in tqdm(dataloader, desc=prefix, dynamic_ncols=True, leave=True, position=0):

                p = float(i + epoch * len_sourceloader) / self.cfg['epochs'] / len_sourceloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                if i%len(target_dataloader)==0:
                    data_target_iter = iter(target_dataloader)

                i += 1

                # source data
                data = batch_data['image']
                label = batch_data['label']
                s_domain_label = torch.zeros(len(label)).long()
                data  = data.cuda() if torch.cuda.is_available() else data
                label = label.cuda() if torch.cuda.is_available() else label
                s_domain_label = s_domain_label.cuda() if torch.cuda.is_available() else s_domain_label
                sc_pred, sc_prob, sd_pred, _ = self.forward(data, alpha)

                patch_info['gt_label']   += label.cpu().numpy().tolist()
                patch_info['prediction'] += torch.argmax(sc_prob, dim=1).cpu().numpy().tolist()
                patch_info['probability'] = np.vstack((patch_info['probability'],
                                                       sc_prob.cpu().numpy()))
                del data

                # forward for unlabeled target data
                batch_data_target = data_target_iter.next()
                t_data = batch_data_target['image']
                t_label = batch_data_target['label']
                t_domain_label = torch.ones(len(t_label)).long()

                t_data  = t_data.cuda() if torch.cuda.is_available() else t_data
                t_label = t_label.cuda() if torch.cuda.is_available() else t_label
                t_domain_label = t_domain_label.cuda() if torch.cuda.is_available() else t_domain_label

                _, _, td_pred, _ = self.forward(t_data, alpha)

                del t_data

                # get loss
                loss_s_label = self.criterion(sc_pred.type(torch.float), label.type(torch.long))
                loss_s_domain = self.criterion_domain(sd_pred.type(torch.float), s_domain_label.type(torch.long))
                loss_t_domain = self.criterion_domain(td_pred.type(torch.float), t_domain_label.type(torch.long))
                loss = loss_s_label + loss_s_domain + loss_t_domain

                # get performance metrics
                loss_ += loss.item() * label.shape[0]

                # clean cache
                del label
                del s_domain_label
                del t_label
                del t_domain_label
                del loss_s_label 
                del loss_s_domain
                del loss_t_domain
                del loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        val_loss = loss_ / len(patch_info['gt_label'])
        # run patch evaluation
        perf = patch_metrics(patch_info, self.cfg['num_classes'])
        # write in tensorflow

        val_auc  = perf['overall_auc']
        val_acc  = perf['overall_acc']
        perf['val_loss'] = val_loss
        self.writer.add_scalar(f'validation/_val_acc', val_acc, global_step=epoch)
        self.writer.add_scalar(f'validation/_val_auc', val_auc, global_step=epoch)
        self.writer.add_scalar(f'validation/_val_loss', val_loss, global_step=epoch)
        print(f"Validation (patch) loss is {val_loss:.4f}, AUC is {val_auc:.4f}, ACC is {val_acc:.4f}.")

        return {'patch': patch_info, 'performance':perf}

    def train(self):
        print(f"\nStart SimpleTrain training for {self.cfg['epochs']} epochs.")
        print(f"Training with {(self.device).upper()} device.\n")

        # save config file
        save_json(self.cfg, os.path.join(self.cfg["checkpoints"], 'training_config.json'))
        # get dataset
        train_dataloader = get_source_dataloader(self.cfg['source_dataset'], self.cfg['source_train_idx'], self.cfg['batch_size'], self.cfg['classification_type'], augment=self.cfg['augment'], shuffle=True, num_workers=self.cfg['num_workers'])
        valid_dataloader = get_source_dataloader(self.cfg['source_dataset'], self.cfg['source_val_idx'], self.cfg['batch_size'], self.cfg['classification_type'], augment=False, shuffle=False, num_workers=self.cfg['num_workers'])
        target_train_dataloader = get_target_dataloader(self.cfg['target_dataset'], self.cfg['target_train_idx'],  self.cfg['batch_size'], self.cfg['classification_type'],  augment=self.cfg['augment'], shuffle=True, num_workers=self.cfg['num_workers'])
        target_valid_dataloader = get_target_dataloader(self.cfg['target_dataset'], self.cfg['target_val_idx'],  self.cfg['batch_size'], self.cfg['classification_type'],  augment=False, shuffle=False, num_workers=self.cfg['num_workers'])
        
        if self.cfg['val_criteria'] == 'val_loss':
            best_criteria_value = np.inf
        else:
            best_criteria_value = -np.inf
        
        # step up early stopping
        last_update = 0
        
        # start training
        for epoch in range(self.cfg['epochs']):
            # train
            self.train_one_epoch(train_dataloader, target_train_dataloader, epoch, self.cfg['epochs'])
            # validate
            info = self.validate_with_target(valid_dataloader, target_valid_dataloader, epoch)
            perf = info['performance']

            # save the model
            if self.cfg['val_criteria'] == 'val_loss':
                if perf[self.cfg['val_criteria']] < best_criteria_value:
                    # save the model weights
                    best_criteria_value = perf[self.cfg['val_criteria']]
                    model_dict = self.model.state_dict() if not isinstance(self.model, nn.DataParallel) else \
                                    self.model.module.state_dict()
                    torch.save({'model': model_dict},
                                os.path.join(self.cfg["checkpoints"], f"model_{self.cfg['val_criteria']}.pth"))
                    print(f"Saved model weights based on patch for {self.cfg['val_criteria']} at epoch {epoch}.")
                    save_dict(info, f"{self.cfg['checkpoints']}/validation_{self.cfg['val_criteria']}.pkl")
                    save_json(info['performance'], f"{self.cfg['checkpoints']}/validation_{self.cfg['val_criteria']}.json")
                    last_update = epoch
            else:
                if perf[self.cfg['val_criteria']] > best_criteria_value:
                    # save the model weights
                    best_criteria_value = perf[self.cfg['val_criteria']]
                    model_dict = self.model.state_dict() if not isinstance(self.model, nn.DataParallel) else \
                                    self.model.module.state_dict()
                    torch.save({'model': model_dict},
                                os.path.join(self.cfg["checkpoints"], f"model_{self.cfg['val_criteria']}.pth"))
                    print(f"Saved model weights based on patch for {self.cfg['val_criteria']} at epoch {epoch}.")
                    save_dict(info, f"{self.cfg['checkpoints']}/validation_{self.cfg['val_criteria']}.pkl")
                    save_json(info['performance'], f"{self.cfg['checkpoints']}/validation_{self.cfg['val_criteria']}.json")
                    last_update = epoch

            # in validation, model finds out the lr needs to be reduced.
            if self.cfg['use_schedular']:
                before_lr = self.scheduler.get_last_lr()
                self.scheduler.step()
                after_lr = self.scheduler.get_last_lr()
                print(f"\nLearning rate is decreased from {before_lr[0]} to {after_lr[0]}!")
            
            # early stopping
            if self.cfg['use_earlystopping']:
                if epoch - last_update > self.cfg['earlystopping_epoch']:
                    break

        print("\nTraining has finished.")

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

        # run test on daaset
        info = self.validate(test_dataloader, test=True)
        test_perf = info['performance']
        # save results
        save_dict(info, f"{self.cfg['checkpoints']}/test_{dataset}_{self.cfg['val_criteria']}.pkl")
        save_json(info['performance'], f"{self.cfg['checkpoints']}/test_{dataset}_{self.cfg['val_criteria']}.json")

        # print results
        for k in ['overall_auc', 'overall_acc', 'overall_f1']:
            print(f"{k}: {test_perf[k]}")
        print(f'\nconfusion matrix')
        print(test_perf['conf_mat'])
        print(f'accuracy for each class')
        print(test_perf['acc_per_subtype'])

        print("\nTesting has finished.")

    def run(self):
        if not self.cfg['only_test']:
            self.train()
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
    # example config file
    # slices = get_colorado_slice('../data/Colorado-10X')
    # print("slice numbers for Colorado dataset:")
    # print(slices)
    # slice = [0, 1, 2, 3, 4, 5, 6, 94, 96, 97, 98, 99]

    # sample config file for the training class
    cfg = {'tensorboard_dir': '../tensorboard_log', # dir to save tensorboard
    'saved_model_path': None,  # path of pretrained model
    'model_name': 'resnet34', # resnet34 or resnet18
    'feature_block': 4, # select the feature layer that is sent to the domain discriminator, int from 1 ~ 4
    'classification_type': 'full', # classify all, or benign vs cancer, or low grade vs high grade. String: 'full', 'cancer' or 'grade'
    'optimizer': 'SGD',  # name of optimizer
    'lr': 0.001,  # learning rate
    'momentum': 0.9, # momentum for SGD
    'wd': 0.001,  # weight decay
    'use_schedular': False,  # bool to select whether to use scheduler
    'use_earlystopping': True,
    'earlystopping_epoch': 50, # early stop the training if no improvement on validation for this number of epochs
    'epochs': 100,  # number of training epochs
    'source_dataset': '../data/VPC-10X',   # path to vancouver dataset
    'source_train_idx': [2,5,6,7],   # indexes of the slides used for training (van dataset)
    'source_val_idx': [3],   # indexes of the slides used for validation (van dataset)
    'source_test_idx': [1],   # indexes of the slides used for testing (van dataset)
    'batch_size': 8,  # batch size
    'augment': False,  # whether use classical cv augmentation
    'target_dataset': '../data/Colorado-10X',  # path to Colorado dataset
    'target_train_idx': [0, 1, 2, 3, 4, 5, 6, 94],  # indexes of the slides used for training (CO dataset)
    'target_val_idx': [96, 98],
    'target_test_idx': [97, 99],  # indexes of the slides used for testing (CO dataset)
    'num_workers': 1, # number of workers
    'val_criteria': 'overall_auc',  # criteria to keep the current best model, can be overall_acc, overall_f1, overall_auc, val_loss
    'checkpoints': '../DANN_full_resnet34_block4',  # dir to save the best model, training configurations and results
    'only_test': False  # select true if only want to do testing

    }

    cfg['num_classes'] = NUMBER_OF_CLASSES[cfg['classification_type']]

    # init trainer
    trainer = SimpleDANNTrain(cfg)
    # run trainer
    trainer.run()