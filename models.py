import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Function



class ReverseLayerF(Function):
    # https://github.com/fungtion/DANN/blob/master/models/functions.py
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANN(nn.Module):
    def __init__(self, model_name, num_classes, feature_block):
    # def __init__(self, num_classes):
        super(DANN, self).__init__()
        
        if model_name == 'resnet34':
            resnet = models.resnet34(pretrained=True)  # resnet34: conv2d -> bm2d -> relu -> maxpool -> block 1 (64 channels) -> block 2 (128 channels) -> block 3 (256 channels) -> block 4 (512 channels) -> adaptive pool -> linear
        elif model_name == 'resnet18':
            resnet = models.resnet18(pretrained=True)  # resnet18: conv2d -> bm2d -> relu -> maxpool -> block 1 (64 channels) -> block 2 (128 channels) -> block 3 (256 channels) -> block 4 (512 channels) -> adaptive pool -> linear
        else:
            raise ValueError("Wrong model name!")

        if feature_block == 1:
            modules = list(resnet.children())[:5]
            self.feat = nn.Sequential(*modules)

            modules = list(resnet.children())[5:-1]
            self.class_classifier_1 = nn.Sequential(*modules)
            self.class_classifier_2 = nn.Sequential()
            self.class_classifier_2.add_module('last_fc', nn.Linear(512, num_classes))
            self.class_classifier_2.add_module('res_softmax', nn.LogSoftmax(dim=1))
            self.domain_classifier = nn.Sequential()
            self.domain_classifier.add_module('d_fc1', nn.Linear(64 * 128 * 128 , 256)) # -3
            self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(256))
            self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
            self.domain_classifier.add_module('d_fc2', nn.Linear(256 , 100))
            self.domain_classifier.add_module('d_bn2', nn.BatchNorm1d(100))
            self.domain_classifier.add_module('d_relu2', nn.ReLU(True))
            self.domain_classifier.add_module('d_fc3', nn.Linear(100, 2))
            self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))
        elif feature_block == 2:
            modules = list(resnet.children())[:6]
            self.feat = nn.Sequential(*modules)

            modules = list(resnet.children())[6:-1]
            self.class_classifier_1 = nn.Sequential(*modules)

            self.class_classifier_2 = nn.Sequential()
            self.class_classifier_2.add_module('last_fc', nn.Linear(512, num_classes))
            self.class_classifier_2.add_module('res_softmax', nn.LogSoftmax(dim=1))
            self.domain_classifier = nn.Sequential()
            self.domain_classifier.add_module('d_fc1', nn.Linear(128 * 64 * 64 , 256)) # -3
            self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(256))
            self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
            self.domain_classifier.add_module('d_fc2', nn.Linear(256 , 100))
            self.domain_classifier.add_module('d_bn2', nn.BatchNorm1d(100))
            self.domain_classifier.add_module('d_relu2', nn.ReLU(True))
            self.domain_classifier.add_module('d_fc3', nn.Linear(100, 2))
            self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))
        elif feature_block == 3:
            modules = list(resnet.children())[:-3]
            self.feat = nn.Sequential(*modules)

            modules = list(resnet.children())[-3:-1]
            self.class_classifier_1 = nn.Sequential(*modules)

            self.class_classifier_2 = nn.Sequential()
            self.class_classifier_2.add_module('last_fc', nn.Linear(512, num_classes))
            self.class_classifier_2.add_module('res_softmax', nn.LogSoftmax(dim=1))
            self.domain_classifier = nn.Sequential()
            self.domain_classifier.add_module('d_fc1', nn.Linear(256 * 32 * 32 , 256)) # -3
            self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(256))
            self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
            self.domain_classifier.add_module('d_fc2', nn.Linear(256 , 100))
            self.domain_classifier.add_module('d_bn2', nn.BatchNorm1d(100))
            self.domain_classifier.add_module('d_relu2', nn.ReLU(True))
            self.domain_classifier.add_module('d_fc3', nn.Linear(100, 2))
            self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))
        elif feature_block == 4:
            modules = list(resnet.children())[:-2]
            self.feat = nn.Sequential(*modules)

            modules = list(resnet.children())[-2:-1]
            self.class_classifier_1 = nn.Sequential(*modules)

            self.class_classifier_2 = nn.Sequential()
            self.class_classifier_2.add_module('last_fc', nn.Linear(512, num_classes))
            self.class_classifier_2.add_module('res_softmax', nn.LogSoftmax(dim=1))
            self.domain_classifier = nn.Sequential()
            self.domain_classifier.add_module('d_fc1', nn.Linear(512 * 16 * 16 , 256)) # -3
            self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(256))
            self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
            self.domain_classifier.add_module('d_fc2', nn.Linear(256 , 100))
            self.domain_classifier.add_module('d_bn2', nn.BatchNorm1d(100))
            self.domain_classifier.add_module('d_relu2', nn.ReLU(True))
            self.domain_classifier.add_module('d_fc3', nn.Linear(100, 2))
            self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))
        else:
            raise ValueError("Incorrect feature layer!")
        

    def forward(self, input_data, alpha=0):
        
        input_data = input_data.expand(input_data.data.shape[0], input_data.data.shape[1], 
                                       input_data.data.shape[2], input_data.data.shape[3])


        feature = self.feat(input_data)
        feat_C = self.class_classifier_1(feature)
        feat_C = feat_C.view(-1, feat_C.shape[1]* feat_C.shape[2] * feat_C.shape[3])
        class_output = self.class_classifier_2(feat_C)
        
        feat_D = ReverseLayerF.apply(feature, alpha)
        feat_D = feat_D.view(-1, feat_D.shape[1] * feat_D.shape[2] * feat_D.shape[3])
                
        domain_output = self.domain_classifier(feat_D)
        
        return class_output, domain_output
    
    def get_feature(self, input_data, alpha=0):
        input_data = input_data.expand(input_data.data.shape[0], input_data.data.shape[1], 
                                       input_data.data.shape[2], input_data.data.shape[3])


        feature = self.feat(input_data)
        feat_C = self.class_classifier_1(feature)
        
        return feat_C, feature

        
if __name__ == "__main__":

    model = DANN('resnet18', 4, 4)
    print(model)
    class_output, domain_output = model(torch.rand(4, 3, 512, 512))
    print(class_output.shape, domain_output.shape)

    # # test loading pretrained model   ## TODO: wait for the code to check
    # model = DANN('resnet18', 5, 4)
    # pretrain_dict = torch.load('pretrained_models/model_patch_overall_auc.pth', map_location='cpu')
    # model.load_state_dict(pretrain_dict['model'])
    