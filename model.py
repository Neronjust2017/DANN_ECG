import torch
import torch.nn as nn
from functions import ReverseLayerF

class CNNModel(nn.Module):

    def __init__(self, num_classes=2, num_d_classes=3):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()

        self.feature.add_module('f_conv1', nn.Conv1d(1 ,32, kernel_size=11))
        self.feature.add_module('f_bn1', nn.BatchNorm1d(32))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_pool1', nn.MaxPool1d(2))
        self.feature.add_module('f_drop1', nn.Dropout(p=0.2))

        self.feature.add_module('f_conv2', nn.Conv1d(32, 64, kernel_size=11))
        self.feature.add_module('f_bn2', nn.BatchNorm1d(64))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_pool2', nn.MaxPool1d(2))
        self.feature.add_module('f_drop2', nn.Dropout(p=0.2))

        self.feature.add_module('f_conv3', nn.Conv1d(64, 128, kernel_size=11))
        self.feature.add_module('f_bn3', nn.BatchNorm1d(128))
        self.feature.add_module('f_relu3', nn.ReLU(True))
        self.feature.add_module('f_pool3', nn.MaxPool1d(2))
        self.feature.add_module('f_drop3', nn.Dropout(p=0.2))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(128, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 80))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(80))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(80, num_classes))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(128, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, num_d_classes))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):

        feature = self.feature(input_data)
        feature = torch.mean(feature, dim=2)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output
