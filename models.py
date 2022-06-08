from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models




###########################################################RAN##########################################################

# https://github.com/kaiwang960112/Challenge-condition-FER-dataset
# https://drive.google.com/file/d/1H421M8mosIVt8KsEWQ1UuYMkQS8X1prf/view - resnet model

## only ResNet without attention module#################################################################################
class Resnet(nn.Module):
    def __init__(self, pretrained_task='faceid', n_classes=1):
        super(Resnet, self).__init__()

        if pretrained_task == 'faceid':
            resnet = models.resnet18(True)
            checkpoint = torch.load('./models/resnet18_msceleb.pth')
            resnet.load_state_dict(checkpoint['state_dict'], strict=True)

        if pretrained_task == 'emotion_sigmoid':
            resnet = ResnetEmotionTrain(final_function='sigmoid')
            resnet.load_state_dict(torch.load('/data/emotion_data/2022_05_12_07_17_27/saved_models/epoch_099.tar')
                                   ['model_state_dict'], strict=True)

        if pretrained_task == 'neutral_sigmoid':
            resnet = ResnetEmotionTrain(final_function='sigmoid', output_size=1)
            resnet.load_state_dict(torch.load('/data/runs/EM_neutral/2022_05_21_09_41_58/saved_models'
                                              '/epoch_032-Validation[ddcf_childefes]F1Score_0.997.ckpt')
                                   ['model_state_dict'], strict=True)

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()

        #resnet basic case
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        img = x

        x = self.features(img)
        x = x.squeeze(3).squeeze(2)

        out = self.fc(x)
        out = self.sig(out)

        return out
## only ResNet without attention module#################################################################################
class Resnet_fusion(nn.Module):
    def __init__(self, pretrained_task='faceid', n_classes=1, input_keypoints=34):
        super(Resnet_fusion, self).__init__()

        if pretrained_task == 'faceid':
            resnet = models.resnet18(True)
            checkpoint = torch.load('./models/resnet18_msceleb.pth')
            resnet.load_state_dict(checkpoint['state_dict'], strict=True)

        if pretrained_task == 'emotion_sigmoid':
            resnet = ResnetEmotionTrain(final_function='sigmoid')
            resnet.load_state_dict(torch.load('/data/emotion_data/2022_05_12_07_17_27/saved_models/epoch_099.tar')
                                   ['model_state_dict'], strict=True)

        if pretrained_task == 'neutral_sigmoid':
            resnet = ResnetEmotionTrain(final_function='sigmoid', output_size=1)
            resnet.load_state_dict(torch.load('/data/runs/EM_neutral/2022_05_21_09_41_58/saved_models'
                                              '/epoch_032-Validation[ddcf_childefes]F1Score_0.997.ckpt')
                                   ['model_state_dict'], strict=True)

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()

        #resnet basic case
        self.fc = nn.Linear(1024, n_classes)

        self.f1 = nn.Linear(input_keypoints, 100)
        self.f2 = nn.Linear(100, 512)

    def forward(self, x):
        img = x[0]
        f = x[1]

        x = self.features(img)
        x = x.squeeze(3).squeeze(2)

        f = self.relu(self.f1(f))
        f = self.relu(self.f2(f))

        out = torch.cat([x, f], dim=1)
        out = self.fc(out)
        out = self.sig(out)

        return out

##################Small network to process the keypoints################################################################
class KeypointNet(nn.Module):

    def __init__(self, output_size=1, input_keypoints=34):
        super().__init__()
        self.f1 = nn.Linear(input_keypoints, 100)
        self.f2 = nn.Linear(100, 100)
        self.f3 = nn.Linear(100, 100)
        self.f4 = nn.Linear(100, output_size)

        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()


    def forward(self, x):
        x = self.relu(self.f1(x))
        x = self.relu(self.f2(x))
        x = self.relu(self.f3(x))
        x = self.sig(self.f4(x))

        return x
        
class ArrayFeatureNet(nn.Module):
    def __init__(self, input_size : int, first_layer = 100, second_layer = 100, third_layer = 200, output_size=1):
        super().__init__()
        self.f1 = nn.Linear(input_size, first_layer)
        self.f2 = nn.Linear(first_layer, second_layer)
        self.f3 = nn.Linear(second_layer, third_layer)
        self.f4 = nn.Linear(third_layer, output_size)

        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.f1(x)
        x = self.relu(x)
        x = self.f2(x)
        x = self.relu(x)
        x = self.f3(x)
        x = self.relu(x)
        x = self.f4(x)
        out = self.sig(x)

        return out
        
        
################## Training backbone with emotion data #######################
# class for resnet backbone train with emotion data        
class ResnetEmotionTrain(nn.Module):
    def __init__(self, pretrained=True, final_function='softmax', output_size=7):
        super(ResnetEmotionTrain, self).__init__()

        resnet = models.resnet18(pretrained)
        if pretrained:
            checkpoint = torch.load('./models/resnet18_msceleb.pth')
            resnet.load_state_dict(checkpoint['state_dict'], strict=True)

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        if final_function == 'sigmoid':
            self.sig = nn.Sigmoid()
        else:
            self.sig = nn.Softmax()

        self.relu = nn.ReLU()

        self.fc = nn.Linear(512, output_size)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(3).squeeze(2)

        out = self.fc(x)
        out = self.sig(out)

        return out


class SimpleCNN(torch.nn.Module):
    """CNN network for image classification.

    Args:
        input_size: Size of the input image.
        conv_filters: List of numbers representing the number of filters in each conv layer.
        dense_units: List of numbers representing the number of units in each dense layer.
        activation: Activation function to apply on conv and dense layers (default=torch.nn.Relu).
        dropout: Dropout rate applied for regularization after each dense layer (default=0.2).
        input_channels: Number of channels of the input image (default=3 (color)) .
        n_classes: Number of output classes (default=1)

    Example:
        model = SimpleCNN(
            input_size=224,
            input_channels=3,
            conv_filters=[32, 48, 64, 64],
            dense_units=[256, 128, 16],
            n_classes=1
        )
    """
    def __init__(self,
                 input_size,
                 conv_filters,
                 dense_units,
                 activation: torch.nn.Module = torch.nn.ReLU(),
                 dropout=0.2,
                 input_channels=3,
                 n_classes=1):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        channels = [input_channels, *conv_filters]
        for i, in_channels in enumerate(channels[:-1]):
            out_channels = channels[i + 1]
            self.layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3)))
            input_size -= 2
            self.layers.append(torch.nn.MaxPool2d(kernel_size=(2, 2)))
            input_size //= 2
            self.layers.append(activation)

        self.layers.append(torch.nn.Flatten())
        input_size = input_size ** 2 * conv_filters[-1]
        features = [input_size, *dense_units]
        for i, in_features in enumerate(features[:-1]):
            out_features = features[i + 1]
            self.layers.append(torch.nn.Linear(in_features, out_features))
            self.layers.append(activation)
            self.layers.append(torch.nn.Dropout(p=dropout))

        self.layers.append(torch.nn.Linear(features[-1], n_classes))
        self.layers.append(torch.nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
