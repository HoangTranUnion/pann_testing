import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from .pytorch_utils import do_mixup, interpolate, pad_framewise_output

from .base_models import ConvBlock, ConvBlock5x5, AttBlock, init_layer, init_bn


def _resnet_conv3x1_wav1d(in_planes, out_planes, dilation):
    # 3x3 convolution with padding
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=dilation, groups=1, bias=False, dilation=dilation)


def _resnet_conv1x1_wav1d(in_planes, out_planes):
    # 1x1 convolution
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class _ResnetBasicBlockWav1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBasicBlockWav1d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError('_ResnetBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in _ResnetBasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.stride = stride

        self.conv1 = _resnet_conv3x1_wav1d(inplanes, planes, dilation=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _resnet_conv3x1_wav1d(planes, planes, dilation=2)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride != 1:
            out = F.max_pool1d(x, kernel_size=self.stride)
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResNetWav1d(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(_ResNetWav1d, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=4)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=4)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=4)
        self.layer5 = self._make_layer(block, 1024, layers[4], stride=4)
        self.layer6 = self._make_layer(block, 1024, layers[5], stride=4)
        self.layer7 = self._make_layer(block, 2048, layers[6], stride=4)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                    _resnet_conv1x1_wav1d(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[0])
                init_bn(downsample[1])
            else:
                downsample = nn.Sequential(
                    nn.AvgPool1d(kernel_size=stride),
                    _resnet_conv1x1_wav1d(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[1])
                init_bn(downsample[2])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        return x


class Res1dNet31(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num):
        super(Res1dNet31, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False)
        self.bn0 = nn.BatchNorm1d(64)

        self.resnet = _ResNetWav1d(_ResnetBasicBlockWav1d, [2, 2, 2, 2, 2, 2, 2])

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv0)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = input[:, None, :]

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.bn0(self.conv0(x))
        x = self.resnet(x)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict


class Res1dNet51(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num):
        super(Res1dNet51, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False)
        self.bn0 = nn.BatchNorm1d(64)

        self.resnet = _ResNetWav1d(_ResnetBasicBlockWav1d, [2, 3, 4, 6, 4, 3, 2])

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv0)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = input[:, None, :]

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.bn0(self.conv0(x))
        x = self.resnet(x)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict