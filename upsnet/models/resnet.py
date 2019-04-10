# ---------------------------------------------------------------------------
# Unified Panoptic Segmentation Network
#
# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Yuwen Xiong
# ---------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import warnings
from upsnet.operators.modules.deform_conv import DeformConv
from upsnet.config.config import config
import torch.utils.checkpoint


if not config.network.backbone_fix_bn and config.network.use_syncbn:
    from upsnet.operators.modules.distbatchnorm import BatchNorm2d
    nn.BatchNorm2d = BatchNorm2d


def get_params(model, prefixs, suffixes, exclude=None):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    for name, module in model.named_modules():
        for prefix in prefixs:
            if name == prefix:
                for n, p in module.named_parameters():
                    n = '.'.join([name, n])
                    if type(exclude) == list and n in exclude:
                        continue
                    if type(exclude) == str and exclude in n:
                        continue

                    for suffix in suffixes:
                        if (n.split('.')[-1].startswith(suffix) or n.endswith(suffix)) and p.requires_grad:
                            yield p
                break

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fix_bn=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if fix_bn:
            self.bn1.eval()
            self.bn2.eval()
            self.bn3.eval()
            for i in self.bn1.parameters():
                i.requires_grad = False
            for i in self.bn2.parameters():
                i.requires_grad = False
            for i in self.bn3.parameters():
                i.requires_grad = False

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DCNBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fix_bn=True, deformable_group=1):
        super(DCNBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2_offset = nn.Conv2d(planes, 18 * deformable_group, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2_offset.weight.data.zero_()
        self.conv2_offset.bias.data.zero_()
        self.conv2 = DeformConv(planes, planes, kernel_size=3, stride=1,
                                padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if fix_bn:
            self.bn1.eval()
            self.bn2.eval()
            self.bn3.eval()
            for i in self.bn1.parameters():
                i.requires_grad = False
            for i in self.bn2.parameters():
                i.requires_grad = False
            for i in self.bn3.parameters():
                i.requires_grad = False

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        offset = self.conv2_offset(out)
        out = self.conv2(out, offset)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class conv1(nn.Module):
    def __init__(self, requires_grad=False):
        super(conv1, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if not requires_grad:
            self.eval()
            for i in self.parameters():
                i.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class res_block(nn.Module):
    def __init__(self, planes, blocks, block=Bottleneck, stride=1, dilation=1, fix_bn=True, with_dpyramid=False):
        super(res_block, self).__init__()
        downsample = None
        self.inplanes = planes * 2 if planes != 64 else planes
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
                
            if fix_bn:
                downsample[1].eval()
                for i in downsample[1].parameters():
                    i.requires_grad = False

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, fix_bn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks - 1):
            layers.append(block(self.inplanes, planes, dilation=dilation, fix_bn=fix_bn))
        if with_dpyramid:
            layers.append(DCNBottleneck(self.inplanes, planes, dilation=dilation, fix_bn=fix_bn))
        else:
            layers.append(block(self.inplanes, planes, dilation=dilation, fix_bn=fix_bn))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

class resnet_rcnn(nn.Module):

    def name_mapping(self, name, resume=False):
        if resume:
            return name if not name.startswith('module.') else name[len('module.'):]
        if name.startswith('conv1') or name.startswith('bn1'):
            return 'resnet_backbone.conv1.' + name
        return name.replace('layer1', 'resnet_backbone.res2.layers')\
                   .replace('layer2', 'resnet_backbone.res3.layers')\
                   .replace('layer3', 'resnet_backbone.res4.layers')\
                   .replace('layer4', 'resnet_backbone.res5.layers')

    def load_state_dict(self, state_dict, resume=False):
        own_state = self.state_dict()

        if 'rcnn.cls_score.weight' in state_dict and own_state['rcnn.cls_score.weight'].shape[0] == 9 and state_dict['rcnn.cls_score.weight'].shape[0] == 81:
            cls_map = {
                0: 0,  # background
                1: 1,  # person
                2: -1,  # rider, ignore
                3: 3,  # car
                4: 8,  # truck
                5: 6,  # bus
                6: 7,  # train
                7: 4,  # motorcycle
                8: 2,  # bicycle
            }
            for weight_name in ['rcnn.cls_score.weight', 'rcnn.cls_score.bias', 'rcnn.bbox_pred.weight', 'rcnn.bbox_pred.bias', 'mask_branch.mask_score.weight', 'mask_branch.mask_score.bias']:
                mean = state_dict[weight_name].mean().item()
                std = state_dict[weight_name].std().item()
                state_dict[weight_name] = state_dict[weight_name].view(*([81, -1] + list(state_dict[weight_name].shape[1:])))
                weight_blobs = ((np.random.randn(*([9] + list(state_dict[weight_name].shape[1:])))) * std + mean).astype(np.float32)

                for i in range(9):
                    cls = cls_map[i]
                    if cls >= 0:
                        weight_blobs[i] = state_dict[weight_name][cls]
                weight_blobs = weight_blobs.reshape([-1] + list(state_dict[weight_name].shape[2:]))
                state_dict[weight_name] = torch.from_numpy(weight_blobs)

        if 'fcn_head.score.weight' in own_state and 'fcn_head.score.weight' in state_dict and own_state['fcn_head.score.weight'].shape[0] == 19 and state_dict['fcn_head.score.weight'].shape[0] == 133:
            cls_map = {
                0: 20,  # road
                1: 43,  # sidewalk (pavement-merged -> sidewalk)
                2: 49,  # building
                3: 51,  # wall
                4: 37,  # fence
                5: -1,  # pole
                6: 62,  # traffic light
                7: -1,  # traffic sign
                8: 36,  # vegetation (tree-merged -> vegetation)
                9: -1,  # terrain
                10: 39,  # sky
                11: 53,  # person
                12: -1,  # rider
                13: 55,  # car
                14: 60,  # truck
                15: 58,  # bus
                16: 59,  # train
                17: 56,  # motorcycle
                18: 54,  # bicycle
            }
            for weight_name in ['fcn_head.score.weight', 'fcn_head.score.bias']:
                mean = state_dict[weight_name].mean().item()
                std = state_dict[weight_name].std().item()
                weight_blobs = ((np.random.randn(*([19] + list(state_dict[weight_name].shape[1:])))) * std + mean).astype(np.float32)

                for i in range(19):
                    cls = cls_map[i]
                    if cls >= 0:
                        weight_blobs[i] = state_dict[weight_name][cls]
                state_dict[weight_name] = torch.from_numpy(weight_blobs)

        for name, param in state_dict.items():
            name = self.name_mapping(name, resume)
            if name not in own_state:
                warnings.warn('unexpected key "{}" in state_dict'.format(name))
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            if own_state[name].shape == param.shape:
                own_state[name].copy_(param)
            else:
                warnings.warn('While copying the parameter named {}, whose dimensions in the models are'
                              ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                              name, own_state[name].size(), param.size()))

        missing = set(own_state.keys()) - set([self.name_mapping(_, resume) for _ in state_dict.keys()])
        if len(missing) > 0:
            warnings.warn('missing keys in state_dict: "{}"'.format(missing))

    def get_params_lr(self):
        raise NotImplementedError()

    def freeze_backbone(self, freeze_at):
        assert freeze_at > 0
        for p in self.resnet_backbone.conv1.parameters():
            p.requires_grad = False
        self.resnet_backbone.conv1.eval()
        for i in range(2, freeze_at + 1):
            for p in eval('self.resnet_backbone.res{}'.format(i)).parameters():
                p.requires_grad = False
            eval('self.resnet_backbone.res{}'.format(i)).eval()
        
class ResNetBackbone(nn.Module):

    def __init__(self, blocks):
        super(ResNetBackbone, self).__init__()

        self.fix_bn = config.network.backbone_fix_bn
        self.with_dilation = config.network.backbone_with_dilation
        self.with_dpyramid = config.network.backbone_with_dpyramid
        self.with_dconv = config.network.backbone_with_dconv
        self.freeze_at = config.network.backbone_freeze_at


        self.conv1 = conv1(requires_grad=False)
        self.res2 = res_block(64, blocks[0], fix_bn=self.fix_bn)
        self.res3 = res_block(128, blocks[1], block=DCNBottleneck if self.with_dconv <= 3 else Bottleneck,
                              stride=2, fix_bn=self.fix_bn, with_dpyramid=self.with_dpyramid)
        self.res4 = res_block(256, blocks[2], block=DCNBottleneck if self.with_dconv <= 4 else Bottleneck,
                              stride=2, fix_bn=self.fix_bn, with_dpyramid=self.with_dpyramid)
        if self.with_dilation:
            res5_stride, res5_dilation = 1, 2
        else:
            res5_stride, res5_dilation = 2, 1
        self.res5 = res_block(512, blocks[3], block=DCNBottleneck if self.with_dconv <= 5 else Bottleneck,
                              stride=res5_stride, dilation=res5_dilation, fix_bn=self.fix_bn)
        if self.freeze_at > 0:
            for p in self.conv1.parameters():
                p.requires_grad = False
            self.conv1.eval()
            for i in range(2, self.freeze_at + 1):
                for p in eval('self.res{}'.format(i)).parameters():
                    p.requires_grad = False
                eval('self.res{}'.format(i)).eval()

    def forward(self, x):

        conv1 = self.conv1(x).detach() if self.freeze_at == 1 else self.conv1(x)
        res2 = self.res2(conv1).detach() if self.freeze_at == 2 else self.res2(conv1)

        res3 = self.res3(res2).detach() if self.freeze_at == 3 else self.res3(res2)
        res4 = self.res4(res3).detach() if self.freeze_at == 4 else self.res4(res3)
        res5 = self.res5(res4).detach() if self.freeze_at == 5 else self.res5(res4)

        return res2, res3, res4, res5
