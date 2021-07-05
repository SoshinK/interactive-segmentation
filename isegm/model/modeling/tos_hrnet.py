'''
Deep Interactive Thin Object Selection,
Jun Hao Liew, Scott Cohen, Brian Price, Long Mai, Jiashi Feng
In: Winter Conference on Applications of Computer Vision (WACV), 2021
https://openaccess.thecvf.com/content/WACV2021/papers/Liew_Deep_Interactive_Thin_Object_Selection_WACV_2021_paper.pdf

https://github.com/liewjunhao/thin-object-selection
'''


""" Thin Object Selection Network (TOS-Net). """

import torch
import torch.nn as nn
import torch.nn.functional as F
# from collections import OrderedDict
# from copy import deepcopy
from torchvision.ops import RoIAlign as ROIAlign
# import os, sys
# # HACK to solve the problem "cannot find layers"
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
# import Layers_WS as L

from isegm.model.modeling.hrnet_ocr import HighResolutionNet
from isegm.utils.pytorch_sobel import sobel

# import matplotlib.pyplot as plt 

""" Weight standardization.

References:
https://arxiv.org/abs/1903.10520
https://github.com/joe-siyuan-qiao/WeightStandardization
https://github.com/MarcoForte/FBA_Matting/blob/master/networks/layers_WS.py
"""


class L_Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)

    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        #std = (weight).view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        #if self.bias is None:
        #    print("WTF", x.shape, weight.shape, self.bias, self.stride, self.padding, self.dilation, self.groups)
        #else:
        #    print("WTF not NONE", x.shape, weight.shape, self.bias.shape, self.stride, self.padding, self.dilation, self.groups)
 
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def l_GroupNorm(num_features):
    num_groups = 32 if (num_features % 32 == 0) else num_features
    return nn.GroupNorm(num_channels=num_features, num_groups=num_groups)


model_urls = {'resnet50': 'weights/resnet50-19c8e357.pth'}
Conv2d = L_Conv2d
norm_layer = l_GroupNorm
lr_size = 80 # size of the low-resolution input

def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    """ 3x3 convolution with padding. """
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                  padding=padding, dilation=dilation, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """ 1x1 convolution. """
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, 1, dilation, dilation)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes*self.expansion)
        self.bn3 = norm_layer(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
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
        else:
            residual = x
        
        out += residual
        out = self.relu(out)
        
        return out


class EncoderBlock(nn.Module):
    def __init__(self, n_inputs, n_channels, n_side_channels, n_layers=2,
                 pool=True, scale=1.0):
        super(EncoderBlock, self).__init__()
        layers = [self._make_block(n_inputs, n_channels, ks=3)]
        for n in range(n_layers):
            layers.append(self._make_block(n_channels, n_channels, ks=3))
        self.main = nn.Sequential(*layers)
        self.side = self._make_block(n_side_channels, n_channels, ks=1)
        self.pool = nn.MaxPool2d((2, 2), stride=2, padding=0) if pool else None
        self.scale = scale

    def _make_block(self, in_channels, out_channels, ks=1):
        if ks == 1:
            conv = conv1x1(in_channels, out_channels)
        elif ks == 3:
            conv = conv3x3(in_channels, out_channels)
        else:
            raise NotImplementedError
        norm = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, norm, relu)

    def forward(self, x, x_side, roi=None):
        x = self.main(x)
        xp = self.pool(x) if self.pool is not None else x

        # if x_side is None:
        #     return xp, xp # ?
        
        x_side = self.side(x_side)
        # print("lol", x_side.shape)
        if roi is None:
            roi = torch.Tensor([[0, 0, 0, lr_size, lr_size] for _ in range(x.shape[0])]).to(x_side.device)
        h, w = xp.size()[2:]
        x_side = ROIAlign((h, w), self.scale, -1)(x_side, roi)

        # print("WTF", xp.shape, x_side.shape)
        xp = torch.cat((xp, x_side), dim=1)
        return xp, x

class EncoderBlock_without_roi(EncoderBlock):
    def __init__(self, n_inputs, n_channels, n_side_channels, n_layers=2,
                 pool=True, scale=1.0):
        super().__init__(n_inputs, n_channels, n_side_channels, n_layers,
                 pool, scale)

    def forward(self, x, x_side):
        x = self.main(x)
        xp = self.pool(x) if self.pool is not None else x

        x_side = self.side(x_side)
        
        h, w = xp.size()[2:]
        # x_side = ROIAlign((h, w), self.scale, -1)(x_side, roi)
        x_side = F.interpolate(x_side, (h, w), mode='bilinear',
                align_corners=True)
        # print("WTF", xp.shape, x_side.shape)
        xp = torch.cat((xp, x_side), dim=1)
        return xp, x

class DecoderBlock(nn.Module):
    def __init__(self, n_inputs, n_channels, n_layers=2):
        super(DecoderBlock, self).__init__()
        self.layer1 = self._make_block(n_inputs, n_channels, ks=1)
        layers = []
        for n in range(n_layers):
            layers.append(self._make_block(n_channels, n_channels, ks=3))
        self.layer2 = nn.Sequential(*layers)

    def _make_block(self, in_channels, out_channels, ks=1):
        if ks == 1:
            conv = conv1x1(in_channels, out_channels)
        elif ks == 3:
            conv = conv3x3(in_channels, out_channels)
        else:
            raise NotImplementedError
        norm = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, norm, relu)

    def forward(self, x, x_side):
        x = self.layer1(x)
        x = F.interpolate(x, size=x_side.size()[2:], mode='bilinear', 
                align_corners=True)
        x = x + x_side # TODO: try concat instead of sum
        x = self.layer2(x)
        return x

class SimpleBottleneck(nn.Module):
    """Similar structure to the bottleneck layer of ResNet but with fixed #channels. """
    def __init__(self, planes):
        super(SimpleBottleneck, self).__init__()
        self.conv1 = conv1x1(planes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

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
        out += residual
        out = self.relu(out)
        return out


class HighResolutionNet_with_hrnetoutput(HighResolutionNet):
    def __init__(self, width, num_classes, ocr_width=256, small=False,
                 norm_layer=nn.BatchNorm2d, align_corners=True):
        super().__init__(width, num_classes, ocr_width, small, norm_layer, align_corners)

    
    def forward(self, x, additional_features=None):
        feats, hrnetoutput = self.compute_hrnet_feats(x, additional_features)
        if self.ocr_width > 0:
            out_aux = self.aux_head(feats)
            feats = self.conv3x3_ocr(feats)

            context = self.ocr_gather_head(feats, out_aux)
            feats = self.ocr_distri_head(feats, context)
            cls_head_out = self.cls_head(feats)
            out = feats
            return [out, out_aux, hrnetoutput, cls_head_out]
        else:
            # return [self.cls_head(feats), None], hrnetoutput
            return [feats, None, hrnetoutput, self.cls_head(feats)] 

    def compute_hrnet_feats(self, x, additional_features):
        x = self.compute_pre_stage_features(x, additional_features)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_num_branches):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_num_branches):
            if self.transition2[i] is not None:
                if i < self.stage2_num_branches:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_num_branches):
            if self.transition3[i] is not None:
                if i < self.stage3_num_branches:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
    
        return self.aggregate_hrnet_features(x), x



class TOS_HRNet(nn.Module):
    def __init__(self, width, num_classes, ocr_width=256, small=False,
                 norm_layer_param=nn.BatchNorm2d, align_corners=True):
        super(TOS_HRNet, self).__init__()
        self.inplanes = 64

  
        # Context branch (ResNet)
        self.context_branch = HighResolutionNet_with_hrnetoutput(width, num_classes, ocr_width, small, norm_layer_param, align_corners)
            
        # # Edge branch
        n_layers = 2
        # self.grad1 = EncoderBlock(4, 16, 18, n_layers, True, 1./4)
        self.grad1 = EncoderBlock_without_roi(4, 16, 18, n_layers, True, 1./4)
        self.grad2 = EncoderBlock_without_roi(32, 32, 36, n_layers, True, 1./8)
        self.grad3 = EncoderBlock_without_roi(64, 64, 72, n_layers, True, 1./8)
        self.grad4 = EncoderBlock_without_roi(128, 128, 144, n_layers, True, 1./16)
        # self.grad4 = EncoderBlock(128, 128, 72, n_layers, True, 1./16)
        # self.grad5 = EncoderBlock(256, 128, 144, n_layers, False, 1./16)
        self.grad4_decoder = DecoderBlock(256, 128, n_layers)
        self.grad3_decoder = DecoderBlock(128, 64, n_layers)
        self.grad2_decoder = DecoderBlock(64, 32, n_layers)
        self.grad1_decoder = DecoderBlock(32, 16, n_layers)
        self.edge = nn.Conv2d(16, num_classes, kernel_size=1, bias=True)

        # # Fusion block
        self.mask_trans = nn.Sequential(conv1x1(128, 48),
                                        norm_layer(48),
                                        nn.ReLU(inplace=True))
        self.img_trans = nn.Sequential(conv1x1(3, 3),
                                       norm_layer(3),
                                       nn.ReLU(inplace=True))
        self.fuse0 = nn.Sequential(conv1x1(48+16+3, 16),
                                   norm_layer(16),
                                   nn.ReLU(inplace=True))
        self.fuse1 = SimpleBottleneck(16)
        self.fuse2 = SimpleBottleneck(16)
        self.fuse3 = SimpleBottleneck(16)
        self.mask = nn.Conv2d(16, num_classes, kernel_size=1, bias=True)


  
    def forward(self, x, additional_features=None, roi=None):
        # print("1", x.shape, additional_features.shape)
        # x_lr = F.interpolate(x, size=(512, 512))
        x_lr = x
        # additional_features = F.interpolate(additional_features, size=(1024, 1024))
        # print("2", x_lr.shape, additional_features.shape)
        # s = sobel(x_lr)[0]
        # plt.figure(figsize=(15,5))
        # plt.subplot(131)
        # plt.imshow(s[0].detach().cpu().numpy())
        # plt.subplot(132)
        # plt.imshow(s[1].detach().cpu().numpy())
        # plt.subplot(133)
        # plt.imshow(s[1].detach().cpu().numpy())
        # plt.show()
        # exit()
        x_grad = torch.cat((x_lr, sobel(x_lr)[:, :1]), dim=1)
        # print("3", x_grad.shape)
        # Context stream
        # x_lr0 = self.conv1(x_lr)
        # x_lr0 = self.bn1(x_lr0)
        # x_lr0 = self.relu(x_lr0)
        # x_lr0 = self.maxpool(x_lr0)
        # x_lr1 = self.layer1(x_lr0)
        # x_lr2 = self.layer2(x_lr1)
        # x_lr3 = self.layer3(x_lr2)
        # x_lr4 = self.layer4(x_lr3)
        # print()
        # mask_lr, x_lr5_aspp, x_lr5 = self.layer5(x_lr1, x_lr4)
        # out, hrnetoutput = self.context_branch(x_lr, additional_features)
        out, out_aux, hrnetoutput, cls_head_out  = self.context_branch(x_lr, additional_features)
        # print("LOL", out.shape, cls_head_out.shape)
        # print("4", out[0].shape, out[1].shape, hrnetoutput[0].shape, hrnetoutput[1].shape, hrnetoutput[2].shape, hrnetoutput[3].shape)
        # out, out_aux = out
        # Edge stream
        x_gr1, x_enc1 = self.grad1(x_grad, hrnetoutput[0])
        # # print("5", x_gr1.shape, x_enc1.shape)
        x_gr2, x_enc2 = self.grad2(x_gr1, hrnetoutput[1])
        # # print("6", x_gr2.shape, x_enc2.shape)
        x_gr3, x_enc3 = self.grad3(x_gr2, hrnetoutput[2])
        # # print("7", x_gr3.shape, x_enc3.shape)
        x_gr4, x_enc4 = self.grad4(x_gr3, hrnetoutput[3])
        # # print("8", x_gr4.shape, x_enc4.shape)
        # x_gr5, x_enc5 = self.grad5(x_gr4, hrnetoutput[3], roi)
        # # print("9", x_gr5.shape, x_enc5.shape)
        dec = self.grad4_decoder(x_gr4, x_enc4)
        # # print("10", dec.shape)
        dec = self.grad3_decoder(dec, x_enc3)
        # # print("11", dec.shape)
        # dec = x_gr2
        dec = self.grad2_decoder(dec, x_enc2)
        # # print("12", dec.shape)
        dec = self.grad1_decoder(dec, x_enc1)
        # # print("13", dec.shape)
        edge = self.edge(dec)

        # # Fusion stream
        # print("14", out.shape)
        x_lr5 = self.mask_trans(out)
        # print("15", x_lr5.shape, out.shape)
        x_img = self.img_trans(x)
        # print("16", x.shape, x_img.shape)
        # if roi is None:
        #     # roi = torch.Tensor([[0, 0, 0, lr_size, lr_size]]).to(x.device)
        #     roi = torch.Tensor([[0, 0, 0, lr_size, lr_size] for _ in range(x.shape[0])]).to(x.device)
        # h, w = x.size()[2:]
        # roipool = ROIAlign((h, w), 1./4, -1)
        # x_lr5 = roipool(x_lr5, roi)
        # x_img = roipool(x_img, roi)
        # print("WTF", x_lr5.shape, dec.shape, x_img.shape)
        # fuse0 = torch.cat((x_lr5, dec, F.interpolate(x_img, x_lr5.shape[2:], mode='bilinear',
        #         align_corners=True)), dim=1)
        fuse0 = torch.cat((F.interpolate(x_lr5, x_img.shape[2:], mode='bilinear',
                align_corners=True), dec, x_img), dim=1)
        fuse0 = self.fuse0(fuse0)
        fuse1 = self.fuse1(fuse0)
        fuse2 = self.fuse2(fuse1)
        fuse3 = self.fuse3(fuse2)
        mask = self.mask(fuse3)
        # print("LOL2", mask.shape)

        return mask, out_aux, cls_head_out, edge

    # def load_pretrained_weights(self, pretrained_path=''):
    #     model_dict = self.state_dict()

    #     if not os.path.exists(pretrained_path):
    #         print(f'\nFile "{pretrained_path}" does not exist.')
    #         print('You need to specify the correct path to the pre-trained weights.\n'
    #               'You can download the weights for HRNet from the repository:\n'
    #               'https://github.com/HRNet/HRNet-Image-Classification')
    #         exit(1)
    #     pretrained_dict = torch.load(pretrained_path, map_location={'cuda:0': 'cpu'})
    #     pretrained_dict = {k.replace('last_layer', 'aux_head').replace('model.', ''): v for k, v in
    #                        pretrained_dict.items()}

    #     pretrained_dict = {k: v for k, v in pretrained_dict.items()
    #                        if k in model_dict.keys()}

    #     model_dict.update(pretrained_dict)
    #     self.load_state_dict(model_dict)








# if __name__ == '__main__':
#     x = torch.rand(1, 4, 512, 512).cuda()
#     x_grad = torch.rand(1, 4, 512, 512).cuda()
#     x_lr = torch.rand(1, 4, 416, 416).cuda()
#     model = tosnet_resnet50(n_inputs=4, n_classes=1).cuda()
#     model.eval()
#     print(model)
#     outputs = model(x, x_grad, x_lr, roi=None)
#     for output in outputs:
#         print(output.size())
