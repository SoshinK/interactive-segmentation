import torch
import torch.nn as nn

from isegm.utils.serialization import serialize
from .is_model import ISModel
from .modeling.tos_hrnet import TOS_HRNet
from isegm.model.modifiers import LRMult

class TOSHRNetModel(ISModel):
    @serialize
    def __init__(self, width=48, ocr_width=256, small=False, hrnet_lr_mult=0.1,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)

        self.feature_extractor = TOS_HRNet(width=width, ocr_width=ocr_width, small=small,
                                                   num_classes=1, norm_layer_param=norm_layer)
        self.feature_extractor.apply(LRMult(1.0))                          
        self.feature_extractor.context_branch.apply(LRMult(hrnet_lr_mult))
        if ocr_width > 0:
            self.feature_extractor.context_branch.ocr_distri_head.apply(LRMult(1.0))
            self.feature_extractor.context_branch.ocr_gather_head.apply(LRMult(1.0))
            self.feature_extractor.context_branch.conv3x3_ocr.apply(LRMult(1.0))

    def backbone_forward(self, image, coord_features=None):
        net_outputs = self.feature_extractor(image, coord_features)

        return {'instances': net_outputs[0], 'instances_aux': net_outputs[1], 'instances_cls_head': net_outputs[2], 'instances_edges': net_outputs[3]}
    
    def forward(self, image, points):
        image, prev_mask = self.prepare_input(image)
        coord_features = self.get_coord_features(image, prev_mask, points)

        if self.rgb_conv is not None:
            x = self.rgb_conv(torch.cat((image, coord_features), dim=1))
            outputs = self.backbone_forward(x)
        else:
            coord_features = self.maps_transform(coord_features)
            outputs = self.backbone_forward(image, coord_features)

        outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:],
                                                         mode='bilinear', align_corners=True)
        if self.with_aux_output:
            outputs['instances_aux'] = nn.functional.interpolate(outputs['instances_aux'], size=image.size()[2:],
                                                             mode='bilinear', align_corners=True)
        outputs['instances_cls_head'] = nn.functional.interpolate(outputs['instances_cls_head'], size=image.size()[2:],
                                                             mode='bilinear', align_corners=True)
        return outputs