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

        return {'instances': net_outputs[0], 'instances_aux': net_outputs[1]}