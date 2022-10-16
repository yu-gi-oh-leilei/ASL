from collections import OrderedDict
import os
import warnings

import torch
from torch.functional import Tensor
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

import models
from models.cls_cvt import build_CvT
from models.vision_transformer.swin_transformer import build_swin_transformer
from models.backbone import build_backbone
from utils.misc import clean_state_dict, clean_body_state_dict
from .position_encoding import build_position_encoding



class Baseline(nn.Module):
    def __init__(self, backbone, num_class):
        """[summary]
    
        Args:
            backbone ([type]): backbone model.
            transfomer ([type]): transformer model.
            num_class ([type]): number of classes. (80 for MSCOCO).
        """
        super().__init__()
        self.backbone = backbone
        self.num_class = num_class
        self.num_channels = backbone.num_channels
        # assert not (self.ada_fc and self.emb_fc), "ada_fc and emb_fc cannot be True at the same time."
        
        self.pool= nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.num_channels, self.num_class)


    def forward(self, input):
        feature = self.backbone(input)
        feature = feature['0']
        feature = self.pool(feature).view(-1, self.num_channels)
        
        # import ipdb; ipdb.set_trace()

    
        out = self.fc(feature)
        return out  


    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(), self.query_embed.parameters())


def build_baseline(args):
    backbone = build_backbone(args)

    model = Baseline(
        backbone = backbone,
        num_class = args.num_class
    )

    return model