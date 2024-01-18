import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, basemodel_name='tf_efficientnet_b5_ap'):
        super(Encoder, self).__init__()

        print(f'Loading base model ({basemodel_name})...')
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch',
                                   basemodel_name,
                                   pretrained=True)
        print('Done.')

        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        self.original_model = basemodel

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if k == 'blocks':
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


