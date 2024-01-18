import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.submodules.encoder import Encoder
from models.submodules.decoder import Decoder


class NNET(nn.Module):
    def __init__(self, args):
        super(NNET, self).__init__()
        # basemodel_name = 'tf_efficientnet_b5_ap'
        # basemodel_name = 'tf_efficientnet_b0_ap'
        basemodel_name = 'tf_efficientnet_lite0'
        self.encoder = Encoder(basemodel_name)
        self.decoder = Decoder(args, basemodel_name)

    def get_encoder_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_decoder_params(self):  # lr learning rate
        return self.decoder.parameters()

    def forward(self, img, **kwargs):
        record_cost_time = True
        print_features_shape = False
        if record_cost_time:
            t0 = time.time_ns()

        feat_s = self.encoder(img)

        if record_cost_time:
            t1 = time.time_ns()
            print(f'\nencoder cost {(t1 - t0) * 1e-6} millisecond.')
        if print_features_shape:
            print(f'Encoder features are:')
            for i, v in enumerate(feat_s):
                print(f'{i} shape is {v.shape}.')
        if record_cost_time:
            t0 = time.time_ns()

        result_s = self.decoder(feat_s, **kwargs)

        if record_cost_time:
            t1 = time.time_ns()
            print(f'decoder cost {(t1 - t0) * 1e-6} millisecond.')
        return result_s
