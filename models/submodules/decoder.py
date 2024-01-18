import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.submodules.submodules import UpSampleBN, UpSampleGN, norm_normalize, sample_points


class Decoder(nn.Module):
    def __init__(self, args, basemodel_name='tf_efficientnet_b5_ap'):
        super(Decoder, self).__init__()

        # hyper-parameter for sampling
        self.sampling_ratio = args.sampling_ratio
        self.importance_ratio = args.importance_ratio

        if basemodel_name in ['tf_efficientnet_b0_ap',
                              'tf_efficientnet_lite0']:
            # feature-map
            self.conv2 = nn.Conv2d(1280, 640, kernel_size=1, stride=1, padding=0)
            if args.architecture == 'BN':
                self.up1 = UpSampleBN(skip_input=640 + 112, output_features=320)
                self.up2 = UpSampleBN(skip_input=320 + 40, output_features=160)
                self.up3 = UpSampleBN(skip_input=160 + 24, output_features=80)
                self.up4 = UpSampleBN(skip_input=80 + 16, output_features=40)

            elif args.architecture == 'GN':
                self.up1 = UpSampleGN(skip_input=640 + 112, output_features=320)
                self.up2 = UpSampleGN(skip_input=320 + 40, output_features=320)
                self.up3 = UpSampleGN(skip_input=160 + 24, output_features=80)
                self.up4 = UpSampleGN(skip_input=80 + 16, output_features=40)

            else:
                raise Exception('invalid architecture')

            # produces 1/8 res output
            self.out_conv_res8 = nn.Conv2d(160, 4, kernel_size=3, stride=1, padding=1)

            # produces 1/4 res output
            self.out_conv_res4 = nn.Sequential(
                nn.Conv1d(160 + 4, 80, kernel_size=1), nn.LeakyReLU(),
                nn.Conv1d(80, 80, kernel_size=1), nn.LeakyReLU(),
                nn.Conv1d(80, 4, kernel_size=1),
            )

            # produces 1/2 res output
            self.out_conv_res2 = nn.Sequential(
                nn.Conv1d(80 + 4, 40, kernel_size=1), nn.LeakyReLU(),
                nn.Conv1d(40, 40, kernel_size=1), nn.LeakyReLU(),
                nn.Conv1d(40, 4, kernel_size=1),
            )

            # produces 1/1 res output
            self.out_conv_res1 = nn.Sequential(
                nn.Conv1d(40 + 4, 20, kernel_size=1), nn.LeakyReLU(),
                nn.Conv1d(20, 20, kernel_size=1), nn.LeakyReLU(),
                nn.Conv1d(20, 4, kernel_size=1),
            )
        elif basemodel_name in ['tf_efficientnet_b5_ap']:
            # feature-map
            self.conv2 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0)
            if args.architecture == 'BN':
                self.up1 = UpSampleBN(skip_input=2048 + 176, output_features=1024)
                self.up2 = UpSampleBN(skip_input=1024 + 64, output_features=512)
                self.up3 = UpSampleBN(skip_input=512 + 40, output_features=256)
                self.up4 = UpSampleBN(skip_input=256 + 24, output_features=128)

            elif args.architecture == 'GN':
                self.up1 = UpSampleGN(skip_input=2048 + 176, output_features=1024)
                self.up2 = UpSampleGN(skip_input=1024 + 64, output_features=512)
                self.up3 = UpSampleGN(skip_input=512 + 40, output_features=256)
                self.up4 = UpSampleGN(skip_input=256 + 24, output_features=128)

            else:
                raise Exception('invalid architecture')

            # produces 1/8 res output
            self.out_conv_res8 = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)

            # produces 1/4 res output
            self.out_conv_res4 = nn.Sequential(
                nn.Conv1d(512 + 4, 128, kernel_size=1), nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
                nn.Conv1d(128, 4, kernel_size=1),
            )

            # produces 1/2 res output
            self.out_conv_res2 = nn.Sequential(
                nn.Conv1d(256 + 4, 128, kernel_size=1), nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
                nn.Conv1d(128, 4, kernel_size=1),
            )

            # produces 1/1 res output
            self.out_conv_res1 = nn.Sequential(
                nn.Conv1d(128 + 4, 128, kernel_size=1), nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
                nn.Conv1d(128, 4, kernel_size=1),
            )
        else:
            print(f'unknown basemodel_name: {basemodel_name}!')
            exit(1)

    def forward(self, features, gt_norm_mask=None, mode='test'):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]

        # generate feature-map

        x_d0 = self.conv2(x_block4)                     # x_d0 : [2, 640, 15, 20]      1/32 res
        x_d1 = self.up1(x_d0, x_block3)                 # x_d1 : [2, 320, 30, 40]      1/16 res
        x_d2 = self.up2(x_d1, x_block2)                 # x_d2 : [2, 160, 60, 80]       1/8 res
        x_d3 = self.up3(x_d2, x_block1)                 # x_d3: [2, 80, 120, 160]      1/4 res
        x_d4 = self.up4(x_d3, x_block0)                 # x_d4: [2, 40, 240, 320]      1/2 res

        # 1/8 res output
        out_res8 = self.out_conv_res8(x_d2)             # out_res8: [2, 4, 60, 80]      1/8 res output
        out_res8 = norm_normalize(out_res8)

        ################################################################################################################
        # out_res4
        ################################################################################################################

        if mode == 'train':
            # upsampling ... out_res8: [2, 4, 60, 80] -> out_res8_res4: [2, 4, 120, 160]
            out_res8_res4 = F.interpolate(out_res8, scale_factor=2, mode='bilinear', align_corners=True)
            B, _, H, W = out_res8_res4.shape

            # samples: [B, 1, N, 2]
            point_coords_res4, rows_int, cols_int = sample_points(out_res8_res4.detach(), gt_norm_mask,
                                                                  sampling_ratio=self.sampling_ratio,
                                                                  beta=self.importance_ratio)

            # output (needed for evaluation / visualization)
            out_res4 = out_res8_res4

            # grid_sample feature-map
            feat_res4 = F.grid_sample(x_d2, point_coords_res4, mode='bilinear', align_corners=True)  # (B, 160, 1, N)
            init_pred = F.grid_sample(out_res8, point_coords_res4, mode='bilinear', align_corners=True)  # (B, 4, 1, N)
            feat_res4 = torch.cat([feat_res4, init_pred], dim=1)  # (B, 160+4, 1, N)

            # prediction (needed to compute loss)
            samples_pred_res4 = self.out_conv_res4(feat_res4[:, :, 0, :])  # (B, 4, N)
            samples_pred_res4 = norm_normalize(samples_pred_res4)

            for i in range(B):
                out_res4[i, :, rows_int[i, :], cols_int[i, :]] = samples_pred_res4[i, :, :]

        else:
            # grid_sample feature-map
            feat_map = F.interpolate(x_d2, scale_factor=2, mode='bilinear', align_corners=True)
            init_pred = F.interpolate(out_res8, scale_factor=2, mode='bilinear', align_corners=True)
            feat_map = torch.cat([feat_map, init_pred], dim=1)  # (B, 160+4, H, W)
            B, C, H, W = feat_map.shape

            # try all pixels
            out_res4 = self.out_conv_res4(feat_map.view(B, C, -1))  # (B, 4, N)
            out_res4 = norm_normalize(out_res4)
            out_res4 = out_res4.view(B, 4, H, W)
            samples_pred_res4 = point_coords_res4 = torch.empty(0) #None

        ################################################################################################################
        # out_res2
        ################################################################################################################

        if mode == 'train':
            # upsampling ... out_res4: [2, 4, 120, 160] -> out_res4_res2: [2, 4, 240, 320]
            out_res4_res2 = F.interpolate(out_res4, scale_factor=2, mode='bilinear', align_corners=True)
            B, _, H, W = out_res4_res2.shape

            # samples: [B, 1, N, 2]
            point_coords_res2, rows_int, cols_int = sample_points(out_res4_res2.detach(), gt_norm_mask,
                                                                  sampling_ratio=self.sampling_ratio,
                                                                  beta=self.importance_ratio)

            # output (needed for evaluation / visualization)
            out_res2 = out_res4_res2

            # grid_sample feature-map
            feat_res2 = F.grid_sample(x_d3, point_coords_res2, mode='bilinear', align_corners=True)  # (B, 80, 1, N)
            init_pred = F.grid_sample(out_res4, point_coords_res2, mode='bilinear', align_corners=True)  # (B, 4, 1, N)
            feat_res2 = torch.cat([feat_res2, init_pred], dim=1)  # (B, 80+4, 1, N)

            # prediction (needed to compute loss)
            samples_pred_res2 = self.out_conv_res2(feat_res2[:, :, 0, :])  # (B, 4, N)
            samples_pred_res2 = norm_normalize(samples_pred_res2)

            for i in range(B):
                out_res2[i, :, rows_int[i, :], cols_int[i, :]] = samples_pred_res2[i, :, :]

        else:
            # grid_sample feature-map
            feat_map = F.interpolate(x_d3, scale_factor=2, mode='bilinear', align_corners=True)
            init_pred = F.interpolate(out_res4, scale_factor=2, mode='bilinear', align_corners=True)
            feat_map = torch.cat([feat_map, init_pred], dim=1)  # (B, 80+4, H, W)
            B, C, H, W = feat_map.shape

            out_res2 = self.out_conv_res2(feat_map.view(B, C, -1))  # (B, 4, N)
            out_res2 = norm_normalize(out_res2)
            out_res2 = out_res2.view(B, 4, H, W)
            samples_pred_res2 = point_coords_res2 = torch.empty(0) #None

        ################################################################################################################
        # out_res1
        ################################################################################################################

        if mode == 'train':
            # upsampling ... out_res4: [2, 4, 120, 160] -> out_res4_res2: [2, 4, 240, 320]
            out_res2_res1 = F.interpolate(out_res2, scale_factor=2, mode='bilinear', align_corners=True)
            B, _, H, W = out_res2_res1.shape

            # samples: [B, 1, N, 2]
            point_coords_res1, rows_int, cols_int = sample_points(out_res2_res1.detach(), gt_norm_mask,
                                                                  sampling_ratio=self.sampling_ratio,
                                                                  beta=self.importance_ratio)

            # output (needed for evaluation / visualization)
            out_res1 = out_res2_res1

            # grid_sample feature-map
            feat_res1 = F.grid_sample(x_d4, point_coords_res1, mode='bilinear', align_corners=True)  # (B, 40, 1, N)
            init_pred = F.grid_sample(out_res2, point_coords_res1, mode='bilinear', align_corners=True)  # (B, 4, 1, N)
            feat_res1 = torch.cat([feat_res1, init_pred], dim=1)  # (B, 40+4, 1, N)

            # prediction (needed to compute loss)
            samples_pred_res1 = self.out_conv_res1(feat_res1[:, :, 0, :])  # (B, 4, N)
            samples_pred_res1 = norm_normalize(samples_pred_res1)

            for i in range(B):
                out_res1[i, :, rows_int[i, :], cols_int[i, :]] = samples_pred_res1[i, :, :]

        else:
            # grid_sample feature-map
            feat_map = F.interpolate(x_d4, scale_factor=2, mode='bilinear', align_corners=True)
            init_pred = F.interpolate(out_res2, scale_factor=2, mode='bilinear', align_corners=True)
            feat_map = torch.cat([feat_map, init_pred], dim=1)  # (B, 40+4, H, W)
            B, C, H, W = feat_map.shape

            out_res1 = self.out_conv_res1(feat_map.view(B, C, -1))  # (B, 4, N)
            out_res1 = norm_normalize(out_res1)
            out_res1 = out_res1.view(B, 4, H, W)
            samples_pred_res1 = point_coords_res1 = torch.empty(0) #None

        return (out_res8, out_res4, out_res2, out_res1), \
               (out_res8, samples_pred_res4, samples_pred_res2, samples_pred_res1), \
               (torch.empty(0), point_coords_res4, point_coords_res2, point_coords_res1)

