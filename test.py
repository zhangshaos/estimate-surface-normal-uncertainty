import argparse
import os
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from data.dataloader_sz import SZLoader
from models.NNET import NNET
import utils.utils as utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def test(model, test_loader, device, results_dir):
    alpha_max = 90

    with torch.no_grad():
        for data_dict in tqdm(test_loader):

            img = data_dict['img'].to(device)
            norm_out_list, _, _ = model(img)
            norm_out = norm_out_list[-1]
            pred_norm = norm_out[:, :3, :, :]
            pred_kappa = norm_out[:, 3:, :, :]

            # to numpy arrays
            pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()        # (B, H, W, 3)
            pred_kappa = pred_kappa.cpu().permute(0, 2, 3, 1).numpy()

            # save results
            img_name = os.path.basename(data_dict['img_name'][0]).replace('.png','')

            # 2. predicted normal
            # 模型使用左x，上y，后z左手坐标系，将其转换为右x，下y，前z相机坐标系
            pred_norm *= -1
            pred_norm_rgb = ((pred_norm + 1) * 0.5) * 255
            pred_norm_rgb = np.clip(pred_norm_rgb, a_min=0, a_max=255)
            pred_norm_rgb = pred_norm_rgb.astype(np.uint8)  # (B, H, W, 3)

            target_path = f'{results_dir}/{img_name}_pred_norm.png'
            plt.imsave(target_path, pred_norm_rgb[0, :, :, :])

            # 4. predicted uncertainty
            pred_alpha = utils.kappa_to_alpha(pred_kappa)
            pred_alpha = np.clip(pred_alpha, 0, alpha_max)
            pred_alpha_gray = (pred_alpha * (255 / 90)).astype(np.uint8)

            target_path = f'{results_dir}/{img_name}_pred_alpha.png'
            #plt.imsave(target_path, pred_alpha_gray[0, :, :, 0], cmap='gray')
            Image.fromarray(pred_alpha_gray[0, :, :, 0]).save(target_path)


if __name__ == '__main__':
    # Arguments ########################################################################################################
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.convert_arg_line_to_args = utils.convert_arg_line_to_args

    parser.add_argument('--architecture', required=True, type=str, help='{BN, GN}')
    parser.add_argument("--pretrained", required=True, type=str, help="{pretrained model path}")
    parser.add_argument('--sampling_ratio', type=float, default=0.4)
    parser.add_argument('--importance_ratio', type=float, default=0.7)
    parser.add_argument('--input_height', default=480, type=int)
    parser.add_argument('--input_width', default=640, type=int)
    parser.add_argument('--imgs_dir', required=True, type=str)

    # read arguments from txt file
    if sys.argv.__len__() == 2 and '.txt' in sys.argv[1]:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    device = torch.device('cuda:0')

    # load checkpoint
    checkpoint = args.pretrained
    print(f'loading checkpoint... {checkpoint}')
    model = NNET(args).to(device)
    model = utils.load_checkpoint(checkpoint, model)
    model.eval()
    print('loading checkpoint... / done')

    # test the model
    results_dir = f'{args.imgs_dir}/results'
    os.makedirs(results_dir, exist_ok=True)
    test_loader = SZLoader(args, 'test').data
    test(model, test_loader, device, results_dir)

