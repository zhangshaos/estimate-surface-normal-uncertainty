import os
import sys
import time
import torch
import argparse
import numpy as np
from PIL import Image
import imageio.v2 as imageio
from tqdm import tqdm

from data.dataloader_vcc import VCC_Loader, VCC_DatasetParams
from models.NNET import NNET
import funcs.utils as utils


def run(model, test_loader, device, output_dir):
    alpha_max = 90
    i = 0
    output_dir = os.path.abspath(output_dir)
    f = open(f'{output_dir}/estimate_normal.txt', 'wt')
    t0 = time.time()
    with torch.no_grad():
        for data_dict in tqdm(test_loader):
            name = data_dict['img_name'][0]
            f.write(f'{name} ')
            img = data_dict['img'].to(device)
            norm_out_list, _, _ = model(img)
            norm_out = norm_out_list[-1]
            pred_norm = norm_out[:, :3, :, :]
            pred_kappa = norm_out[:, 3:, :, :]

            # to numpy arrays
            pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()  # (B, H, W, 3)
            pred_kappa = pred_kappa.cpu().permute(0, 2, 3, 1).numpy()

            # save results
            img_name = os.path.basename(data_dict['img_name'][0]).replace('.png', '')

            # 2. predicted normal
            # 模型使用左x，上y，后z左手坐标系，将其转换为右x，下y，前z相机坐标系
            pred_norm *= -1  # (B, H, W, 3)
            pred_norm_tiff = pred_norm[0, :, :, :].astype(np.float32)
            target_path = f'{output_dir}/{i}_{img_name}_pred_norm.tiff'
            imageio.imwrite(target_path, pred_norm_tiff)
            f.write(f'{target_path} ')

            pred_norm_rgb = ((pred_norm + 1) * 0.5) * 255
            pred_norm_rgb = np.clip(pred_norm_rgb, a_min=0, a_max=255)
            pred_norm_rgb = pred_norm_rgb.astype(np.uint8)  # (B, H, W, 3)
            target_path = f'{output_dir}/{i}_{img_name}_pred_norm_vis.png'
            Image.fromarray(pred_norm_rgb[0, :, :, :]).save(target_path)
            # f.write(f'{target_path} ')

            # 4. predicted uncertainty
            pred_alpha = utils.kappa_to_alpha(pred_kappa)  # (B, H, W, 1)
            pred_alpha = np.clip(pred_alpha, 0, alpha_max)
            pred_alpha_gray = (pred_alpha * (255 / 90)).astype(np.uint8)
            target_path = f'{output_dir}/{i}_{img_name}_pred_alpha.png'
            Image.fromarray(pred_alpha_gray[0, :, :, 0]).save(target_path)
            # f.write(f'{target_path} ')
            f.write('\n')
            f.flush()
            # print(f'\nhandle {name}.\n')
            i += 1
    t1 = time.time()
    print(f'\nhandle {i} images.\n')
    print(f'cost {t1 - t0} seconds.\n')
    f.close()


if __name__ == '__main__':
    # Arguments #################################################################################
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.convert_arg_line_to_args = utils.convert_arg_line_to_args

    parser.add_argument('--architecture', required=True, type=str, help='{BN, GN}')
    parser.add_argument('--pretrained', required=True, type=str, help="{pretrained model path}")
    parser.add_argument('--sampling_ratio', type=float, default=0.4)
    parser.add_argument('--importance_ratio', type=float, default=0.7)
    parser.add_argument('--input_height', default=480, type=int)
    parser.add_argument('--input_width', default=640, type=int)
    parser.add_argument('--input_data_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

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

    # run the model
    os.makedirs(args.output_dir, exist_ok=True)
    params = VCC_DatasetParams()
    params.mode = 'test'
    params.input_height = args.input_height
    params.input_width = args.input_width
    params.data_record_file = args.input_data_file
    params.need_scene = True
    params.need_normal = False
    params.need_depth = False
    loader = VCC_Loader(params).data
    run(model, loader, device, args.output_dir)
