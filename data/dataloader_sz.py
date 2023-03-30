import os
import sys
import shutil
import random

import numpy as np
from PIL import Image

import torch
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import data.utils as data_utils
# 手动导入 utils/utils.py 包
import importlib.util as iutil
visual_utils_spec = iutil.spec_from_file_location('visual_utils', f'{os.path.dirname(__file__)}/../utils/utils.py')
visual_utils = iutil.module_from_spec(visual_utils_spec)
visual_utils_spec.loader.exec_module(visual_utils)
import matplotlib.pyplot as plt


# Modify the following
SZ_PATH = f'{os.path.dirname(__file__)}/../data_split'


class SZLoader(object):
    def __init__(self, args, mode):
        """mode: {'train',      # train set
                  'test'}       # test set
        """
        self.t_samples = SZDataset(args, mode)

        # train, train_big
        if 'train' in mode:
            self.data = DataLoader(self.t_samples,
                                   args.batch_size,
                                   shuffle=True,
                                   # num_workers=args.num_threads,
                                   pin_memory=True,
                                   drop_last=True)

        else:
            self.data = DataLoader(self.t_samples,
                                   1,
                                   shuffle=False,
                                   # num_workers=1,
                                   pin_memory=False)


class SZDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.filenames = []
        i = 0
        while True:
            name = f'{SZ_PATH}/{mode}/{i}_scene.png'
            if not os.path.exists(name):
                break
            self.filenames.append(name)
            i += 1
        print(f'SZDataset load {i} images.')
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.dataset_path = f'{SZ_PATH}/{mode}'
        self.input_height = args.input_height
        self.input_width = args.input_width

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # img path and norm path
        img_path = self.filenames[idx]
        norm_path = f'{self.dataset_path}/{idx}_normal.png'
        mask_path = f'{self.dataset_path}/{idx}_mask.png'
        mode_name = self.mode

        # read img / normal
        img = Image.open(img_path).convert("RGB")
        if img.width != self.input_width or img.height != self.input_height:
            img = img.resize(
                size=(self.input_width, self.input_height),
                resample=Image.LANCZOS,
                reducing_gap=4.0)
        norm_gt = Image.open(norm_path).convert("RGB")
        if norm_gt.width != self.input_width or norm_gt.height != self.input_height:
            norm_gt = norm_gt.resize(
                size=(self.input_width, self.input_height),
                resample=Image.NEAREST,
                reducing_gap=4.0)
        norm_valid_mask = Image.open(mask_path).convert('L')
        if norm_valid_mask.width != self.input_width \
            or norm_valid_mask.height != self.input_height:
            norm_valid_mask = norm_valid_mask.resize(
                size=(self.input_width, self.input_height),
                resample=Image.NEAREST,
                reducing_gap=4.0)

        # to array
        img = np.array(img).astype(np.float32) / 255.0
        norm_gt = np.array(norm_gt).astype(np.uint8)
        norm_gt = ((norm_gt.astype(np.float32) / 255.0) * 2.0) - 1.0
        norm_valid_mask = np.array(norm_valid_mask).astype(np.uint8)
        # norm_valid_mask = np.logical_not(
        #     np.logical_and(
        #         np.logical_and(
        #             norm_valid_mask[:, :, 0] == 0,
        #             norm_valid_mask[:, :, 1] == 0),
        #         norm_valid_mask[:, :, 2] == 0))
        norm_valid_mask = norm_valid_mask.astype(np.bool_) #只排除0代表的天空
        norm_valid_mask = norm_valid_mask[:, :, np.newaxis]

        if 'train' in self.mode:
            # horizontal flip (default: True)
            if self.args.data_augmentation_hflip:
                DA_hflip = random.random() > 0.5
                if DA_hflip:
                    img = np.fliplr(img)
                    norm_gt = np.fliplr(norm_gt)
                    norm_valid_mask = np.fliplr(norm_valid_mask)
                    # RGB is XYZ，前X 右Y 上Z
                    norm_gt[:, :, 1] = - norm_gt[:, :, 1]

            # random crop (default: False)
            if self.args.data_augmentation_random_crop:
                img, norm_gt, norm_valid_mask = data_utils.random_crop(
                    img, norm_gt, norm_valid_mask, height=416, width=544)

            # color augmentation (default: True)
            if self.args.data_augmentation_color:
                DA_color = random.random() > 0.5
                if DA_color:
                    img = data_utils.color_augmentation(img, indoors=False)

        # to tensors
        img = self.normalize(torch.from_numpy(img.copy()).permute(2, 0, 1))            # (3, H, W)
        norm_gt = torch.from_numpy(norm_gt.copy()).permute(2, 0, 1)                    # (3, H, W)
        norm_valid_mask = torch.from_numpy(norm_valid_mask.copy()).permute(2, 0, 1)    # (1, H, W)

        # 预训练模型使用 左x，上y，后z 左手坐标系
        norm_gt = torch.stack((-norm_gt[1,:,:], norm_gt[2,:,:], -norm_gt[0,:,:]), dim=0)

        sample = {'img': img,
                  'norm': norm_gt,
                  'norm_valid_mask': norm_valid_mask,
                  'scene_name': mode_name,
                  'img_name': img_path}

        return sample


def preprocess(srcDir: str, destDir: str, width=640, height=480):
    """图片预处理：放缩图片、并将图片复制到指定目录下"""
    assert os.path.exists(srcDir)
    if os.path.exists(destDir):
        shutil.rmtree(destDir, ignore_errors=True)
    os.makedirs(destDir)
    cameraFile = f'{srcDir}/cameras.txt'
    assert os.path.exists(cameraFile)
    shutil.copy(cameraFile, destDir)
    i = 0
    while True:
        scene   = f'{srcDir}/{i}_scene.png'
        normal = f'{srcDir}/{i}_normal.png'
        mask   = f'{srcDir}/{i}_mask.png'
        seg    = f'{srcDir}/{i}_seg.png'
        if not os.path.exists(scene):
            break
        Image.open(scene).convert('RGB') \
            .resize((width, height), Image.LANCZOS, reducing_gap=4.0) \
            .save(f'{destDir}/{i}_scene.png')
        Image.open(normal).convert('RGB') \
            .resize((width, height), Image.NEAREST, reducing_gap=4.0) \
            .save(f'{destDir}/{i}_normal.png')
        Image.open(mask).convert('L') \
            .resize((width, height), Image.NEAREST, reducing_gap=4.0) \
            .save(f'{destDir}/{i}_mask.png')
        Image.open(seg).convert('RGB') \
            .resize((width, height), Image.NEAREST, reducing_gap=4.0) \
            .save(f'{destDir}/{i}_seg.png')
        i += 1
    print(f'\nHandled {i} pictures...\n')


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        exit(0)
    cmd = sys.argv[1]
    if cmd == 'moveDataset':
        preprocess(
            r'E:\Py_Projects\VCCPlaneRecon\datasets\generator\raw_data',
            r'..\data_split\train')
    elif cmd == 'testDataLoad':
        class Args:
            def __init__(self):
                self.input_width=640
                self.input_height=480
                self.data_augmentation_hflip = True
                self.data_augmentation_random_crop = False
                self.data_augmentation_color = True
        dataSet = SZDataset(Args(), 'train')
        for i, dat in enumerate(dataSet):
            # {'img'            : img,
            #  'norm'           : norm_gt,
            #  'norm_valid_mask': norm_valid_mask,
            #  'scene_name'     : mode_name,
            #  'img_name'       : img_path}
            img = dat['img'].permute(1,2,0).numpy()
            norm = dat['norm'].permute(1,2,0).numpy()
            mask = dat['norm_valid_mask'].permute(1,2,0).numpy()

            img = visual_utils.unnormalize(img)
            norm = ((norm + 1) * 0.5) * 255
            norm = np.clip(norm, a_min=0, a_max=255)
            norm = norm.astype(np.uint8)  # (B, H, W, 3)
            mask = mask.astype(np.uint8).squeeze()

            plt.imsave('t_img.png', img)
            plt.imsave('t_norm.png', norm)
            plt.imsave('t_mask.png', mask, cmap='gray')
            print(f'Save the result from dataset...')
            break
    pass