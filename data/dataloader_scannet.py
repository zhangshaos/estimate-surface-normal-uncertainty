import random
import numpy as np
from PIL import Image
import os
import shutil
import sys

import torch
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader

import data.utils as data_utils
import matplotlib.pyplot as plt


# Modify the following
SCANNET_PATH = 'E:/datasets/scannet/data'


class ScannetLoader(object):
    def __init__(self, args, mode):
        """mode: {'train',      # train set
                  'test'}       # test set
        """
        self.t_samples = ScannetDataset(args, mode)

        # train, test
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


class ScannetDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.filenames = []
        i = 0
        while True:
            name = f'{SCANNET_PATH}/{mode}/{i}_scene.png'
            if not os.path.exists(name):
                break
            self.filenames.append(name)
            i += 1
        print(f'ScannetDataset load {i} images.')
        self.mode = mode
        if args.use_clahe:
            # 使用clahe图像增强算法保证图像白平衡
            self.clahe = data_utils.clahe_process()
        else:
            self.clahe = None
        self.dataset_path = f'{SCANNET_PATH}/{mode}'
        self.input_height = args.input_height
        self.input_width = args.input_width

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # img path and norm path
        img_path = self.filenames[idx]
        norm_path = f'{self.dataset_path}/{idx}_normal.png'
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

        # to array
        if self.clahe is not None:
            img = np.array(img, np.uint8)
            img = self.clahe(img)
            img = img.astype(np.float32) / 255.0
        else:
            img = np.array(img).astype(np.float32) / 255.0
        norm_gt = np.array(norm_gt).astype(np.uint8)
        norm_gt = ((norm_gt.astype(np.float32) / 255.0) * 2.0) - 1.0
        norm_valid_mask = (np.abs(norm_gt).sum(-1) >= 0.5)
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
        img = torch.from_numpy(img.copy()).permute(2, 0, 1)                            # (3, H, W)
        norm_gt = data_utils.norm_normalize(norm_gt)
        norm_gt = torch.from_numpy(norm_gt.copy()).permute(2, 0, 1)                    # (3, H, W)
        norm_valid_mask = torch.from_numpy(norm_valid_mask.copy()).permute(2, 0, 1)    # (1, H, W)

        # 预训练模型使用 左x，上y，后z 左手坐标系
        norm_gt = norm_gt * -1

        sample = {'img': img,
                  'norm': norm_gt,
                  'norm_valid_mask': norm_valid_mask,
                  'scene_name': mode_name,
                  'img_name': img_path}

        return sample


# 数据预处理：放缩图片、并将图片复制到指定目录下，并构建数据集
def preprocess(srcDir: str, destDir: str, width=640, height=480):
    assert os.path.exists(srcDir)
    dst_train_dir, dst_test_dir = f'{destDir}/train', f'{destDir}/test'
    if os.path.exists(dst_train_dir):
        shutil.rmtree(dst_train_dir, ignore_errors=True)
    os.makedirs(dst_train_dir)
    if os.path.exists(dst_test_dir):
        shutil.rmtree(dst_test_dir, ignore_errors=True)
    os.makedirs(dst_test_dir)
    train_i = 0
    test_i = 0
    for cur_dir, dirs, files in os.walk(srcDir):
        for f in files:
            if 'color' not in f:
                continue
            color_file: str = f'{cur_dir}/{f}'
            normal_file: str = color_file.replace('color', 'normal')
            if not os.path.exists(normal_file):
                continue
            scene = Image.open(color_file).convert('RGB')
            if scene.height != height or scene.width != width:
                scene = scene.resize((width, height), Image.LANCZOS, reducing_gap=4.0)
            normal = Image.open(normal_file).convert('RGB')
            if normal.height != height or normal.width != width:
                normal = normal.resize((width, height), Image.NEAREST, reducing_gap=4.0)
            # 80%的图片作为训练集，20%作为测试集
            if random.random() > 0.2:
                scene.save(f'{dst_train_dir}/{train_i}_scene.png')
                normal.save(f'{dst_train_dir}/{train_i}_normal.png')
                train_i += 1
            else:
                scene.save(f'{dst_test_dir}/{test_i}_scene.png')
                normal.save(f'{dst_test_dir}/{test_i}_normal.png')
                test_i += 1
    print(f'\nHandled {train_i + test_i} pictures...\n'
          f'train={train_i}; test={test_i}\n')


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        exit(0)
    cmd = sys.argv[1]
    if cmd == 'moveDataset':
        preprocess(
            r'E:\datasets\scannet\scannet-frames',
            r'E:\datasets\scannet\data')
    elif cmd == 'testDataLoad':
        class Args:
            def __init__(self):
                self.input_width=640
                self.input_height=480
                self.data_augmentation_hflip = True
                self.data_augmentation_random_crop = False
                self.data_augmentation_color = True
                self.use_clahe = False
        dataSet = ScannetDataset(Args(), 'train')
        for i, dat in enumerate(dataSet):
            # {'img'            : img,
            #  'norm'           : norm_gt,
            #  'norm_valid_mask': norm_valid_mask,
            #  'scene_name'     : mode_name,
            #  'img_name'       : img_path}
            img = dat['img'].permute(1,2,0).numpy()
            norm = dat['norm'].permute(1,2,0).numpy()
            mask = dat['norm_valid_mask'].permute(1,2,0).numpy()

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