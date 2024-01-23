import os
import random
import numpy as np
from PIL import Image
import imageio.v2 as imageio
import torch
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import data.utils as data_utils
import funcs.utils as img_utils
import matplotlib.pyplot as plt


class VCC_DatasetParams:
    def __init__(self):
        self.mode = 'train'
        self.batch_size = 1
        self.num_threads = 1
        self.data_record_file = 'data.txt'
        self.input_width = 640
        self.input_height = 480
        self.data_augmentation_hflip = False
        self.data_augmentation_random_crop = False
        self.data_augmentation_color = False


class VCC_Loader(object):
    def __init__(self, params: VCC_DatasetParams):
        self.samples = VCC_Dataset(params)
        if 'train' == params.mode:
            self.data = DataLoader(self.samples,
                                   params.batch_size,
                                   shuffle=True,
                                   num_workers=params.num_threads,
                                   pin_memory=True,
                                   drop_last=True)

        elif 'test' == params.mode:
            self.data = DataLoader(self.samples,
                                   1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False)


class VCC_Dataset(Dataset):
    def __init__(self, params: VCC_DatasetParams):
        self.params = params
        self.scene_images = []
        self.normal_images = []
        self.num_image = 0
        with open(params.data_record_file, 'rt') as f:
            while True:
                line = f.readline().strip()
                scene_file = line
                line = f.readline().strip()
                depth_file = line
                line = f.readline().strip()
                normal_file = line
                if not (os.path.exists(scene_file) and
                        os.path.exists(depth_file) and
                        os.path.exists(normal_file)):
                    break
                self.scene_images.append(scene_file)
                self.normal_images.append(normal_file)
                self.num_image += 1
        print(f'VCC_Dataset found {self.num_image} images.')
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return self.num_image

    def __getitem__(self, idx):
        scene_file = self.scene_images[idx]
        normal_file = self.normal_images[idx]

        # read img / normal
        scene_img = Image.open(scene_file).convert("RGB")
        if not (scene_img.width == self.params.input_width and
                scene_img.height == self.params.input_height):
            print(f'VCC_Dataset: expect image size is '
                  f'({self.params.input_height},{self.params.input_width}),'
                  f' but is actually '
                  f'({scene_img.width},{scene_img.height}).')
            exit(1)
        scene_img = np.array(scene_img).astype(np.float32) / 255.0  # to array
        normal_img = imageio.imread(normal_file)  # 3 channel float32 tiff
        if not (normal_img.shape[0] == self.params.input_height and
                normal_img.shape[1] == self.params.input_width):
            print(f'VCC_Dataset: expect normal.shape is '
                  f'({self.params.input_height},{self.params.input_width}),'
                  f' but is actually '
                  f'({normal_img.shape[0]},{normal_img.shape[1]}).')
            exit(1)
        invalid_mask = np.sum(np.abs(normal_img), axis=2) < 0.1
        # 右X，下Y，前Z，将所有无效的法向量变成(0,0,1)
        normal_img[invalid_mask, :] = np.array([0,0,1], dtype=np.float32)
        invalid_mask = invalid_mask[:, :, np.newaxis]
        invalid_mask.fill(False)

        if 'train' == self.params.mode:
            # horizontal flip (default: True)
            if self.params.data_augmentation_hflip:
                DA_hflip = random.random() > 0.5
                if DA_hflip:
                    scene_img = np.fliplr(scene_img)
                    normal_img = np.fliplr(normal_img)
                    invalid_mask = np.fliplr(invalid_mask)
                    # RGB is XYZ，右X，下Y，前Z
                    normal_img[:, :, 0] = - normal_img[:, :, 0]

            # random crop (default: False)
            if self.params.data_augmentation_random_crop:
                scene_img, normal_img, invalid_mask = data_utils.random_crop(
                    scene_img, normal_img, invalid_mask, height=416, width=544)

            # color augmentation (default: True)
            if self.params.data_augmentation_color:
                DA_color = random.random() > 0.5
                if DA_color:
                    scene_img = data_utils.color_augmentation(scene_img, indoors=False)

        # to tensors
        scene_img = self.normalize(torch.from_numpy(scene_img.copy()).permute(2, 0, 1))    # (3, H, W)
        normal_img = torch.from_numpy(normal_img.copy()).permute(2, 0, 1)                  # (3, H, W)
        valid_mask = torch.from_numpy(np.logical_not(invalid_mask.copy())).permute(2, 0, 1)# (1, H, W)

        # 预训练模型使用 左X，上Y，后Z 左手坐标系
        normal_img = - normal_img

        sample = {'img': scene_img,
                  'norm': normal_img,
                  'norm_valid_mask': valid_mask,
                  'scene_name': scene_file,
                  'img_name': scene_file}
        return sample


if __name__ == '__main__':
    params = VCC_DatasetParams()
    params.mode = 'train'
    params.input_width = 640
    params.input_height = 480
    params.data_augmentation_hflip = True
    params.data_augmentation_random_crop = False
    params.data_augmentation_color = True
    params.data_record_file = f'../data_split/data.txt'
    data_set = VCC_Dataset(params)
    for i, dat in enumerate(data_set):
        # {'img'            : img,
        #  'norm'           : norm_gt,
        #  'norm_valid_mask': norm_valid_mask,
        #  'scene_name'     : mode_name,
        #  'img_name'       : img_path}
        img = dat['img'].permute(1, 2, 0).numpy()
        norm = dat['norm'].permute(1, 2, 0).numpy()
        mask = dat['norm_valid_mask'].permute(1, 2, 0).squeeze().numpy()

        img = img_utils.unnormalize(img)
        norm = ((norm + 1) * 0.5) * 255
        norm = np.clip(norm, a_min=0, a_max=255)
        norm = norm.astype(np.uint8)
        mask = np.where(mask, np.uint8(255), np.uint8(0))

        plt.imsave('t_img.png', img)
        plt.imsave('t_norm.png', norm)
        plt.imsave('t_mask.png', mask, cmap='gray')
        print(f'Save the result from dataset...')
        break
