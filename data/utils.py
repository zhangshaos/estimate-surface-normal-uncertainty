import random
import numpy as np
import cv2


def random_crop(img, norm, norm_mask, height, width):
    """randomly crop the input image & surface normal
    """
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y + height, x:x + width, :]
    norm = norm[y:y + height, x:x + width, :]
    norm_mask = norm_mask[y:y + height, x:x + width, :]
    return img, norm, norm_mask


def color_augmentation(image, indoors=True):
    """color augmentation
    """
    # gamma augmentation
    gamma = random.uniform(0.9, 1.1)
    image_aug = image ** gamma

    # brightness augmentation
    if indoors:
        brightness = random.uniform(0.75, 1.25)
    else:
        brightness = random.uniform(0.9, 1.1)
    image_aug = image_aug * brightness

    # color augmentation
    colors = np.random.uniform(0.9, 1.1, size=3)
    white = np.ones((image.shape[0], image.shape[1]))
    color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
    image_aug *= color_image
    image_aug = np.clip(image_aug, 0, 1)
    return image_aug


class clahe_process:
    def __init__(self):
        self.clahe = cv2.createCLAHE(6.0, (8, 8))  # 和Explore项目保持一致

    def __call__(self, image: np.ndarray) -> np.ndarray:
        assert image.dtype == np.uint8
        image_r, image_g, image_b = image[:,:,0], image[:,:,1], image[:,:,2]
        image_r = self.clahe.apply(image_r)
        image_g = self.clahe.apply(image_g)
        image_b = self.clahe.apply(image_b)
        img = np.stack([image_r, image_g, image_b], axis=2)
        return img


def norm_normalize(normals: np.ndarray):
    """normals shape is (?,?,3,...)"""
    norm_x, norm_y, norm_z = np.split(normals, 3, axis=2)
    norm = np.sqrt(norm_x ** 2.0 + norm_y ** 2.0 + norm_z ** 2.0) + 1e-10
    final_out = np.concatenate([norm_x / norm, norm_y / norm, norm_z / norm], axis=2)
    return final_out
