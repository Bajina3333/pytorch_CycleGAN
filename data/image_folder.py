"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data
from pathlib import Path
from PIL import Image
import pydicom
import numpy as np
import torch
from data.base_dataset import BaseDataset

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tif",
    ".TIF",
    ".tiff",
    ".TIFF",
    ".dcm", ".DCM",
    ".dicom", ".DICOM",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    dir_path = Path(dir)
    assert dir_path.is_dir(), f"{dir} is not a valid directory"

    for path in sorted(dir_path.rglob("*")):
        if path.is_file() and is_image_file(path.name):
            images.append(str(path))
    return images[: min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert("RGB")


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False, loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n" "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        # 使用 pydicom 讀取 DICOM 檔案
        A_dcm = pydicom.dcmread(A_path)
        B_dcm = pydicom.dcmread(B_path)
        
        # 提取像素矩陣並轉為 float32
        A_img = A_dcm.pixel_array.astype(np.float32)
        B_img = B_dcm.pixel_array.astype(np.float32)

        # 定義醫療 HU 值範圍
        HU_MIN = -1000.0
        HU_MAX = 3000.0
        
        # 對 A (CBCT) 與 B (CT) 進行極端值截斷，避免金屬偽影的超高數值破壞比例
        A_img = np.clip(A_img, HU_MIN, HU_MAX)
        B_img = np.clip(B_img, HU_MIN, HU_MAX)
        
        # 線性映射到 CycleGAN 規定的 [-1, 1] 區間
        A_img = 2.0 * (A_img - HU_MIN) / (HU_MAX - HU_MIN) - 1.0
        B_img = 2.0 * (B_img - HU_MIN) / (HU_MAX - HU_MIN) - 1.0

        # 轉為 PyTorch Tensor，並加上 Channel 維度變成 [3]
        A = torch.from_numpy(A_img).unsqueeze(0)
        B = torch.from_numpy(B_img).unsqueeze(0)
        
        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        return len(self.imgs)
