import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import pydicom
import random
import numpy as np
import torch


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    
    20260505：將__getitem__函式中用Image.open讀取並轉換為RGB的程式碼替換為讀取Numpy並轉為單通道Tensor的寫法
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == "BtoA"
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        
        """unpaired"""
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        
        
        """paired
        index_B = index % self.B_size
        B_path = self.B_paths[index_B]
        """

        # 使用 pydicom 讀取 DICOM 檔案
        A_dcm = pydicom.dcmread(A_path)
        B_dcm = pydicom.dcmread(B_path)
        
        # 提取像素矩陣並轉為 float32
        A_img = A_dcm.pixel_array.astype(np.float32)
        B_img = B_dcm.pixel_array.astype(np.float32)

        # 執行數值正規化
        HU_MIN = -1000.0
        HU_MAX = 3000.0
        A_img = np.clip(A_img, HU_MIN, HU_MAX)
        B_img = np.clip(B_img, HU_MIN, HU_MAX)
        A_img = 2.0 * (A_img - HU_MIN) / (HU_MAX - HU_MIN) - 1.0
        B_img = 2.0 * (B_img - HU_MIN) / (HU_MAX - HU_MIN) - 1.0
        
        # 嚴格將尺寸控制在 512x512
        def pad_or_crop_to_512(img):
            h, w = img.shape
            
            # 計算長寬距離 512 還差多少
            pad_h = max(512 - h, 0)
            pad_w = max(512 - w, 0)
            
            # 分配到上下左右 (置中 Padding)
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            
            # 使用 -1.0 (代表空氣 HU=-1000) 來填充邊緣
            img = np.pad(img, ((top, bottom), (left, right)), mode='constant', constant_values=-1.0)
            
            # 萬一遇到少數長或寬大於 512 的極端病患，進行中心裁剪 (Center Crop)
            h, w = img.shape
            if h > 512 or w > 512:
                start_h = (h - 512) // 2
                start_w = (w - 512) // 2
                img = img[start_h : start_h+512, start_w : start_w+512]
                
            return img

        # 對 A 與 B 套用尺寸校正
        A_img = pad_or_crop_to_512(A_img)
        B_img = pad_or_crop_to_512(B_img)

        # 轉為 PyTorch Tensor，並加上 Channel 維度
        A = torch.from_numpy(A_img).unsqueeze(0)
        B = torch.from_numpy(B_img).unsqueeze(0)

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
