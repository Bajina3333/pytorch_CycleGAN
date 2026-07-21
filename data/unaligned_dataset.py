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
        
    def load_medical_slice(self, path):
        """通用載入函式：自動判斷副檔名並返回範圍 [-1, 1] 的 2D float32 矩陣"""
        ext = os.path.splitext(path)[1].lower()
        
        # 情況 A：前處理好的 .npy 檔案
        if ext == '.npy':
            img = np.load(path)
            
        # 情況 B：原始 .dcm 檔案
        elif ext in ['.dcm', '.dicom']:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array.astype(np.float32)
            
            # 還原真實 HU 值 (Slope / Intercept)
            slope = getattr(dcm, 'RescaleSlope', 1.0)
            intercept = getattr(dcm, 'RescaleIntercept', 0.0)
            img = img * slope + intercept
            
            # 進行 HU 值 Clip 與歸一化到 [-1, 1]
            HU_MIN, HU_MAX = -1000.0, 3000.0
            img = np.clip(img, HU_MIN, HU_MAX)
            img = 2.0 * (img - HU_MIN) / (HU_MAX - HU_MIN) - 1.0
        else:
            raise ValueError(f"不支援的檔案格式: {path}")

        # 尺寸保護：確保矩陣大小為 512x512
        h, w = img.shape
        if h != 512 or w != 512:
            pad_h = max(512 - h, 0)
            pad_w = max(512 - w, 0)
            top, left = pad_h // 2, pad_w // 2
            bottom, right = pad_h - top, pad_w - left
            img = np.pad(img, ((top, bottom), (left, right)), mode='constant', constant_values=-1.0)
            h, w = img.shape
            if h > 512 or w > 512:
                sh, sw = (h - 512) // 2, (w - 512) // 2
                img = img[sh:sh+512, sw:sw+512]
                
        return img

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

        # 自動依據檔案類型載入並前處理
        A_img = self.load_medical_slice(A_path)
        B_img = self.load_medical_slice(B_path)

        # 轉為 PyTorch Single-Channel Tensor (1, 512, 512)
        A = torch.from_numpy(A_img).unsqueeze(0).float()
        B = torch.from_numpy(B_img).unsqueeze(0).float()

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
