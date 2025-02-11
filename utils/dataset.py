import os
import os.path
import numpy as np
import random
import h5py
import torch
import torch.utils.data as udata
import PIL.Image as Image
from numpy.random import RandomState
import scipy.io as sio
import PIL
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.utils as tvu
import time

def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    tvu.save_image(img, file_directory)

def image_get_minmax():
    return 0.0, 1.0

def proj_get_minmax():
    return 0.0, 4.0

def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    data = data.astype(np.float32)
    data = data*255.0
    data = np.transpose(np.expand_dims(data, 2), (2, 0, 1))
    return data

def test_image(data_path, model):
    #txtdir = os.path.join(data_path, 'test_640geo_dir.txt')
    txtdir = os.path.join(data_path, 'test.txt')
    mat_files = open(txtdir, 'r').readlines()
    total = 0
    transform_input = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
    transform_gt = Compose([ToTensor()])
    total_time = 0
    for line in range(len(mat_files)):
        gt_dir = mat_files[line]
        file_dir = gt_dir[:-6]
        for i in range(10):
            data_file = file_dir + str(i) + '.h5'
            abs_dir = os.path.join(data_path, 'test_640geo/', data_file)
            gt_absdir = os.path.join(data_path, 'test_640geo/', gt_dir[:-1])
            #print(f"gt_absdir = {gt_absdir}, abs_dir={abs_dir}")
            gt_file = h5py.File(gt_absdir, 'r')
            Xgt = gt_file['image'][()]
            gt_file.close()
            file = h5py.File(abs_dir, 'r')
            XLI = file['ma_CT'][()]
            file.close()
            input = transform_input(XLI)
            gt = transform_gt(Xgt)
            input = input.unsqueeze(0).cuda()
            #print(f"input.shape={input.shape}")
            gt = gt.cuda()
            start_time = time.time()
            output = model(input)
            end_time = time.time() - start_time
            total_time += end_time
            total += 1
            save_image(gt, os.path.join("./eva/gt", f"{line}_{i}.png"))
            save_image(output, os.path.join("./eva/output", f"{line}_{i}.png"))
    return total, total_time / total


class MARTrainDataset(udata.Dataset):
    def __init__(self, crop_size, train_data_dir, random_flip, random_rotate):
        super().__init__()
        self.dir = train_data_dir
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.crop_size = crop_size
        self.txtdir = os.path.join(self.dir, 'train_640geo_dir.txt')
        self.mat_files = open(self.txtdir, 'r').readlines()
        self.file_num = len(self.mat_files)
        self.rand_state = RandomState(66)
    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        crop_width, crop_height = self.crop_size
        gt_dir = self.mat_files[idx]
        #random_mask = random.randint(0, 89)  # include 89
        random_mask = random.randint(0, 9)  # for demo
        file_dir = gt_dir[:-6]
        data_file = file_dir + str(random_mask) + '.h5'
        abs_dir = os.path.join(self.dir, 'train_640geo/', data_file)
        gt_absdir = os.path.join(self.dir,'train_640geo/', gt_dir[:-1])
        gt_file = h5py.File(gt_absdir, 'r')
        Xgt = gt_file['image'][()]
        gt_file.close()
        file = h5py.File(abs_dir, 'r')
        XLI =file['ma_CT'][()]
        file.close()

        width, height = XLI.shape
        input_img = XLI
        gt_img = Xgt
        if width < crop_width and height < crop_height:
            input_img = XLI.resize((crop_width, crop_height), Image.LANCZOS)
            gt_img = Xgt.resize((crop_width, crop_height), Image.LANCZOS)
        elif width < crop_width:
            input_img = XLI.resize((crop_width, height), Image.LANCZOS)
            gt_img = Xgt.resize((crop_width, height), Image.LANCZOS)
        elif height < crop_height:
            input_img = XLI.resize((width, crop_height), Image.LANCZOS)
            gt_img = Xgt.resize((width, crop_height), Image.LANCZOS)

        width, height = input_img.shape
        # random crop
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        input_crop_img = input_img[y:y + crop_height, x:x + crop_width]
        gt_crop_img = gt_img[y:y + crop_height, x:x + crop_width]

        transform_input = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_crop_img)
        gt = transform_gt(gt_crop_img)

        if list(input_im.shape)[0] != 1 or list(gt.shape)[0] != 1:
            raise Exception('input.shape={}, gt.shape={}'.format(input_crop_img.shape, gt_crop_img.shape))

        if self.random_flip and random.random() < 0.5:
            input_im = torch.flip(input_im, dims=[-1])
            gt = torch.flip(gt, dims=[-1])

        if self.random_rotate:
            r = random.random()
            # 随机旋转
            if r < 0.25:
                # 90
                input_im = torch.rot90(input_im, k=1, dims=(1, 2))
                gt = torch.rot90(gt, k=1, dims=(1, 2))

            elif r < 0.5:
                # 270
                input_im = torch.rot90(input_im, k=3, dims=(1, 2))
                gt = torch.rot90(gt, k=3, dims=(1, 2))
            elif r < 0.75:
                # 180
                input_im = torch.rot90(input_im, k=2, dims=(1, 2))
                gt = torch.rot90(gt, k=2, dims=(1, 2))
        
        return input_im, gt
"""
def test_image(data_path, model):
    txtdir = os.path.join(data_path, 'test_640geo_dir.txt')
    mat_files = open(txtdir, 'r').readlines()
    total = 0
    transform_input = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
    transform_gt = Compose([ToTensor()])
    for line in range(len(mat_files)):
        gt_dir = mat_files[line]
        file_dir = gt_dir[:-6]
        for i in range(0, 9):
            data_file = file_dir + str(i) + '.h5'
            abs_dir = os.path.join(data_path, 'test_640geo/', data_file)
            gt_absdir = os.path.join(data_path, 'test_640geo/', gt_dir[:-1])
            #print(f"gt_absdir = {gt_absdir}, abs_dir={abs_dir}")
            gt_file = h5py.File(gt_absdir, 'r')
            Xgt = gt_file['image'][()]
            gt_file.close()
            file = h5py.File(abs_dir, 'r')
            XLI = file['LI_CT'][()]
            file.close()
            input = transform_input(XLI)
            gt = transform_gt(Xgt)
            input_im_freq = torch.fft.fft2(input) 
            input = torch.cat([input_im_freq.real, input_im_freq.imag], dim=0)
            #print(f"input.shape={input.shape}")
            input = input.unsqueeze(0).cuda()
            gt = gt.cuda()
            output = model(input)
            output = output[:, 0, :, :] + 1j * output[:, 1, :, :]
            output = torch.fft.ifft2(output)
            output = output.real * 0.5 + 0.5
            total += 1
            save_image(gt, os.path.join("./eva/gt", f"{line}_{i}.png"))
            save_image(output, os.path.join("./eva/output", f"{line}_{i}.png"))
    return total
"""