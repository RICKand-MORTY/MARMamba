import sys
sys.path.append('../')
import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.val_data_functions import ValData
from utils.metrics import calculate_psnr, calculate_ssim, calculate_rmse
import os
import numpy as np
import random
from torchvision.transforms import Compose, ToTensor, Normalize
from model.mamba import MambaFormer
import torchvision.utils as tvu
import cv2
import h5py
import PIL
import lpips
from PIL import Image


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-save_place', help='directory for saving the networks of the experiment', type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument('-checkpoint', help='select checkpoint of model', type=str)
parser.add_argument('-val_data_dir', help='test dataset path', type=str)
parser.add_argument('-test_mode', help='test mode:[no_metal, has_metal]', type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_batch_size = args.val_batch_size
save_place = args.save_place
test_mode = args.test_mode
lpips_fn = lpips.LPIPS(net='vgg').to(device)
lp_total = 0


def to_tensor(img_np):
    # img_np: numpy array [H,W,3], BGR (cv2 默认)
    # 1. 转换为 RGB
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    # 2. 转换为 float32
    img_np = img_np.astype('float32') / 127.5 - 1.0   # 映射到 [-1,1]
    # 3. 转换为 tensor，形状 [1,3,H,W]
    img_t = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0)
    return img_t

def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    tvu.save_image(img, file_directory)

def test_metal_image(data_path, model, save_place):
    metal_large = [1, 7, 6]
    metal_medium = [2, 0]
    metal_small = [9, 8, 4]
    metal_tiny = [3, 5]
    txtdir = os.path.join(data_path, 'test_640geo_dir.txt')
    #txtdir = os.path.join(data_path, 'test.txt')
    large_dir = os.path.join(save_place, 'Large')
    medium_dir = os.path.join(save_place, 'Medium')
    small_dir = os.path.join(save_place, 'Small')
    tiny_dir = os.path.join(save_place, 'Tiny')

    mat_files = open(txtdir, 'r').readlines()
    total = 0
    transform_input = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
    transform_gt = Compose([ToTensor()])
    time_list = []
    test_mask = np.load(os.path.join(data_path, 'testmask.npy'))
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
            M512 = test_mask[:, :, i]
            M = np.array(Image.fromarray(M512).resize((416, 416), PIL.Image.BILINEAR))
            Mask = M.astype(np.float32)
            #Mask = np.transpose(np.expand_dims(Mask, 2), (2, 0, 1))
            #print(f"mask.shape={Mask.shape}")
            input = transform_input(XLI)
            gt = transform_gt(Xgt)
            Mask = transform_gt(Mask)
            #print(f"mask.shape={Mask.shape}")
            input = input.unsqueeze(0).cuda()
            #print(f"input.shape={input.shape}")
            gt = gt.cuda()
            start_time = time.time()
            output = model(input)
            end_time = time.time()
            dur_time =  end_time - start_time
            #total_time += end_time
            if total > 0:
                time_list.append(dur_time)
            else:
                print("drop first!")
            total += 1
            print(f"process No.{total}.png, spend {dur_time}")
            if i in metal_large:
                save_dir = large_dir
            elif i in metal_medium:
                save_dir = medium_dir
            elif i in metal_small:
                save_dir = small_dir
            elif i in metal_tiny:
                save_dir = tiny_dir

            Mask = Mask.unsqueeze(0) 
            gt = gt.unsqueeze(0)
            #print(f"mask.shape={Mask.shape}, output.shape={output.shape}, gt.shape={gt.shape}")
            output[Mask==1] = 0
            gt[Mask==1] = 0
            
            save_image(gt, os.path.join(os.path.join(save_dir, "gt"), f"{line}_{i}.png"))
            save_image(output, os.path.join(os.path.join(save_dir, "output"), f"{line}_{i}.png"))
            save_image(Mask, os.path.join(os.path.join(save_dir, "mask"), f"{line}_{i}.png"))
    avg_time = np.mean(time_list)
    std_time = np.std(time_list)
    return total, avg_time, std_time, len(time_list)

def test_full_image(data_path, model, save_place):
    metal_large = [1, 7, 6]
    metal_medium = [2, 0]
    metal_small = [9, 8, 4]
    metal_tiny = [3, 5]
    txtdir = os.path.join(data_path, 'test_640geo_dir.txt')
    #txtdir = os.path.join(data_path, 'test.txt')
    large_dir = os.path.join(save_place, 'Large')
    medium_dir = os.path.join(save_place, 'Medium')
    small_dir = os.path.join(save_place, 'Small')
    tiny_dir = os.path.join(save_place, 'Tiny')

    mat_files = open(txtdir, 'r').readlines()
    total = 0
    transform_input = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
    transform_gt = Compose([ToTensor()])
    time_list = []
    test_mask = np.load(os.path.join(data_path, 'testmask.npy'))
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
            M512 = test_mask[:, :, i]
            M = np.array(Image.fromarray(M512).resize((416, 416), PIL.Image.BILINEAR))
            Mask = M.astype(np.float32)
            #Mask = np.transpose(np.expand_dims(Mask, 2), (2, 0, 1))
            #print(f"mask.shape={Mask.shape}")
            input = transform_input(XLI)
            gt = transform_gt(Xgt)
            Mask = transform_gt(Mask)
            #print(f"mask.shape={Mask.shape}")
            input = input.unsqueeze(0).cuda()
            #print(f"input.shape={input.shape}")
            gt = gt.cuda()
            start_time = time.time()
            output = model(input)
            end_time = time.time()
            dur_time =  end_time - start_time
            #total_time += end_time
            if total > 0:
                time_list.append(dur_time)
            else:
                print("drop first!")
            total += 1
            print(f"process No.{total}.png, spend {dur_time}")
            if i in metal_large:
                save_dir = large_dir
            elif i in metal_medium:
                save_dir = medium_dir
            elif i in metal_small:
                save_dir = small_dir
            elif i in metal_tiny:
                save_dir = tiny_dir

            Mask = Mask.unsqueeze(0) 
            gt = gt.unsqueeze(0)
            #print(f"mask.shape={Mask.shape}, output.shape={output.shape}, gt.shape={gt.shape}")
            
            save_image(gt, os.path.join(os.path.join(save_dir, "gt"), f"{line}_{i}.png"))
            save_image(output, os.path.join(os.path.join(save_dir, "output"), f"{line}_{i}.png"))
            save_image(Mask, os.path.join(os.path.join(save_dir, "mask"), f"{line}_{i}.png"))
    avg_time = np.mean(time_list)
    std_time = np.std(time_list)
    return total, avg_time, std_time, len(time_list)

# set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print('Seed:\t{}'.format(seed))

# --- Set category-specific hyper-parameters  --- #
val_data_dir = args.val_data_dir

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# --- Define the network --- #

net = MambaFormer(in_channels=1)
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)

# --- Load the network weight --- #
net.load_state_dict(torch.load(args.checkpoint))
total = sum([param.nelement() for param in net.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))
# --- Use the evaluation model in testing --- #
net.eval()


os.makedirs(args.save_place)
large_dir = os.path.join(args.save_place, 'Large')
os.makedirs(large_dir)
os.makedirs(os.path.join(large_dir, 'output'))
os.makedirs(os.path.join(large_dir, 'gt'))
os.makedirs(os.path.join(large_dir, 'mask'))
medium_dir = os.path.join(args.save_place, 'Medium')
os.makedirs(medium_dir)
os.makedirs(os.path.join(medium_dir, 'output'))
os.makedirs(os.path.join(medium_dir, 'gt'))
os.makedirs(os.path.join(medium_dir, 'mask'))
small_dir = os.path.join(args.save_place, 'Small')
os.makedirs(small_dir)
os.makedirs(os.path.join(small_dir, 'output'))
os.makedirs(os.path.join(small_dir, 'gt'))
os.makedirs(os.path.join(small_dir, 'mask'))
tiny_dir = os.path.join(args.save_place, 'Tiny')
os.makedirs(tiny_dir)
os.makedirs(os.path.join(tiny_dir, 'output'))
os.makedirs(os.path.join(tiny_dir, 'gt'))
os.makedirs(os.path.join(tiny_dir, 'mask'))


total_image = 0

print('--- Testing starts! ---')
with torch.no_grad():
    #start_time = time.time()
    if test_mode == 'has_metal':
        total_image, avg_time, std_time, len_time_list =test_full_image(val_data_dir, net, save_place)
    elif test_mode == 'no_metal':
        total_image, avg_time, std_time, len_time_list = test_metal_image(val_data_dir, net, save_place)

print('average validation time is {0:.4f}, std time is {1:.4f}, total image is {2}, time_list_len is {3}'.format(avg_time, std_time, total_image, len_time_list))

with open(os.path.join(args.save_place, "log.txt"), "a+") as f:
    f.write('average validation time is {0:.4f}, std time is {1:.4f}, total image is {2}, time_list_len is {3}\n'.format(avg_time, std_time, total_image, len_time_list))
    

print('--- Evaluating! ---')
for dir in [large_dir, medium_dir, small_dir, tiny_dir]:
    ssim_list = []
    psnr_list = []
    rmse_list = []
    lpips_list = []
    count = 0
    results_path = os.path.join(dir, "output")
    gt_path = os.path.join(dir, "gt")
    mask_path = os.path.join(dir, "mask")
    imgsName = sorted(os.listdir(results_path))
    gtsName = sorted(os.listdir(gt_path))
    masksName = sorted(os.listdir(mask_path))
    assert len(imgsName) == len(gtsName) and len(imgsName) == len(masksName)
    print(f"dir={dir}, total_image={len(imgsName)}")
    
    for i in range(len(imgsName)):
        res = cv2.imread(os.path.join(results_path, imgsName[i]), cv2.IMREAD_COLOR)
        gt = cv2.imread(os.path.join(gt_path, gtsName[i]), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(mask_path, masksName[i]), cv2.IMREAD_COLOR)
        cur_psnr = calculate_psnr(res, gt, test_y_channel=True)
        cur_ssim = calculate_ssim(res, gt, test_y_channel=True)
        rmse = calculate_rmse(res, gt)
        res_t = to_tensor(res).to(device)
        gt_t = to_tensor(gt).to(device)
        with torch.no_grad():
            lp = lpips_fn(res_t, gt_t)
            lp_val = lp.item()
        count += 1
        del res_t, gt_t, lp
        torch.cuda.empty_cache()
        print('PSNR is %.4f and SSIM is %.4f, RMSE is %.4f, LPIPS is %.4f' % (cur_psnr, cur_ssim, rmse, lp_val))
        ssim_list.append(cur_ssim)
        psnr_list.append(cur_psnr)
        rmse_list.append(rmse)
        lpips_list.append(lp_val)
    
    print(f"Total image:{count}")
    assert len(imgsName) == len(gtsName) and len(imgsName) == count
    ssim_mean, ssim_std   = np.mean(ssim_list), np.std(ssim_list)
    psnr_mean, psnr_std   = np.mean(psnr_list), np.std(psnr_list)
    rmse_mean, rmse_std   = np.mean(rmse_list), np.std(rmse_list)
    lpips_mean, lpips_std = np.mean(lpips_list), np.std(lpips_list)
    print(f"SSIM:  {ssim_mean:.4f} ± {ssim_std:.4f}")
    print(f"PSNR:  {psnr_mean:.2f} ± {psnr_std:.2f} dB")
    print(f"RMSE:  {rmse_mean:.2f} ± {rmse_std:.2f}")
    print(f"LPIPS: {lpips_mean:.4f} ± {lpips_std:.4f}")

    
    with open(os.path.join(args.save_place, "log.txt"), "a+") as f:
        f.write(f"{dir}:\n")
        f.write(f"SSIM:  {ssim_mean:.4f} ± {ssim_std:.4f}\n")
        f.write(f"PSNR:  {psnr_mean:.2f} ± {psnr_std:.2f} dB\n")
        f.write(f"RMSE:  {rmse_mean:.2f} ± {rmse_std:.2f}\n")
        f.write(f"LPIPS: {lpips_mean:.4f} ± {lpips_std:.4f}\n")
        f.write('image counts is {0}\n'.format(len(imgsName)))
        f.write('-----------------------------------------------------------------------------------------------\n')
        f.close()
