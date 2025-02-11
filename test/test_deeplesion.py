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
from PIL import Image


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-save_place', help='directory for saving the networks of the experiment', type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument('-checkpoint', help='select checkpoint of model', type=str)
parser.add_argument('-val_data_dir', help='test dataset path', type=str)
parser.add_argument('-test_mode', help='test mode:[metal, full_image]', type=str)
args = parser.parse_args()

val_batch_size = args.val_batch_size
save_place = args.save_place
test_mode = args.test_mode


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
    total_time = 0
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
            end_time = time.time() - start_time
            total_time += end_time
            total += 1
            print(f"process No.{total}.png, spend {end_time}")
            if i in metal_large:
                save_dir = large_dir
            elif i in metal_medium:
                save_dir = medium_dir
            elif i in metal_small:
                save_dir = small_dir
            elif i in metal_tiny:
                save_dir = tiny_dir

            save_image(gt, os.path.join(os.path.join(save_dir, "gt"), f"{line}_{i}.png"))
            save_image(output, os.path.join(os.path.join(save_dir, "output"), f"{line}_{i}.png"))
            save_image(Mask, os.path.join(os.path.join(save_dir, "mask"), f"{line}_{i}.png"))

    return total, total_time / total

def test_image(data_path, model, save_place):
    txtdir = os.path.join(data_path, 'test_640geo_dir.txt')
    #txtdir = os.path.join(data_path, 'test.txt')
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
            print(f"process No.{total}.png, spend {end_time}")
            #Not use plt.imsave since it will auto-scale the pixels
            save_image(gt, os.path.join(os.path.join(save_place, "gt"), f"{line}_{i}.png"))
            save_image(output, os.path.join(os.path.join(save_place, "output"), f"{line}_{i}.png"))
    return total, total_time / total

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

if test_mode == 'full_image':
    if os.path.exists(args.save_place) == False:
        os.makedirs(args.save_place)
        os.makedirs(os.path.join(args.save_place, 'output'))
        os.makedirs(os.path.join(args.save_place, 'gt'))

elif test_mode == 'metal':
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
time_avg = 0

print('--- Testing starts! ---')
with torch.no_grad():
    #start_time = time.time()
    if test_mode == 'full_image':
        total_image, time_avg=test_image(val_data_dir, net, save_place)
    elif test_mode == 'metal':
        total_image, time_avg = test_metal_image(val_data_dir, net, save_place)
    #end_time = time.time() - start_time
    #print(f"process {total_image}.png, spend {time_avg}")

print('average validation time is {0:.4f}, total image is {1}'.format(time_avg, total_image))

with open(os.path.join(args.save_place, "log.txt"), "a+") as f:
    f.write('average validation time is {0:.4f}, total image is {1}\n'.format(time_avg, total_image))
    
cumulative_psnr, cumulative_ssim = 0, 0
rmse_all = 0

print('--- Evaluating! ---')
if test_mode == 'full_image':
    results_path = os.path.join(save_place, "output")
    gt_path = os.path.join(save_place, "gt")
    imgsName = sorted(os.listdir(results_path))
    gtsName = sorted(os.listdir(gt_path))
    assert len(imgsName) == len(gtsName) and len(imgsName) == total_image
    print(f"len(imgsName)={len(imgsName)}, total_image={total_image}")

    for i in range(len(imgsName)):
        res = cv2.imread(os.path.join(results_path, imgsName[i]), cv2.IMREAD_COLOR)
        gt = cv2.imread(os.path.join(gt_path, gtsName[i]), cv2.IMREAD_COLOR)
        cur_psnr = calculate_psnr(res, gt, test_y_channel=True)
        cur_ssim = calculate_ssim(res, gt, test_y_channel=True)
        rmse = calculate_rmse(res, gt)
        print('PSNR is %.4f and SSIM is %.4f, RMSE is %.4f' % (cur_psnr, cur_ssim, rmse))
        cumulative_psnr += cur_psnr
        cumulative_ssim += cur_ssim
        rmse_all += rmse

    print('psnr_avg: {0:.4f}, ssim_avg: {1:.4f}, rmse_avg: {2:.4f}'.format(cumulative_psnr / len(imgsName), cumulative_ssim / len(imgsName), rmse_all / len(imgsName)))

    with open(os.path.join(args.save_place, "log.txt"), "a+") as f:
        f.write('psnr_avg: {0:.4f}, ssim_avg: {1:.4f}, rmse_avg: {2:.4f}\n'.format(cumulative_psnr / len(imgsName), cumulative_ssim / len(imgsName), rmse_all / len(imgsName)))
        f.write('average validation time is {0:.4f}, total image is {1}\n'.format(time_avg, total_image))
        f.close()

elif test_mode == 'metal':
    for dir in [large_dir, medium_dir, small_dir, tiny_dir]:
        results_path = os.path.join(dir, "output")
        gt_path = os.path.join(dir, "gt")
        mask_path = os.path.join(dir, "mask")
        imgsName = sorted(os.listdir(results_path))
        gtsName = sorted(os.listdir(gt_path))
        masksName = sorted(os.listdir(mask_path))
        assert len(imgsName) == len(gtsName) and len(imgsName) == len(masksName)
        print(f"Large metal: total_image={len(imgsName)}")

        cumulative_psnr, cumulative_ssim = 0, 0
        rmse_all = 0
        for i in range(len(imgsName)):
            res = cv2.imread(os.path.join(results_path, imgsName[i]), cv2.IMREAD_COLOR)
            gt = cv2.imread(os.path.join(gt_path, gtsName[i]), cv2.IMREAD_COLOR)
            mask = cv2.imread(os.path.join(mask_path, masksName[i]), cv2.IMREAD_COLOR)
            cur_psnr = calculate_psnr(res * (1 - mask), gt * (1 - mask), test_y_channel=True)
            cur_ssim = calculate_ssim(res * (1 - mask), gt * (1 - mask), test_y_channel=True)
            rmse = calculate_rmse(res * (1 - mask), gt * (1 - mask))
            print('PSNR is %.4f and SSIM is %.4f, RMSE is %.4f' % (cur_psnr, cur_ssim, rmse))
            cumulative_psnr += cur_psnr
            cumulative_ssim += cur_ssim
            rmse_all += rmse

        print('psnr_avg: {0:.4f}, ssim_avg: {1:.4f}, rmse_avg: {2:.4f}'.format(cumulative_psnr / len(imgsName),
                                                                               cumulative_ssim / len(imgsName),
                                                                               rmse_all / len(imgsName)))
        with open(os.path.join(args.save_place, "log.txt"), "a+") as f:
            f.write(f"{dir}:\n")
            f.write('psnr_avg: {0:.4f}, ssim_avg: {1:.4f}, rmse_avg: {2:.4f}\n'.format(cumulative_psnr / len(imgsName),
                                                                                       cumulative_ssim / len(imgsName),
                                                                                       rmse_all / len(imgsName)))
            f.write('image counts is {0}\n'.format(len(imgsName)))
            f.write('-----------------------------------------------------------------------------------------------\n')
            f.close()