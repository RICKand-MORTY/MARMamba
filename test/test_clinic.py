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
import nibabel
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description="CLINIC-metal test")
parser.add_argument("--checkpoint", type=str, help='path to checkpoint')
parser.add_argument("--data_path", type=str, help='path to CLINIC-metal dataset')
parser.add_argument("--save_path", type=str, default="./save_real", help='path to save results')
parser.add_argument('-seed', help='set random seed', default=19, type=int)
opt = parser.parse_args()
model_dir = opt.checkpoint
data_path = opt.data_path
save_path = opt.save_path

os.makedirs(data_path, exist_ok=True)
os.makedirs(save_path, exist_ok=True)

output_dir = save_path + '/output/'
input_dir = save_path + '/input_hu/'
output_hu_dir = save_path + '/output_hu/'
mask_dir = save_path + '/mask/'
#nometal_dir = save_path + '/no_metal_input/'
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_hu_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)
#os.makedirs(nometal_dir, exist_ok=True)

# set seed
seed = opt.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print('Seed:\t{}'.format(seed))

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #

net = MambaFormer(in_channels=1)
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)

# --- Load the network weight --- #
net.load_state_dict(torch.load(model_dir))
total = sum([param.nelement() for param in net.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))
# --- Use the evaluation model in testing --- #
net.eval()

transform_save = Compose([ToTensor()])
transform_input = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])

def save_image(file_directory, img):
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
    data = data * 255.0
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)),0)
    return data
    
def tohu(X):           # display window as [-175HU, 275HU]
    CT = (X - 0.192) * 1000 / 0.192
    CT_win = CT.clamp_(-175, 275)
    CT_winnorm = (CT_win +175) / (275+175)
    return CT_winnorm

print('--- Saving images! ---')
count_nii = 0
for file_name in os.listdir(data_path):
    print(f"---------{file_name}---------")
    file_path = data_path+'/'+file_name
    img_nii = nibabel.load(file_path)
    img_data = img_nii.get_fdata()
    total_img = img_data.shape[2]
    mask_thre = 2500 /1000 * 0.192 + 0.192
    #print(f"shape={img_data.shape}")
    for id in range(total_img):
        #input = img_data[..., id]
        input = np.zeros((416, 416), dtype='float32')
        image = np.array(Image.fromarray(img_data[:,:,id]).resize((416, 416), PIL.Image.BILINEAR))
        image[image < -1000] = -1000
        image = image / 1000 * 0.192 + 0.192
        M  = np.zeros((416, 416), dtype='float32')
        [rowindex, colindex] = np.where(image > mask_thre)
        M[rowindex, colindex] = 1
        input = image
        ori_input = torch.Tensor(np.copy(image))
        #remove metal region
        no_metal = 1 - M
        input = image * no_metal
        #input = normalize(input, image_get_minmax()) 
        input = transform_input(input)
        input = input.unsqueeze(0).cuda()
        output = net(input)
        output = output.cpu()
        no_metal = torch.Tensor(no_metal)
        output = output * no_metal + ori_input * (1 - no_metal)
        output_hu = tohu(output)
        ori_input_hu = tohu(ori_input)
        #ori_input_hu = tohu(ori_input)
        
        #image_input = normalize(image, image_get_minmax()) 
        #output = normalize(output.data.cpu().numpy(), image_get_minmax())
        #output = output / 255.0
        #print(f"shape={input.shape}")

        #save_image(input_dir + str(count_nii) +'_' + str(id) + '.png', torch.Tensor(image))
        #save_image(output_dir + str(count_nii) +'_' + str(id) + '.png', output)
        
        #save_image(input_dir + str(count_nii) +'_' + str(id) + '.png', torch.Tensor(image_input))
        #save_image(output_dir + str(count_nii) +'_' + str(id) + '.png', torch.Tensor(output))
        plt.imsave(input_dir + str(count_nii) +'_' + str(id) + '.png', ori_input_hu, cmap="gray")
        plt.imsave(mask_dir + str(count_nii) +'_' + str(id) + '.png', M, cmap="gray")
        plt.imsave(output_dir + str(count_nii) +'_' + str(id) + '.png', output.data.numpy().squeeze(), cmap="gray")
        plt.imsave(output_hu_dir + str(count_nii) +'_' + str(id) + '.png', output_hu.data.numpy().squeeze(), cmap="gray")
        print(f"test {id}.png")
    count_nii += 1
print("Saving finish!")
print('--- Testing Start! ---')

"""
with torch.no_grad():
    for name in os.listdir(data_path):
        file = data_path + '/' + name
        image = Image.open(file).convert('L')
        input = transform_input(image)
        input = input.unsqueeze(0).cuda()
        output = net(input)
        #plt.imsave(os.path.join(output_dir, name), output.data.cpu().numpy().squeeze(), cmap="gray")
        plt.imsave(os.path.join(output_dir, name), output.data.cpu().numpy().squeeze(), cmap="gray")
        #save_image(os.path.join(output_dir, name), output)

"""

print("Test finish!")
        
        
    


