#reference from weather diffusion
import sys
sys.path.append('../')
import os
import cv2
from utils.metrics import calculate_psnr, calculate_ssim, calculate_rmse

def main():
    # Sample script to calculate PSNR and SSIM metrics from saved images in two directories
    # using calculate_psnr and calculate_ssim functions from: https://github.com/JingyunLiang/SwinIR

    gt_path = 'all/gt'
    results_path = 'all/output'

    imgsName = sorted(os.listdir(results_path))
    gtsName = sorted(os.listdir(gt_path))
    print(len(imgsName))
    print(len(gtsName))
    assert len(imgsName) == len(gtsName)
    count=0
    cumulative_psnr, cumulative_ssim = 0, 0
    rmse_total = 0
    for i in range(len(imgsName)):
        print('Processing image: %s' % (imgsName[i]))
        res = cv2.imread(os.path.join(results_path, imgsName[i]), cv2.IMREAD_COLOR)
        gt = cv2.imread(os.path.join(gt_path, gtsName[i]), cv2.IMREAD_COLOR)
        print(f"image:{imgsName[i]}, gt:{gtsName[i]}")
        cur_psnr = calculate_psnr(res, gt, test_y_channel=True)
        cur_ssim = calculate_ssim(res, gt, test_y_channel=True)
        rmse = calculate_rmse(res, gt)
        count += 1
        print('PSNR is %.4f and SSIM is %.4f, RMSE is %.4f' % (cur_psnr, cur_ssim, rmse))
        cumulative_psnr += cur_psnr
        cumulative_ssim += cur_ssim
        rmse_total += rmse
    print('Testing set, PSNR is %.4f and SSIM is %.4f, RMSE is %.4f' % (cumulative_psnr / len(imgsName), cumulative_ssim / len(imgsName), rmse_total / len(imgsName)))
    print(f"Total image:{count}")
    assert len(imgsName) == len(gtsName) and len(imgsName) == count

def measure(steps, results_path, gt_path):
    imgsName = sorted(os.listdir(results_path))
    gtsName = sorted(os.listdir(gt_path))
    assert len(imgsName) == len(gtsName)

    cumulative_psnr, cumulative_ssim = 0, 0
    for i in range(len(imgsName)):
        #logger('Processing image: %s' % (imgsName[i]))
        res = cv2.imread(os.path.join(results_path, imgsName[i]), cv2.IMREAD_COLOR)
        gt = cv2.imread(os.path.join(gt_path, gtsName[i]), cv2.IMREAD_COLOR)
        #logger(f"image:{imgsName[i]}, gt:{gtsName[i]}")
        cur_psnr = calculate_psnr(res, gt, test_y_channel=True)
        cur_ssim = calculate_ssim(res, gt, test_y_channel=True)
        #logger('PSNR is %.4f and SSIM is %.4f' % (cur_psnr, cur_ssim))
        cumulative_psnr += cur_psnr
        cumulative_ssim += cur_ssim
    return cumulative_psnr / len(imgsName), cumulative_ssim / len(imgsName)

if __name__ == "__main__":
    main()

