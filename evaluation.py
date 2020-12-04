import cv2
import numpy as np
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from imutils.paths import list_images
import imutils
import cv2
import tensorlayer as tl
import os

def load_dataset(path):
    train_hr_img_list = list(list_images(path))
    train_hr_imgs = tl.vis.read_images(train_hr_img_list,printable=False,n_threads=32)
    return train_hr_imgs

def compute_score(path_ori, path_sr):
    ori_imgs = load_dataset(path_ori)
    sr_imgs = load_dataset(path_sr)
    PSNR_scores = []
    SSIM_scores = []
    for ori_img,sr_img in zip(ori_imgs,sr_imgs):
        PSNR_scores.append(PSNR(ori_img,sr_img))
        SSIM_scores.append(SSIM(ori_img,sr_img, multichannel=True))
    mean_PSNR_score = np.mean(np.array(PSNR_scores))
    mean_SSIM_score = np.mean(np.array(SSIM_scores))
    return mean_PSNR_score, mean_SSIM_score

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-hr", "--high-reso", type=str, default='archive/Set5', help="path to high resolution images directory")
    ap.add_argument("-sr", "--super-reso", type=str, default='archive/Set5_sr', help="path to super resolution images directory")

    args = vars(ap.parse_args())

    PSNR_score, SSIM_score = compute_score(args['high_reso'], args['super_reso'])
    print('PSNR_score = ',PSNR_score)
    print('SSIM_score = ',SSIM_score)    