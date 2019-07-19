import cv2
import random
import numpy as np
import os
from math import *
import glob
savePath = 'D:/job/fridge/small_quantou'

def Gauss_noise(iamge_name_dir):
    img = cv2.imread(iamge_name_dir)
    src = img.copy()
    image = np.array(src/ 255, dtype=float)
    mean = 0
    var = 0.001
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    path = os.path.split(iamge_name_dir)
    first_path = path[1].split('.')[0] + '_1.jpg'
    final_path = os.path.join(path[0], first_path)
    cv2.imwrite(final_path,out)
def img_resize(iamge_name_dir):
    src = cv2.imread(iamge_name_dir)
    img = src.copy()
    out1 = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    path = os.path.split(iamge_name_dir)
    first_path = path[1].split('.')[0] + '_2.jpg'
    final_path = os.path.join(path[0], first_path)
    cv2.imwrite(final_path,out1)
    out2 = cv2.resize(img, (0, 0), fx=1.25, fy=1.25, interpolation=cv2.INTER_NEAREST)
    path = os.path.split(iamge_name_dir)
    first_path = path[1].split('.')[0] + '_3.jpg'
    final_path = os.path.join(path[0], first_path)
    cv2.imwrite(final_path,out2)
def img_rotation(iamge_name_dir):
    src = cv2.imread(iamge_name_dir)
    img = src.copy()
    height, width = img.shape[0:2]
    degree = random.randint(0,180)
    # 旋转后的尺寸
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2

    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    path = os.path.split(iamge_name_dir)
    first_path = path[1].split('.')[0] + '_4.jpg'
    final_path = os.path.join(path[0], first_path)
    cv2.imwrite(final_path, imgRotation)

def main(save_path):
    image_name = glob.glob(save_path + '/**/*.jpg',recursive=True)
    image_num = len(image_name)
    for i in range(image_num):
        Gauss_noise(image_name[i])
        img_resize(image_name[i])
        img_rotation(image_name[i])



