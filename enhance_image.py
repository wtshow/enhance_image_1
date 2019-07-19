import numpy as np
import os
import xml.etree.ElementTree as ET
import glob
import cv2
import random
import re
from pascal_voc_io import *
import deal_image
quantou_image_path = 'D:/job/fridge/0423quantou'
deal_image_path = 'D:/job/fridge/beef'

def write_pascalvoc_xml(ret,filename, path,imgSize):
    writer = PascalVocWriter('mixImagesSet', filename, imgSize, localImgPath=path)
    difficult = 0
    for r in ret:
        xmin = int(r[0])
        ymin = int(r[1])
        xmax = int(r[2])
        ymax = int(r[3])
        classname = r[4]
        writer.addBndBox(xmin, ymin, xmax, ymax, classname, difficult)
    writer.save(path.replace(".jpg", ".xml"))

def get_bounding_box_list(input_xml_file):
    tree = ET.parse(input_xml_file.strip())
    root = tree.getroot()
    size = root.find('size')
    box = [0,0,0,0]
    BoxList = []
    for obj in root.iter('object'):
        xmlbox = obj.find('bndbox')
        xmlbox_name = obj.find('name').text
        xmin = int(xmlbox.find('xmin').text)
        xmax = int(xmlbox.find('xmax').text)
        ymin = int(xmlbox.find('ymin').text)
        ymax = int(xmlbox.find('ymax').text)
        box = [xmin, ymin, xmax, ymax, xmlbox_name]
        BoxList.append(box)
    #print("get_bounding_box_list: get " + str(len(BoxList)) + " boxes")
    #print (BoxList)
    return BoxList
def crop_image(input_image_path, bounding_box):
    x = bounding_box[0]
    y = bounding_box[1]
    w = bounding_box[2] - bounding_box[0]
    h = bounding_box[3] - bounding_box[1]
    img = cv2.imread(input_image_path.strip())
    crop_img = img[y:y+h, x:x+w]
    return crop_img

def iou_calculation(BB_GT, BB_Pred):
    # BB_Pred = np.array(obj_struct["pred_bbox"]).astype(float)
    #print("BB_GT:", BB_GT)
    #print("BB_Pred:", BB_Pred)
    ixmin = np.maximum(BB_GT[0], BB_Pred[0])
    iymin = np.maximum(BB_GT[1], BB_Pred[1])
    ixmax = np.minimum(BB_GT[2], BB_Pred[2])
    iymax = np.minimum(BB_GT[3], BB_Pred[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih
    # union
    uni = ((BB_Pred[2] - BB_Pred[0] + 1.) * (BB_Pred[3] - BB_Pred[1] + 1.) +
           (BB_GT[2] - BB_GT[0] + 1.) *
           (BB_GT[3] - BB_GT[1] + 1.) - inters)
    overlaps = inters / uni
    return overlaps
def find_combineImg(fn,savePath,deal_image_set,deal_xml_set):
    GT_BBox = get_bounding_box_list(deal_xml_set[fn])
    combine_image_name = os.listdir(savePath)
    combine_image_num = len(combine_image_name)
    n = random.randint(3,7)
    combine_image_BBox_list = []
    combine_image_BBox_list.append(GT_BBox)
    for i in range(n):
        random_num = random.randint(0,combine_image_num-1)
        select_combineimg_name = combine_image_name[random_num]
        object_name = re.split(r'[_.]',select_combineimg_name)[-3]#如果用处理之后的图片，注意变成-3
        input_image = cv2.imread(deal_image_set[fn])
        deal_image_h,deal_image_w = input_image.shape[0:2]
        combine_iamge = cv2.imread(os.path.join(savePath,select_combineimg_name))
        h,w = combine_iamge.shape[0:2]
        attempt = 300
        while(True):
            combine_x = random.randint(0,deal_image_w-w)
            combine_y = random.randint(0,deal_image_h-h)
            combine_image_size = [combine_x,combine_y,combine_x+w,combine_y+h]
            combine_image_size.append(object_name)
            iou_result = 0
            for bbox_host in combine_image_BBox_list[0]:
                iou = iou_calculation(bbox_host,combine_image_size)
                iou_result += iou
            attempt -= 1
            if iou_result == 0:
                e = round(random.random(),2)
                f = round(random.random(),2)
                out_image = cv2.addWeighted(input_image[combine_y:combine_y+h,combine_x:combine_x+w],e,combine_iamge,f,0)
                input_image[combine_y:combine_y+h,combine_x:combine_x+w] = out_image
                output_image = input_image
                cv2.imwrite(deal_image_set[fn],input_image)
                combine_image_BBox_list[0].append(combine_image_size)
                break
            if attempt <= 0:
                print("try " + str(attempt) + " times, failed")
                return input_image, combine_image_BBox_list, False
    return output_image, combine_image_BBox_list


def iou_calculation(BB_GT, BB_Pred):
    # BB_Pred = np.array(obj_struct["pred_bbox"]).astype(float)
    #print("BB_GT:", BB_GT)
    #print("BB_Pred:", BB_Pred)
    ixmin = np.maximum(BB_GT[0], BB_Pred[0])
    iymin = np.maximum(BB_GT[1], BB_Pred[1])
    ixmax = np.minimum(BB_GT[2], BB_Pred[2])
    iymax = np.minimum(BB_GT[3], BB_Pred[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih
    # union
    uni = ((BB_Pred[2] - BB_Pred[0] + 1.) * (BB_Pred[3] - BB_Pred[1] + 1.) +
           (BB_GT[2] - BB_GT[0] + 1.) *
           (BB_GT[3] - BB_GT[1] + 1.) - inters)
    overlaps = inters / uni
    return overlaps


def main():
    f = os.listdir(quantou_image_path)
    image_set_test = []
    xml_set_test = []
    for file_name in f:
        file_pic = file_name.split('.')
        if file_pic[1] == 'jpg':
            image_set_test.append(file_name)
        else:
            xml_set_test.append(file_name)
    if len(image_set_test) != len(xml_set_test):
        print('数据不匹配')
    else:
        for i in range(len(image_set_test)):

            if os.path.split(image_set_test[i])[0] != os.path.split(xml_set_test[i])[0]:
                print('数据名不匹配')
#####################get quantou and deal image with resize、make noise、rotation and so on#######################################
    image_set = glob.glob(quantou_image_path + "/**/*.jpg", recursive=True)
    xml_set = glob.glob(quantou_image_path + "/**/*.xml", recursive = True)
    xml_num = len(xml_set)
    BBoxList = []
    present_dir = os.getcwd()
    savePath = present_dir + '/small_quantou'
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    for i in range(xml_num):
        BBox_host_list = get_bounding_box_list(xml_set[i])
        select_image_path = xml_set[i].replace('.xml','.jpg')
        if not os.path.exists(select_image_path.strip()):
            print(select_image_path+'not exit')
        list_num = len(BBox_host_list)
        for i in range(list_num):
            crop_select_image = crop_image(select_image_path, BBox_host_list[i])
            src_name = os.path.split(select_image_path)[1]
            real_path = src_name.split('.')[0]+'_'+str(i)+'_'+BBox_host_list[i][-1]+'.jpg'
            cv2.imwrite(os.path.join(savePath, real_path), crop_select_image)
    # deal_image_name = glob.glob(savePath+'/**/*.jpg')
    #deal_image.main(savePath)
    #找到要融合图片的信息
    # deal_image_set: 背景图
    # combine_image_name：将要进行融合的图（小图）
    deal_image_set = glob.glob(deal_image_path + "/**/*.jpg", recursive=True)
    deal_xml_set = glob.glob(deal_image_path + "/**/*.xml", recursive = True)
    deal_image_num = len(deal_image_set)
    for fn in range(deal_image_num):
        output_image, combine_image_BBox_list = find_combineImg(fn,savePath,deal_image_set,deal_xml_set)
        output_image_size = output_image.shape
        write_pascalvoc_xml(combine_image_BBox_list[0], deal_image_set[fn], deal_xml_set[fn], output_image_size)

if __name__ == '__main__':
    main()