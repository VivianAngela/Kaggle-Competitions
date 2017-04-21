from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2
import numpy as np

def usage():
    if (len(sys.argv) != 2):
        print('Usage: resize_rotate_imgs.py <TOS_DIR>')
        sys.exit(3)

    tos_dir = sys.argv[1]
    return tos_dir

# 680 * 512

def resize_src_img(img_path):
    image = cv2.imread(img_path)
    dim = (680, 512)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
    name = img_path.split('/')[1 : ][-1]
    cv2.imwrite(edited_imgs + "/" + name, resized)
    print("[INFO]: resized image %s" % name)


def rotate_src_img(img_path):
    image = cv2.imread(img_path)
    height, width = image.shape[ : 2]
    dim = (680, 512)
    if width > height:
        center = (width / 2, height / 2)
        matrix_2d = cv2.getRotationMatrix2D(center, 90, 1.0)
        rotated = cv2.warpAffine(image, matrix_2d, (width, height))
        name = img_path.split('/')[1 : ][-1]
        cv2.imwrite(edited_imgs + "/" + name, rotated)
        print("[INFO]: rotated image %s" % name)

def rotate_resize_src_img(img_path):
    image = cv2.imread(img_path)
    height, width = image.shape[ : 2]
    dim = (680, 512)
    if width > height:
        center = (width / 2, height / 2)
        matrix_2d = cv2.getRotationMatrix2D(center, 90, 1.0)
        rotated = cv2.warpAffine(image, matrix_2d, (width, height))
        resized = cv2.resize(rotated, dim, interpolation=cv2.INTER_LINEAR)
        name = img_path.split('/')[1 : ][-1]
        cv2.imwrite(edited_imgs + "/" + name, resized)
        print("[INFO]: rotated and resized image %s" % name)
    else:
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
        name = img_path.split('/')[1 : ][-1]
        cv2.imwrite(edited_imgs + "/" + name, resized)
        print("[INFO]: resized image %s" % name)

def edit_src_imgs(src_dir):
    for lists in os.listdir(src_dir):
        obj = os.path.join(src_dir, lists)
        if os.path.isdir(obj):
            edit_src_imgs(obj)
        else:
            if obj.endswith(".jpg"):
                # rotate_src_img(obj)
                # resize_src_img(obj)
                rotate_resize_src_img(obj)
                print("=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=")


if __name__ == "__main__":
    print("[INFO]: resizing and rotating images")
    tos_dir = usage()
    print("[INFO]: tos_dir: %s" % tos_dir)
    edited_imgs = '../edited_imgs'
    if os.path.exists(edited_imgs):
        print("[INFO]: folder of the edited images exists: %s" % edited_imgs)
    else:
        try:
            os.mkdir(edited_imgs)
            print("[INFO]: created folder %s for the edited images" % edited_imgs)
        except:
            print("[ERROR]: Failed to create folder %s for the edited images" % edited_imgs)
            sys.exit(1)

    try:
        src_dir = '%s' % tos_dir
        edit_src_imgs(src_dir)
    except IOError as detail:
        print("[ERROR]: ", detail)
    except ValueError:
        raise
    except:
        raise
    else:
        print("[INFO]: edited all images, exit")
