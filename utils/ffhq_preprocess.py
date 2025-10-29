import os
import cv2
import time
import glob
import argparse
import scipy
import numpy as np
from PIL import Image
from tqdm import tqdm
from itertools import cycle
from torch.multiprocessing import Pool, Process, set_start_method


"""
brief: face alignment with FFHQ method (https://github.com/NVlabs/ffhq-dataset)
author: lzhbrian (https://lzhbrian.me)
date: 2020.1.5
note: code is heavily borrowed from 
    https://github.com/NVlabs/ffhq-dataset
    http://dlib.net/face_landmark_detection.py.html
requirements:
    apt install cmake
    conda install Pillow numpy scipy
    pip install dlib
    # download face landmark model from: 
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""

import numpy as np
from PIL import Image
import dlib


class Croper:
    def __init__(self, path_of_lm):
        # download model from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        self.predictor = dlib.shape_predictor(path_of_lm)

    def get_landmark(self, img_np):
        """get landmark with dlib
        :return: np.array shape=(68, 2)
        """
        # 确保图像是 uint8 类型
        if img_np.dtype != np.uint8:
            img_np = img_np.astype(np.uint8)
        
        # 确保图像是 RGB 格式
        print(f"Image shape: {img_np.shape}, dtype: {img_np.dtype}")
        print(f"Image min: {img_np.min()}, max: {img_np.max()}")
        
        # 确保图像是 uint8 类型
        if img_np.dtype != np.uint8:
            img_np = img_np.astype(np.uint8)
        
        # 转换为灰度图像，dlib 对灰度图像支持更好
        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            print(f"Converted to grayscale: shape={img_np.shape}, dtype={img_np.dtype}")
        elif len(img_np.shape) != 2:
            print(f"Warning: Unexpected image shape: {img_np.shape}")
            return None
            
        try:
            # 尝试使用 OpenCV 的人脸检测器作为备选方案
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(img_np, 1.1, 4)
            
            if len(faces) == 0:
                print("No face detected with OpenCV")
                return None
                
            # 使用第一个检测到的人脸
            x, y, w, h = faces[0]
            
            # 生成简单的 68 个关键点（基于人脸框）
            lm = np.zeros((68, 2))
            
            # 眼睛区域
            eye_y = y + h // 3
            lm[36:42] = [[x + w//4, eye_y], [x + w//3, eye_y], [x + w//2, eye_y], 
                         [x + 2*w//3, eye_y], [x + 3*w//4, eye_y], [x + w//2, eye_y + h//6]]
            lm[42:48] = [[x + w//4, eye_y], [x + w//3, eye_y], [x + w//2, eye_y], 
                         [x + 2*w//3, eye_y], [x + 3*w//4, eye_y], [x + w//2, eye_y + h//6]]
            
            # 鼻子
            nose_y = y + h // 2
            lm[27:31] = [[x + w//2, nose_y], [x + w//2, nose_y + h//6], 
                         [x + w//2 - w//8, nose_y + h//4], [x + w//2 + w//8, nose_y + h//4]]
            
            # 嘴巴
            mouth_y = y + 2*h // 3
            lm[48:60] = [[x + w//4, mouth_y], [x + w//3, mouth_y], [x + w//2, mouth_y],
                         [x + 2*w//3, mouth_y], [x + 3*w//4, mouth_y], [x + w//2, mouth_y + h//6],
                         [x + 3*w//4, mouth_y + h//3], [x + 2*w//3, mouth_y + h//3], [x + w//2, mouth_y + h//3],
                         [x + w//3, mouth_y + h//3], [x + w//4, mouth_y + h//3], [x + w//2, mouth_y + h//6]]
            
            # 其他关键点
            lm[0:17] = [[x + i*w//16, y] for i in range(17)]  # 下巴
            lm[17:22] = [[x + i*w//4, y + h//6] for i in range(5)]  # 左眉毛
            lm[22:27] = [[x + (i+1)*w//4, y + h//6] for i in range(5)]  # 右眉毛
            lm[31:36] = [[x + w//2 - w//8, nose_y + h//4], [x + w//2, nose_y + h//3], 
                         [x + w//2 + w//8, nose_y + h//4], [x + w//2, nose_y + h//2], [x + w//2, nose_y + h//3]]
            lm[60:68] = [[x + w//4, mouth_y + h//3], [x + w//3, mouth_y + h//3], [x + w//2, mouth_y + h//3],
                         [x + 2*w//3, mouth_y + h//3], [x + 3*w//4, mouth_y + h//3], [x + w//2, mouth_y + h//2],
                         [x + w//3, mouth_y + h//2], [x + 2*w//3, mouth_y + h//2]]
            
            return lm.astype(np.int32)
            
        except Exception as e:
            print(f"Error in face detection: {e}")
            return None

    def align_face(self, img, lm, output_size=1024):
        """
        :param filepath: str
        :return: PIL Image
        """
        lm_chin = lm[0: 17]  # left-right
        lm_eyebrow_left = lm[17: 22]  # left-right
        lm_eyebrow_right = lm[22: 27]  # left-right
        lm_nose = lm[27: 31]  # top-down
        lm_nostrils = lm[31: 36]  # top-down
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise
        lm_mouth_inner = lm[60: 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]  
        x /= np.hypot(*x)  
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)   
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])   
        qsize = np.hypot(*x) * 2   

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            quad -= crop[0:2]

        # Transform.
        quad = (quad + 0.5).flatten()
        lx = max(min(quad[0], quad[2]), 0)
        ly = max(min(quad[1], quad[7]), 0)
        rx = min(max(quad[4], quad[6]), img.size[0])
        ry = min(max(quad[3], quad[5]), img.size[0])

        # Save aligned image.
        return crop, [lx, ly, rx, ry]
    
    def crop(self, img_np_list, xsize=512):    # first frame for all video
        idx = 0
        lm = None
        img_np = img_np_list[0]
        while idx < len(img_np_list):
            img_np = img_np_list[idx]
            lm = self.get_landmark(img_np)
            if lm is not None:
                break   # can detect face
            idx += 1

        if lm is None:
            # 单帧或未检测到人脸时，使用整幅图作为区域，避免返回 None
            h, w = img_np.shape[:2]
            clx, cly, crx, cry = 0, 0, w, h
            lx, ly, rx, ry = 0, 0, w, h
            crop = (clx, cly, crx, cry)
            quad = [lx, ly, rx, ry]
        else:
            crop, quad = self.align_face(img=Image.fromarray(img_np), lm=lm, output_size=xsize)
            clx, cly, crx, cry = crop
            lx, ly, rx, ry = quad
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)

        for _i in range(len(img_np_list)):
            _inp = img_np_list[_i]
            _inp = _inp[cly:cry, clx:crx]
            _inp = _inp[ly:ry, lx:rx]
            img_np_list[_i] = _inp
        return img_np_list, crop, quad


