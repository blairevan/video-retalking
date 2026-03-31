import os
import cv2
import time
import glob
import argparse
import face_alignment
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from itertools import cycle

from torch.multiprocessing import Pool, Process, set_start_method


class KeypointExtractor():
    AUTO_FALLBACK_BOX = (10, 610, 100, 1100)

    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, device=device)

    @staticmethod
    def _normalize_crop_box(image_shape, crop_box):
        if crop_box is None:
            return None
        if len(crop_box) != 4:
            return None
        values = [int(v) for v in crop_box]
        h, w = image_shape[:2]

        # 标准格式: (top, bottom, left, right)
        top, bottom, left, right = values
        if top >= 0 and bottom >= 0 and left >= 0 and right >= 0:
            n_top = max(0, min(top, h))
            n_bottom = max(0, min(bottom, h))
            n_left = max(0, min(left, w))
            n_right = max(0, min(right, w))
            if n_bottom - n_top >= 2 and n_right - n_left >= 2:
                return (n_top, n_bottom, n_left, n_right)

        return None

    def _detect_landmarks(self, image_np):
        landmarks = self.detector.get_landmarks_from_image(image_np)
        if landmarks is None or len(landmarks) == 0:
            return None
        return landmarks[0]

    def extract_keypoint(self, images, name=None, info=True, crop_box=None):
        if isinstance(images, list):
            keypoints = []
            if info:
                i_range = tqdm(images, desc='landmark Det:')
            else:
                i_range = images

            for image in i_range:
                current_kp = self.extract_keypoint(image, crop_box=crop_box)
                if np.mean(current_kp) == -1 and keypoints:
                    keypoints.append(keypoints[-1])
                else:
                    keypoints.append(current_kp[None])

            keypoints = np.concatenate(keypoints, 0)
            np.savetxt(os.path.splitext(name)[0]+'.txt', keypoints.reshape(-1))
            return keypoints
        else:
            image_np = np.array(images)
            normalized_box = self._normalize_crop_box(image_np.shape, crop_box)
            while True:
                try:
                    keypoints = None
                    if normalized_box is not None:
                        y1, y2, x1, x2 = normalized_box
                        roi = image_np[y1:y2, x1:x2]
                        roi_keypoints = self._detect_landmarks(roi)
                        if roi_keypoints is not None:
                            keypoints = roi_keypoints.copy()
                            keypoints[:, 0] += x1
                            keypoints[:, 1] += y1

                    if keypoints is None:
                        keypoints = self._detect_landmarks(image_np)
                    if keypoints is None and normalized_box is None:
                        # 未指定 --box 且全图检测失败时，自动尝试兜底 ROI
                        auto_box = self._normalize_crop_box(
                            image_np.shape, self.AUTO_FALLBACK_BOX)
                        if auto_box is not None:
                            y1, y2, x1, x2 = auto_box
                            roi = image_np[y1:y2, x1:x2]
                            roi_keypoints = self._detect_landmarks(roi)
                            if roi_keypoints is not None:
                                keypoints = roi_keypoints.copy()
                                keypoints[:, 0] += x1
                                keypoints[:, 1] += y1
                    if keypoints is None:
                        raise TypeError('No face detected')
                    break
                except RuntimeError as e:
                    if str(e).startswith('CUDA'):
                        print("Warning: out of memory, sleep for 1s")
                        time.sleep(1)
                    else:
                        print(e)
                        break
                except TypeError:
                    print('No face detected in this image')
                    shape = [68, 2]
                    keypoints = -1. * np.ones(shape)
                    break
            if name is not None:
                np.savetxt(os.path.splitext(name)[
                           0]+'.txt', keypoints.reshape(-1))
            return keypoints


def read_video(filename):
    frames = []
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
        else:
            break
    cap.release()
    return frames


def run(data):
    filename, opt, device = data
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    kp_extractor = KeypointExtractor()
    images = read_video(filename)
    name = filename.split('/')[-2:]
    os.makedirs(os.path.join(opt.output_dir, name[-2]), exist_ok=True)
    kp_extractor.extract_keypoint(
        images,
        name=os.path.join(opt.output_dir, name[-2], name[-1])
    )


if __name__ == '__main__':
    set_start_method('spawn')
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir', type=str,
                        help='the folder of the input files')
    parser.add_argument('--output_dir', type=str,
                        help='the folder of the output files')
    parser.add_argument('--device_ids', type=str, default='0,1')
    parser.add_argument('--workers', type=int, default=4)

    opt = parser.parse_args()
    filenames = list()
    VIDEO_EXTENSIONS_LOWERCASE = {'mp4'}
    VIDEO_EXTENSIONS = VIDEO_EXTENSIONS_LOWERCASE.union(
        {f.upper() for f in VIDEO_EXTENSIONS_LOWERCASE})
    extensions = VIDEO_EXTENSIONS

    for ext in extensions:
        os.listdir(f'{opt.input_dir}')
        print(f'{opt.input_dir}/*.{ext}')
        filenames = sorted(glob.glob(f'{opt.input_dir}/*.{ext}'))
    print('Total number of videos:', len(filenames))
    pool = Pool(opt.workers)
    args_list = cycle([opt])
    device_ids = opt.device_ids.split(",")
    device_ids = cycle(device_ids)
    for data in tqdm(pool.imap_unordered(run, zip(filenames, args_list, device_ids))):
        None
