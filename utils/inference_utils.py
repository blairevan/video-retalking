import numpy as np
import cv2
import argparse
import torch
import torchvision.transforms.functional as TF

from models import load_network, load_DNet
from tqdm import tqdm
from PIL import Image
from scipy.spatial import ConvexHull
from third_part import face_detection
from third_part.face3d.models import networks

import warnings
warnings.filterwarnings("ignore")

from utils.audio import parse_bool

AUTO_FALLBACK_BOX = (10, 610, 100, 1100)


def options():
    parser = argparse.ArgumentParser(
        description='Inference code to lip-sync videos in the wild using Wav2Lip models')

    parser.add_argument('--DNet_path', type=str, default='checkpoints/DNet.pt')
    parser.add_argument('--LNet_path', type=str,
                        default='checkpoints/LNet.pth')
    parser.add_argument('--ENet_path', type=str,
                        default='checkpoints/ENet.pth')
    parser.add_argument('--face3d_net_path', type=str,
                        default='checkpoints/face3d_pretrain_epoch_20.pth')
    parser.add_argument(
        '--face', type=str, help='Filepath of video/image that contains faces to use', required=True)
    parser.add_argument(
        '--audio', type=str, help='Filepath of video/audio file to use as raw audio source', required=True)
    parser.add_argument('--exp_img', type=str,
                        help='Expression template. neutral, smile or image path', default='neutral')
    parser.add_argument('--outfile', type=str,
                        help='Video path to save result')

    parser.add_argument(
        '--fps', type=float, help='Can be specified only if input is a static image (default: 25)', default=25., required=False)
    parser.add_argument('--pads', nargs='+', type=int, default=[
                        0, 20, 0, 0], help='Padding (top, bottom, left, right). Please adjust to include chin at least')
    parser.add_argument('--face_det_batch_size', type=int,
                        help='Batch size for face detection', default=4)
    parser.add_argument('--LNet_batch_size', type=int,
                        help='Batch size for LNet', default=16)
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                        help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
                        'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')
    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                        help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                        'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')
    parser.add_argument('--nosmooth', default=False, action='store_true',
                        help='Prevent smoothing face detections over a short temporal window')
    parser.add_argument('--static', default=False, action='store_true')

    parser.add_argument('--up_face', default='original')
    parser.add_argument('--one_shot', action='store_true')
    parser.add_argument('--without_rl1', default=False,
                        action='store_true', help='Do not use the relative l1')
    parser.add_argument('--tmp_dir', type=str, default='temp',
                        help='Folder to save tmp results')
    parser.add_argument('--re_preprocess', action='store_true')
    parser.add_argument('--close_mouth_start', type=parse_bool, default=True,
                        help='Close mouth during leading silence (true/false)')
    parser.add_argument('--close_mouth_end', type=parse_bool, default=False,
                        help='Close mouth during trailing silence (true/false)')

    args = parser.parse_args()
    return args


exp_aus_dict = {        # AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU07_r, AU09_r, AU10_r, AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU23_r, AU25_r, AU26_r, AU45_r.
    'sad': torch.Tensor([[0,     0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0]]),
    'angry': torch.Tensor([[0,     0,      0.3,    0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0]]),
    'surprise': torch.Tensor([[0, 0,      0,      0.2,    0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0]])
}


def mask_postprocess(mask, thres=20):
    mask[:thres, :] = 0
    mask[-thres:, :] = 0
    mask[:, :thres] = 0
    mask[:, -thres:] = 0
    mask = cv2.GaussianBlur(mask, (101, 101), 11)
    mask = cv2.GaussianBlur(mask, (101, 101), 11)
    return mask.astype(np.float32)


def trans_image(image):
    image = TF.resize(
        image, size=256, interpolation=Image.BICUBIC)
    image = TF.to_tensor(image)
    image = TF.normalize(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    return image


def obtain_seq_index(index, num_frames):
    seq = list(range(index-13, index+13))
    seq = [min(max(item, 0), num_frames-1) for item in seq]
    return seq


def transform_semantic(semantic, frame_index, crop_norm_ratio=None):
    index = obtain_seq_index(frame_index, semantic.shape[0])

    coeff_3dmm = semantic[index, ...]
    ex_coeff = coeff_3dmm[:, 80:144]  # expression # 64
    angles = coeff_3dmm[:, 224:227]  # euler angles for pose
    translation = coeff_3dmm[:, 254:257]  # translation
    crop = coeff_3dmm[:, 259:262]  # crop param

    if crop_norm_ratio:
        crop[:, -3] = crop[:, -3] * crop_norm_ratio

    coeff_3dmm = np.concatenate([ex_coeff, angles, translation, crop], 1)
    return torch.Tensor(coeff_3dmm).permute(1, 0)


def find_crop_norm_ratio(source_coeff, target_coeffs):
    alpha = 0.3
    # mean different exp
    exp_diff = np.mean(
        np.abs(target_coeffs[:, 80:144] - source_coeff[:, 80:144]), 1)
    angle_diff = np.mean(np.abs(
        # mean different angle
        target_coeffs[:, 224:227] - source_coeff[:, 224:227]), 1)
    # find the smallerest index
    index = np.argmin(alpha*exp_diff + (1-alpha)*angle_diff)
    crop_norm_ratio = source_coeff[:, -3] / target_coeffs[index:index+1, -3]
    return crop_norm_ratio


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def _normalize_box_for_image(image_shape, box):
    if box is None or len(box) != 4:
        return None
    values = [int(v) for v in box]
    h, w = image_shape[:2]

    # 标准格式: (top, bottom, left, right)
    y1, y2, x1, x2 = values
    if y1 >= 0 and y2 >= 0 and x1 >= 0 and x2 >= 0:
        ny1 = max(0, min(y1, h))
        ny2 = max(0, min(y2, h))
        nx1 = max(0, min(x1, w))
        nx2 = max(0, min(x2, w))
        if ny2 - ny1 >= 2 and nx2 - nx1 >= 2:
            return (ny1, ny2, nx1, nx2)

    return None


def face_detect(images, args, jaw_correction=False, detector=None):
    if detector == None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                                flip_input=False, device=device)

    batch_size = args.face_det_batch_size
    box = getattr(args, 'box', None)
    normalized_box = _normalize_box_for_image(images[0].shape, box) if len(images) > 0 else None
    explicit_box = normalized_box is not None
    if explicit_box:
        print(f'[Info] FaceDet uses --box ROI: {normalized_box}')

    while 1:
        predictions = []
        try:
            if explicit_box:
                y1, y2, x1, x2 = normalized_box
                roi_images = [image[y1:y2, x1:x2] for image in images]
                for i in tqdm(range(0, len(roi_images), batch_size), desc='FaceDet:'):
                    batch_predictions = detector.get_detections_for_batch(
                        np.array(roi_images[i:i + batch_size]))
                    for rect in batch_predictions:
                        if rect is None:
                            predictions.append(None)
                        else:
                            predictions.append(
                                [rect[0] + x1, rect[1] + y1, rect[2] + x1, rect[3] + y1]
                            )
            else:
                for i in tqdm(range(0, len(images), batch_size), desc='FaceDet:'):
                    predictions.extend(detector.get_detections_for_batch(
                        np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    if explicit_box:
        # ROI 未检测到时回退全图检测，避免框不准导致整段失败
        for i, rect in enumerate(predictions):
            if rect is not None:
                continue
            predictions[i] = detector.get_detections_for_batch(np.array([images[i]]))[0]
    else:
        # 未显式设置 --box 时，若全图检测失败，则自动尝试兜底 ROI
        auto_box = _normalize_box_for_image(images[0].shape, AUTO_FALLBACK_BOX) if len(images) > 0 else None
        if auto_box is not None:
            miss_indices = [i for i, rect in enumerate(predictions) if rect is None]
            if len(miss_indices) > 0:
                print(f'[Info] FaceDet fallback ROI active: {auto_box}, missing={len(miss_indices)}')
                y1, y2, x1, x2 = auto_box
                for idx in miss_indices:
                    roi = images[idx][y1:y2, x1:x2]
                    fallback = detector.get_detections_for_batch(np.array([roi]))[0]
                    if fallback is not None:
                        predictions[idx] = [fallback[0] + x1, fallback[1] + y1, fallback[2] + x1, fallback[3] + y1]

    results = []
    pady1, pady2, padx1, padx2 = args.pads if jaw_correction else (0, 20, 0, 0)
    for rect, image in zip(predictions, images):
        if rect is None:
            # check this frame where the face was not detected.
            cv2.imwrite('temp/faulty_frame.jpg', image)
            raise ValueError(
                'Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth:
        boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)]
               for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    torch.cuda.empty_cache()
    return results


def _load(checkpoint_path, device):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def split_coeff(coeffs):
    """
    Return:
        coeffs_dict     -- a dict of torch.tensors

    Parameters:
        coeffs          -- torch.tensor, size (B, 256)
    """
    id_coeffs = coeffs[:, :80]
    exp_coeffs = coeffs[:, 80: 144]
    tex_coeffs = coeffs[:, 144: 224]
    angles = coeffs[:, 224: 227]
    gammas = coeffs[:, 227: 254]
    translations = coeffs[:, 254:]
    return {
        'id': id_coeffs,
        'exp': exp_coeffs,
        'tex': tex_coeffs,
        'angle': angles,
        'gamma': gammas,
        'trans': translations
    }


def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels=6):
    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpA = [gpA[num_levels-1]]
    lpB = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in range(num_levels-1, 0, -1):
        # Laplacian: subtract upscaled version of lower level from current level
        # to get the high frequencies
        LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i-1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1])  # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la, lb, gm in zip(lpA, lpB, gpMr):
        gm = gm[:, :, np.newaxis]
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1, num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    return ls_


def load_model(args, device):
    D_Net = load_DNet(args).to(device)
    model = load_network(args).to(device)
    return D_Net, model


def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(
            kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(
            kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}
    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(
                kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(
                jacobian_diff, kp_source['jacobian'])
    return kp_new


def load_face3d_net(ckpt_path, device):
    net_recon = networks.define_net_recon(
        net_recon='resnet50', use_last_fc=False, init_path='').to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    net_recon.load_state_dict(checkpoint['net_recon'])
    net_recon.eval()
    return net_recon
