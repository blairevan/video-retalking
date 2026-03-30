import warnings
import gc
from utils.inference_utils import Laplacian_Pyramid_Blending_with_mask, face_detect, load_model, options, split_coeff, \
    trans_image, transform_semantic, find_crop_norm_ratio, load_face3d_net, exp_aus_dict
from utils.alignment_stit import crop_faces, calc_alignment_coefficients, paste_image
from utils.ffhq_preprocess import Croper
from utils import audio
from third_part.ganimation_replicate.model.ganimation import GANimationModel
from third_part.GFPGAN.gfpgan import GFPGANer
from third_part.GPEN.gpen_face_enhancer import FaceEnhancement
from third_part.face3d.extract_kp_videos import KeypointExtractor
from third_part.face3d.util.load_mats import load_lm3d
from third_part.face3d.util.preprocess import align_img
import numpy as np
import cv2
import os
import sys
import subprocess
import platform
import torch
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat

sys.path.insert(0, 'third_part')
sys.path.insert(0, 'third_part/GPEN')
sys.path.insert(0, 'third_part/GFPGAN')

# 3dmm extraction
# face enhancement
# expression control

warnings.filterwarnings("ignore")

args = options()


def get_gpu_free_memory():
    """获取 GPU 可用显存（GB）"""
    if not torch.cuda.is_available():
        return 0
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        free_mb = int(result.stdout.strip().split('\n')[0])
        return free_mb / 1024
    except Exception:
        # 备用方案：使用 PyTorch 估算
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        return max(0, total - reserved - 1)  # 保守估计，预留1GB


def get_optimal_batch_size(default_batch_size=16):
    """根据可用 GPU 显存自动调整批量大小"""
    if not torch.cuda.is_available():
        return default_batch_size

    free_memory = get_gpu_free_memory()
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print(
        f'[Info] GPU Memory: Total={total_memory:.1f}GB, Free={free_memory:.1f}GB')

    # 根据可用显存调整批量大小
    # LNet 每批次约需要 0.8-1.2GB 显存
    if free_memory < 2:
        batch_size = 1
    elif free_memory < 3:
        batch_size = 2
    elif free_memory < 5:
        batch_size = 4
    elif free_memory < 8:
        batch_size = 8
    else:
        batch_size = min(default_batch_size, 16)

    print(
        f'[Info] Auto LNet_batch_size: {batch_size} (free={free_memory:.1f}GB)')
    return batch_size


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[Info] Using {} for inference.'.format(device))
    os.makedirs(os.path.join('temp', args.tmp_dir), exist_ok=True)

    # 自动调整批量大小以避免 OOM
    if device == 'cuda':
        args.LNet_batch_size = get_optimal_batch_size(args.LNet_batch_size)

    # 面部增强配置：降低upscale避免过度增强和过亮
    # upscale=1: 标准增强，适合大多数场景
    # upscale=2: 2倍放大增强，可能导致过亮
    enhancer = FaceEnhancement(base_dir='checkpoints', size=512, model='GPEN-BFR-512', use_sr=False,
                               sr_model='rrdb_realesrnet_psnr', channel_multiplier=2, narrow=1, device=device)
    restorer = GFPGANer(model_path='checkpoints/GFPGANv1.3.pth', upscale=1, arch='clean',
                        channel_multiplier=2, bg_upsampler=None)

    base_name = args.face.split('/')[-1]
    if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        args.static = True
    if not os.path.isfile(args.face):
        raise ValueError(
            '--face argument must be a valid path to video/image file')
    elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps
    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            y1, y2, x1, x2 = args.crop
            if x2 == -1:
                x2 = frame.shape[1]
            if y2 == -1:
                y2 = frame.shape[0]
            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)

    print("[Step 0] Number of frames available for inference: " +
          str(len(full_frames)))
    # face detection & cropping, cropping the first frame as the style of FFHQ
    croper = Croper('checkpoints/shape_predictor_68_face_landmarks.dat')
    full_frames_RGB = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                       for frame in full_frames]
    full_frames_RGB, crop, quad = croper.crop(full_frames_RGB, xsize=512)

    clx, cly, crx, cry = crop
    lx, ly, rx, ry = quad
    lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
    oy1, oy2, ox1, ox2 = cly + \
        ly, min(cly+ry, full_frames[0].shape[0]), clx + \
        lx, min(clx+rx, full_frames[0].shape[1])
    # original_size = (ox2 - ox1, oy2 - oy1)
    frames_pil = [Image.fromarray(cv2.resize(frame, (256, 256)))
                  for frame in full_frames_RGB]

    # get the landmark according to the detected face.
    if not os.path.isfile('temp/'+base_name+'_landmarks.txt') or args.re_preprocess:
        print('[Step 1] Landmarks Extraction in Video.')
        kp_extractor = KeypointExtractor()
        if len(args.box) == 4 and min(args.box) >= 0:
            print(f'[Info] Landmark uses --box ROI: {args.box}')
        lm = kp_extractor.extract_keypoint(
            frames_pil, './temp/'+base_name+'_landmarks.txt', crop_box=args.box)
    else:
        print('[Step 1] Using saved landmarks.')
        lm = np.loadtxt('temp/'+base_name+'_landmarks.txt').astype(np.float32)
        lm = lm.reshape([len(full_frames), -1, 2])

    if not os.path.isfile('temp/'+base_name+'_coeffs.npy') or args.exp_img is not None or args.re_preprocess:
        net_recon = load_face3d_net(args.face3d_net_path, device)
        lm3d_std = load_lm3d('checkpoints/BFM')

        video_coeffs = []
        for idx in tqdm(range(len(frames_pil)), desc="[Step 2] 3DMM Extraction In Video:"):
            frame = frames_pil[idx]
            W, H = frame.size
            lm_idx = lm[idx].reshape([-1, 2])
            if np.mean(lm_idx) == -1:
                lm_idx = (lm3d_std[:, :2]+1) / 2.
                lm_idx = np.concatenate(
                    [lm_idx[:, :1] * W, lm_idx[:, 1:2] * H], 1)
            else:
                lm_idx[:, -1] = H - 1 - lm_idx[:, -1]

            trans_params, im_idx, lm_idx, _ = align_img(
                frame, lm_idx, lm3d_std)
            trans_params = np.array(
                [float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
            im_idx_tensor = torch.tensor(np.array(
                im_idx)/255., dtype=torch.float32).permute(2, 0, 1).to(device).unsqueeze(0)
            with torch.no_grad():
                coeffs = split_coeff(net_recon(im_idx_tensor))

            pred_coeff = {key: coeffs[key].cpu().numpy() for key in coeffs}
            pred_coeff = np.concatenate([pred_coeff['id'], pred_coeff['exp'], pred_coeff['tex'], pred_coeff['angle'],
                                         pred_coeff['gamma'], pred_coeff['trans'], trans_params[None]], 1)
            video_coeffs.append(pred_coeff)
        semantic_npy = np.array(video_coeffs)[:, 0]
        np.save('temp/'+base_name+'_coeffs.npy', semantic_npy)
    else:
        print('[Step 2] Using saved coeffs.')
        semantic_npy = np.load(
            'temp/'+base_name+'_coeffs.npy').astype(np.float32)

    # generate the 3dmm coeff from a single image
    if args.exp_img is not None and ('.png' in args.exp_img or '.jpg' in args.exp_img):
        print('extract the exp from', args.exp_img)
        exp_pil = Image.open(args.exp_img).convert('RGB')
        lm3d_std = load_lm3d('third_part/face3d/BFM')

        W, H = exp_pil.size
        kp_extractor = KeypointExtractor()
        lm_exp = kp_extractor.extract_keypoint(
            [exp_pil], 'temp/'+base_name+'_temp.txt')[0]
        if np.mean(lm_exp) == -1:
            lm_exp = (lm3d_std[:, :2] + 1) / 2.
            lm_exp = np.concatenate(
                [lm_exp[:, :1] * W, lm_exp[:, 1:2] * H], 1)
        else:
            lm_exp[:, -1] = H - 1 - lm_exp[:, -1]

        trans_params, im_exp, lm_exp, _ = align_img(exp_pil, lm_exp, lm3d_std)
        trans_params = np.array(
            [float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
        im_exp_tensor = torch.tensor(np.array(
            im_exp)/255., dtype=torch.float32).permute(2, 0, 1).to(device).unsqueeze(0)
        with torch.no_grad():
            expression = split_coeff(net_recon(im_exp_tensor))['exp'][0]
        del net_recon
    elif args.exp_img == 'smile':
        expression = torch.tensor(
            loadmat('checkpoints/expression.mat')['expression_mouth'])[0]
    else:
        print('using expression center')
        expression = torch.tensor(
            loadmat('checkpoints/expression.mat')['expression_center'])[0]

    # load DNet, model(LNet and ENet)
    D_Net, model = load_model(args, device)

    if not os.path.isfile('temp/'+base_name+'_stablized.npy') or args.re_preprocess:
        imgs = []
        for idx in tqdm(range(len(frames_pil)), desc="[Step 3] Stabilize the expression In Video:"):
            if args.one_shot:
                source_img = trans_image(frames_pil[0]).unsqueeze(0).to(device)
                semantic_source_numpy = semantic_npy[0:1]
            else:
                source_img = trans_image(
                    frames_pil[idx]).unsqueeze(0).to(device)
                semantic_source_numpy = semantic_npy[idx:idx+1]
            ratio = find_crop_norm_ratio(semantic_source_numpy, semantic_npy)
            coeff = transform_semantic(
                semantic_npy, idx, ratio).unsqueeze(0).to(device)

            # hacking the new expression
            coeff[:, :64, :] = expression[None, :64, None].to(device)
            with torch.no_grad():
                output = D_Net(source_img, coeff)
            img_stablized = np.uint8((output['fake_image'].squeeze(
                0).permute(1, 2, 0).cpu().clamp_(-1, 1).numpy() + 1)/2. * 255)
            imgs.append(cv2.cvtColor(img_stablized, cv2.COLOR_RGB2BGR))
        np.save('temp/'+base_name+'_stablized.npy', imgs)
        del D_Net
    else:
        print('[Step 3] Using saved stabilized video.')
        imgs = np.load('temp/'+base_name+'_stablized.npy')
        # 使用缓存时也要释放 D_Net
        del D_Net

    torch.cuda.empty_cache()
    gc.collect()

    if not args.audio.endswith('.wav'):
        command = '/usr/bin/ffmpeg -loglevel error -y -i {} -strict -2 {}'.format(
            args.audio, 'temp/{}/temp.wav'.format(args.tmp_dir))
        subprocess.call(command, shell=True)
        args.audio = 'temp/{}/temp.wav'.format(args.tmp_dir)
    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError(
            'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_step_size, mel_idx_multiplier, i, mel_chunks = 16, 80./fps, 0, []
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1

    print("[Step 4] Load audio; Length of mel chunks: {}".format(len(mel_chunks)))
    imgs = imgs[:len(mel_chunks)]
    full_frames = full_frames[:len(mel_chunks)]
    lm = lm[:len(mel_chunks)]

    imgs_enhanced = []
    for idx in tqdm(range(len(imgs)), desc='[Step 5] Reference Enhancement'):
        img = imgs[idx]
        # 完全跳过 GPEN 增强，直接使用原图 - 避免过亮
        # 如果需要增强，取消下面的注释
        # pred, _, _ = enhancer.process(
        #     img, img, face_enhance=False, possion_blending=False)
        # imgs_enhanced.append(pred)

        # 直接使用原图，不做任何增强
        imgs_enhanced.append(img)

    # ========== 🧹 显存清理：Step 6 前释放不需要的模型 ==========
    print('[Info] Cleaning up GPU memory before Lip Synthesis...')
    # 释放 enhancer 和 restorer（占用约 2-3GB 显存）
    del enhancer
    del restorer
    # 释放不再需要的中间变量
    del frames_pil
    del imgs
    # 强制垃圾回收
    gc.collect()
    torch.cuda.empty_cache()

    # 显示清理后的显存状态
    if torch.cuda.is_available():
        free_mem = get_gpu_free_memory()
        print(f'[Info] GPU Memory after cleanup: Free={free_mem:.2f}GB')
    # ========== 显存清理结束 ==========

    gen = datagen(imgs_enhanced.copy(), mel_chunks,
                  full_frames, None, (oy1, oy2, ox1, ox2))

    frame_h, frame_w = full_frames[0].shape[:-1]
    print(f"[Info] Output video resolution: {frame_w}x{frame_h}")
    out = cv2.VideoWriter('temp/{}/result.mp4'.format(args.tmp_dir),
                          cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))

    if args.up_face != 'original':
        instance = GANimationModel()
        instance.initialize()
        instance.setup()

    kp_extractor = KeypointExtractor()
    for i, (img_batch, mel_batch, frames, coords, img_original, f_frames) in enumerate(tqdm(gen, desc='[Step 6] Lip Synthesis:', total=int(np.ceil(float(len(mel_chunks)) / args.LNet_batch_size)))):
        img_batch = torch.FloatTensor(
            np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(
            np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        img_original = torch.FloatTensor(np.transpose(
            img_original, (0, 3, 1, 2))).to(device)/255.  # BGR -> RGB

        with torch.no_grad():
            incomplete, reference = torch.split(img_batch, 3, dim=1)
            pred, low_res = model(mel_batch, img_batch, reference)
            pred = torch.clamp(pred, 0, 1)

            if args.up_face in ['sad', 'angry', 'surprise']:
                tar_aus = exp_aus_dict[args.up_face]
            else:
                pass

            if args.up_face == 'original':
                cur_gen_faces = img_original
            else:
                test_batch = {'src_img': torch.nn.functional.interpolate((img_original * 2 - 1), size=(128, 128), mode='bilinear'),
                              'tar_aus': tar_aus.repeat(len(incomplete), 1)}
                instance.feed_batch(test_batch)
                instance.forward()
                cur_gen_faces = torch.nn.functional.interpolate(
                    instance.fake_img / 2. + 0.5, size=(384, 384), mode='bilinear')

            if args.without_rl1 is not False:
                incomplete, reference = torch.split(img_batch, 3, dim=1)
                mask = torch.where(incomplete == 0, torch.ones_like(
                    incomplete), torch.zeros_like(incomplete))
                pred = pred * mask + cur_gen_faces * (1 - mask)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        # 清理显存避免长视频累积OOM
        del img_batch, mel_batch, img_original, incomplete, reference, low_res
        if args.up_face != 'original':
            del cur_gen_faces
        torch.cuda.empty_cache()

        for p, f, xf, c in zip(pred, frames, f_frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            ff = xf.copy()

            # ===== 🎭 椭圆形边缘羽化 - 消除方框边界（更自然）=====
            # 使用椭圆形羽化 mask，更贴合人脸形状
            face_h, face_w = p.shape[:2]

            # 羽化强度（0.0-1.0）- 值越大羽化范围越广
            feather_ratio = 0.15  # 👈 可调整：0.1-0.3（推荐 0.15）

            # 创建椭圆形渐变 mask
            center_y, center_x = face_h / 2, face_w / 2
            mask = np.zeros((face_h, face_w), dtype=np.float32)

            # 椭圆的半轴长度
            radius_y = face_h / 2
            radius_x = face_w / 2

            # 为每个像素计算到椭圆中心的归一化距离
            for y in range(face_h):
                for x in range(face_w):
                    # 归一化坐标（椭圆方程）
                    dy = (y - center_y) / radius_y
                    dx = (x - center_x) / radius_x

                    # 到中心的椭圆距离
                    distance = np.sqrt(dx**2 + dy**2)

                    if distance <= 1.0 - feather_ratio:
                        # 中心区域：完全使用合成面部
                        mask[y, x] = 1.0
                    elif distance <= 1.0:
                        # 羽化区域：从1.0平滑过渡到0.0
                        fade = (1.0 - distance) / feather_ratio
                        mask[y, x] = fade
                    else:
                        # 外部区域：完全使用原图
                        mask[y, x] = 0.0

            # 将 mask 扩展到3通道
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

            # 使用椭圆 mask 进行加权混合
            original_face_region = ff[y1:y2, x1:x2].astype(np.float32)
            p_float = p.astype(np.float32)
            blended = p_float * mask_3d + original_face_region * (1 - mask_3d)

            ff[y1:y2, x1:x2] = blended.astype(np.uint8)
            # ===== 结束椭圆形羽化 =====

            # GFPGAN 增强已关闭 - 直接使用合成结果
            # 如果要启用 GFPGAN，取消下面的注释
            # cropped_faces, restored_faces, restored_img = restorer.enhance(
            #     ff, has_aligned=False, only_center_face=True, paste_back=True)
            # # 0,   1,   2,   3,   4,   5,   6,   7,   8,  9, 10,  11,  12,
            # mm = [0,   0,   0,   0,   0,   0,   0,   0,
            #       0,  0, 255, 255, 255, 0, 0, 0, 0, 0, 0]
            # mouse_mask = np.zeros_like(restored_img)
            # tmp_mask = enhancer.faceparser.process(
            #     restored_img[y1:y2, x1:x2], mm)[0]
            # mouse_mask[y1:y2, x1:x2] = cv2.resize(
            #     tmp_mask, (x2 - x1, y2 - y1))[:, :, np.newaxis] / 255.
            # height, width = ff.shape[:2]
            # restored_img, ff, full_mask = [cv2.resize(x, (512, 512)) for x in (
            #     restored_img, ff, np.float32(mouse_mask))]
            # img = Laplacian_Pyramid_Blending_with_mask(
            #     restored_img, ff, full_mask[:, :, 0], 8)
            # pp = np.uint8(cv2.resize(np.clip(img, 0, 255), (width, height)))

            # 不使用 GFPGAN，直接输出合成结果
            pp = ff

            # GPEN 处理也完全关闭 - 避免任何额外增强导致过亮
            # 如果需要 GPEN 的 Poisson 混合，取消下面的注释
            # pp, orig_faces, enhanced_faces = enhancer.process(
            #     pp, xf, bbox=c, face_enhance=False, possion_blending=True)

            # ===== 🎨 亮度和对比度校正 =====
            # 面部区域亮度调整 - 解决过曝问题
            # brightness: 0.0-2.0, 1.0=原始, <1.0变暗, >1.0变亮
            # contrast: 0.0-2.0, 1.0=原始, <1.0降低对比度, >1.0增加对比度
            brightness = 1.0   # 👈 不调整亮度，保持原始
            contrast = 1.0     # 👈 不调整对比度，保持原始

            # 只对面部区域应用调整
            y1, y2, x1, x2 = c[0], c[1], c[2], c[3]
            face_region = pp[y1:y2, x1:x2].astype(np.float32)

            # 调整亮度和对比度
            face_region = face_region * brightness  # 亮度
            face_region = (face_region - 127.5) * contrast + 127.5  # 对比度
            face_region = np.clip(face_region, 0, 255).astype(np.uint8)

            pp[y1:y2, x1:x2] = face_region
            # ===== 结束亮度校正 =====

            out.write(pp)
    out.release()

    if not os.path.isdir(os.path.dirname(args.outfile)):
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    # 首先合成带音频的视频
    temp_output = 'temp/{}/temp_output.mp4'.format(args.tmp_dir)
    command = '/usr/bin/ffmpeg -loglevel error -y -i {} -i {} -strict -2 -q:v 1 {}'.format(
        args.audio, 'temp/{}/result.mp4'.format(args.tmp_dir), temp_output)
    subprocess.call(command, shell=platform.system() != 'Windows')

    # 然后进行转码以确保浏览器兼容性
    # CRF值：0-51，越低质量越高
    # 4K推荐: crf=23 (高质量平衡), crf=20 (极高质量), crf=26 (标准质量)
    # preset: fast(快速低内存) medium(平衡) slow(高质量高内存)
    target_w = frame_w if frame_w % 2 == 0 else frame_w - 1
    target_h = frame_h if frame_h % 2 == 0 else frame_h - 1

    # GPU 编码参数 - 匹配高质量输入
    # 目标：保持 50-60 Mbps 码率（接近输入的 62.5 Mbps）
    gpu_preset = 'p5'          # 高质量预设
    gpu_cq = 15                # 极高质量（CQ越低码率越高）
    gpu_profile = 'high'       # High profile（更好的压缩）
    gpu_rc_mode = 'vbr'        # 可变码率模式

    # 根据分辨率设置目标码率（确保足够高）
    if max(target_w, target_h) >= 2160:  # 4K
        gpu_bitrate = '50M'    # 50 Mbps 目标码率
        cpu_crf = 18           # CPU 对应的高质量
    elif max(target_w, target_h) >= 1080:  # 1080p
        gpu_bitrate = '15M'    # 15 Mbps
        cpu_crf = 18
    else:  # 720p及以下
        gpu_bitrate = '8M'     # 8 Mbps
        cpu_crf = 20

    # 尝试使用 GPU 编码（NVIDIA），失败则回退到 CPU
    print(
        f"[Info] Encoding final video at {target_w}x{target_h} resolution (GPU: preset={gpu_preset} cq={gpu_cq} profile={gpu_profile} bitrate={gpu_bitrate})...")

    # NVIDIA GPU 编码（速度快10-20倍，内存占用低）
    # 使用 scale_cuda 在 GPU 上缩放（如果需要），比 -s 更高效
    if target_w != frame_w or target_h != frame_h:
        # 需要缩放：使用 scale_cuda 在 GPU 上进行
        gpu_command = '/usr/bin/ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i {} -vf "scale_cuda={}:{}" -c:v h264_nvenc -preset {} -profile:v {} -rc:v {} -cq {} -b:v {} -maxrate {} -bufsize {} -c:a aac -b:a 192k -movflags +faststart {}'.format(
            temp_output, target_w, target_h, gpu_preset, gpu_profile, gpu_rc_mode, gpu_cq, gpu_bitrate, gpu_bitrate, gpu_bitrate, args.outfile)
    else:
        # 不需要缩放：直接编码（最高效）
        gpu_command = '/usr/bin/ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i {} -c:v h264_nvenc -preset {} -profile:v {} -rc:v {} -cq {} -b:v {} -maxrate {} -bufsize {} -c:a aac -b:a 192k -movflags +faststart {}'.format(
            temp_output, gpu_preset, gpu_profile, gpu_rc_mode, gpu_cq, gpu_bitrate, gpu_bitrate, gpu_bitrate, args.outfile)

    # CPU 编码（备用方案）
    if target_w != frame_w or target_h != frame_h:
        cpu_command = '/usr/bin/ffmpeg -i {} -vf "scale={}:{}" -c:v libx264 -profile:v high -crf {} -preset medium -c:a aac -b:a 192k -movflags +faststart {}'.format(
            temp_output, target_w, target_h, cpu_crf, args.outfile)
    else:
        cpu_command = '/usr/bin/ffmpeg -i {} -c:v libx264 -profile:v high -crf {} -preset medium -c:a aac -b:a 192k -movflags +faststart {}'.format(
            temp_output, cpu_crf, args.outfile)

    # 打印命令用于调试
    print(f"[Debug] GPU command: {gpu_command}")
    print(f"[Debug] CPU command: {cpu_command}")

    # 先尝试 GPU，失败则使用 CPU
    print("[Info] Trying GPU encoding...")
    result = subprocess.call(gpu_command, shell=platform.system() != 'Windows')
    if result != 0:
        print("[Warning] GPU encoding failed, falling back to CPU encoding...")
        subprocess.call(cpu_command, shell=platform.system() != 'Windows')
    else:
        print("[Info] GPU encoding completed successfully!")

    # 清理临时文件
    if os.path.exists(temp_output):
        os.remove(temp_output)

    print('outfile:', args.outfile)


# frames:256x256, full_frames: original size
def datagen(frames, mels, full_frames, frames_pil, cox):
    img_batch, mel_batch, frame_batch, coords_batch, ref_batch, full_frame_batch = [
    ], [], [], [], [], []
    base_name = args.face.split('/')[-1]
    image_size = 256

    # original frames
    kp_extractor = KeypointExtractor()
    fr_pil = [Image.fromarray(frame) for frame in frames]
    lms = kp_extractor.extract_keypoint(
        fr_pil, 'temp/'+base_name+'x12_landmarks.txt', crop_box=args.box)
    # frames is the croped version of modified face
    frames_pil = [(lm, frame) for frame, lm in zip(fr_pil, lms)]
    crops, orig_images, quads = crop_faces(
        image_size, frames_pil, scale=1.0, use_fa=True)
    inverse_transforms = [calc_alignment_coefficients(
        quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]]) for quad in quads]
    del kp_extractor.detector

    oy1, oy2, ox1, ox2 = cox
    face_det_results = face_detect(full_frames, args, jaw_correction=True)

    # 【方案3：延迟计算】按需计算ref，不缓存，彻底避免内存累积
    # 每个ref会被计算约5-6次，但内存使用最低
    def get_ref(idx):
        """按需生成ref，不缓存，用完即释放"""
        inverse_transform = inverse_transforms[idx]
        crop = crops[idx]
        full_frame = full_frames[idx]
        face_det = face_det_results[idx]

        imc_pil = paste_image(inverse_transform, crop, Image.fromarray(
            cv2.resize(full_frame[int(oy1):int(oy2), int(ox1):int(ox2)], (256, 256))))

        ff = full_frame.copy()
        ff[int(oy1):int(oy2), int(ox1):int(ox2)] = cv2.resize(
            np.array(imc_pil.convert('RGB')), (ox2 - ox1, oy2 - oy1))
        oface, coords = face_det
        y1, y2, x1, x2 = coords
        return ff[y1: y2, x1:x2]  # 直接返回，不缓存

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face = get_ref(idx)  # 按需获取ref
        oface, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))
        oface = cv2.resize(oface, (args.img_size, args.img_size))

        img_batch.append(oface)
        ref_batch.append(face)
        mel_batch.append(m)
        coords_batch.append(coords)
        frame_batch.append(frame_to_save)
        full_frame_batch.append(full_frames[idx].copy())

        if len(img_batch) >= args.LNet_batch_size:
            img_batch, mel_batch, ref_batch = np.asarray(
                img_batch), np.asarray(mel_batch), np.asarray(ref_batch)
            img_masked = img_batch.copy()
            img_original = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0
            img_batch = np.concatenate((img_masked, ref_batch), axis=3) / 255.
            mel_batch = np.reshape(
                mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch
            img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch, ref_batch = [
            ], [], [], [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch, ref_batch = np.asarray(
            img_batch), np.asarray(mel_batch), np.asarray(ref_batch)
        img_masked = img_batch.copy()
        img_original = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0
        img_batch = np.concatenate((img_masked, ref_batch), axis=3) / 255.
        mel_batch = np.reshape(
            mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        yield img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch


if __name__ == '__main__':
    main()
