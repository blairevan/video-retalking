"""
面部增强配置文件
用于调整面部清晰度和亮度
"""

# ==================== 配置选项 ====================

# 1. GFPGAN 配置
GFPGAN_CONFIG = {
    # upscale: 1 或 2
    # 1 = 不放大，速度快，适合大多数情况 ✅ 当前设置
    # 2 = 2倍放大，质量更高但可能过度增强
    'upscale': 1,  # ✅ 当前生效值（inference.py 第 47 行）

    # arch: 'clean', 'original', 或 'bilinear'
    # clean = 清晰，适合大多数情况
    # original = 原始版本，可能更柔和
    'arch': 'clean',

    # channel_multiplier: 1 或 2
    # 2 = 更高质量但更慢
    'channel_multiplier': 2,
}

# 2. GPEN 配置
GPEN_CONFIG = {
    # size: 512 或 256
    # 512 = 高质量
    # 256 = 快速
    'size': 512,

    # use_sr: 是否使用超分辨率
    'use_sr': False,

    # 增强强度（通过调整其他参数实现）
    'narrow': 1,  # 窄度系数，1=标准
}

# 3. 面部增强开关（当前所有增强都已关闭）
ENHANCEMENT_SWITCHES = {
    # 是否在参考帧使用 GPEN 增强
    'use_gpen_on_reference': False,  # ✅ 已完全关闭（inference.py 第 241 行）
    # 直接使用原图，不调用 enhancer.process()

    # 是否在最终输出使用 GPEN 增强
    'use_gpen_on_output': False,  # ✅ 已完全关闭（inference.py 第 320-321 行）
    # 不调用 enhancer.process()，直接输出

    # 是否使用 GFPGAN 增强（嘴部区域）
    'use_gfpgan': False,  # ✅ 已完全关闭（inference.py 第 296-313 行）
    # 整个 restorer.enhance() 调用已注释

    # 是否使用 Poisson 混合
    'use_poisson_blending': False,  # ✅ 已关闭（所有增强都关闭了）
}

# 4. 🎭 椭圆形边缘羽化配置（消除方框边界，更自然）
EDGE_FEATHERING = {
    # 羽化强度（0.0-1.0）：椭圆边缘渐变的范围比例
    # - 0.10-0.12: 轻微羽化，边缘清晰
    # - 0.12-0.15: 适度羽化，适合一般场景
    # - 0.15-0.20: 标准羽化，推荐值 ✅
    # - 0.20-0.25: 强羽化，适合背景复杂的场景
    # - 0.25-0.30: 极强羽化，边缘非常柔和
    'feather_ratio': 0.15,  # 👈 当前设置：0.15（15% 椭圆边缘羽化）（inference.py 第 304 行）

    # 📍 修改位置：inference.py 第 304 行
    # 💡 效果：使用椭圆形渐变 mask（而非矩形），更贴合人脸形状
    # 💡 优势：消除方框边界，过渡更自然，符合人脸椭圆轮廓

    # 形状说明：
    # - 旧版：矩形羽化 ▢（有明显的方角）
    # - 新版：椭圆羽化 ⬭（贴合人脸，更自然）
}

# 5. 🎨 亮度和对比度校正（解决过曝问题）
BRIGHTNESS_CORRECTION = {
    # 亮度调整：0.0-2.0
    # - 1.0 = 原始亮度（不调整）
    # - <1.0 = 变暗（如 0.85 = 降低 15% 亮度）
    # - >1.0 = 变亮（如 1.15 = 增加 15% 亮度）
    'brightness': 1.0,   # 👈 当前设置：不调整亮度（原始）（inference.py 第 369 行）

    # 对比度调整：0.0-2.0
    # - 1.0 = 原始对比度（不调整）
    # - <1.0 = 降低对比度（柔和）
    # - >1.0 = 增加对比度（锐利）
    'contrast': 1.0,     # 👈 当前设置：不调整对比度（原始）（inference.py 第 370 行）

    # 📍 修改位置：inference.py 第 369-370 行
    # 💡 当前配置：保持原始亮度和对比度，不做任何调整
    # 💡 调整建议：
    #    - 如果太亮：brightness 降到 1.0-1.1
    #    - 如果还是偏暗：brightness 提升到 1.3-1.4
    #    - 如果想要更清晰：contrast 提升到 1.10-1.15
    #    - 如果太锐利：contrast 降到 0.95-1.00
}

# 6. FFmpeg 编码配置
FFMPEG_CONFIG = {
    # 分辨率保持：输出视频将自动保持源视频分辨率
    # 📍 实现位置：inference.py 第 362-367 行
    # 自动调整为偶数尺寸（H.264 编码要求）

    # CRF: 0-51，值越小质量越高
    # 推荐: 15-18 高质量, 18-23 标准质量
    'crf': 1,  # ✅ 当前设置：1（极致无损模式）⚠️ 文件会非常大！（inference.py 第 365 行）

    # preset: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
    'preset': 'veryslow',  # ✅ 当前设置：veryslow（最高质量编码）（inference.py 第 365 行）

    # 音频比特率
    'audio_bitrate': '320k',  # ✅ 当前设置：320k（最高音质）（inference.py 第 365 行）
}

# 7. Laplacian 金字塔混合配置
PYRAMID_CONFIG = {
    # 主混合金字塔层数 (6-8 推荐)
    'main_levels': 8,

    # GPEN 混合金字塔层数 (6-8 推荐)
    'gpen_levels': 8,
}

# ==================== 预设配置 ====================

PRESETS = {
    # 原始设置（可能过亮）
    'original': {
        'gfpgan_upscale': 2,
        'use_gpen_on_reference': True,
        'use_gpen_on_output': False,
        'use_poisson_blending': True,
    },

    # 推荐设置（减少过度增强）
    'recommended': {
        'gfpgan_upscale': 1,  # 减少放大
        'use_gpen_on_reference': True,
        'use_gpen_on_output': False,
        'use_poisson_blending': True,
    },

    # 柔和模式（最自然）
    'soft': {
        'gfpgan_upscale': 1,
        'use_gpen_on_reference': False,  # 关闭 GPEN
        'use_gpen_on_output': False,
        'use_poisson_blending': False,  # 简单混合
    },

    # 高清晰度模式（可能更亮）
    'sharp': {
        'gfpgan_upscale': 2,
        'use_gpen_on_reference': True,
        'use_gpen_on_output': True,  # 额外增强
        'use_poisson_blending': True,
    },
}

# ==================== 使用说明 ====================
"""
使用方法：

1. 快速调整：
   修改 inference.py 中的相应参数

2. 如果面部过亮：
   - 将 GFPGAN upscale 从 2 改为 1
   - 将 use_gpen_on_reference 改为 False
   - 将 use_poisson_blending 改为 False

3. 如果面部不够清晰：
   - 保持 GFPGAN upscale = 2
   - 将 use_gpen_on_output 改为 True
   - 确保 use_poisson_blending = True

4. 推荐的调整顺序：
   a. 先将 upscale 从 2 改为 1
   b. 如果还是亮，关闭 use_gpen_on_reference
   c. 如果需要更清晰，开启 use_gpen_on_output
   d. 如果边缘不自然，关闭 use_poisson_blending
"""
