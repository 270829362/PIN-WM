# 这段代码是 PIN-WM 实验时序结果可视化的核心工具，专门处理 3D 高斯渲染或物理仿真生成的 RGB 图像序列，核心价值是 “统一格式转换 + 轻量化结果输出”，关键特点与应用场景如下：
# 导入核心依赖库
import numpy as np  # 数值数组处理（图像数据格式转换、异常值处理）
import os  # 文件路径操作（未直接使用，预留输出目录创建扩展）
import enum  # 枚举类（未直接使用，备用）
import types  # 类型相关（未直接使用，备用）
from typing import List, Mapping, Optional, Text, Tuple, Union  # 类型注解
import copy  # 深拷贝（未直接使用，备用）
from PIL import Image  # 图像处理（未直接使用，预留单帧图像保存扩展）
import mediapy as media  # 视频写入工具（核心：生成H264编码视频）
from matplotlib import cm  # 色彩映射（未直接使用，预留图像色彩调整扩展）
from tqdm import tqdm  # 进度条（未直接使用，预留大批量图像处理进度显示）
import imageio  # GIF生成工具（核心：将图像序列打包为GIF）


def create_videos_with_rgbs(rgbs, out_dir):
    """
    将RGB图像序列（PyTorch张量）保存为H264编码的视频文件
    核心用途：PIN-WM实验中记录3D仿真过程、高斯渲染结果时序动画（如物体翻转、姿态变化）
    适配输入：PyTorch张量格式的图像序列（单帧形状[C, H, W]，C=3（RGB），数值范围0~1）
    Args:
        rgbs: RGB图像序列（列表或批量张量，格式：[B, C, H, W] 或 [单帧1[C,H,W], 单帧2[C,H,W], ...]）
              B=帧数，C=通道数（3），H=图像高度，W=图像宽度，数值范围0~1
        out_dir: 输出视频路径（含文件名，如"./output/simulation_video.mp4"）
    """
    # 配置视频参数（H264编码，保证兼容性和画质）
    video_kwargs = {
      'shape': rgbs[0].shape[1:3],  # 视频分辨率（H, W）：从第一帧图像提取（跳过通道维度C）
      'codec': 'h264',  # 视频编码器（H264：主流兼容格式，支持大多数播放器）
      'fps': 6,  # 视频帧率（每秒6帧，适配仿真/渲染的时序节奏，可按需调整）
      'crf': 18,  # 视频质量参数（0~51，值越低质量越好，18为平衡质量和文件大小的推荐值）
    }
    input_format = 'rgb'  # 输入图像的通道顺序（RGB：与PyTorch张量通道顺序一致）

    # 创建视频写入器（上下文管理器自动处理资源释放，避免文件损坏）
    with media.VideoWriter(
        out_dir, **video_kwargs, input_format=input_format) as writer:
        # 遍历图像序列中的每一张图
        for rgb in rgbs:
            # 步骤1：张量维度转换 + 格式转换（PyTorch张量[C, H, W] → NumPy数组[H, W, C]）
            # permute(1, 2, 0)：交换维度顺序（C→H→W 转为 H→W→C，适配视频写入器的输入要求）
            # cpu().detach().numpy()：从GPU张量→CPU张量→解除计算图绑定→转为NumPy数组（视频工具仅支持NumPy）
            rgb = rgb.permute(1, 2, 0).cpu().detach().numpy()
            
            # 步骤2：图像数据清洗 + 数值范围转换（0~1 → 0~255 uint8）
            # np.nan_to_num(rgb)：将张量中的NaN/inf值转为0（避免异常值导致视频编码失败）
            # np.clip(..., 0., 1.)：限制数值范围在0~1（防止渲染过程中数值溢出）
            # *255.：将归一化数值转为图像标准的0~255范围
            # astype(np.uint8)：转为8位无符号整数（图像/视频的标准数据类型）
            rgb = (np.clip(np.nan_to_num(rgb), 0., 1.) * 255.).astype(np.uint8)
            
            # 步骤3：将处理后的单帧图像写入视频
            writer.add_image(rgb)


def create_gif_with_rgbs(rgbs, out_dir):
    """
    将RGB图像序列（PyTorch张量）保存为GIF动图文件
    核心用途：PIN-WM实验中快速生成轻量化时序结果（如论文插图、实验报告动图、网页展示）
    适配输入：与create_videos_with_rgbs完全一致（PyTorch张量序列，0~1范围）
    Args:
        rgbs: RGB图像序列（列表或批量张量，格式：[B, C, H, W] 或 [单帧1[C,H,W], 单帧2[C,H,W], ...]）
        out_dir: 输出GIF路径（含文件名，如"./output/render_result.gif"）
    """
    # 步骤1：批量处理所有图像（维度转换 + 格式转换，逻辑同视频函数）
    # 列表推导式遍历所有帧，统一转为[H, W, C]格式的NumPy数组
    rgbs = [rgb.permute(1, 2, 0).cpu().detach().numpy() for rgb in rgbs]
    
    # 步骤2：批量数据清洗 + 数值范围转换（0~1 → 0~255 uint8，逻辑同视频函数）
    rgbs = (np.clip(np.nan_to_num(rgbs), 0., 1.) * 255.).astype(np.uint8)
    
    # 步骤3：将处理后的图像序列保存为GIF
    # imageio.mimsave：批量写入GIF的核心函数
    # fps=6：GIF播放帧率（与视频保持一致，保证时序节奏统一）
    imageio.mimsave(out_dir, rgbs, fps=6)