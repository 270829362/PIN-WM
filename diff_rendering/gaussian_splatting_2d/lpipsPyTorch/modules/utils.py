# 该代码是 LPIPS 损失函数的核心辅助工具，包含两个关键函数
from collections import OrderedDict  # 有序字典：保证权重键名顺序与网络结构一致

import torch


def normalize_activation(x, eps=1e-10):
    """
    特征激活归一化（L2归一化，按通道维度）：LPIPS核心步骤，消除特征尺度差异
    数学公式：norm_x = x / (||x||_2 + eps)，其中||x||_2是通道维度的L2范数
    Args:
        x (torch.Tensor): 输入特征（形状[B, C, H, W]，B=批量数，C=通道数，H/W=特征图尺寸）
        eps (float): 极小值，避免分母为0（默认1e-10）
    Returns:
        torch.Tensor: 归一化后的特征（形状与输入一致，通道维度L2范数为1）
    """
    # 计算通道维度的L2范数：sum(x², dim=1, keepdim=True) → [B, 1, H, W]
    # sqrt后得到每个空间位置的通道L2范数（保持维度，方便广播除法）
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    # 特征除以L2范数（广播除法），加eps避免分母为0
    return x / (norm_factor + eps)


def get_state_dict(net_type: str = 'alex', version: str = '0.1'):
    """
    从LPIPS官方仓库下载预训练的线性层权重，并适配自定义LinLayers的键名格式
    官方权重键名（如'lin0.model.0.weight'）→ 自定义键名（如'0.1.weight'）
    Args:
        net_type (str): 特征提取网络类型（'alex'/'squeeze'/'vgg'），默认'alex'
        version (str): LPIPS版本，默认'0.1'（唯一稳定版本）
    Returns:
        OrderedDict: 重命名后的预训练权重（适配自定义LinLayers类）
    """
    # 1. 构建官方权重下载链接（LPIPS官方仓库的预训练权重地址）
    url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/' \
        + f'master/lpips/weights/v{version}/{net_type}.pth'

    # 2. 下载权重（自动适配设备：有GPU则下载到GPU，否则到CPU）
    # torch.hub.load_state_dict_from_url：PyTorch内置的权重下载函数，自动缓存
    old_state_dict = torch.hub.load_state_dict_from_url(
        url, progress=True,  # progress=True：显示下载进度条
        # 设备映射：GPU可用则用默认设备，否则用CPU
        map_location=None if torch.cuda.is_available() else torch.device('cpu')
    )

    # 3. 重命名权重键名（解决官方权重与自定义LinLayers的键名不匹配问题）
    new_state_dict = OrderedDict()  # 有序字典：保证权重顺序与LinLayers一致
    for key, val in old_state_dict.items():
        new_key = key
        # 移除官方键名中的'lin'前缀（如'lin0'→'0'）
        new_key = new_key.replace('lin', '')
        # 移除官方键名中的'model.'（如'0.model.0.weight'→'0.0.weight'）
        new_key = new_key.replace('model.', '')
        # 保存重命名后的键值对
        new_state_dict[new_key] = val

    return new_state_dict