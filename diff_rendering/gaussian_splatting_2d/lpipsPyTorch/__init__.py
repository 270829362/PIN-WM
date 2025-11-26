import torch  # 导入PyTorch：用于张量运算和设备管理（适配GPU/CPU计算）

# 从自定义模块导入LPIPS损失类：该类封装了预训练特征网络和相似度计算逻辑
from .modules.lpips import LPIPS


def lpips(x: torch.Tensor,
          y: torch.Tensor,
          net_type: str = 'alex',
          version: str = '0.1'):
    r"""计算 Learned Perceptual Image Patch Similarity（LPIPS）：感知图像相似度指标
    核心优势：相比L2损失（像素级误差），能更精准捕捉人类视觉敏感的纹理、结构差异，
    适合作为渲染对齐的损失项（确保渲染结果在视觉上与真实一致）

    Arguments:
        x (torch.Tensor): 第一个输入图像张量，形状需为 [N, C, H, W]
            - N: 批量大小（如1张图像则N=1）
            - C: 图像通道数（RGB图像为3，灰度图为1）
            - H/W: 图像高度/宽度（需与y一致）
        y (torch.Tensor): 第二个输入图像张量，形状需与x完全一致（如“渲染图像”与“真实图像”）
        net_type (str): 用于提取特征的预训练网络类型（影响感知差异的计算粒度）
            - 'alex': 基于AlexNet（轻量、速度快，论文中默认选择，平衡精度与效率）
            - 'squeeze': 基于SqueezeNet（更轻量，适合资源受限场景）
            - 'vgg': 基于VGGNet（特征维度更高，精度略优但速度慢）
            默认值: 'alex'
        version (str): LPIPS的版本（对应不同的预训练权重，确保计算逻辑与权重匹配）
            默认值: '0.1'（论文中使用的主流稳定版本）
    """
    # 1. 获取输入张量的设备（GPU/CPU）：确保LPIPS损失函数与输入在同一设备（避免数据搬运错误）
    device = x.device
    # 2. 初始化LPIPS损失函数：加载预训练网络，移到目标设备
    criterion = LPIPS(net_type, version).to(device)
    # 3. 计算并返回LPIPS值：值越小表示x与y的感知相似度越高（视觉差异越小）
    return criterion(x, y)