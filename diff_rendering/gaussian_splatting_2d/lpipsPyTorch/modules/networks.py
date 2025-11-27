# 该代码是 LPIPS 损失函数的特征提取网络核心实现，基于 PyTorch 官方的预训练 CNN（AlexNet/SqueezeNet/VGG16）改造：
from typing import Sequence  # 序列类型注解

from itertools import chain  # 迭代器拼接工具

import torch
import torch.nn as nn
from torchvision import models  # PyTorch官方预训练模型

from .utils import normalize_activation  # 特征激活归一化（如L2归一化）


def get_network(net_type: str):
    """
    根据网络类型获取对应的特征提取网络（LPIPS专用）
    Args:
        net_type: 网络类型，可选['alex', 'squeeze', 'vgg']
    Returns:
        BaseNet子类实例：冻结参数的预训练特征提取网络
    """
    if net_type == 'alex':
        return AlexNet()
    elif net_type == 'squeeze':
        return SqueezeNet()
    elif net_type == 'vgg':
        return VGG16()
    else:
        raise NotImplementedError('choose net_type from [alex, squeeze, vgg].')


class LinLayers(nn.ModuleList):
    """
    LPIPS的线性层列表：对每个特征层级的平方差进行1×1卷积加权（通道数→1）
    所有参数固定（不参与训练），仅作为预训练权重的载体
    """
    def __init__(self, n_channels_list: Sequence[int]):
        """
        初始化线性层列表
        Args:
            n_channels_list: 各特征层级的输出通道数（如AlexNet为[64,192,384,256,256]）
        """
        super(LinLayers, self).__init__([
            # 每个层级对应：Identity（占位） + 1×1卷积（无偏置，通道数映射为1）
            nn.Sequential(
                nn.Identity(),  # 无操作，仅为了保持结构统一
                nn.Conv2d(nc, 1, 1, 1, 0, bias=False)  # 1×1卷积：C→1，无偏置
            ) for nc in n_channels_list
        ])

        # 冻结所有参数：线性层权重由预训练权重加载，训练过程中不更新
        for param in self.parameters():
            param.requires_grad = False


class BaseNet(nn.Module):
    """
    LPIPS特征提取网络的基类：封装通用逻辑（图像归一化、特征提取、参数冻结）
    所有子类（AlexNet/SqueezeNet/VGG16）继承此类，仅需指定特征层和通道数
    """
    def __init__(self):
        super(BaseNet, self).__init__()

        # 注册缓冲区（buffer）：保存均值/标准差（不参与训练，但随模型保存/加载）
        # 数值来源：LPIPS官方实现，匹配预训练网络的输入分布（归一化到[-1,1]后的统计）
        self.register_buffer(
            'mean', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])  # [1,3,1,1]
        self.register_buffer(
            'std', torch.Tensor([.458, .448, .450])[None, :, None, None])     # [1,3,1,1]

    def set_requires_grad(self, state: bool):
        """
        设置网络参数和缓冲区的梯度状态（核心：冻结特征提取网络）
        Args:
            state: True=允许梯度更新，False=冻结（LPIPS中固定为False）
        """
        # chain拼接参数和缓冲区的迭代器，统一设置requires_grad
        for param in chain(self.parameters(), self.buffers()):
            param.requires_grad = state

    def z_score(self, x: torch.Tensor):
        """
        图像z-score归一化：匹配预训练网络的输入分布
        Args:
            x: 输入图像（[B,3,H,W]，已归一化到[-1,1]）
        Returns:
            归一化后的图像（均值为0，标准差为1）
        """
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor):
        """
        前向传播：提取指定层级的卷积特征，并进行激活归一化
        Args:
            x: 输入图像（[B,3,H,W]，需归一化到[-1,1]）
        Returns:
            list[torch.Tensor]: 各目标层级的归一化特征（如AlexNet返回5层特征）
        """
        # 1. 图像z-score归一化（匹配预训练网络的输入分布）
        x = self.z_score(x)

        output = []  # 存储各目标层级的特征
        # 遍历网络的卷积层（按名称遍历，保证顺序）
        for i, (_, layer) in enumerate(self.layers._modules.items(), 1):
            x = layer(x)  # 前向传播通过当前层
            # 若当前层是目标层，提取特征并归一化
            if i in self.target_layers:
                # normalize_activation：特征激活归一化（如L2归一化），LPIPS关键步骤
                output.append(normalize_activation(x))
            # 提取完所有目标层特征后提前终止（避免多余计算）
            if len(output) == len(self.target_layers):
                break
        return output


class SqueezeNet(BaseNet):
    """
    SqueezeNet特征提取网络（LPIPS专用）
    基于torchvision预训练SqueezeNet1_1，仅保留卷积层，冻结所有参数
    """
    def __init__(self):
        super(SqueezeNet, self).__init__()

        # 加载预训练的SqueezeNet1_1特征层（去掉分类头）
        self.layers = models.squeezenet1_1(True).features
        # 目标特征层：LPIPS论文中指定的层级（对应SqueezeNet的关键卷积层）
        self.target_layers = [2, 5, 8, 10, 11, 12, 13]
        # 各目标层的输出通道数（与LinLayers的输入匹配）
        self.n_channels_list = [64, 128, 256, 384, 384, 512, 512]

        # 冻结所有参数：特征提取网络作为固定提取器，不参与训练
        self.set_requires_grad(False)


class AlexNet(BaseNet):
    """
    AlexNet特征提取网络（LPIPS专用，默认选择）
    基于torchvision预训练AlexNet，仅保留卷积层，冻结所有参数
    """
    def __init__(self):
        super(AlexNet, self).__init__()

        # 加载预训练的AlexNet特征层（去掉分类头）
        self.layers = models.alexnet(True).features
        # 目标特征层：LPIPS论文中指定的层级（AlexNet的conv1-conv5）
        self.target_layers = [2, 5, 8, 10, 12]
        # 各目标层的输出通道数
        self.n_channels_list = [64, 192, 384, 256, 256]

        # 冻结所有参数
        self.set_requires_grad(False)


class VGG16(BaseNet):
    """
    VGG16特征提取网络（LPIPS专用）
    基于torchvision预训练VGG16，仅保留卷积层，冻结所有参数
    """
    def __init__(self):
        super(VGG16, self).__init__()

        # 加载预训练的VGG16特征层（PyTorch新版权重加载方式）
        self.layers = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        # 目标特征层：LPIPS论文中指定的层级（VGG16的conv1_2, conv2_2, conv3_3, conv4_3, conv5_3）
        self.target_layers = [4, 9, 16, 23, 30]
        # 各目标层的输出通道数
        self.n_channels_list = [64, 128, 256, 512, 512]

        # 冻结所有参数
        self.set_requires_grad(False)