""" 该代码实现了Learned Perceptual Image Patch Similarity (LPIPS) 损失函数，核心是通过预训练的卷积神经网络（如 AlexNet/SqueezeNet/VGG）提取图像的感知特征，计算特征空间的逐通道平方差异，再通过预训练的线性层加权求和，最终得到贴合人类视觉感知的图像相似度指标（值越小表示图像越相似）。
LPIPS 相比传统的 MSE（均方误差）更符合人类视觉认知，广泛用于图像生成、风格迁移、超分辨率等任务的损失函数，是感知质量评估的核心工具。 """
import torch
import torch.nn as nn

from .networks import get_network, LinLayers  # 导入特征提取网络/线性层定义
from .utils import get_state_dict  # 导入预训练权重加载工具


class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    def __init__(self, net_type: str = 'alex', version: str = '0.1'):
        """
        初始化LPIPS损失函数
        Args:
            net_type: 特征提取网络类型（'alex'/'squeeze'/'vgg'），默认'alex'（性能/速度平衡）
            version: LPIPS版本，仅支持'0.1'（官方发布的唯一稳定版本）
        """
        # 版本校验：仅支持v0.1（当前实现仅适配该版本的预训练权重）
        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        # 1. 加载预训练的特征提取网络（去掉分类头，保留卷积层）
        # 作用：提取图像不同层级的感知特征（如AlexNet的conv1-conv5）
        self.net = get_network(net_type)

        # 2. 初始化线性层（LinLayers）：对不同层级的特征差异进行加权
        # self.net.n_channels_list：特征提取网络各层的输出通道数（如AlexNet为[64,192,384,256,256]）
        self.lin = LinLayers(self.net.n_channels_list)
        
        # 3. 加载线性层的预训练权重（与net_type/version匹配）
        # 预训练权重是LPIPS的核心：通过人类主观评分训练，使加权后的特征差贴合视觉感知
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        前向传播：计算两张图像的LPIPS距离
        Args:
            x: 输入图像1（形状：[B, 3, H, W]，需归一化到[-1,1]，与预训练网络输入一致）
            y: 输入图像2（形状与x相同）
        Returns:
            torch.Tensor: LPIPS距离（形状：[B, 1]，值越小表示图像感知相似度越高）
        """
        # 1. 提取两张图像的多层感知特征
        # feat_x/feat_y：列表，每个元素是对应层级的特征图（如AlexNet返回5层特征）
        feat_x, feat_y = self.net(x), self.net(y)

        # 2. 计算各层级特征的逐通道平方差
        # diff：列表，每个元素形状与对应层级特征一致（[B, C, H, W]）
        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]

        # 3. 线性层加权 + 空间维度平均
        # l(d)：线性层对平方差特征加权（将通道数映射为1）
        # mean((2,3), True)：对H/W维度求平均，保留维度（[B, 1, 1, 1]）
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        # 4. 所有层级的结果拼接并求和
        # torch.cat(res, 0)：按通道维度拼接（[num_layers, B, 1, 1]）
        # sum(0, True)：按层级求和，保留维度（[1, B, 1, 1]）→ 最终输出[B, 1]
        return torch.sum(torch.cat(res, 0), 0, True)