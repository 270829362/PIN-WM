""" 这段代码是论文中渲染质量评估的核心工具函数，直接服务于 PIN-WM 的 “可微分渲染 - 观测损失” 逻辑：
评估对象：输入的images是高斯溅射渲染器生成的虚拟图像，gt_images是真实数据集采集的图像（如机器人摄像头拍摄的画面）；
核心作用：
训练阶段：可作为观测损失的一部分（或辅助损失），引导高斯渲染器学习 “生成与真实图像一致的虚拟观测”；
评估阶段：作为量化指标，在论文实验表格中展示渲染质量（如 PSNR 值对比，证明渲染器的有效性）；
关键特性：
支持批量计算（适配深度学习的批量训练流程）；
双精度累加（避免多次迭代后的精度丢失）；
与 PyTorch 张量兼容（可融入自动微分计算图，支持端到端训练）。
简单说，这个函数的核心价值是 “量化渲染图像与真实图像的差距”，是连接高斯渲染模块与模型优化 / 评估的关键指标工具。
 """
# 导入核心依赖与工具函数
from diff_rendering.gaussian_splatting_2d.utils.image_utils import psnr
# 导入四元数标准化工具（注：当前函数未使用，可能是冗余导入或后续扩展预留）
from utils.quaternion_utils import quaternion_standardize
import torch  # PyTorch框架（支持张量计算、自动微分，适配模型训练）

def mean_psnr(images, gt_images):
    """
    计算批量图像的平均峰值信噪比（PSNR）—— 评估预测图像与真实图像的相似度
    核心意义：PSNR是图像质量评估的经典指标，值越高表示图像越接近（误差越小）
    适配场景：论文中用于评估高斯渲染器生成的图像（images）与真实观测图像（gt_images）的匹配度
    （PSNR计算逻辑：基于像素值误差，公式为 PSNR = 10×log10(MAX²/MSE)，MAX为像素最大值（通常255或1.0））
    
    Args:
        images: 预测图像批量（Tensor类型，形状通常为[B, H, W, C]或[B, C, H, W]，B=批量数，H=高，W=宽，C=通道数）
                论文中特指高斯渲染器生成的场景图像
        gt_images: 真实图像批量（Tensor类型，形状与images完全一致）
                论文中特指数据集采集的真实观测图像（ground truth）
    Returns:
        psnr_mean: 批量图像的平均PSNR值（Tensor类型，标量，值越高表示渲染质量越好）
    """
    # 初始化PSNR累加器（用于汇总所有图像对的PSNR）
    psnr_mean = 0.0
    
    # 遍历批量中的每一对预测图像与真实图像
    for image, gt_image in zip(images, gt_images):
        # 1. 计算单张图像的PSNR（调用image_utils中的psnr函数，基于像素MSE误差）
        # 2. .mean()：对单张图像的所有像素/通道的PSNR取均值（确保输出标量）
        # 3. .double()：转换为double精度（避免累加时的精度损失）
        # 4. 累加到总PSNR
        psnr_mean += psnr(image, gt_image).mean().double()
    
    # 计算批量平均PSNR（总PSNR ÷ 图像数量）
    psnr_mean /= len(images)
    
    # 返回平均PSNR（用于训练损失计算或实验结果评估）
    return psnr_mean