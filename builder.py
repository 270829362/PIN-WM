""" 这个Builder类是 PIN-WM 模型的 “组装核心”，串联了论文的三大关键技术：
高斯溅射场景表示：通过build_static_2dgs加载预训练高斯模型，将 3D 场景编码为可微分的高斯点云；
可微分渲染：通过build_renderer配置渲染器，生成与真实观测一致的图像，为 “无需状态估计的观测损失” 提供基础；
物理参数端到端优化：通过build_optimizer收集可训练的物理参数（质量、摩擦、惯性），设置合理学习率，用 Adam 优化器实现物理参数与渲染参数的联合优化。 """
# 导入系统/基础模块
import os  # 系统路径、文件操作
from argparse import Namespace  # 命令行参数/配置参数的结构化存储
import torch  # PyTorch深度学习框架（核心计算、自动微分）
from functools import partial  # 函数偏置，用于固定渲染函数的默认参数
import numpy as np  # 数值计算、数组处理
import trimesh  # 3D网格/点云处理（论文3D物体表示依赖）
import random  # 随机数生成（配合种子固定保证可复现）
import json  # JSON文件处理（可能用于读取配置/数据）
import copy  # 对象深拷贝（避免参数修改冲突）

# 导入论文核心模块（对应GitHub目录结构）
# 1. 可微分渲染模块：高斯溅射（2D场景适配，论文核心观测损失计算依赖）
from diff_rendering.gaussian_splatting_2d.gaussian_renderer import GaussianModel  # 高斯模型（场景的3D高斯表示）
from diff_rendering.gaussian_splatting_2d.train_2dgs import train_static_2dgs  # 2D高斯溅射训练函数（此处未直接调用，可能备用）
from diff_rendering.gaussian_splatting_2d.utils.system_utils import searchForMaxIteration  # 查找高斯模型最新迭代的点云文件

import diff_rendering.gaussian_splatting_2d.gaussian_renderer as Gaussian_Renderer  # 高斯渲染器核心（生成渲染图像）

# 2. 可微分仿真模块：3D刚体动力学模拟（论文物理参数优化依赖）
from diff_simulation.body.body_mesh import Body_Mesh  # 刚体网格类（描述物体的3D几何形状）
from diff_simulation.force.constant_force import Constant_Force  # 恒定力类（模拟机器人推/拨等操作的作用力）
from diff_simulation.simulator import Simulator  # 可微分仿真器核心（模拟物理运动，支持梯度反向传播）

# 3. 工具模块：配置解析（统一管理实验参数）
from utils.cfg_utils import get_gaussian_args  # 读取高斯溅射相关的配置参数（数据集、渲染设置等）

# 4. 环境模块：强化学习/交互环境（可能用于机器人任务场景定义）
import gym


class Builder():
    """
    核心组件构建器类（对应论文PIN-WM的模块组装逻辑）
    作用：统一初始化/构建项目关键组件，包括：
    - 高斯溅射模型（3D场景表示）
    - 可微分渲染器（生成观测图像，计算观测损失）
    - 优化器（优化物理参数：质量、摩擦系数、惯性等）
    简化组件依赖管理，实现模块化复用
    """
    def __init__(self, all_args):
        """
        初始化构建器，解析所有配置参数
        Args:
            all_args: 全局配置字典（包含数据、渲染、仿真、系统等所有参数，来自配置文件/命令行）
        """
        # 解析各模块配置，转换为Namespace（方便通过属性访问，如self.data_args.batch_size）
        self.data_args = Namespace(**all_args['data_args'])  # 数据集配置（数据路径、批次大小、预处理设置等）
        self.render_args = Namespace(**all_args['render_args'])  # 渲染配置（分辨率、渲染管线参数等）
        self.sim_args = Namespace(**all_args['sim_args'])  # 仿真配置（物理引擎参数、仿真步长等）
        self.sys_args = Namespace(**all_args['sys_args'])  # 系统配置（日志路径、GPU设备、随机种子等）
        
        # 专门获取高斯溅射的配置（整合数据/渲染/系统配置，适配高斯模型需求）
        self.gaussian_args = get_gaussian_args(self.data_args, self.render_args, self.sys_args)
        
        # 固定随机种子（保证实验可复现，论文中所有实验的基础要求）
        self.set_seed(self.sys_args.seed)
    
    def set_seed(self, seed):
        """
        固定所有随机数生成器的种子（Python/numpy/torch/GPU）
        Args:
            seed: 随机种子值（从配置中读取，如1234）
        """
        os.environ['PYTHONHASHSEED'] = str(seed)  # Python哈希种子（影响字典等的迭代顺序）
        random.seed(seed)  # Python随机数种子
        np.random.seed(seed)  # numpy随机数种子
        torch.manual_seed(seed)  # PyTorch CPU随机数种子
        torch.cuda.manual_seed(seed)  # PyTorch单个GPU随机数种子
        torch.cuda.manual_seed_all(seed)  # PyTorch多GPU随机数种子
        torch.backends.cudnn.benchmark = False  # 禁用cuDNN自动优化（避免不同运行的算法差异）
        torch.backends.cudnn.deterministic = True  # 强制cuDNN使用确定性算法（保证结果一致）

    def load_gaussian(self, gaussian_path):
        """
        加载预训练的高斯模型（.ply格式点云文件，存储3D高斯分布的参数）
        Args:
            gaussian_path: 高斯模型文件路径（如point_cloud.ply）
        Returns:
            gaussians: 初始化后的GaussianModel实例（包含场景的3D高斯表示）
        """
        # 初始化高斯模型：sh_degree是球谐函数阶数（控制高斯模型的光照表示精度，论文配置参数）
        gaussians = GaussianModel(self.gaussian_args.dataset.sh_degree)
        # 从.ply文件加载高斯点云参数（位置、颜色、尺度、旋转等）
        gaussians.load_ply(gaussian_path)   
        return gaussians     

    def build_static_2dgs(self):
        """
        构建静态2D高斯溅射模型（论文中3D场景的2D适配渲染基础）
        逻辑：查找最新迭代的预训练高斯模型，加载并返回
        Returns:
            gaussians: 加载完成的高斯模型（用于后续可微分渲染）
        """
        # 查找高斯模型训练的最新迭代目录（模型训练会保存多个迭代，取最后一个即最优）
        loaded_iter = searchForMaxIteration(
            os.path.join(self.gaussian_args.dataset.model_path, "point_cloud")
        )
        # 拼接最新迭代的高斯点云文件路径
        gaussian_path = os.path.join(
            self.gaussian_args.dataset.model_path,  # 高斯模型根路径
            "point_cloud",  # 点云存储目录
            "iteration_" + str(loaded_iter),  # 最新迭代目录（如iteration_30000）
            "point_cloud.ply"  # 高斯点云文件（核心参数存储文件）
        )
        # 调用加载函数，返回高斯模型
        gaussians = self.load_gaussian(gaussian_path)
        return gaussians
    
    def build_renderer(self):
        """
        构建可微分渲染器（论文核心组件，用于生成观测图像，计算观测损失）
        逻辑：配置背景色，固定渲染函数的默认参数，返回渲染器
        Returns:
            Gaussian_Renderer: 配置完成的高斯可微分渲染器
        """
        # 根据配置设置背景色（白色背景[1,1,1]或黑色背景[0,0,0]，论文实验可能用白色避免阴影干扰）
        bg_color = [1.0, 1.0, 1.0] if self.gaussian_args.dataset.white_background else [0.0, 0.0, 0.0]
        # 转换为GPU张量（渲染计算在GPU上进行，提升速度）
        background = torch.tensor(bg_color, device="cuda")
        
        # 用partial固定render函数的默认参数：渲染管线（pipe）和背景色
        # 目的：后续调用渲染时无需重复传入这两个参数，简化观测损失计算流程
        Gaussian_Renderer.render = partial(
            Gaussian_Renderer.render, 
            pipe=self.gaussian_args.pipe,  # 渲染管线配置（如分辨率、抗锯齿等）
            bg_color=background  # 固定背景色
        )
        return Gaussian_Renderer

    def build_optimizer(self, simulator: Simulator):
        """
        构建物理参数优化器（论文核心优化逻辑，端到端优化物理参数）
        逻辑：从仿真器中收集可训练的物理参数，为不同参数设置学习率，创建Adam优化器
        Args:
            simulator: 可微分仿真器实例（包含物理材料属性、刚体运动状态）
        Returns:
            optimizer: Adam优化器（用于优化物理参数：质量、摩擦系数、惯性等）
        """
        # 存储需要优化的参数组（每个元素是字典，包含params、name、lr）
        optim_params = []
        
        # 从仿真器中获取所有物体的物理材料属性（包含质量、摩擦系数、惯性等参数）
        all_physical_materials = simulator.get_all_physical_materials()
        
        # 遍历每个物体的物理材料属性
        for physical_materials in all_physical_materials:
            # 遍历物理材料中的所有参数（key：参数名，value：参数张量）
            for param_name, param_tensor in physical_materials.all.items():
                # 只选择需要梯度更新的参数（requires_grad=True表示可训练）
                if param_tensor.requires_grad is not True:
                    continue

                # 为不同物理参数设置不同学习率（根据参数敏感性调整，提升优化稳定性）
                if param_name == "inertia":  # 惯性参数（对运动影响敏感，用较小学习率）
                    lr = 2e-3  # 学习率：0.002
                else:  # 其他参数（质量、摩擦系数等，用较大学习率加速收敛）
                    lr = 2e-1  # 学习率：0.2
                
                # 构造参数组（PyTorch优化器支持为不同参数设置独立学习率）
                param_group = {
                    'params': param_tensor,  # 待优化的参数张量
                    'name': param_name,  # 参数名（用于日志打印/调试）
                    'lr': lr  # 该参数的学习率
                }
                optim_params.append(param_group)
        
        # 创建Adam优化器（论文常用优化器，自适应学习率，适合物理参数优化）
        optimizer = torch.optim.Adam(optim_params)
        return optimizer