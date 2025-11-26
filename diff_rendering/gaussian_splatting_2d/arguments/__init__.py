# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    """空参数容器类，用于存储从命令行/配置文件解析后的参数组实例"""
    pass

class ParamGroup:
    """参数组基类，封装argparse参数添加逻辑，支持子类继承定义特定参数组"""
    def __init__(self, parser: ArgumentParser, name: str, fill_none = False):
        # 给argparse添加一个命名参数组（方便命令行帮助信息分类）
        self.group = parser.add_argument_group(name)
        # 遍历子类中定义的参数（key=参数名，value=默认值）
        for key, value in vars(self).items():
            shorthand = False  # 是否启用命令行短选项（如--source-path简写为-s）
            if key.startswith("_"):
                shorthand = True
                key = key[1:]  # 去掉下划线前缀，得到最终参数名（如_source_path→source_path）
            param_type = type(value)  # 获取参数类型（用于argparse类型校验）
            # 若fill_none为True，将默认值设为None（用于后续参数合并）
            value = value if not fill_none else None
            
            # 根据是否启用短选项、是否为布尔类型，添加argparse参数
            if shorthand:
                if param_type == bool:
                    # 布尔类型参数：使用action="store_true"（默认False，指定则为True）
                    self.group.add_argument(
                        "--" + key, ("-" + key[0:1]),  # 长选项+短选项（取参数名首字母）
                        default=value, action="store_true"
                    )
                else:
                    # 非布尔类型：指定类型和默认值
                    self.group.add_argument(
                        "--" + key, ("-" + key[0:1]),
                        default=value, type=param_type
                    )
            else:
                if param_type == bool:
                    self.group.add_argument("--" + key, default=value, action="store_true")
                else:
                    self.group.add_argument("--" + key, default=value, type=param_type)

    def extract(self, args):
        """从解析后的所有参数中，提取当前参数组的相关参数，返回GroupParams实例"""
        group = GroupParams()
        for arg_name, arg_value in vars(args).items():
            # 匹配子类中定义的参数（含下划线前缀的原始名）
            if arg_name in vars(self) or ("_" + arg_name) in vars(self):
                setattr(group, arg_name, arg_value)
        return group

class ModelParams(ParamGroup): 
    """模型加载参数组：管理数据读取、模型保存、设备配置等基础参数（对应论文2DGS渲染初始化）"""
    def __init__(self, parser, sentinel=False):
        # 球谐函数阶数（SH_degree）：用于2DGS的方向光照和材质表征（论文3.2节渲染对齐部分）
        self.sh_degree = 3
        # 数据源路径（必填）：存储真实场景图像的根目录（论文中机器人拍摄的多视角图像路径）
        self._source_path = ""
        # 模型保存路径（必填）：2DGS优化后的参数保存目录
        self._model_path = ""
        # 图像文件夹名：数据源路径下存储图像的子文件夹名（默认"images"）
        self._images = "images"
        # 图像分辨率：-1表示使用原始分辨率，否则缩放为指定尺寸
        self._resolution = -1
        # 是否使用白色背景：论文中为简化渲染对齐，默认启用白色背景
        self._white_background = True
        # 数据处理设备：默认使用CUDA（GPU加速，2DGS优化需大量计算）
        self.data_device = "cuda"
        # 是否为评估模式：eval=True时仅加载模型，不执行优化
        self.eval = False
        # 调用父类构造函数，添加参数组到argparse
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        """重写提取方法，将数据源路径转换为绝对路径（避免相对路径错误）"""
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    """流水线执行参数组：管理2DGS渲染流水线的核心逻辑开关（对应论文渲染函数I的实现）"""
    def __init__(self, parser):
        # 是否用Python实现SH转换：默认False（使用C++加速，提升渲染效率）
        self.convert_SHs_python = False
        # 是否用Python计算3D协方差：默认False（C++加速）
        self.compute_cov3D_python = False
        # 深度比率：用于深度相关的渲染优化（论文中未明确调整，保持默认0.0）
        self.depth_ratio = 0.0
        # 调试模式：True时输出渲染中间结果（用于开发调试）
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    """优化参数组：管理2DGS的高斯点参数优化配置（对应论文3.2节渲染参数α的优化）"""
    def __init__(self, parser):
        # 优化迭代次数：论文中默认5000次（原注释30000为备选值）
        self.iterations = 5_000
        # 位置学习率（初始）：高斯点位置的初始更新步长
        self.position_lr_init = 0.00016
        # 位置学习率（最终）：迭代末期的位置更新步长（衰减后）
        self.position_lr_final = 0.0000016
        # 位置学习率延迟因子：前N步学习率按该因子缩小（避免初始震荡）
        self.position_lr_delay_mult = 0.01
        # 位置学习率最大衰减步数：学习率从初始值衰减到最终值的总步数
        self.position_lr_max_steps = 30_000
        # 特征学习率：高斯点颜色/特征向量的更新步长
        self.feature_lr = 0.0025
        # 透明度学习率：高斯点不透明度的更新步长
        self.opacity_lr = 0.05
        # 缩放学习率：高斯点椭圆缩放尺寸的更新步长
        self.scaling_lr = 0.005
        # 旋转学习率：高斯点旋转角度的更新步长
        self.rotation_lr = 0.001
        # 稠密率百分比：初始高斯点的稠密程度（0.01表示稀疏初始化，后续动态致密化）
        self.percent_dense = 0.01
        # DSSIM损失权重：论文中用于渲染对齐的核心损失（与L2损失结合为Lc）
        self.lambda_dssim = 0.2
        # 距离损失权重：用于约束高斯点位置的深度一致性（论文Ld）
        self.lambda_dist = 100.0
        # 法向量损失权重：用于约束高斯点法向量的一致性（论文Ln）
        self.lambda_normal = 0.001
        # 透明度损失权重：用于约束高斯点不透明度的合理性（避免过度透明/不透明）
        self.lambda_opacity = 0.0001
        # 透明度裁剪阈值：低于该阈值的高斯点被裁剪（减少冗余计算）
        self.opacity_cull = 0.05

        # 致密化间隔：每100次迭代补充新的高斯点（提升渲染细节）
        self.densification_interval = 100
        # 透明度重置间隔：每3000次迭代重置高斯点透明度（避免过早收敛）
        self.opacity_reset_interval = 3000
        # 致密化开始迭代：从第500次迭代开始补充高斯点（初始迭代先优化现有点）
        self.densify_from_iter = 500
        # 致密化结束迭代：到第15000次迭代停止补充（后续仅优化现有点）
        self.densify_until_iter = 15_000
        # 致密化梯度阈值：梯度大于该值的区域补充高斯点（聚焦高细节区域）
        self.densify_grad_threshold = 0.0002
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser, output_static_2dgs_path):
    """合并命令行参数和配置文件参数（优先使用命令行参数覆盖配置文件）
    
    Args:
        parser: argparse解析器实例
        output_static_2dgs_path: 2DGS模型保存路径（从外部传入，指定参数保存目录）
    
    Returns:
        Namespace: 合并后的所有参数
    """
    # 读取命令行参数（此处原代码空列表表示仅解析配置文件，命令行参数通过外部传入覆盖）
    args_cmdline = parser.parse_args([])
    # 强制设置模型保存路径（从外部传入，确保参数保存到指定目录）
    args_cmdline.model_path = output_static_2dgs_path

    # 尝试读取模型保存目录下的配置文件（cfg_args）：存储历史优化参数
    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found")
        cfgfile_string = "Namespace()"  # 无配置文件时初始化空参数

    # 解析配置文件中的参数（cfg_args存储为Namespace字符串）
    args_cfgfile = eval(cfgfile_string)

    # 合并参数：先复制配置文件参数，再用命令行参数覆盖非None值
    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v

    # 返回合并后的参数实例
    return Namespace(**merged_dict)