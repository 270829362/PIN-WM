""" 这段代码是高斯溅射（2DGS）模块的参数 “总调度”，解决了 “多来源参数整合” 和 “参数结构化管理” 两个核心问题，为前序Builder类提供关键输入：
参数来源整合：将数据集配置（data_args）、渲染配置（render_args）、系统配置（sys_args）合并，同时补充高斯溅射专用的默认参数（如保存迭代次数、TensorBoard 端口），避免参数冲突；
参数结构化：通过Gaussian_args（NamedTuple）将零散的参数按功能分类（数据集、优化、渲染管线），后续Builder类加载高斯模型（GaussianModel）、构建渲染器时，可直接通过属性访问所需参数（如gaussian_args.pipe.image_size获取渲染分辨率）；
适配论文需求：补充的 3000 次迭代保存、中期 / 最终迭代测试，是论文实验的标准配置（确保模型训练过程可追溯、性能可评估）；统一的模型保存路径（sys_args.output_path/static）也符合论文代码的工程化规范。 """
# 导入参数解析相关模块（用于命令行参数/配置参数处理）
from argparse import ArgumentParser, Namespace
from typing import NamedTuple  # 用于定义结构化的参数类（不可变，访问更清晰）
import os  # 系统路径操作（用于拼接模型保存路径）
import argparse  # 命令行参数解析核心模块

# 导入高斯溅射（2DGS）的专用参数类（来自论文的diff_rendering模块）
# 这些类封装了高斯溅射的核心配置项，按功能拆分便于管理
from diff_rendering.gaussian_splatting_2d.arguments import (
    ModelParams,    # 模型参数（数据集路径、球谐函数阶数等）
    PipelineParams, # 渲染管线参数（分辨率、抗锯齿、背景色等）
    OptimizationParams,  # 优化参数（学习率、迭代次数、正则化等）
    GroupParams     # 参数组基类（上述三个类的父类，提供参数提取方法）
)

def config_parser():
    '''
    定义顶层命令行参数解析器（仅用于读取配置文件路径）
    作用：让用户通过命令行传入配置文件路径（--config参数），后续从配置文件加载详细参数
    Returns:
        parser: 命令行参数解析器实例
    '''
    # 创建解析器，formatter_class指定参数帮助信息的格式（显示默认值）
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 添加必填参数--config：指定配置文件路径（如json/yaml文件，存储所有实验参数）
    parser.add_argument('--config', required=True,
                        help='config file path (配置文件路径，存储数据、渲染、仿真等所有参数)')
    return parser

def get_combined_args(args1, args2):
    '''
    合并两个Namespace类型的参数对象（后一个参数覆盖前一个的重复字段）
    作用：整合不同来源的参数（如配置文件参数、命令行参数、模块专属参数），避免参数冲突
    Args:
        args1: 基础参数对象（被覆盖的参数）
        args2: 补充/覆盖参数对象（用于覆盖args1中的重复参数）
    Returns:
        Namespace: 合并后的参数对象（结构化，支持属性访问）
    '''
    # 将Namespace转换为字典（便于遍历和修改参数）
    args1_dict = vars(args1)
    args2_dict = vars(args2)
    
    # 遍历args2的所有参数，若参数在args1中存在则覆盖，不存在则新增
    for k, v in args2_dict.items():
        if k in args1_dict:
            args1_dict[k] = v
    
    # 将合并后的字典转回Namespace（保持参数访问的一致性）
    return Namespace(**args1_dict)

class Gaussian_args(NamedTuple):
    '''
    结构化的高斯溅射（2DGS）参数类（基于NamedTuple，不可变，参数清晰）
    作用：统一存储高斯溅射所需的所有核心参数，避免零散传递，便于后续模块调用
    字段说明：
        dataset: GroupParams实例 → 数据集/模型相关参数（从ModelParams提取）
        opt: GroupParams实例 → 优化相关参数（从OptimizationParams提取）
        pipe: GroupParams实例 → 渲染管线相关参数（从PipelineParams提取）
        testing_iterations: list → 需要进行测试的迭代次数（如[1500, 3000]，在这些迭代后评估模型）
        saving_iterations: list → 需要保存模型的迭代次数（如[1000, 3000]，在这些迭代后保存 checkpoint）
        checkpoint_iterations: list → 需要加载checkpoint的迭代次数（用于断点续训）
        checkpoint: str → 初始checkpoint路径（用于断点续训或预训练模型加载）
    '''
    dataset: GroupParams
    opt: GroupParams
    pipe: GroupParams
    testing_iterations: int
    saving_iterations: int
    checkpoint_iterations: int
    checkpoint: str

def get_gaussian_args(data_args, render_args, sys_args):
    '''
    核心函数：整合多来源参数，生成高斯溅射（2DGS）的最终配置（Gaussian_args实例）
    逻辑：1. 初始化高斯溅射的参数解析器 → 2. 合并外部参数（data_args/render_args）→ 3. 补充默认参数 → 4. 构造结构化参数对象
    Args:
        data_args: 数据集相关参数（来自配置文件的data_args字段，如数据路径、批次大小）
        render_args: 渲染相关参数（来自配置文件的render_args字段，如背景色、分辨率）
        sys_args: 系统相关参数（来自配置文件的sys_args字段，如输出路径、GPU设备）
    Returns:
        Gaussian_args: 结构化的高斯溅射参数对象（供GaussianModel、渲染器等模块使用）
    '''
    # 1. 创建高斯溅射专用的命令行参数解析器（用于定义2DGS所需的所有参数）
    parser = ArgumentParser(description="2dgs script parameters (高斯溅射模块的参数解析器)")
    
    # 2. 注册高斯溅射的专用参数类（自动给parser添加对应参数）
    lp = ModelParams(parser)    # 注册模型参数（如--model_path、--sh_degree等）
    op = OptimizationParams(parser)  # 注册优化参数（如--lr、--iterations等）
    pp = PipelineParams(parser) # 注册渲染管线参数（如--image_size、--white_background等）
    
    # 3. 添加额外的辅助参数（高斯溅射训练/测试所需的补充配置）
    parser.add_argument('--ip', type=str, default="127.0.0.1", help='TensorBoard服务IP（默认本地）')
    parser.add_argument('--port', type=int, default=6009, help='TensorBoard服务端口（默认6009）')
    # parser.add_argument('--detect_anomaly', action='store_true', default=False, help='启用PyTorch梯度异常检测')
    # 测试迭代次数：默认空列表，后续补充（在指定迭代次数后运行测试）
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[], help='需要执行测试的迭代次数列表')
    # 保存迭代次数：默认仅保存第1次迭代，后续补充更多关键迭代
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1], help='需要保存模型的迭代次数列表')
    parser.add_argument("--quiet", action="store_true", help='静默模式（减少日志输出）')
    # 断点续训迭代次数：默认空列表（不加载历史checkpoint）
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[], help='需要加载checkpoint的迭代次数列表')
    # 初始checkpoint路径：默认None（从头训练，若指定则从该路径加载模型继续训练）
    parser.add_argument("--start_checkpoint", type=str, default=None, help='初始checkpoint路径（断点续训用）')
    
    # 4. 解析空列表（关键！：此处不读取命令行参数，仅用parser的参数定义来初始化默认值）
    # 因为参数已通过data_args/render_args/sys_args传入，无需再从命令行读取
    args = parser.parse_args([])
    
    # 5. 合并外部参数（优先级：render_args > data_args > 默认参数，后传入的覆盖前一个）
    # 合并数据参数（data_args）：更新数据集路径、预处理设置等
    args = get_combined_args(args, data_args)
    # 合并渲染参数（render_args）：更新渲染分辨率、背景色等
    args = get_combined_args(args, render_args)
    
    # 6. 补充关键的保存/测试迭代次数（确保核心节点的模型被保存和测试）
    args.save_iterations.append(3000)  # 固定保存第3000次迭代的模型（论文常用训练迭代节点）
    args.save_iterations.append(args.iterations)  # 保存最终迭代（训练结束时的模型）
    args.save_iterations.append(args.iterations / 2)  # 保存中期迭代（训练中途的模型，便于对比）
    # 补充测试迭代次数（在最终迭代和中期迭代时执行测试，评估模型性能）
    args.test_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations / 2)
    
    # 7. 若系统配置指定了输出路径，更新模型保存路径（统一存储到输出目录下的static文件夹）
    if sys_args.output_path is not None:
        args.model_path = os.path.join(sys_args.output_path, "static")  # 拼接路径：output/static（存储高斯模型）
    
    # 8. 从合并后的args中提取各模块专用参数，构造结构化的Gaussian_args对象并返回
    return Gaussian_args(
        dataset=lp.extract(args),  # 提取模型/数据集参数（ModelParams的extract方法）
        opt=op.extract(args),      # 提取优化参数（OptimizationParams的extract方法）
        pipe=pp.extract(args),     # 提取渲染管线参数（PipelineParams的extract方法）
        testing_iterations=args.test_iterations,  # 测试迭代次数列表
        saving_iterations=args.save_iterations,    # 保存迭代次数列表
        checkpoint_iterations=args.checkpoint_iterations,  # 断点续训迭代次数列表
        checkpoint=args.start_checkpoint  # 初始checkpoint路径
    )