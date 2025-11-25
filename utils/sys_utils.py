# 这段代码是深度学习实验的日志管理核心工具，专门适配 PIN-WM 这类需要记录物理参数、仿真结果、渲染图像的复杂实验，核心价值是 “统一输出管理 + 多格式日志记录 + 实验可复现”，关键作用可概括为 3 类：
# 导入核心依赖库
import os  # 文件/目录操作（创建输出文件夹、路径拼接）
import uuid  # 生成唯一标识符（用于实验输出目录命名）
from argparse import Namespace  # 命令行参数解析辅助（此处备用）
import sys  # 系统相关操作（标准输出stdout）
import warnings  # 警告信息处理
import time  # 时间相关操作（生成时间戳）
from collections import defaultdict  # 带默认值的字典（日志键值对存储）
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union  # 类型注解
import datetime  # 日期时间处理（日志目录时间戳）
import numpy as np  # 数值计算（日志中数组数据处理）
import torch  # 深度学习框架（日志中张量数据处理）
from matplotlib import pyplot as plt  # 绘图库（日志中图表数据处理）

# 导入wandb（注释未启用，用于云端日志管理，如实验可视化、多人协作）
# import wandb

# 尝试导入TensorBoard日志写入器（若未安装则设为None）
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# 日志级别常量定义（控制日志输出阈值，数值越大输出越精简）
DEBUG = 10    # 调试级别（最详细，用于开发调试）
INFO = 20     # 信息级别（常规实验信息，如训练进度、参数）
WARN = 30     # 警告级别（非致命错误，如参数不匹配、数据异常）
ERROR = 40    # 错误级别（致命错误，如文件不存在、内存溢出）
DISABLED = 50 # 禁用级别（关闭所有日志输出）


def prepare_output_and_logger(all_args, need_logger=False):
    """
    实验初始化核心函数：创建输出目录 + 保存实验参数 + 初始化多模块日志器
    核心作用：统一管理实验输出（参数、日志、结果文件），确保实验可复现、可追溯
    适配场景：PIN-WM论文的物理参数优化实验（需记录训练损失、物理参数、渲染图像等）
    Args:
        all_args: 实验所有参数（字典格式，包含系统参数sys_args、数据参数data_args等）
        need_logger: 是否需要初始化日志器（True=训练阶段，False=仅创建输出目录）
    Returns:
        all_args: 更新后的参数（补充输出目录路径）
        loggers: 多模块日志器字典（key=模块名，value=Logger实例，支持多模块独立日志）
    """
    # 步骤1：生成唯一输出目录（若未指定）
    if not all_args['sys_args']['output_path']:
        # 优先使用OAR集群的任务ID（若在集群运行），否则生成UUID（通用唯一标识符）
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        # 输出目录路径：./output/唯一标识符前10位（简洁且唯一）
        all_args['sys_args']['output_path'] = os.path.join("./output", unique_str[0:10])
    
    # 步骤2：创建输出目录（exist_ok=True避免目录已存在报错）
    print("Output folder: {}".format(all_args['sys_args']['output_path']))
    os.makedirs(all_args['sys_args']['output_path'], exist_ok=True)
    
    # 步骤3：保存所有实验参数到文件（确保实验可复现，记录参数配置）
    with open(os.path.join(all_args['sys_args']['output_path'], "all_args"), 'w') as args_log_f:
        args_log_f.write(str((all_args)))

    # 步骤4：初始化多模块日志器（仅当need_logger=True时）
    loggers = {}
    if need_logger:
        # 生成精确时间戳（用于日志目录命名，区分同一实验的多次运行）
        timestamp = time.time()
        dt_obj = datetime.datetime.fromtimestamp(timestamp)
        formatted_time = dt_obj.strftime("%Y-%m-%d_%H-%M-%S")  # 格式：年-月-日_时-分-秒
        # 日志根目录：输出目录/logs/时间戳
        logs_dir = os.path.join(all_args['sys_args']['output_path'], "logs", formatted_time)
        os.makedirs(logs_dir, exist_ok=True)
        print("log_dir:", logs_dir)

        # 定义需要日志记录的模块（PIN-WM实验核心模块）
        modules_name = ["static", "dynamic", "policy"]
        # 日志输出格式（支持TensorBoard可视化，可扩展stdout/log等）
        format_strings = ["tensorboard"]
        
        # 为每个模块创建独立日志目录和日志器
        for module_name in modules_name:
            # 创建模块结果保存目录（用于存储该模块的中间结果/模型）
            os.makedirs(os.path.join(all_args['sys_args']['output_path'], module_name), exist_ok=True)
            # 模块日志目录：日志根目录/模块名
            log_module_name_dir = os.path.join(logs_dir, module_name)
            os.makedirs(log_module_name_dir, exist_ok=True)
            
            # 特殊处理dynamic模块：若有多个训练样本（train_num），为每个样本创建独立日志器
            if module_name == "dynamic":
                loggers[module_name] = []
                if "train_num" in all_args['data_args']:
                    for index in range(all_args['data_args']["train_num"]):
                        log_sub_dir = os.path.join(log_module_name_dir, str(index))  # 每个样本的日志子目录
                        loggers[module_name].append(configure_logger(log_sub_dir, format_strings))
            else:
                # static/policy模块：单日志器（全局记录该模块日志）
                loggers[module_name] = configure_logger(log_module_name_dir, format_strings)

    return all_args, loggers


class Video(object):
    """
    视频数据封装类——统一日志系统中的视频数据格式
    核心用途：将视频帧和帧率打包，方便日志器（如TensorBoard）统一处理
    适配场景：PIN-WM实验中记录物体运动仿真视频、渲染结果视频等
    """
    def __init__(self, frames: Union[torch.Tensor, np.ndarray], fps: Union[float, int]):
        """
        Args:
            frames: 视频帧数据（Tensor或np.array，形状通常为[T, C, H, W]，T=帧数，C=通道数）
            fps: 视频帧率（每秒帧数，控制视频播放速度）
        """
        self.frames = frames  # 视频帧数据
        self.fps = fps        # 帧率


class Figure(object):
    """
    图表数据封装类——统一日志系统中的matplotlib图表格式
    核心用途：将matplotlib绘制的图表（如损失曲线、参数趋势图）打包，支持自动关闭图表
    适配场景：PIN-WM实验中记录训练损失曲线、物理参数收敛趋势图等
    """
    def __init__(self, figure: plt.figure, close: bool):
        """
        Args:
            figure: matplotlib图表对象（plt.plot()生成的figure）
            close: 是否在日志记录后关闭图表（True=避免内存泄漏，False=保留图表供后续修改）
        """
        self.figure = figure  # matplotlib图表
        self.close = close    # 记录后是否关闭图表


class Image(object):
    """
    图像数据封装类——统一日志系统中的图像数据格式
    核心用途：将图像数据和数据格式（如HWC/NCHW）打包，避免日志器处理时格式混淆
    适配场景：PIN-WM实验中记录高斯渲染图像、真实观测图像、接触点可视化图像等
    """
    def __init__(self, image: Union[torch.Tensor, np.ndarray, str], dataformats: str):
        """
        Args:
            image: 图像数据（Tensor/np.array/图像路径字符串）
            dataformats: 图像数据格式（如'HWC'=高×宽×通道，'NCHW'=批量×通道×高×宽，适配TensorBoard）
        """
        self.image = image        # 图像数据
        self.dataformats = dataformats  # 数据格式说明


class FormatUnsupportedError(NotImplementedError):
    """
    自定义异常类——处理不支持的日志格式
    核心用途：当日志数据类型与输出格式不匹配时（如视频数据用文本日志输出），抛出明确错误
    """
    def __init__(self, unsupported_formats: Sequence[str], value_description: str):
        if len(unsupported_formats) > 1:
            format_str = f"formats {', '.join(unsupported_formats)} are"
        else:
            format_str = f"format {unsupported_formats[0]} is"
        # 调用父类构造函数，生成详细错误信息
        super(FormatUnsupportedError, self).__init__(
            f"The {format_str} not supported for the {value_description} value logged.\n"
            f"You can exclude formats via the `exclude` parameter of the logger's `record` function."
        )


class KVWriter(object):
    """
    键值对日志写入器基类——定义键值对日志（如loss=0.1, acc=0.9）的统一接口
    所有支持键值对输出的日志格式（如TensorBoard、文本日志）都需继承此类并实现write/close
    """
    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:
        """
        写入键值对日志（需子类实现具体逻辑）
        Args:
            key_values: 键值对字典（如{'train/loss': 0.5, 'val/acc': 0.8}）
            key_excluded: 排除的输出格式（如{'train/loss': ('stdout',)}表示该键不输出到控制台）
            step: 日志步骤（如训练迭代次数，用于TensorBoard的时序可视化）
        """
        raise NotImplementedError

    def close(self) -> None:
        """关闭日志资源（需子类实现，如关闭文件句柄、TensorBoard写入器）"""
        raise NotImplementedError


class SeqWriter(object):
    """
    序列日志写入器基类——定义序列日志（如打印调试信息、错误信息）的统一接口
    所有支持序列输出的日志格式（如控制台、文本日志）都需继承此类并实现write_sequence
    """
    def write_sequence(self, sequence: List) -> None:
        """
        写入序列日志（需子类实现具体逻辑）
        Args:
            sequence: 日志序列（如["训练开始", "迭代100次", "损失0.3"]）
        """
        raise NotImplementedError


class HumanOutputFormat(KVWriter, SeqWriter):
    """
    人类可读日志格式——支持键值对（文本表格）和序列（纯文本）日志，输出到控制台或文本文件
    核心特点：格式简洁易读，适合实时查看实验进度（如训练损失、迭代次数）
    输出示例：
    ----------------------------
    | train/loss               | 0.5     |
    | val/acc                  | 0.8     |
    ----------------------------
    """
    def __init__(self, filename_or_file: Union[str, TextIO], max_length: int = 36):
        """
        Args:
            filename_or_file: 输出目标（文本文件路径或文件对象，如sys.stdout=控制台）
            max_length: 键/值的最大长度（超过截断，避免日志过长）
        """
        self.max_length = max_length
        # 若输入为文件路径，创建文件对象并标记为"自有文件"（需手动关闭）
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            self.own_file = True
        else:
            # 若输入为文件对象（如sys.stdout），直接使用，不标记为自有文件
            assert hasattr(filename_or_file, "write"), f"Expected file or str, got {filename_or_file}"
            self.file = filename_or_file
            self.own_file = False

    def write(self, key_values: Dict, key_excluded: Dict, step: int = 0) -> None:
        """写入键值对日志（文本表格格式）"""
        key2str = []
        tag = None
        tags = set()
        # 遍历所有键值对和排除规则
        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):
            # 跳过需要排除当前格式的键（如exclude=('stdout',)则不输出到控制台）
            if excluded is not None and ("stdout" in excluded or "log" in excluded):
                continue

            # 不支持视频/图表/图像的文本输出，抛出异常
            elif isinstance(value, Video):
                raise FormatUnsupportedError(["stdout", "log"], "video")
            elif isinstance(value, Figure):
                raise FormatUnsupportedError(["stdout", "log"], "figure")
            elif isinstance(value, Image):
                raise FormatUnsupportedError(["stdout", "log"], "image")

            # 格式化数值（浮点数保留3位有效数字，其他转为字符串）
            elif isinstance(value, float):
                value_str = f"{value:<8.3g}"
            else:
                value_str = str(value)

            # 处理键的标签前缀（如"train/loss"中的"train/"，作为分组标题）
            if key.find("/") > 0:
                tag = key[: key.find("/") + 1]
                if tag not in tags:
                    tags.add(tag)
                    key2str.append((self._truncate(tag), ""))  # 标签行，值为空
            # 移除键中的标签前缀，使日志更简洁
            if tag is not None and tag in key:
                key = str("   " + key[len(tag) :])
            # 记录截断后的键和值
            key2str.append((self._truncate(key), self._truncate(value_str)))

        # 若没有可输出的键值对，警告并返回
        if len(key2str) == 0:
            warnings.warn("Tried to write empty key-value dict")
            return
        else:
            # 计算键和值的最大长度，用于表格对齐
            keys, vals = list(zip(*key2str))
            key_width = max(map(len, keys))
            val_width = max(map(len, vals))

        # 生成表格格式的日志内容
        dashes = "-" * (key_width + val_width + 7)  # 分隔线
        lines = [dashes]
        for key, value in key2str:
            key_space = " " * (key_width - len(key))  # 键的填充空格
            val_space = " " * (val_width - len(value))  # 值的填充空格
            lines.append(f"| {key}{key_space} | {value}{val_space} |")
        lines.append(dashes)
        # 写入文件并刷新（确保实时输出）
        self.file.write("\n".join(lines) + "\n")
        self.file.flush()

    def _truncate(self, string: str) -> str:
        """截断过长的字符串（超过max_length时末尾加...）"""
        if len(string) > self.max_length:
            string = string[: self.max_length - 3] + "..."
        return string

    def write_sequence(self, sequence: List) -> None:
        """写入序列日志（纯文本，元素间用空格分隔）"""
        sequence = list(sequence)
        for i, elem in enumerate(sequence):
            self.file.write(elem)
            if i < len(sequence) - 1:  # 最后一个元素后不加空格
                self.file.write(" ")
        self.file.write("\n")
        self.file.flush()

    def close(self) -> None:
        """关闭日志文件（仅关闭自有文件）"""
        if self.own_file:
            self.file.close()


class TensorBoardOutputFormat(KVWriter):
    """
    TensorBoard日志格式——支持键值对日志（标量、图像、视频、图表），输出到TensorBoard可可视化目录
    核心特点：支持时序可视化（如损失曲线）、图像/视频预览，是深度学习实验的核心日志格式
    适配场景：PIN-WM实验中可视化训练损失、物理参数趋势、渲染图像对比、仿真视频等
    """
    def __init__(self, folder: str):
        """
        Args:
            folder: TensorBoard日志保存目录（通过tensorboard --logdir=该目录启动可视化）
        """
        # 检查TensorBoard是否安装
        assert SummaryWriter is not None, (
            "tensorboard is not installed, you can use "
            "pip install tensorboard to do so"
        )
        # 初始化TensorBoard写入器
        self.writer = SummaryWriter(log_dir=folder)

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:
        """写入键值对日志（支持标量、图像、视频、图表）"""
        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):
            # 跳过需要排除TensorBoard格式的键
            if excluded is not None and "tensorboard" in excluded:
                continue

            # 处理标量/文本（如损失、准确率、字符串信息）
            if isinstance(value, (np.ScalarType, torch.Tensor)):
                if isinstance(value, str):
                    self.writer.add_text(key, value, step)  # 文本日志
                else:
                    self.writer.add_scalar(key, value, step)  # 标量日志（时序可视化）

            # 处理视频（如仿真视频、渲染视频）
            if isinstance(value, Video):
                self.writer.add_video(key, value.frames, step, value.fps)

            # 处理matplotlib图表（如损失曲线、参数趋势图）
            if isinstance(value, Figure):
                self.writer.add_figure(key, value.figure, step, close=value.close)

            # 处理图像（如渲染图像、真实图像、接触点可视化图）
            if isinstance(value, Image):
                self.writer.add_image(
                    key, value.image, step, dataformats=value.dataformats
                )

        # 刷新写入器（确保日志实时写入文件）
        self.writer.flush()

    def close(self) -> None:
        """关闭TensorBoard写入器，释放资源"""
        if self.writer:
            self.writer.close()
            self.writer = None


# class WANDBOutputFormat(KVWriter):
#     """
#     WandB日志格式——支持云端日志管理（注释未启用，需安装wandb）
#     核心特点：云端存储、多人协作、实验对比、自动生成报告
#     适配场景：大规模实验、团队协作、论文实验结果展示
#     """
#     def __init__(self):
#         if wandb.run is None:
#             raise ValueError("WandB is not initialized")

#     def write(
#         self,
#         key_values: Dict[str, Any],
#         key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
#         step: int = 0,
#     ) -> None:
#         wandb_dict = {}
#         for (key, value), (_, excluded) in zip(
#             sorted(key_values.items()), sorted(key_excluded.items())
#         ):
#             if excluded is not None and "wandb" in excluded:
#                 continue
#             if isinstance(value, np.ScalarType):
#                 wandb_dict[key] = value
#             if isinstance(value, torch.Tensor):
#                 wandb_dict[key] = wandb.Histogram(value)  # 张量分布日志
#             if isinstance(value, Video):
#                 wandb_dict[key] = wandb.Video(value.frames, fps=value.fps, format="gif")  # 视频日志
#             if isinstance(value, Image):
#                 wandb_dict[key] = wandb.Image(value.image)  # 图像日志
#         wandb.log(wandb_dict)  # 上传到WandB云端

#     def close(self) -> None:
#         pass


def make_output_format(_format: str, log_dir: str, log_suffix: str = "") -> KVWriter:
    """
    根据格式字符串创建对应的日志写入器
    核心作用：统一日志格式的创建逻辑，支持扩展新格式（如添加CSV格式）
    Args:
        _format: 日志格式（"stdout"=控制台，"log"=文本文件，"tensorboard"=TensorBoard）
        log_dir: 日志保存目录（仅对文件类格式有效）
        log_suffix: 日志文件后缀（如".txt"，默认空）
    Returns:
        对应的日志写入器实例（KVWriter子类）
    """
    os.makedirs(log_dir, exist_ok=True)  # 确保日志目录存在
    if _format == "stdout":
        return HumanOutputFormat(sys.stdout)  # 控制台输出（人类可读格式）
    elif _format == "log":
        # 文本文件输出（人类可读格式，路径：log_dir/log{log_suffix}.txt）
        return HumanOutputFormat(os.path.join(log_dir, f"log{log_suffix}.txt"))
    elif _format == "tensorboard":
        return TensorBoardOutputFormat(log_dir)  # TensorBoard格式
    # elif _format == "wandb":
    #     return WANDBOutputFormat()  # WandB格式（未启用）
    else:
        raise ValueError(f"Unknown format specified: {_format}")


class Logger(object):
    """
    日志核心类——统一管理多格式日志写入，提供记录、刷新、打印等接口
    核心功能：
    1. 记录键值对日志（标量、图像、视频、图表）
    2. 支持平均值记录（如批量平均损失）
    3. 按步骤刷新日志到所有格式（如同时输出到控制台和TensorBoard）
    4. 支持不同级别日志打印（DEBUG/INFO/WARN/ERROR）
    """
    def __init__(self, folder: Optional[str], output_formats: List[KVWriter]):
        """
        Args:
            folder: 日志保存目录（None=无文件输出）
            output_formats: 日志格式列表（如[HumanOutputFormat, TensorBoardOutputFormat]）
        """
        self.name_to_value = defaultdict(float)  # 存储当前迭代的键值对（最后一次记录的值）
        self.name_to_count = defaultdict(int)    # 存储平均值记录的计数（用于record_mean）
        self.name_to_excluded = defaultdict(str) # 存储每个键的排除格式
        self.level = INFO                        # 默认日志级别（INFO及以上才输出）
        self.dir = folder                        # 日志目录
        self.output_formats = output_formats     # 多格式写入器列表

    def record(
        self,
        key: str,
        value: Any,
        exclude: Optional[Union[str, Tuple[str, ...]]] = None,
    ) -> None:
        """
        记录单个键值对（覆盖式，多次调用取最后一次值）
        Args:
            key: 日志键（如"train/loss"、"render/image"）
            value: 日志值（标量、Tensor、Image、Video、Figure等）
            exclude: 排除的日志格式（如("stdout",)表示该键不输出到控制台）
        """
        self.name_to_value[key] = value
        self.name_to_excluded[key] = exclude

    def record_mean(
        self,
        key: str,
        value: Any,
        exclude: Optional[Union[str, Tuple[str, ...]]] = None,
    ) -> None:
        """
        记录键值对的平均值（累加式，多次调用计算平均值）
        适配场景：批量训练中，记录一个batch内的平均损失、平均准确率
        Args:
            key: 日志键（如"train/batch_loss"）
            value: 单次值（如单个样本的损失）
            exclude: 排除的日志格式
        """
        if value is None:
            self.name_to_value[key] = None
            return
        # 计算累计平均值：old_val * count / (count+1) + new_val / (count+1)
        old_val, count = self.name_to_value[key], self.name_to_count[key]
        self.name_to_value[key] = old_val * count / (count + 1) + value / (count + 1)
        self.name_to_count[key] = count + 1  # 更新计数
        self.name_to_excluded[key] = exclude

    def dump(self, step: int = 0) -> None:
        """
        刷新日志——将当前所有记录的键值对写入所有格式，并清空缓存
        Args:
            step: 日志步骤（如训练迭代次数、epoch数，用于时序可视化）
        """
        if self.level == DISABLED:  # 若日志级别为禁用，直接返回
            return
        # 遍历所有日志格式，写入键值对
        for _format in self.output_formats:
            if isinstance(_format, KVWriter):
                _format.write(self.name_to_value, self.name_to_excluded, step)
        # 清空当前迭代的缓存（准备下一次记录）
        self.name_to_value.clear()
        self.name_to_count.clear()
        self.name_to_excluded.clear()

    def log(self, *args, level: int = INFO) -> None:
        """
        打印序列日志（如调试信息、实验进度），支持按级别过滤
        Args:
            *args: 日志内容（如"训练开始"、f"迭代{step}次"）
            level: 日志级别（默认INFO，低于当前logger级别则不输出）
        """
        if self.level <= level:  # 仅输出级别<=当前logger级别的日志
            self._do_log(args)

    def debug(self, *args) -> None:
        """打印DEBUG级别的日志（最详细，用于开发调试）"""
        self.log(*args, level=DEBUG)

    def info(self, *args) -> None:
        """打印INFO级别的日志（常规实验信息）"""
        self.log(*args, level=INFO)

    def warn(self, *args) -> None:
        """打印WARN级别的日志（警告信息）"""
        self.log(*args, level=WARN)

    def error(self, *args) -> None:
        """打印ERROR级别的日志（错误信息）"""
        self.log(*args, level=ERROR)

    # 日志级别配置
    def set_level(self, level: int) -> None:
        """设置日志级别（控制输出阈值）"""
        self.level = level

    def get_dir(self) -> str:
        """获取日志保存目录"""
        return self.dir

    def close(self) -> None:
        """关闭所有日志格式的写入器，释放资源"""
        for _format in self.output_formats:
            _format.close()

    def _do_log(self, args) -> None:
        """内部方法：将序列日志写入所有支持的格式"""
        for _format in self.output_formats:
            if isinstance(_format, SeqWriter):
                _format.write_sequence(map(str, args))


def configure_logger(folder: str, format_strings: List[str] = None) -> Logger:
    """
    配置单个日志器——创建日志目录+多格式写入器，返回Logger实例
    核心作用：简化单个模块的日志器创建，统一配置逻辑
    Args:
        folder: 日志保存目录
        format_strings: 日志格式列表（如["stdout", "tensorboard"]）
    Returns:
        配置好的Logger实例
    """
    assert isinstance(folder, str)
    os.makedirs(folder, exist_ok=True)  # 确保日志目录存在

    log_suffix = ""  # 日志文件后缀（默认空）
    # 过滤空格式字符串，确保格式有效
    format_strings = list(filter(None, format_strings))
    # 根据格式字符串创建对应的写入器列表
    output_formats = [make_output_format(f, folder, log_suffix) for f in format_strings]

    # 创建Logger实例
    logger = Logger(folder=folder, output_formats=output_formats)
    # 若日志格式包含文件类格式（非仅stdout），打印日志目录信息
    if len(format_strings) > 0 and format_strings != ["stdout"]:
        logger.log(f"Logging to {folder}")
    return logger