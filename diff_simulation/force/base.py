#外部力模拟的抽象基类，定义了所有力（如重力、推力、恒力等）的通用结构与接口
from abc import ABCMeta, abstractmethod  # 用于定义抽象基类（统一力的接口）
import torch  # 用于张量计算（力向量的表示与运算）


class Force(metaclass=ABCMeta):
    """
    力的抽象基类，定义所有外部力（重力、推力、恒力等）的通用属性与接口
    功能：统一管理力的方向、大小、作用时间范围，提供力的时间有效性判断与个性化计算接口
    对应论文1-42段“3D刚体动力学”中“外力作用”的核心逻辑，为LCP求解（1-64段）提供标准化外力输入，支撑刚体运动模拟
    """

    def __init__(self, direction, magnitude=10.0, starttime=0.0, endtime=1e5):
        """
        初始化力的基本属性
        Args:
            direction: 力的方向向量（torch.Tensor, shape=[3]），如重力方向为[0,0,-1]（Z轴负方向）
            magnitude: 力的大小（标量，默认10.0），需与方向结合生成最终力向量（力向量=方向×大小）
            starttime: 力的开始作用时间（默认0.0，即模拟开始时生效）
            endtime: 力的结束作用时间（默认1e5，即长期生效，避免模拟中力意外消失）
        """
        self.direction = direction  # 力的方向向量（需归一化，确保方向与大小分离控制）
        self.magnitude = magnitude  # 力的大小（标量，控制力的强度）
        self.starttime = starttime  # 力的生效起始时间（对应模拟时间步）
        self.endtime = endtime      # 力的生效结束时间（对应模拟时间步）

    def apply(self, cur_time):
        """
        根据当前模拟时间，判断力是否生效并返回最终力向量
        核心逻辑：仅在[starttime, endtime]时间范围内施加力，否则返回零向量（无作用力）
        Args:
            cur_time: 当前模拟时间（对应论文中动力学模拟的时间步t，如1-62段的时间 horizon H）
        Returns:
            力向量（torch.Tensor, shape=[3]）：生效则返回“方向×大小”，不生效则返回零向量
        """
        # 时间判断：当前时间超出力的作用范围 → 返回零向量（无作用力）
        if cur_time < self.starttime or cur_time > self.endtime:
            return self.direction * 0  # 零向量（方向向量×0，保持维度一致）
        # 时间在作用范围内 → 调用子类实现的force_function计算具体力向量
        else:
            return self.force_function()
        
    @abstractmethod
    def force_function(self, *args, **kwargs):
        """
        抽象方法：计算具体的力向量（由子类实现，支撑不同类型力的个性化逻辑）
        作用：根据力的类型（如恒力、变力、随时间变化的力）定义力向量的计算规则
        示例：恒定力（如重力）需返回“方向×大小”的固定向量，变力需根据时间/位置动态计算
        Raises:
            NotImplementedError: 未实现该方法时触发（强制子类必须定义具体力计算逻辑）
        """
        raise NotImplementedError  # 抽象方法必须被子类重写，否则无法实例化