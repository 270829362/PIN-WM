# 定力（大小和方向不随时间变化的力）的具体实现类
from .base import Force  # 导入力的抽象基类，继承通用属性（方向、大小、作用时间）


class Constant_Force(Force):
    """
    恒定力的具体实现类，继承自Force抽象基类
    功能：模拟大小和方向不随时间/位置变化的外力（如重力、恒定推力）
    对应论文1-42段“3D刚体动力学”中“恒定外力”场景，例如刚体的重力（Body.add_gravity方法）、机器人末端的固定推力
    为LCP求解（1-64段）提供稳定的外力输入，确保刚体在恒定外力作用下的运动符合牛顿力学规律
    """

    def __init__(self, direction, magnitude, starttime=0.0, endtime=1e5):
        """
        初始化恒定力（直接复用父类Force的构造逻辑，无额外扩展）
        Args:
            direction: 力的恒定方向向量（torch.Tensor, shape=[3]，需归一化以确保方向与大小独立控制）
            magnitude: 力的恒定大小（标量，如重力大小为“质量×9.8”，推力大小为机器人输出力）
            starttime: 力的开始作用时间（默认0.0，模拟启动即生效）
            endtime: 力的结束作用时间（默认1e5，长期生效，避免模拟中力意外中断）
        """
        # 调用父类Force的构造函数，初始化方向、大小、作用时间范围（无新增属性，直接复用父类逻辑）
        super().__init__(direction, magnitude, starttime, endtime)

    def force_function(self):
        """
        实现恒定力的计算逻辑（重写父类抽象方法）
        核心：力向量 = 恒定方向 × 恒定大小，不随时间/位置变化，符合“恒定力”物理定义
        Returns:
            恒定力向量（torch.Tensor, shape=[3]）：方向与大小固定，直接用于刚体动力学计算
        """
        # 例如：重力场景中，direction=[0,0,-1]（Z轴负方向），magnitude=mass×9.8 → 力向量=[0,0,-mass×9.8]
        # 该向量将传入Body.apply_external_forces方法，参与合力/合力矩计算，最终影响刚体动量与运动状态
        return self.direction * self.magnitude