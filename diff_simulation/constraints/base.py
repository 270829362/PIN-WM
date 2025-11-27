""" 该代码是论文《PIN-WM: Learning Physics-INformed World Models for Non-Prehensile Manipulation》中刚体动力学约束系统的核心定义代码，包含两部分核心内容：
Joint_Type枚举类：定义常见的刚体关节约束类型，为后续具体约束实现提供类型标识，对应论文 1-64 段中 “关节约束（Joint Constraints）” 的分类（如固定约束、旋转限制约束等）；
Constraint抽象基类：封装约束的通用属性（关联刚体 ID、约束维度）与接口（约束雅可比矩阵计算），是所有具体约束（如固定约束、接触约束）的父类，支撑论文中基于线性互补问题（LCP） 的刚体动力学求解（1-62、1-64 段），雅可比矩阵（由J方法计算）是 LCP 中描述 “约束 - 速度关系” 的核心组件。 """
from abc import ABCMeta, abstractmethod  # 用于定义抽象基类（Abstract Base Class）

from enum import Enum  # 用于定义枚举类（约束类型标识）


class Joint_Type(Enum):
    """
    关节约束类型枚举类，定义刚体间常见的运动约束类型，对应论文1-64段“关节约束（Joint Constraints）”的分类
    用于后续具体约束实现时，标识约束对刚体运动的限制范围
    """
    # 固定约束：限制两个刚体间的所有相对运动（平移+旋转），如刚体与地面的固定连接
    FIX_CONSTRAINT = 0
    # 无旋转约束：允许两个刚体间的相对平移，但限制所有相对旋转（仅保留平移自由度）
    NO_ROT_CONSTRATNT = 1
    # 无Z轴平移约束：限制两个刚体在X、Y轴的相对平移，允许Z轴相对平移和所有旋转（仅开放Z轴平移自由度）
    NO_TRANS_Z_CONSTRATNT = 2


class Constraint(metaclass=ABCMeta):
    """
    约束抽象基类，所有具体约束（如关节约束、接触约束、摩擦约束）的父类
    封装约束的通用属性，定义约束核心计算接口，支撑论文中LCP（线性互补问题）的动力学求解（1-62、1-64段）
    """
    def __init__(self, body1_id, body2_id, constraint_dim):
        """
        初始化约束实例
        Args:
            body1_id: 约束关联的第一个刚体ID（如“机器人末端执行器”ID）
            body2_id: 约束关联的第二个刚体ID（如“目标物体”或“地面”ID）
            constraint_dim: 约束维度（即约束方程的数量，如固定约束维度为6，限制3个平移+3个旋转自由度）
        """
        self.id = None  # 约束唯一标识ID（后续由外部管理器设置，用于多约束场景的区分与管理）
        self.body1_id = body1_id  # 约束关联的第一个刚体ID
        self.body2_id = body2_id  # 约束关联的第二个刚体ID
        self.constraint_dim = constraint_dim  # 约束维度（约束方程数量）

    @abstractmethod
    def J(self, *args, **kwargs):
        """
        抽象方法：计算约束雅可比矩阵（Jacobian Matrix）——论文LCP求解的核心组件（1-64段）
        雅可比矩阵描述“刚体速度”与“约束满足度变化率”的关系，用于构建LCP中的约束方程（如J·ξ = 0）
        Args:
            *args, **kwargs: 需传入的动态参数（如两个刚体的当前姿态、速度，因具体约束类型而异）
        Returns:
            约束雅可比矩阵，shape=(constraint_dim, 12)（12=2个刚体×6维速度向量（3线速度+3角速度））
        """
        pass

    def set_id(self, id):
        """
        设置约束的唯一标识ID，用于多约束场景（如多个接触点约束）的管理与索引
        Args:
            id: 约束的唯一标识（如整数序号）
        """
        self.id = id