# 该代码是论文《PIN-WM: Learning Physics-INformed World Models for Non-Prehensile Manipulation》中固定约束（FIX_CONSTRAINT）的具体实现类，继承自抽象约束基类Constraint。固定约束的核心作用是限制单个刚体的所有 6 个自由度（3 个平移自由度 + 3 个旋转自由度），使其保持静止（如地面、固定基座等），对应论文 1-64 段 “关节约束（Joint Constraints）” 中 “固定连接” 的场景。其J方法返回的单位矩阵是 LCP（线性互补问题）求解中固定约束的核心数学表达，确保约束刚体的速度向量满足 “全零” 条件（即无任何运动），支撑 3D 刚体动力学的精确模拟。
from diff_simulation.constraints.base import Constraint  # 导入约束抽象基类
import torch  # 用于张量计算（雅可比矩阵构建）


class Fix_Constraint(Constraint):
    """
    固定约束的具体实现类，继承自Constraint抽象基类
    功能：限制单个刚体的所有6个自由度（3平移+3旋转），使其完全固定（如地面、固定支架）
    对应论文1-64段“关节约束（Joint Constraints）”中的固定连接场景，是LCP求解固定刚体运动的核心组件
    """

    def __init__(self, body_id):
        """
        初始化固定约束
        Args:
            body_id: 需被固定的刚体ID（仅约束单个刚体，故第二个刚体ID设为None）
        """
        # 调用父类Constraint的构造函数：
        # - body1_id：被固定的刚体ID（如地面ID）
        # - body2_id：None（固定约束仅作用于单个刚体，无需关联第二个刚体）
        # - constraint_dim：6（固定约束限制6个自由度，需6个约束方程）
        super().__init__(body1_id=body_id, body2_id=None, constraint_dim=6)
        
    def J(self):
        """
        计算固定约束的雅可比矩阵（Jacobian Matrix）——对应论文1-64段LCP中的约束方程J·ξ=0
        固定约束的雅可比矩阵为6×6单位矩阵，确保刚体的速度向量（3线速度+3角速度）全为0（无运动）
        Returns:
            torch.eye(self.constraint_dim): 6×6单位矩阵（约束第一个刚体的速度）
            None: 无第二个刚体，故返回None（无需约束第二个刚体的速度）
        """
        # 单位矩阵I满足：I · ξ = 0 → ξ = 0（ξ为刚体的6维速度向量：[角速度x,角速度y,角速度z,线速度x,线速度y,线速度z]）
        # 即强制被约束刚体的所有速度分量为0，实现“固定”效果
        return torch.eye(self.constraint_dim), None