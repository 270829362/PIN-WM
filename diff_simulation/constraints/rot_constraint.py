# 该代码是论文《PIN-WM: Learning Physics-INformed World Models for Non-Prehensile Manipulation》中无旋转约束（NO_ROT_CONSTRATNT）的具体实现类，继承自抽象约束基类Constraint。其核心作用是限制单个刚体的3 个旋转自由度（绕 X、Y、Z 轴的转动），但保留 3 个平移自由度（沿 X、Y、Z 轴的移动），对应论文 1-64 段 “关节约束（Joint Constraints）” 中 “限制旋转、允许平移” 的场景（如平面推动任务中，需物体仅沿平面平移而不翻转）。其J方法构建的雅可比矩阵是 LCP（线性互补问题）求解的关键，通过 “约束旋转速度、放行平移速度” 的数学表达，确保刚体运动符合任务所需的自由度限制，支撑非抓握操作中精准的运动约束控制。
from diff_simulation.constraints.base import Constraint  # 导入约束抽象基类
import torch  # 用于张量计算（构建约束雅可比矩阵）


class Rot_Constraint(Constraint):
    """
    无旋转约束的具体实现类，继承自Constraint抽象基类
    功能：限制单个刚体的3个旋转自由度（绕X/Y/Z轴转动），保留3个平移自由度（沿X/Y/Z轴移动）
    对应论文1-64段“关节约束（Joint Constraints）”中“限制旋转、允许平移”的场景（如平面推物不翻转）
    是LCP求解中控制刚体旋转自由度的核心组件，确保运动符合非抓握操作的任务需求
    """

    def __init__(self, body_id):
        """
        初始化无旋转约束
        Args:
            body_id: 需被限制旋转的刚体ID（仅约束单个刚体，故第二个刚体ID设为None）
        """
        # 调用父类Constraint的构造函数：
        # - body1_id：被限制旋转的刚体ID（如目标推动物体ID）
        # - body2_id：None（无旋转约束仅作用于单个刚体，无需关联第二个刚体）
        # - constraint_dim：3（仅限制3个旋转自由度，需3个约束方程）
        super().__init__(body1_id=body_id, body2_id=None, constraint_dim=3)
        
    def J(self):
        """
        计算无旋转约束的雅可比矩阵（Jacobian Matrix）——对应论文1-64段LCP中的约束方程J·ξ=0
        雅可比矩阵设计逻辑：约束旋转速度（设为0）、放行平移速度（不限制），实现“无旋转、可平移”效果
        Returns:
            J: 3×6雅可比矩阵（3个约束方程×6维速度向量），约束旋转速度为0
            None: 无第二个刚体，故返回None（无需约束第二个刚体的速度）
        """
        # 1. 旋转约束部分：3×3单位矩阵（对应速度向量的前3维——绕X/Y/Z轴的角速度）
        # 单位矩阵确保：J_rot · [ω_x, ω_y, ω_z]^T = 0 → 角速度全为0（限制旋转）
        J_rot = torch.eye(3)
        
        # 2. 平移放行部分：3×3零矩阵（对应速度向量的后3维——沿X/Y/Z轴的线速度）
        # 零矩阵确保：J_trans · [v_x, v_y, v_z]^T = 0 → 对平移速度无约束（允许自由移动）
        J_trans = torch.zeros([3, 3])
        
        # 3. 拼接旋转约束与平移放行部分，形成完整的3×6雅可比矩阵
        # 速度向量ξ格式：[ω_x, ω_y, ω_z, v_x, v_y, v_z]（6维）
        # J·ξ = J_rot·[ω_x,ω_y,ω_z]^T + J_trans·[v_x,v_y,v_z]^T = 0 → 仅限制旋转速度
        J = torch.cat([J_rot, J_trans], dim=1)        
        return J, None