# 该代码是论文《PIN-WM: Learning Physics-INformed World Models for Non-Prehensile Manipulation》中无 Z 轴平移约束（NO_TRANS_Z_CONSTRATNT）的具体实现类，继承自抽象约束基类Constraint。其核心作用是限制单个刚体的1 个 Z 轴平移自由度（沿 Z 轴的上下移动），但保留 3 个旋转自由度（绕 X/Y/Z 轴转动）和 2 个 X/Y 轴平移自由度（沿平面的水平移动），对应论文 1-64 段 “关节约束（Joint Constraints）” 中 “限制垂直移动、允许平面运动 + 旋转” 的场景（如桌面推物任务中，需物体仅在 XY 平面内移动 + 旋转，不脱离桌面或向上浮动）。其J方法构建的 1×6 雅可比矩阵是 LCP（线性互补问题）求解的关键，通过 “仅约束 Z 轴线速度” 的数学表达，确保刚体运动符合非抓握操作的平面场景需求，避免因垂直方向位移导致的动力学模拟偏差。
from diff_simulation.constraints.base import Constraint  # 导入约束抽象基类
import torch  # 用于张量计算（构建约束雅可比矩阵）


class TransZ_Constraint(Constraint):
    """
    无Z轴平移约束的具体实现类，继承自Constraint抽象基类
    功能：限制单个刚体的1个Z轴平移自由度（沿Z轴上下移动），保留3个旋转自由度+2个X/Y轴平移自由度
    对应论文1-64段“关节约束（Joint Constraints）”中“限制垂直移动、允许平面运动”的场景（如桌面推物不脱离平面）
    是LCP求解中控制刚体垂直方向自由度的核心组件，确保非抓握操作（如推、拨）在平面场景下的动力学合理性
    """

    def __init__(self, body_id):
        """
        初始化无Z轴平移约束
        Args:
            body_id: 需被限制Z轴平移的刚体ID（仅约束单个刚体，故第二个刚体ID设为None）
        """
        # 调用父类Constraint的构造函数：
        # - body1_id：被限制Z轴平移的刚体ID（如桌面推物任务中的目标物体ID）
        # - body2_id：None（无Z轴平移约束仅作用于单个刚体，无需关联第二个刚体）
        # - constraint_dim：1（仅限制1个Z轴平移自由度，需1个约束方程）
        super().__init__(body1_id=body_id, body2_id=None, constraint_dim=1)
        
    def J(self):
        """
        计算无Z轴平移约束的雅可比矩阵（Jacobian Matrix）——对应论文1-64段LCP中的约束方程J·ξ=0
        雅可比矩阵设计逻辑：仅约束Z轴线速度（设为0），放行旋转速度与X/Y轴线速度，实现“平面内自由运动、无垂直位移”效果
        Returns:
            J: 1×6雅可比矩阵（1个约束方程×6维速度向量），仅约束Z轴线速度为0
            None: 无第二个刚体，故返回None（无需约束第二个刚体的速度）
        """
        # 速度向量ξ的标准格式（6维）：[ω_x, ω_y, ω_z, v_x, v_y, v_z]
        # 其中：ω_x/ω_y/ω_z = 绕X/Y/Z轴的角速度（旋转自由度）；v_x/v_y/v_z = 沿X/Y/Z轴的线速度（平移自由度）
        # 雅可比矩阵每行对应1个约束方程：J·ξ = 0
        # 此处矩阵设计：仅Z轴线速度（ξ第6个元素）的系数为1.0，其余均为0.0
        # → 约束方程：0*ω_x + 0*ω_y + 0*ω_z + 0*v_x + 0*v_y + 1*v_z = 0 → v_z = 0（Z轴线速度为0，限制垂直移动）
        J = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])
        return J, None