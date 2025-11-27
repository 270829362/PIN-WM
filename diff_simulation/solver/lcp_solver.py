# 该代码是论文《PIN-WM: Learning Physics-INformed World Models for Non-Prehensile Manipulation》中线性互补问题（LCP）求解器的核心实现，是 3D 刚体动力学模拟的 “核心引擎”。其核心逻辑是将刚体系统的约束问题（关节约束 + 接触 / 摩擦约束）转化为二次规划（QP）问题，通过可微分内点法（PDIPM） 求解，同时手动实现反向传播以支撑 PIN-WM 的端到端优化。
import torch
from torch.autograd import Function  # 用于自定义可微分函数（手动实现forward/backward）

from diff_simulation.constraints.base import Constraint  # 约束抽象基类
from torch.nn.functional import normalize  # 向量归一化（摩擦方向构建）
from diff_simulation.solver.util import bger, expandParam, extract_nBatch, orthogonal  # 工具函数：外积/参数扩展/正交向量计算
import diff_simulation.solver.cvxpy as cvxpy  # CVXPY求解器（备用）
import diff_simulation.solver.batch as pdipm_b  # 批处理内点法（PDIPM）求解器（核心）

from enum import Enum  # 枚举求解器类型


class QPSolvers(Enum):
    """
    QP求解器类型枚举：对应LCP转化后的QP问题求解策略
    - PDIPM_BATCHED：批处理内点法（论文默认，速度快、支持可微分）
    - CVXPY：凸优化库CVXPY（精度高但速度慢，用于验证/调试）
    """
    PDIPM_BATCHED = 1
    CVXPY = 2


class LCP_Solver:
    """
    线性互补问题（LCP）求解器：处理刚体系统的约束动力学（关节约束+接触/摩擦约束）
    核心：将LCP转化为QP问题，通过可微分内点法求解，支撑PIN-WM的端到端优化
    """

    def __init__(self, simulator):
        """
        初始化LCP求解器
        Args:
            simulator: 模拟器实例（包含刚体、关节、接触信息等全局状态）
        """
        from diff_simulation.simulator import Simulator
        self.simulator: Simulator = simulator  # 关联模拟器，获取全局状态（刚体、关节、时间步等）
        self.device = simulator.device  # 计算设备（CPU/GPU）

    def Je(self):
        """
        构建关节约束的雅可比矩阵（Je）——对应论文1-64段“关节约束的速度限制”
        雅可比矩阵形状：[总关节约束维度, 系统总速度维度]
        系统总速度维度 = 刚体数 × 6（每个刚体6维速度：3线速度+3角速度）
        Returns:
            Je: 关节约束雅可比矩阵（torch.Tensor）
        """
        sum_constraint_dim = self.simulator.get_sum_constraint_dim()  # 所有关节约束的总维度
        # 初始化雅可比矩阵（全零）：[总约束维度, 系统总速度维度]
        Je = torch.zeros(
            (sum_constraint_dim, self.simulator.velocity_dim * len(self.simulator.bodies)),
            device=self.device
        )
        row = 0  # 当前填充的行索引（逐约束填充）
        for joint in self.simulator.joints:
            # 调用关节约束的J方法，获取两个刚体的雅可比矩阵（J1=刚体1，J2=刚体2）
            J1, J2 = joint.J()
            # 获取刚体在模拟器中的索引（用于定位速度维度）
            body1_index = self.simulator.get_body_list_index(joint.body1_id)
            body2_index = self.simulator.get_body_list_index(joint.body2_id)
            
            # 填充刚体1的雅可比矩阵：对应行×对应速度维度列
            Je[row:row + J1.size(0), 
               body1_index * self.simulator.velocity_dim:(body1_index + 1) * self.simulator.velocity_dim] = J1       
            # 若存在刚体2的约束（如两个刚体间的关节），填充J2
            if J2 is not None:
                Je[row:row + J2.size(0),
                   body2_index * self.simulator.velocity_dim:(body2_index + 1) * self.simulator.velocity_dim] = J2     
            row += J1.size(0)  # 更新行索引
        return Je            

    def Jc(self, contact_infos):
        """
        构建接触法向约束的雅可比矩阵（Jc）——对应论文1-64段“碰撞法向速度限制”
        雅可比矩阵形状：[接触数, 系统总速度维度]
        Args:
            contact_infos: 接触信息列表，每个元素包含（接触参数, 刚体1ID, 刚体2ID）
                           接触参数：(法向量, 刚体1接触点, 刚体2接触点)
        Returns:
            Jc: 接触法向约束雅可比矩阵（torch.Tensor）
        """
        # 初始化雅可比矩阵（全零）：[接触数, 系统总速度维度]
        Jc = torch.zeros(
            (len(contact_infos), self.simulator.velocity_dim * len(self.simulator.bodies)),
            device=self.device
        )
        for i, contact_info in enumerate(contact_infos):
            # 解析接触参数：法向量、刚体1接触点、刚体2接触点
            c = contact_info[0]
            normal = c[0]
            p_a, p_b = c[1], c[2]
            # 获取接触刚体的索引
            body1_index = self.simulator.get_body_list_index(contact_info[1])
            body2_index = self.simulator.get_body_list_index(contact_info[2])
            
            # 构建单个接触的雅可比矩阵（6维：3角速度项+3线速度项）
            # 角速度项：接触点叉乘法向量（角运动对法向速度的贡献）；线速度项：法向量（线运动对法向速度的贡献）
            J1 = torch.cat([torch.cross(p_a, normal), normal])
            J2 = - torch.cat([torch.cross(p_b, normal), normal])  # 刚体2的雅可比为负（相对运动）
            
            # 填充雅可比矩阵
            Jc[i, body1_index * self.simulator.velocity_dim:(body1_index + 1) * self.simulator.velocity_dim] = J1
            Jc[i, body2_index * self.simulator.velocity_dim:(body2_index + 1) * self.simulator.velocity_dim] = J2        
        return Jc
    
    def Jf(self, contact_infos):
        """
        构建接触摩擦约束的雅可比矩阵（Jf）——对应论文1-64段“摩擦锥约束（多方向近似）”
        雅可比矩阵形状：[接触数×摩擦方向数, 系统总速度维度]
        摩擦方向数（fric_dirs）：4/8方向（近似摩擦锥，避免连续优化）
        Args:
            contact_infos: 接触信息列表（同Jc）
        Returns:
            Jf: 接触摩擦约束雅可比矩阵（torch.Tensor）
        """
        # 初始化雅可比矩阵（全零）：[接触数×摩擦方向数, 系统总速度维度]
        Jf = torch.zeros(
            (len(contact_infos) * self.simulator.fric_dirs, self.simulator.velocity_dim * len(self.simulator.bodies)),
            device=self.device
        )
        for i, contact_info in enumerate(contact_infos):
            # 解析接触参数和刚体索引
            c = contact_info[0]
            body1_index = self.simulator.get_body_list_index(contact_info[1])
            body2_index = self.simulator.get_body_list_index(contact_info[2])
            
            # 构建摩擦方向（近似摩擦锥）：
            # 1. 法向量的正交方向dir1；2. dir1叉乘法向量得到dir2（垂直于法向的平面内两个正交方向）
            dir1 = normalize(orthogonal(c[0]), dim=0)
            dir2 = normalize(torch.cross(dir1, c[0]), dim=0)
            dirs = torch.stack([dir1, dir2])
            
            # 若摩擦方向数为8，扩展为4个基础方向+4个反方向（更精细近似摩擦锥）
            if self.simulator.fric_dirs == 8:
                dir3 = normalize(dir1 + dir2, dim=0)
                dir4 = normalize(torch.cross(dir3, c[0]), dim=0)
                dirs = torch.cat([dirs, torch.stack([dir3, dir4])], dim=0)
            dirs = torch.cat([dirs, -dirs], dim=0)  # 加入反方向（覆盖整个摩擦锥）

            # 构建摩擦约束的雅可比矩阵（每个摩擦方向对应6维）
            # 角速度项：接触点叉乘摩擦方向；线速度项：摩擦方向
            J1 = torch.cat([torch.cross(c[1].expand(self.simulator.fric_dirs, -1), dirs), dirs], dim=1)
            J2 = torch.cat([torch.cross(c[2].expand(self.simulator.fric_dirs, -1), dirs), dirs], dim=1)

            # 填充雅可比矩阵（刚体2的雅可比为负）
            Jf[i * self.simulator.fric_dirs:(i + 1) * self.simulator.fric_dirs, 
               body1_index * self.simulator.velocity_dim:(body1_index + 1) * self.simulator.velocity_dim] = J1
            Jf[i * self.simulator.fric_dirs:(i + 1) * self.simulator.fric_dirs, 
               body2_index * self.simulator.velocity_dim:(body2_index + 1) * self.simulator.velocity_dim] = -J2
        return Jf

    def M(self):
        """
        构建刚体系统的质量矩阵（M）——块对角矩阵，每个块对应单个刚体的6×6质量矩阵（惯性矩阵+质量项）
        Returns:
            M: 系统质量矩阵（torch.Tensor，块对角）
        """
        # 按刚体顺序拼接每个刚体的世界坐标系质量矩阵，形成块对角矩阵
        M = torch.block_diag(*[b.get_M_world() for b in self.simulator.bodies])
        return M
    
    def E(self, contact_infos):
        """
        构建摩擦约束的扩展矩阵（E）——用于将法向力与摩擦力关联（摩擦力≤μ×法向力）
        矩阵形状：[接触数×摩擦方向数, 接触数]
        Args:
            contact_infos: 接触信息列表（同Jc）
        Returns:
            E: 摩擦扩展矩阵（torch.Tensor）
        """
        num_contacts = len(contact_infos)
        n = self.simulator.fric_dirs * num_contacts  # 总摩擦约束数
        E = torch.zeros((n, num_contacts), device=self.device)
        # 每个接触的摩擦约束对应同一列（关联到该接触的法向力）
        for i in range(num_contacts):
            E[i * self.simulator.fric_dirs: (i + 1) * self.simulator.fric_dirs, i] += 1
        return E

    def restitutions(self, contact_infos):
        """
        计算接触的恢复系数（restitution）——控制碰撞反弹程度（0=完全非弹性，1=完全弹性）
        Args:
            contact_infos: 接触信息列表（同Jc）
        Returns:
            restitutions: 每个接触的恢复系数（torch.Tensor，形状[接触数]）
        """
        restitutions = torch.zeros((len(contact_infos)), device=self.device)
        for i, contact_info in enumerate(contact_infos):
            # 获取接触刚体的恢复系数，取乘积（也可改为平均/平方根，根据物理需求）
            body1_index = self.simulator.get_body_list_index(contact_info[1])
            body2_index = self.simulator.get_body_list_index(contact_info[2])
            r1 = self.simulator.bodies[body1_index].restitution
            r2 = self.simulator.bodies[body2_index].restitution
            restitutions[i] = (r1 * r2)  # 两个刚体恢复系数的乘积
        return restitutions

    def mu(self, contact_infos):
        """
        计算接触的摩擦系数矩阵（mu）——对角矩阵，每个对角元对应一个接触的摩擦系数
        Args:
            contact_infos: 接触信息列表（同Jc）
        Returns:
            mu: 摩擦系数对角矩阵（torch.Tensor，形状[接触数, 接触数]）
        """
        mu = torch.zeros((len(contact_infos)), device=self.device)
        for i, contact_info in enumerate(contact_infos):
            # 获取接触刚体的摩擦系数，取乘积（也可改为平均/平方根）
            body1_index = self.simulator.get_body_list_index(contact_info[1])
            body2_index = self.simulator.get_body_list_index(contact_info[2])
            f1 = self.simulator.bodies[body1_index].friction_coefficient
            f2 = self.simulator.bodies[body2_index].friction_coefficient
            mu[i] = f1 * f2
        return torch.diag(mu)  # 转为对角矩阵

    def solve_constraint(self, contact_infos):
        """
        求解LCP约束问题（核心入口）：区分“无接触”和“有接触”场景
        Args:
            contact_infos: 接触信息列表（空则无接触）
        Returns:
            x: LCP的解（系统速度增量，形状[系统总速度维度]）
        """
        if not contact_infos:
            # 无接触：仅处理关节约束，直接求解简化LCP
            inv, u = self.create_lcp_no_contact()
            x = self.solve_lcp_no_contact(inv, u)
        else:
            # 有接触：处理关节+接触+摩擦约束，转化为QP求解
            M, u, G, h, Je, b, F = self.create_lcp_with_contact(contact_infos)
            # 调用可微分QP求解器，返回速度增量（负号为方向修正）
            x = - self.solve_lcp_with_contact(max_iter=10, verbose=-1)(M, u, G, h, Je, b, F)
            x = x.to(torch.float32)  # 转为float32（适配后续计算）
        return x
        
    def create_lcp_no_contact(self):
        """
        构建无接触时的LCP问题（仅关节约束）
        LCP形式：M·Δv + dtime·f = u, Je·Δv = 0 → 转化为线性方程组求解
        Returns:
            inv: 增广矩阵的逆；u: 右侧向量（动量+外力冲量）
        """
        # 1. 构建系统质量矩阵M
        M = self.M()
        # 2. 计算外力冲量：M·v + dtime·f（v为当前速度，f为合外力）
        f = self.simulator.apply_external_forces(self.simulator.cur_time)
        u = torch.matmul(M, self.simulator.get_vel_vec()) + self.simulator.dtime * f     
        
        # 3. 若有关节约束，构建增广矩阵（融合M和Je）
        if len(self.simulator.joints) > 0:
            Je = self.Je()
            sum_constraint_dim = Je.size(0)
            # 扩展u：加入关节约束的零向量（Je·Δv=0）
            u = torch.cat([u, u.new_zeros(sum_constraint_dim)])
            # 构建增广矩阵P：[M, -Je^T; Je, 0]（KKT系统形式）
            P = torch.cat([
                torch.cat([M, -Je.t()], dim=1),
                torch.cat([Je, Je.new_zeros(sum_constraint_dim, sum_constraint_dim)], dim=1)
            ])        
        else :
            # 无关节约束，增广矩阵即为M
            P = M
        
        # 4. 计算增广矩阵的逆（用于直接求解线性方程组）
        inv = torch.inverse(P)
        return inv, u

    def solve_lcp_no_contact(self, inv, u):
        """
        求解无接触时的LCP问题（直接矩阵乘法）
        Args:
            inv: 增广矩阵的逆；u: 右侧向量
        Returns:
            x: 速度增量Δv（P·x = u → x = inv·u）
        """
        return torch.matmul(inv, u)

    def create_lcp_with_contact(self, contact_infos):
        """
        构建有接触时的LCP问题（关节+接触+摩擦约束），并转化为QP问题
        QP形式：min 1/2 z^T M z + u^T z, s.t. Gz ≤ h, Je z = b
        Args:
            contact_infos: 接触信息列表
        Returns:
            M/u/G/h/Je/b/F: QP问题的系数矩阵（M=二次项，u=一次项，G=不等式约束，h=不等式右侧，Je=等式约束，b=等式右侧，F=互补修正）
        """
        # 1. 构建接触法向/摩擦雅可比矩阵、恢复系数、摩擦系数等
        Jc = self.Jc(contact_infos)
        vel_vec = self.simulator.get_vel_vec()
        v = torch.matmul(Jc, vel_vec) * self.restitutions(contact_infos)  # 法向速度×恢复系数（碰撞反弹）
        M = self.M()
        f = self.simulator.apply_external_forces(self.simulator.cur_time)
        u = torch.matmul(M, vel_vec) + self.simulator.dtime * f  # 动量+外力冲量
        
        # 2. 处理关节约束（等式约束）
        if len(self.simulator.joints) > 0:
            Je = self.Je()
            b = Je.new_zeros(Je.size(0)).unsqueeze(0)  # 等式约束右侧（Je·z=0）
            Je = Je.unsqueeze(0)  # 扩展batch维度（适配批处理QP求解）
        else:
            b = torch.tensor([])
            Je = torch.tensor([])         
        
        # 3. 扩展所有矩阵的batch维度（适配批处理QP求解）
        Jc = Jc.unsqueeze(0)
        v = v.unsqueeze(0)
        E = self.E(contact_infos).unsqueeze(0)
        mu = self.mu(contact_infos).unsqueeze(0)
        Jf = self.Jf(contact_infos).unsqueeze(0)        
        
        # 4. 构建不等式约束矩阵G（融合法向/摩擦约束）
        # G = [Jc; Jf; 0]：法向约束+摩擦约束+摩擦锥约束
        G = torch.cat([Jc, Jf, Jf.new_zeros(Jf.size(0), mu.size(1), Jf.size(2))], dim=1)
        
        # 5. 构建互补修正矩阵F（关联法向力和摩擦力：|f| ≤ μ·n）
        F = G.new_zeros(G.size(1), G.size(1)).unsqueeze(0)
        F[:, Jc.size(1):-E.size(2), -E.size(2):] = E  # 摩擦力→法向力的关联
        F[:, -mu.size(1):, :mu.size(2)] = mu  # 摩擦系数矩阵
        F[:, -mu.size(1):, mu.size(2):mu.size(2) + E.size(1)] = -E.transpose(1, 2)  # 摩擦锥约束修正
        
        # 6. 构建不等式约束右侧h（法向速度约束+摩擦速度约束+摩擦锥约束）
        h = torch.cat([v, v.new_zeros(v.size(0), Jf.size(1) + mu.size(1))], 1)
        
        return M, u, G, h, Je, b, F

    def solve_lcp_with_contact(self, eps=1e-12, verbose=0, notImprovedLim=3,
                        max_iter=20, solver=1, check_Q_spd=True):
        """
        封装可微分QP求解器：自定义torch.autograd.Function，实现forward/backward
        核心：将LCP转化的QP问题通过批处理内点法求解，并手动实现反向传播（支撑PIN-WM端到端优化）
        Returns:
            LCPFunctionFn.apply: 可微分的QP求解函数
        """
        class LCPFunctionFn(Function):
            @staticmethod
            def forward(ctx, Q_, p_, G_, h_, A_, b_, F_):
                """
                QP求解的前向传播：调用批处理内点法（PDIPM）求解QP问题
                Args:
                    Q_: QP二次项矩阵（系统质量矩阵M）
                    p_: QP一次项向量（动量+外力冲量u）
                    G_: QP不等式约束矩阵（接触/摩擦约束）
                    h_: QP不等式右侧向量
                    A_: QP等式约束矩阵（关节约束Je）
                    b_: QP等式右侧向量
                    F_: 互补修正矩阵
                Returns:
                    zhats: QP的最优解（速度增量Δv）
                """
                # 转为float64（提升求解精度）
                Q_ = Q_.to(torch.float64)
                p_ = p_.to(torch.float64)
                G_ = G_.to(torch.float64)
                h_ = h_.to(torch.float64)
                A_ = A_.to(torch.float64)
                b_ = b_.to(torch.float64)
                F_ = F_.to(torch.float64)
                
                # 提取batch维度，扩展所有矩阵到统一batch大小
                nBatch = extract_nBatch(Q_, p_, G_, h_, A_, b_, F_)
                Q, _ = expandParam(Q_, nBatch, 3)
                p, _ = expandParam(p_, nBatch, 2)
                G, _ = expandParam(G_, nBatch, 3)
                h, _ = expandParam(h_, nBatch, 2)
                A, _ = expandParam(A_, nBatch, 3)
                b, _ = expandParam(b_, nBatch, 2)
                F, _ = expandParam(F_, nBatch, 3)

                # 检查Q矩阵是否为对称正定（SPD）——QP有解的必要条件
                if check_Q_spd:
                    try:
                        torch.linalg.cholesky(Q)
                    except:
                        raise RuntimeError('Q is not SPD (Symmetric Positive Definite).')

                # 解析约束维度
                _, nineq, nz = G.size()
                neq = A.size(1) if A.nelement() > 0 else 0
                assert(neq > 0 or nineq > 0)  # 至少有一个约束
                ctx.neq, ctx.nineq, ctx.nz = neq, nineq, nz
                
                # 选择QP求解器：批处理内点法（论文默认）
                if solver == QPSolvers.PDIPM_BATCHED.value:
                    # 预计算KKT系统的固定部分（加速迭代）
                    ctx.Q_LU, ctx.S_LU, ctx.R = pdipm_b.pre_factor_kkt(Q, G, A, F)
                    # 调用内点法求解QP
                    zhats, ctx.nus, ctx.lams, ctx.slacks = pdipm_b.forward(
                        Q, p, G, h, A, b, F, ctx.Q_LU, ctx.S_LU, ctx.R,
                        eps, verbose, notImprovedLim, max_iter)                        
 
                # 保存中间结果（用于反向传播）
                ctx.save_for_backward(zhats, Q_, p_, G_, h_, A_, b_, F_)
                return zhats

            @staticmethod
            def backward(ctx, dl_dzhat):
                """
                QP求解的反向传播：手动计算梯度（适配PIN-WM的端到端优化）
                Args:
                    dl_dzhat: 损失对QP解zhats的梯度
                Returns:
                    grads: 各输入矩阵的梯度（dQs, dps, dGs, dhs, dAs, dbs, dFs）
                """
                # 加载前向传播保存的中间结果
                zhats, Q, p, G, h, A, b, F = ctx.saved_tensors
                # 扩展矩阵到统一batch大小
                nBatch = extract_nBatch(Q, p, G, h, A, b, F)
                Q, Q_e = expandParam(Q, nBatch, 3)
                p, p_e = expandParam(p, nBatch, 2)
                G, G_e = expandParam(G, nBatch, 3)
                h, h_e = expandParam(h, nBatch, 2)
                A, A_e = expandParam(A, nBatch, 3)
                b, b_e = expandParam(b, nBatch, 2)
                F, F_e = expandParam(F, nBatch, 3)

                neq, nineq = ctx.neq, ctx.nineq

                # 计算d=λ/s（对偶变量/松弛变量，避免除零），更新KKT系统的LU分解
                d = torch.clamp(ctx.lams, min=1e-8) / torch.clamp(ctx.slacks, min=1e-8)
                pdipm_b.factor_kkt(ctx.S_LU, ctx.R, d)

                # 求解KKT系统的梯度（dx=z的梯度，dlam=λ的梯度，dnu=等式对偶变量的梯度）
                dx, _, dlam, dnu = pdipm_b.solve_kkt(
                    ctx.Q_LU, d, G, A, ctx.S_LU,
                    dl_dzhat, torch.zeros((nBatch, nineq), device=self.device).type_as(G),
                    torch.zeros((nBatch, nineq), device=self.device).type_as(G),
                    torch.zeros((nBatch, neq), device=self.device).type_as(G) if neq > 0 else torch.Tensor().to(self.device))

                # 计算各输入矩阵的梯度（利用外积bger，适配batch维度）
                dps = dx  # p的梯度=dx
                dFs = bger(dlam, ctx.lams)  # F的梯度=λ·dλ^T
                if F_e: 
                    dFs = dFs.mean(0)  # 无batch则取平均
                dGs = bger(dlam, zhats) + bger(ctx.lams, dx)  # G的梯度=λ·z^T + z·dλ^T
                if G_e: 
                    dGs = dGs.mean(0)
                dhs = -dlam  # h的梯度=-dλ
                if h_e:
                    dhs = dhs.mean(0)
                # 等式约束的梯度（若有）
                if neq > 0:
                    dAs = bger(dnu, zhats) + bger(ctx.nus, dx)  # A的梯度=ν·z^T + z·dν^T
                    dbs = -dnu  # b的梯度=-dν
                    if A_e:
                        dAs = dAs.mean(0)
                    if b_e:
                        dbs = dbs.mean(0)
                else:
                    dAs, dbs = None, None
                # Q的梯度=0.5*(dx·z^T + z·dx^T)（二次项梯度）
                dQs = 0.5 * (bger(dx, zhats) + bger(zhats, dx))
                if Q_e:
                    dQs = dQs.mean(0)
                if p_e:
                    dps = dps.mean(0)

                # 封装所有梯度
                grads = (dQs, dps, dGs, dhs, dAs, dbs, dFs)
                return grads
        return LCPFunctionFn.apply