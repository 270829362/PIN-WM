#
# Copyright 2024 Max-Planck-Gesellschaft
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ========================================================================
# 代码来源：融合lcp-physics和qpth项目，适配新版PyTorch
# 主要修改者：Rama Krishna Kandukuri（MPI-IS）
# 用途：可微分二次规划（QP）求解，支撑刚体物理约束优化（如碰撞、摩擦）
#

import torch
from enum import Enum
from diff_simulation.solver.util import get_sizes, bdiag  # 工具函数：获取约束维度、构造块对角矩阵


def lu_hack(x):
    """
    兼容PyTorch的LU分解工具：处理GPU/CPU上的pivot（主元）差异
    PyTorch在GPU上默认不返回pivots，需手动生成；CPU上正常返回pivots
    Args:
        x (torch.Tensor): 待分解的矩阵（形状：[nBatch, n, n] 或 [n, n]）
    Returns:
        tuple: (LU分解结果, pivots主元索引)
    """
    # 调用PyTorch的LU分解（CPU返回pivots，GPU不返回）
    data, pivots = torch.linalg.lu_factor(x, pivot=not x.is_cuda)
    if x.is_cuda:  # 若在GPU上，手动生成pivots（1-based索引，按顺序取主元）
        if x.ndimension() == 2:  # 单矩阵（无批量）
            pivots = torch.arange(1, 1 + x.size(0)).int().to(x.device)
        elif x.ndimension() == 3:  # 批量矩阵
            pivots = torch.arange(1, 1 + x.size(1)).unsqueeze(0).repeat(x.size(0), 1).int().to(x.device)
        else:
            assert False, "仅支持2D/3D矩阵"
    return (data, pivots)


# 解精度不足时的警告信息
INACC_ERR = """
--------
qpth warning: Returning an inaccurate and potentially incorrect solution.

Some residual is large.
Your problem may be infeasible or difficult.

You can try using the CVXPY solver to see if your problem is feasible
and you can use the verbose option to check the convergence status of
our solver while increasing the number of iterations.

Advanced users:
You can also try to enable iterative refinement in the solver:
https://github.com/locuslab/qpth/issues/6
--------
"""


class KKTSolvers(Enum):
    """
    KKT系统求解器类型枚举：对应不同的LU分解策略（平衡效率与精度）
    - LU_FULL：全矩阵LU分解（精度高但速度慢，适合小规模问题）
    - LU_PARTIAL：部分LU分解（预计算固定部分，迭代中更新可变部分，论文中默认用此方式）
    - IR_UNOPT：带迭代精修（Iterative Refinement）的未优化版本（提升精度，适合病态问题）
    """
    LU_FULL = 1
    LU_PARTIAL = 2
    IR_UNOPT = 3


def forward(Q, p, G, h, A, b, F, Q_LU, S_LU, R, eps=1e-12, verbose=0, notImprovedLim=3,
            max_iter=20, solver=KKTSolvers.LU_PARTIAL):
    """
    内点法求解二次规划（QP）问题的主函数：迭代优化变量，满足约束并最小化目标函数
    Args:
        Q (torch.Tensor): 目标函数二次项系数（形状：[nBatch, nz, nz]，nz为优化变量x的维度）
        p (torch.Tensor): 目标函数一次项系数（形状：[nBatch, nz]）
        G (torch.Tensor): 不等式约束矩阵（形状：[nBatch, nineq, nz]，nineq为不等式约束数）
        h (torch.Tensor): 不等式约束右侧（形状：[nBatch, nineq]）
        A (torch.Tensor): 等式约束矩阵（形状：[nBatch, neq, nz]，neq为等式约束数；None表示无等式约束）
        b (torch.Tensor): 等式约束右侧（形状：[nBatch, neq]；None表示无等式约束）
        F (torch.Tensor): 互补条件修正矩阵（形状：[nBatch, nineq, nineq]，用于摩擦等复杂约束）
        Q_LU (tuple): Q矩阵的预计算LU分解结果（由pre_factor_kkt生成）
        S_LU (tuple): KKT系统中S矩阵的预计算LU分解结果（由pre_factor_kkt生成）
        R (torch.Tensor): KKT系统中R矩阵（预计算的固定部分，由pre_factor_kkt生成）
        eps (float): 收敛阈值（残差小于此值认为收敛）
        verbose (int): 日志输出等级（0=无输出，1=打印迭代信息）
        notImprovedLim (int): 残差未提升的最大迭代次数（超过则停止）
        max_iter (int): 最大迭代次数
        solver (KKTSolvers): KKT系统求解器类型
    Returns:
        tuple: (x, y, z, s) —— QP问题的最优解
            x: 优化变量（形状：[nBatch, nz]）
            y: 等式约束的对偶变量（形状：[nBatch, neq]；None表示无等式约束）
            z: 不等式约束的对偶变量（形状：[nBatch, nineq]，z≥0）
            s: 不等式约束的松弛变量（形状：[nBatch, nineq]，s≥0）
    """
    # 1. 解析约束维度：nineq=不等式数，nz=优化变量数，neq=等式数，nBatch=批量数
    nineq, nz, neq, nBatch = get_sizes(G, A)

    # 2. 生成初始解（不同求解器的初始化解策略不同）
    if solver == KKTSolvers.LU_FULL:
        # 全LU分解：初始化D为单位矩阵（松弛变量s的系数）
        D = torch.eye(nineq).repeat(nBatch, 1, 1).type_as(Q)
        x, s, z, y = factor_solve_kkt(
            Q, D, G, A, p,
            torch.zeros(nBatch, nineq).type_as(Q),  # rs初始为0
            -h, -b if b is not None else None)     # rz=-h，ry=-b（构造初始KKT系统）
    elif solver == KKTSolvers.LU_PARTIAL:
        # 部分LU分解：初始化d为全1向量（d=z/s，初始s/z均为1）
        d = torch.ones(nBatch, nineq).type_as(Q)
        factor_kkt(S_LU, R, d)  # 完成S矩阵的LU分解（迭代中可变部分）
        x, s, z, y = solve_kkt(
            Q_LU, d, G, A, S_LU,
            p, torch.zeros(nBatch, nineq).type_as(Q),
            -h, -b if neq > 0 else None)
    elif solver == KKTSolvers.IR_UNOPT:
        # 迭代精修版本：初始化D为单位矩阵
        D = torch.eye(nineq).repeat(nBatch, 1, 1).type_as(Q)
        x, s, z, y = solve_kkt_ir(
            Q, D, G, A, F, p,
            torch.zeros(nBatch, nineq).type_as(Q),
            -h, -b if b is not None else None)
    else:
        assert False, "不支持的求解器类型"

    # 3. 初始解修正：确保松弛变量s和对偶变量z均≥1（满足内点法的“内点”要求，避免初始解在约束边界）
    # 修正s：若最小s<0，整体偏移使s≥1
    M = torch.min(s, 1)[0].view(M.size(0), 1).repeat(1, nineq)  # 每个batch的最小s
    I = M < 0  # 需要修正的batch索引
    s[I] -= M[I] - 1  # 偏移：s = s - (M-1) → 最小s变为1
    # 修正z：同理确保z≥1
    M = torch.min(z, 1)[0].view(M.size(0), 1).repeat(1, nineq)
    I = M < 0
    z[I] -= M[I] - 1

    # 4. 迭代优化：内点法核心（仿射方向+中心方向+线搜索）
    best = {'resids': None, 'x': None, 'z': None, 's': None, 'y': None}  # 保存最优解
    nNotImproved = 0  # 残差未提升的迭代次数计数器

    for i in range(max_iter):
        # 4.1 计算当前残差（评估解的可行性与最优性）
        # rx：对偶残差（目标函数梯度约束残差）
        rx = (torch.bmm(y.unsqueeze(1), A).squeeze(1) if neq > 0 else 0.) + \
            torch.bmm(z.unsqueeze(1), G).squeeze(1) + \
            torch.bmm(x.unsqueeze(1), Q.transpose(1, 2)).squeeze(1) + p
        # rs：互补残差1（z = -rs → 初始互补条件残差）
        rs = z
        # rz：原始残差（不等式约束残差：Gx + s - h - Fz = rz）
        rz = torch.bmm(x.unsqueeze(1), G.transpose(1, 2)).squeeze(1) + s - h - \
            torch.bmm(z.unsqueeze(1), F.transpose(1, 2)).squeeze(1)
        # ry：等式约束残差（Ax - b = ry）
        ry = torch.bmm(x.unsqueeze(1), A.transpose(1, 2)).squeeze(1) - b if neq > 0 else 0.0

        # 4.2 计算互补度mu（z^T s / nineq，衡量互补条件满足程度，mu越小越优）
        mu = torch.abs((s * z).sum(1).squeeze() / nineq)
        # 计算总残差（原始残差+对偶残差+互补度，用于判断收敛）
        z_resid = torch.norm(rz, 2, 1).squeeze()  # 不等式约束残差范数
        y_resid = torch.norm(ry, 2, 1).squeeze() if neq > 0 else 0  # 等式约束残差范数
        pri_resid = y_resid + z_resid  # 原始残差总和
        dual_resid = torch.norm(rx, 2, 1).squeeze()  # 对偶残差范数
        resids = pri_resid + dual_resid + nineq * mu  # 总残差（含互补度惩罚）

        # 4.3 更新最优解（保留残差最小的解）
        d = z / s  # d=z/s：用于后续KKT系统求解（松弛变量与对偶变量的比值）
        try:
            factor_kkt(S_LU, R, d)  # 完成S矩阵的LU分解（更新可变部分）
        except:
            return best['x'], best['y'], best['z'], best['s']  # 分解失败时返回当前最优解

        # 打印迭代信息（verbose=1时）
        if verbose == 1:
            print('iter: {}, pri_resid: {:.5e}, dual_resid: {:.5e}, mu: {:.5e}'.format(
                i, pri_resid.mean(), dual_resid.mean(), mu.mean()))

        # 初始化或更新最优解
        if best['resids'] is None:
            best['resids'] = resids
            best['x'] = x.clone()
            best['z'] = z.clone()
            best['s'] = s.clone()
            best['y'] = y.clone() if y is not None else None
            nNotImproved = 0
        else:
            I = resids < best['resids']  # 残差更优的batch索引
            if I.sum() > 0:
                nNotImproved = 0  # 有更优解，重置计数器
            else:
                nNotImproved += 1  # 无更优解，计数器加1
            # 按batch更新最优解（仅更新残差更优的部分）
            I_nz = I.repeat(nz, 1).t()
            I_nineq = I.repeat(nineq, 1).t()
            best['resids'][I] = resids[I]
            best['x'][I_nz] = x[I_nz]
            best['z'][I_nineq] = z[I_nineq]
            best['s'][I_nineq] = s[I_nineq]
            if neq > 0:
                I_neq = I.repeat(neq, 1).t()
                best['y'][I_neq] = y[I_neq]

        # 4.4 判断收敛条件（残差足够小/残差未提升/互补度过大）
        if nNotImproved == notImprovedLim or best['resids'].max() < eps or mu.min() > 1e32:
            if best['resids'].max() > 1. and verbose >= 0:
                print(INACC_ERR)  # 残差过大时打印警告
            return best['x'], best['y'], best['z'], best['s']

        # 4.5 求解仿射方向（Affine Scaling Direction）：满足约束的梯度方向
        if solver == KKTSolvers.LU_FULL:
            D = bdiag(d)  # 构造d的块对角矩阵
            dx_aff, ds_aff, dz_aff, dy_aff = factor_solve_kkt(
                Q, D, G, A, rx, rs, rz, ry)
        elif solver == KKTSolvers.LU_PARTIAL:
            dx_aff, ds_aff, dz_aff, dy_aff = solve_kkt(
                Q_LU, d, G, A, S_LU, rx, rs, rz, ry)
        elif solver == KKTSolvers.IR_UNOPT:
            D = bdiag(d)
            dx_aff, ds_aff, dz_aff, dy_aff = solve_kkt_ir(
                Q, D, G, A, F, rx, rs, rz, ry)
        else:
            assert False

        # 4.6 求解中心方向（Centering Direction）：向约束中心靠近，避免边界震荡
        # 计算线搜索步长alpha（确保z+alpha*dz_aff ≥0，s+alpha*ds_aff ≥0）
        alpha = torch.min(torch.min(get_step(z, dz_aff), get_step(s, ds_aff)), torch.ones(nBatch).type_as(Q))
        alpha_nineq = alpha.repeat(nineq, 1).t()
        # 计算中心参数sig（基于当前互补度，控制中心方向权重）
        t1 = s + alpha_nineq * ds_aff
        t2 = z + alpha_nineq * dz_aff
        t3 = torch.sum(t1 * t2, 1).squeeze()
        t4 = torch.sum(s * z, 1).squeeze()
        sig = (t3 / t4) ** 3  # sig∈[0,1]，越接近1越偏向中心方向

        # 构造中心方向的KKT系统残差（仅rs非零，其他残差为0）
        rx = torch.zeros(nBatch, nz).type_as(Q)
        rs = ((-mu * sig).repeat(nineq, 1).t() + ds_aff * dz_aff) / s  # 中心方向的互补残差
        rz = torch.zeros(nBatch, nineq).type_as(Q)
        ry = torch.zeros(nBatch, neq).type_as(Q) if neq > 0 else torch.Tensor()

        # 求解中心方向
        if solver == KKTSolvers.LU_FULL:
            D = bdiag(d)
            dx_cor, ds_cor, dz_cor, dy_cor = factor_solve_kkt(Q, D, G, A, rx, rs, rz, ry)
        elif solver == KKTSolvers.LU_PARTIAL:
            dx_cor, ds_cor, dz_cor, dy_cor = solve_kkt(Q_LU, d, G, A, S_LU, rx, rs, rz, ry)
        elif solver == KKTSolvers.IR_UNOPT:
            D = bdiag(d)
            dx_cor, ds_cor, dz_cor, dy_cor = solve_kkt_ir(Q, D, G, A, F, rx, rs, rz, ry)
        else:
            assert False

        # 4.7 合并方向（仿射方向+中心方向）并线搜索更新变量
        dx = dx_aff + dx_cor  # 优化变量x的更新方向
        ds = ds_aff + ds_cor  # 松弛变量s的更新方向
        dz = dz_aff + dz_cor  # 对偶变量z的更新方向
        dy = dy_aff + dy_cor if neq > 0 else None  # 对偶变量y的更新方向

        # 计算最大安全步长（避免变量变负，违反s≥0、z≥0约束）
        alpha = torch.min(0.999 * torch.min(get_step(z, dz), get_step(s, ds)), torch.ones(nBatch).type_as(Q))
        # 按变量维度重复步长（批量更新）
        alpha_nineq = alpha.repeat(nineq, 1).t()
        alpha_neq = alpha.repeat(neq, 1).t() if neq > 0 else None
        alpha_nz = alpha.repeat(nz, 1).t()

        # 更新变量
        x += alpha_nz * dx
        s += alpha_nineq * ds
        z += alpha_nineq * dz
        y = y + alpha_neq * dy if neq > 0 else None

    # 迭代结束：返回最优解（若残差过大，打印警告）
    if best['resids'].max() > 1. and verbose >= 0:
        print(INACC_ERR)
    return best['x'], best['y'], best['z'], best['s']


def get_step(v, dv):
    """
    计算线搜索的最大安全步长：确保更新后变量v + alpha*dv ≥ 0（避免违反非负约束）
    Args:
        v (torch.Tensor): 当前变量值（如z或s，形状：[nBatch, dim]）
        dv (torch.Tensor): 变量更新方向（如dz或ds，形状与v一致）
    Returns:
        torch.Tensor: 每个batch的最大安全步长（形状：[nBatch]）
    """
    a = -v / dv  # 理论步长（使v+alpha*dv=0的alpha）
    a[dv > 0] = max(1.0, a.max())  # dv>0时变量会增大，步长无限制（设为1.0以上）
    return a.min(1)[0].squeeze()  # 取每个batch的最小步长（确保所有维度非负）


def unpack_kkt(v, nz, nineq, neq):
    """
    解包KKT系统的变量向量v：将扁平向量拆分为x、s、z、y
    Args:
        v (torch.Tensor): KKT系统的解向量（形状：[nBatch, nz+2*nineq+neq]）
        nz (int): 优化变量x的维度
        nineq (int): 不等式约束数（s和z的维度）
        neq (int): 等式约束数（y的维度）
    Returns:
        tuple: (x, s, z, y) —— 解包后的变量
    """
    i = 0
    x = v[:, i:i + nz]
    i += nz
    s = v[:, i:i + nineq]
    i += nineq
    z = v[:, i:i + nineq]
    i += nineq
    y = v[:, i:i + neq]
    return x, s, z, y


def kkt_resid_reg(Q_tilde, D_tilde, G, A, F_tilde, eps, dx, ds, dz, dy, rx, rs, rz, ry):
    """
    计算带正则化的KKT系统残差（用于迭代精修，提升解的稳定性）
    Args:
        Q_tilde (torch.Tensor): 正则化后的Q矩阵
        D_tilde (torch.Tensor): 正则化后的D矩阵
        其他参数：与forward函数中一致（更新方向、残差）
    Returns:
        tuple: 正则化后的KKT残差（resx, ress, resz, resy）
    """
    # 统一添加维度（适配批量矩阵乘法）
    dx, ds, dz, dy, rx, rs, rz, ry = [
        x.unsqueeze(2) if x is not None else None for x in
        [dx, ds, dz, dy, rx, rs, rz, ry]
    ]
    # 计算各变量的残差
    resx = Q_tilde.bmm(dx) + G.transpose(1, 2).bmm(dz) + rx
    if dy is not None:
        resx += A.transpose(1, 2).bmm(dy)  # 等式约束对偶变量贡献
    ress = D_tilde.bmm(ds) + dz + rs
    resz = G.bmm(dx) + ds - eps * dz + F_tilde.bmm(dz) + rz
    resy = A.bmm(dx) - eps * dy + ry if dy is not None else None
    # 移除多余维度
    resx, ress, resz, resy = (v.squeeze(2) if v is not None else None for v in (resx, ress, resz, resy))
    return resx, ress, resz, resy


def solve_kkt_ir(Q, D, G, A, F, rx, rs, rz, ry, niter=1):
    """
    带迭代精修（IR）的KKT系统求解：提升病态问题的解精度（未优化版本，速度较慢）
    Args:
        niter (int): 迭代精修次数（次数越多精度越高，默认1次）
    Returns:
        tuple: (dx, ds, dz, dy) —— KKT系统的解（更新方向）
    """
    nineq, nz, neq, nBatch = get_sizes(G, A)

    eps = 1e-7  # 正则化系数（避免矩阵奇异）
    # 构造正则化后的矩阵（添加小的对角项，提升数值稳定性）
    Q_tilde = Q + eps * torch.eye(nz).type_as(Q).repeat(nBatch, 1, 1)
    D_tilde = D + eps * torch.eye(nineq).type_as(Q).repeat(nBatch, 1, 1)

    # 构造正则化后的互补条件矩阵C_tilde
    C_tilde = -eps * torch.eye(neq + nineq).type_as(Q_tilde).repeat(nBatch, 1, 1)
    if F is not None:
        C_tilde[:, :nineq, :nineq] -= F  # 融入F矩阵的修正
    F_tilde = C_tilde[:, :nineq, :nineq]

    # 初始求解（带正则化）
    dx, ds, dz, dy = factor_solve_kkt_reg(
        Q_tilde, D_tilde, G, A, C_tilde, rx, rs, rz, ry, eps)
    # 迭代精修：修正初始解的残差
    res = kkt_resid_reg(Q, D, G, A, F_tilde, eps, dx, ds, dz, dy, rx, rs, rz, ry)
    resx, ress, resz, resy = res
    for k in range(niter):
        # 求解残差的修正量
        ddx, dds, ddz, ddy = factor_solve_kkt_reg(
            Q_tilde, D_tilde, G, A, C_tilde, -resx, -ress, -resz,
            -resy if resy is not None else None, eps)
        # 更新解（累加修正量）
        dx, ds, dz, dy = [v + dv if v is not None else None
                          for v, dv in zip((dx, ds, dz, dy), (ddx, dds, ddz, ddy))]
        # 重新计算残差
        res = kkt_resid_reg(Q, D, G, A, F_tilde, eps, dx, ds, dz, dy, rx, rs, rz, ry)
        resx, ress, resz, resy = res

    return dx, ds, dz, dy


def factor_solve_kkt_reg(Q_tilde, D_tilde, G, A, C_tilde, rx, rs, rz, ry, eps):
    """
    带正则化的KKT系统求解（通过全LU分解）：适配迭代精修场景
    内部逻辑：将KKT系统转化为增广矩阵，通过LU分解求解
    Returns:
        tuple: (dx, ds, dz, dy) —— KKT系统的解
    """
    nineq, nz, neq, nBatch = get_sizes(G, A)

    # 构造增广矩阵H_（Q_tilde和D_tilde的块对角组合）
    H_ = torch.zeros(nBatch, nz + nineq, nz + nineq).type_as(Q_tilde)
    H_[:, :nz, :nz] = Q_tilde
    H_[:, -nineq:, -nineq:] = D_tilde

    # 构造约束矩阵A_和右侧向量g_、h_（区分有无等式约束）
    if neq > 0:
        A_ = torch.cat([
            torch.cat([G, torch.eye(nineq).type_as(Q_tilde).repeat(nBatch, 1, 1)], 2),
            torch.cat([A, torch.zeros(nBatch, neq, nineq).type_as(Q_tilde)], 2)
        ], 1)
        g_ = torch.cat([rx, rs], 1)
        h_ = torch.cat([rz, ry], 1)
    else:
        A_ = torch.cat([G, torch.eye(nineq).type_as(Q_tilde).repeat(nBatch, 1, 1)], 2)
        g_ = torch.cat([rx, rs], 1)
        h_ = rz

    # LU分解增广矩阵H_
    H_LU = lu_hack(H_)
    # 求解H^{-1}A^T和H^{-1}g（用于后续S矩阵求解）
    invH_A_ = torch.linalg.lu_solve(*H_LU, A_.transpose(1, 2))
    invH_g_ = torch.linalg.lu_solve(*H_LU, g_.unsqueeze(2)).squeeze(2)

    # 构造S矩阵（KKT系统的Schur补）并分解
    S_ = torch.bmm(A_, invH_A_)
    S_ -= C_tilde
    S_ -= eps * torch.eye(neq + nineq).type_as(Q_tilde).repeat(nBatch, 1, 1)
    S_LU = lu_hack(S_)
    # 求解对偶变量w_（z和y的组合）
    t_ = torch.bmm(invH_g_.unsqueeze(1), A_.transpose(1, 2)).squeeze(1) - h_
    w_ = torch.linalg.lu_solve(*S_LU, -t_.unsqueeze(2)).squeeze(2)
    # 求解原变量v_（x和s的组合）
    t_ = -g_ - w_.unsqueeze(1).bmm(A_).squeeze()
    v_ = torch.linalg.lu_solve(*H_LU, t_.unsqueeze(2)).squeeze(2)

    # 解包变量（v_拆分为x和s，w_拆分为z和y）
    dx = v_[:, :nz]
    ds = v_[:, nz:]
    dz = w_[:, :nineq]
    dy = w_[:, nineq:] if neq > 0 else None

    return dx, ds, dz, dy


def factor_solve_kkt(Q, D, G, A, C_tilde, rx, rs, rz, ry):
    """
    无正则化的KKT系统求解（全LU分解）：对应KKTSolvers.LU_FULL求解器
    逻辑与factor_solve_kkt_reg类似，仅移除正则化项
    Returns:
        tuple: (dx, ds, dz, dy) —— KKT系统的解
    """
    nineq, nz, neq, nBatch = get_sizes(G, A)

    # 构造增广矩阵H_
    H_ = torch.zeros(nBatch, nz + nineq, nz + nineq).type_as(Q)
    H_[:, :nz, :nz] = Q
    H_[:, -nineq:, -nineq:] = D

    # 构造约束矩阵A_和右侧向量
    if neq > 0:
        A_ = torch.cat([
            torch.cat([G, torch.eye(nineq).type_as(Q).repeat(nBatch, 1, 1)], 2),
            torch.cat([A, torch.zeros(nBatch, neq, nineq).type_as(Q)], 2)
        ], 1)
        g_ = torch.cat([rx, rs], 1)
        h_ = torch.cat([rz, ry], 1)
    else:
        A_ = torch.cat([G, torch.eye(nineq).type_as(Q)], 1)
        g_ = torch.cat([rx, rs], 1)
        h_ = rz

    # LU分解与求解
    H_LU = lu_hack(H_)
    invH_A_ = torch.linalg.lu_solve(*H_LU, A_.transpose(1, 2))
    invH_g_ = torch.linalg.lu_solve(*H_LU, g_.unsqueeze(2)).squeeze(2)

    S_ = torch.bmm(A_, invH_A_) + C_tilde
    S_LU = lu_hack(S_)
    t_ = torch.bmm(invH_g_.unsqueeze(1), A_.transpose(1, 2)).squeeze(1) - h_
    w_ = torch.linalg.lu_solve(*S_LU, -t_.unsqueeze(2)).squeeze(2)
    t_ = -g_ - w_.unsqueeze(1).bmm(A_).squeeze()
    v_ = torch.linalg.lu_solve(*H_LU, t_.unsqueeze(2)).squeeze(2)

    # 解包变量
    dx = v_[:, :nz]
    ds = v_[:, nz:]
    dz = w_[:, :nineq]
    dy = w_[:, nineq:] if neq > 0 else None

    return dx, ds, dz, dy


def init_s_z(Q_LU, d, G, A, S_LU, rx, rs, rz, ry):
    """初始化松弛变量s和对偶变量z：仅求解KKT系统中s和z的更新方向"""
    nineq, nz, neq, nBatch = get_sizes(G, A)

    # 求解Q^{-1}rx（Q的LU分解已预计算）
    invQ_rx = torch.linalg.lu_solve(*Q_LU, rx.unsqueeze(2)).squeeze(2)
    # 构造右侧向量h（区分有无等式约束）
    if neq > 0:
        h = torch.cat((
            invQ_rx.unsqueeze(1).bmm(A.transpose(1, 2)).squeeze(1) - ry,
            invQ_rx.unsqueeze(1).bmm(G.transpose(1, 2)).squeeze(1) + rs / d - rz
        ), 1)
    else:
        h = invQ_rx.unsqueeze(1).bmm(G.transpose(1, 2)).squeeze(1) + rs / d - rz

    # 求解对偶变量w（含z和y）
    w = torch.linalg.lu_solve(*S_LU, -h.unsqueeze(2)).squeeze(2)
    # 计算s和z的更新方向
    g2 = -rs - w[:, neq:]
    ds = g2 / d
    dz = w[:, neq:]

    return ds, dz


def solve_kkt(Q_LU, d, G, A, S_LU, rx, rs, rz, ry):
    """
    部分LU分解的KKT系统求解：对应KKTSolvers.LU_PARTIAL求解器（论文默认）
    核心优化：预计算Q和S矩阵的固定部分，仅迭代更新可变部分（d=z/s），提升效率
    Returns:
        tuple: (dx, ds, dz, dy) —— KKT系统的解
    """
    nineq, nz, neq, nBatch = get_sizes(G, A)

    # 1. 求解Q^{-1}rx（Q的LU分解已预计算，无需重复分解）
    invQ_rx = torch.linalg.lu_solve(*Q_LU, rx.unsqueeze(2)).squeeze(2)

    # 2. 构造右侧向量h（融合等式/不等式约束残差）
    if neq > 0:
        h = torch.cat((
            # 等式约束部分：Q^{-1}rx * A^T - ry
            invQ_rx.unsqueeze(1).bmm(A.transpose(1, 2)).squeeze(1) - ry,
            # 不等式约束部分：Q^{-1}rx * G^T + rs/d - rz
            invQ_rx.unsqueeze(1).bmm(G.transpose(1, 2)).squeeze(1) + rs / d - rz
        ), 1)
    else:
        h = invQ_rx.unsqueeze(1).bmm(G.transpose(1, 2)).squeeze(1) + rs / d - rz

    # 3. 求解对偶变量w（S矩阵的LU分解已通过factor_kkt更新，直接使用）
    try:
        w = torch.linalg.lu_solve(*S_LU, -h.unsqueeze(2)).squeeze(2)
    except RuntimeError as err:
        # 兼容PyTorch批量LU分解的bug（单batch时特殊处理）
        if h.shape[0] == 1:
            w = -(h[0].unsqueeze(1).lu_solve(*[lu[0] for lu in S_LU])).squeeze(1).unsqueeze(0)
        else:
            raise err

    # 4. 计算原变量x和松弛变量s的更新方向
    # g1：x的更新方向残差（-rx - z*G - y*A）
    g1 = -rx - w[:, neq:].unsqueeze(1).bmm(G).squeeze(1)
    if neq > 0:
        g1 -= w[:, :neq].unsqueeze(1).bmm(A).squeeze(1)
    # g2：s的更新方向残差（-rs - z）
    g2 = -rs - w[:, neq:]

    # 5. 求解x的更新方向（Q的LU分解复用）
    dx = torch.linalg.lu_solve(*Q_LU, g1.unsqueeze(2)).squeeze(2)
    # 计算s和z的更新方向
    ds = g2 / d
    dz = w[:, neq:]
    dy = w[:, :neq] if neq > 0 else None

    return dx, ds, dz, dy


def solve_kkt_backward(Q_LU, d, G, A, S_LU, rx, rs, rz, ry):
    """solve_kkt的反向传播适配版本：确保梯度能正确回传（逻辑与solve_kkt一致，仅符号微调）"""
    nineq, nz, neq, nBatch = get_sizes(G, A)

    invQ_rx = rx.unsqueeze(2).lu_solve(*Q_LU).squeeze(2)
    if neq > 0:
        h = torch.cat((
            invQ_rx.unsqueeze(1).bmm(A.transpose(1, 2)).squeeze(1) - ry,
            invQ_rx.unsqueeze(1).bmm(G.transpose(1, 2)).squeeze(1) + rs / d - rz
        ), 1)
    else:
        h = invQ_rx.unsqueeze(1).bmm(G.transpose(1, 2)).squeeze(1) + rs / d - rz

    try:
        w = -(h.unsqueeze(2).lu_solve(*S_LU)).squeeze(2)
    except RuntimeError as err:
        if h.shape[0] == 1:
            w = -(h[0].unsqueeze(1).lu_solve(*[lu[0] for lu in S_LU])).squeeze(1).unsqueeze(0)
        else:
            raise err

    g1 = -rx - w[:, neq:].unsqueeze(1).bmm(G).squeeze(1)
    if neq > 0:
        g1 -= w[:, :neq].unsqueeze(1).bmm(A).squeeze(1)
    g2 = -rs - w[:, neq:]

    dx = g1.unsqueeze(2).lu_solve(*Q_LU).squeeze(2)
    ds = g2 / d
    dz = w[:, neq:]
    dy = w[:, :neq] if neq > 0 else None

    return dx, ds, dz, dy


def pre_factor_kkt(Q, G, A, F):
    """
    预计算KKT系统的固定部分：为部分LU分解（LU_PARTIAL）加速，减少迭代中的重复计算
    核心：预分解Q矩阵、计算S矩阵的固定部分（与d无关的部分）
    Args:
        Q, G, A, F: 与forward函数中一致（QP问题的系数矩阵）
    Returns:
        tuple: (Q_LU, S_LU, R) —— 预计算的LU分解结果和固定矩阵
    """
    nineq, nz, neq, nBatch = get_sizes(G, A)

    # 1. 预分解Q矩阵（QP目标函数的二次项系数，迭代中固定）
    try:
        Q_LU = lu_hack(Q)
    except:
        raise RuntimeError("""
qpth Error: Cannot perform LU factorization on Q.
Please make sure that your Q matrix is PSD and has
a non-zero diagonal.
""")

    # 2. 计算S矩阵的固定部分（G Q^{-1} G^T + F）
    try:
        # G * Q^{-1} * G^T（复用Q的LU分解）
        G_invQ_GT = torch.bmm(G, torch.linalg.lu_solve(*Q_LU, G.transpose(1, 2))) + F
    except RuntimeError as err:
        # 兼容单batch的bug
        if G.shape[0] == 1:
            G_invQ_GT = torch.bmm(
                G, G.squeeze(0).t().lu_solve(*[lu.squeeze(0) for lu in Q_LU]).unsqueeze(0)
            ) + F
        else:
            raise err

    R = G_invQ_GT.clone()  # R：S矩阵中与d无关的固定部分
    # 初始化S矩阵的pivots（主元索引，1-based）
    S_LU_pivots = torch.IntTensor(range(1, 1 + neq + nineq)).unsqueeze(0) \
        .repeat(nBatch, 1).type_as(Q).int()

    # 3. 若存在等式约束，计算S矩阵的块LU分解（分块处理，提升效率）
    if neq > 0:
        # 计算Q^{-1} A^T 和 A Q^{-1} A^T
        invQ_AT = torch.linalg.lu_solve(*Q_LU, A.transpose(1, 2))
        A_invQ_AT = torch.bmm(A, invQ_AT)
        G_invQ_AT = torch.bmm(G, invQ_AT)

        # 分解A Q^{-1} A^T（S矩阵的(1,1)块）
        LU_A_invQ_AT = lu_hack(A_invQ_AT)
        P_A_invQ_AT, L_A_invQ_AT, U_A_invQ_AT = torch.lu_unpack(*LU_A_invQ_AT)
        P_A_invQ_AT = P_A_invQ_AT.type_as(A_invQ_AT)

        # 构造S矩阵的LU分解数据（分块存储）
        S_LU_11 = LU_A_invQ_AT[0]  # S矩阵(1,1)块的LU分解
        # 计算U_A_invQ_AT的逆（用于后续块分解）
        U_A_invQ_AT_inv = torch.linalg.lu_solve(
            *LU_A_invQ_AT, P_A_invQ_AT.bmm(L_A_invQ_AT)
        )
        S_LU_21 = G_invQ_AT.bmm(U_A_invQ_AT_inv)  # S矩阵(2,1)块
        # 计算S矩阵(1,2)块
        try:
            T = torch.linalg.lu_solve(*LU_A_invQ_AT, G_invQ_AT.transpose(1, 2))
        except RuntimeError as err:
            if G_invQ_AT.shape[0] == 1:
                T = G_invQ_AT.squeeze(0).t().lu_solve(*[lu.squeeze(0) for lu in LU_A_invQ_AT]).unsqueeze(0)
            else:
                raise err
        S_LU_12 = U_A_invQ_AT.bmm(T)
        S_LU_22 = torch.zeros(nBatch, nineq, nineq).type_as(Q)  # S矩阵(2,2)块（迭代中更新）

        # 合并S矩阵的LU分解数据
        S_LU_data = torch.cat((
            torch.cat((S_LU_11, S_LU_12), 2),
            torch.cat((S_LU_21, S_LU_22), 2)
        ), 1)
        # 更新pivots（复用A_invQ_AT的pivots）
        S_LU_pivots[:, :neq] = LU_A_invQ_AT[1]

        # 更新R矩阵（减去与等式约束相关的部分）
        R -= G_invQ_AT.bmm(T)
    else:
        # 无等式约束时，S矩阵仅含nineq维度，初始化空数据
        S_LU_data = torch.zeros(nBatch, nineq, nineq).type_as(Q)

    # 封装S矩阵的LU分解结果（数据+pivots）
    S_LU = [S_LU_data, S_LU_pivots]
    return Q_LU, S_LU, R


# 全局变量：缓存单位矩阵（避免重复创建，提升效率）
factor_kkt_eye = None


def factor_kkt(S_LU, R, d):
    """
    完成S矩阵的LU分解（迭代中可变部分）：S矩阵(2,2)块 = R + diag(1/d)
    仅更新S矩阵中与d（z/s）相关的部分，复用预计算的固定部分
    Args:
        S_LU (tuple): 预计算的S矩阵LU分解结果（数据+pivots）
        R (torch.Tensor): S矩阵的固定部分
        d (torch.Tensor): 迭代中更新的d=z/s（形状：[nBatch, nineq]）
    """
    nBatch, nineq = d.size()
    neq = S_LU[1].size(1) - nineq  # 等式约束数（从pivots维度推导）

    # 缓存单位矩阵（匹配d的形状，避免重复创建）
    global factor_kkt_eye
    if factor_kkt_eye is None or factor_kkt_eye.size() != d.size():
        factor_kkt_eye = torch.eye(nineq).repeat(nBatch, 1, 1).type_as(R).bool()

    # 构造S矩阵(2,2)块：R + diag(1/d)（仅更新对角元素）
    T = R.clone()
    T[factor_kkt_eye] += (1. / d).squeeze().view(-1)  # 对角元素 += 1/d

    # LU分解S矩阵(2,2)块
    T_LU = lu_hack(T)

    # 若在CPU上，更新pivots（GPU上已手动生成pivots，无需更新）
    if not T.is_cuda:
        # 提取旧pivots并调整索引（S矩阵(2,2)块的pivots是整体pivots的后nineq个）
        oldPivotsPacked = S_LU[1][:, -nineq:] - neq
        oldPivots, _, _ = torch.lu_unpack(T_LU[0], oldPivotsPacked, unpack_data=False)
        newPivotsPacked = T_LU[1]
        newPivots, _, _ = torch.lu_unpack(T_LU[0], newPivotsPacked, unpack_data=False)

        # 若有等式约束，更新S矩阵(2,1)块的pivots（重新排列）
        if neq > 0:
            S_LU_21 = S_LU[0][:, -nineq:, :neq]
            S_LU[0][:, -nineq:, :neq] = newPivots.transpose(1, 2).bmm(oldPivots.bmm(S_LU_21))

        # 更新S矩阵的整体pivots（后nineq个为新pivots）
        S_LU[1][:, -nineq:] = newPivotsPacked + neq

    # 将T的LU分解结果写入S矩阵的(2,2)块
    S_LU[0][:, -nineq:, -nineq:] = T_LU[0]