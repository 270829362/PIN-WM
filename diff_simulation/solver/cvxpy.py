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
# 核心用途：基于CVXPY的二次规划（QP）求解，作为基准工具验证自定义求解器正确性
#

import cvxpy as cp  # 导入凸优化库CVXPY（声明式优化建模工具）
import numpy as np  # 导入NumPy（处理数值计算和数组）


def forward_single_np(Q, p, G, h, A, b):
    """
    求解单样本（无批量）的二次规划（QP）问题：最小化二次目标函数，满足线性等式和不等式约束
    QP问题的标准数学形式：
        最小化：(1/2) * z^T * Q * z + p^T * z  （二次目标函数）
        约束条件：
            A * z == b          （线性等式约束，可选）
            G * z + slacks == h  （线性不等式约束，通过松弛变量slacks转化为等式，slacks ≥ 0）
    
    Args:
        Q (np.ndarray): 目标函数二次项系数矩阵（形状：[nz, nz]，nz为优化变量维度，需为对称半正定矩阵）
        p (np.ndarray): 目标函数一次项系数向量（形状：[nz,]）
        G (np.ndarray): 不等式约束矩阵（形状：[nineq, nz]，nineq为不等式约束数量；无不等式约束时需保证逻辑兼容）
        h (np.ndarray): 不等式约束右侧向量（形状：[nineq,]）
        A (np.ndarray): 等式约束矩阵（形状：[neq, nz]，neq为等式约束数量；None表示无等式约束）
        b (np.ndarray): 等式约束右侧向量（形状：[neq,]；None表示无等式约束）
    
    Returns:
        tuple: QP问题的求解结果，包含5个元素：
            - prob.value: 目标函数的最优值（float）
            - zhat: 优化变量z的最优解（np.ndarray，形状：[nz,]）
            - nu: 等式约束的对偶变量（np.ndarray，形状：[neq,]；None表示无等式约束）
            - lam: 不等式约束的对偶变量（np.ndarray，形状：[nineq,]；None表示无不等式约束）
            - slacks: 不等式约束的松弛变量（np.ndarray，形状：[nineq,]；None表示无不等式约束）
    """
    # 1. 解析问题维度：nz=优化变量数，neq=等式约束数，nineq=不等式约束数
    nz = p.shape[0]  # 优化变量z的维度（从一次项系数p的长度推导）
    neq = A.shape[0] if A is not None else 0  # 等式约束数量（A为None时无等式约束）
    nineq = G.shape[0]  # 不等式约束数量（从G的行数推导）

    # 2. 定义优化变量：z为QP问题的核心优化变量
    z_ = cp.Variable(nz)

    # 3. 定义目标函数：(1/2)*z^T*Q*z + p^T*z（CVXPY的quad_form用于表示二次型）
    obj = cp.Minimize(0.5 * cp.quad_form(z_, Q) + p.T * z_)

    # 4. 定义约束条件：分等式约束、不等式约束（含松弛变量）
    # 4.1 等式约束：A*z == b（仅当存在等式约束时定义）
    eqCon = A * z_ == b if neq > 0 else None  # eqCon为None表示无等式约束

    # 4.2 不等式约束：将G*z ≤ h转化为G*z + slacks == h（slacks ≥ 0，松弛变量）
    if nineq > 0:
        slacks = cp.Variable(nineq)  # 定义松弛变量（数量=不等式约束数）
        ineqCon = G * z_ + slacks == h  # 不等式约束转化为等式约束
        slacksCon = slacks >= 0  # 松弛变量非负约束（确保原不等式G*z ≤ h成立）
    else:
        # 无不等式约束时，相关变量设为None
        ineqCon = None
        slacks = None
        slacksCon = None

    # 5. 整合所有有效约束：过滤掉None值（无对应约束时）
    cons = [x for x in [eqCon, ineqCon, slacksCon] if x is not None]

    # 6. 构建并求解QP问题
    prob = cp.Problem(obj, cons)  # 创建优化问题实例（目标函数+约束）
    prob.solve()  # 调用默认求解器（SCS）求解；可手动指定solver=cp.SCS等，调整max_iters等参数
    # 可选：若求解精度不足，可增加迭代次数并开启日志：
    # prob.solve(solver=cp.SCS, max_iters=10000, verbose=True)

    # 7. 验证求解状态：确保问题可行且求解成功（仅处理“optimal”状态，排除不可行、无界等情况）
    assert 'optimal' in prob.status, f"QP求解失败，状态：{prob.status}（可能问题不可行或无界）"

    # 8. 提取求解结果：将CVXPY变量值转换为NumPy数组（ravel()确保为1维数组）
    zhat = np.array(z_.value).ravel()  # 优化变量z的最优解

    # 提取等式约束的对偶变量（对偶变量对应约束的“拉格朗日乘子”，反映约束的松紧程度）
    nu = np.array(eqCon.dual_value).ravel() if eqCon is not None else None

    # 提取不等式约束的对偶变量和松弛变量
    if ineqCon is not None:
        lam = np.array(ineqCon.dual_value).ravel()  # 不等式约束的对偶变量
        slacks = np.array(slacks.value).ravel()      # 松弛变量的最优值
    else:
        lam = None
        slacks = None

    # 返回完整求解结果
    return prob.value, zhat, nu, lam, slacks