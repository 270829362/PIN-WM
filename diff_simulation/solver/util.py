#该代码是可微分 LCP/QP 求解的核心工具函数集，适配 PyTorch 批量张量计算，支撑 3D 刚体动力学模拟中 “约束矩阵构建、维度统一、张量操作、正交向量计算” 等底层逻辑。所有函数均为之前LCP_Solver类的配套工具：
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
# 功能：LCP/QP求解的核心工具函数，支撑3D刚体动力学模拟的张量操作/维度处理/数学计算
# 修改者：Rama Krishna Kandukuri（MPI-IS）
#
import torch
import numpy as np


def print_header(msg):
    """
    日志打印辅助函数：统一格式打印关键信息（如求解器迭代状态、错误提示）
    Args:
        msg: 待打印的字符串（如"开始求解LCP问题"）
    """
    print('===>', msg)


def to_np(t):
    """
    张量转numpy数组：适配调试/可视化场景（PyTorch张量→NumPy数组）
    处理边界情况：None/空张量/正常张量
    Args:
        t (torch.Tensor/None): 待转换的张量
    Returns:
        np.array/None: 转换后的NumPy数组（CPU上）
    """
    if t is None:
        return None
    elif t.nelement() == 0:  # 空张量（如无等式约束时的A矩阵）
        return np.array([])
    else:
        return t.cpu().numpy()  # 转到CPU后转numpy（避免GPU张量无法直接转换）


def bger(x, y):
    """
    批量外积（Batch Outer Product）：计算两个张量的批量外积，适配QP求解的梯度计算
    公式：bger(x,y) = x ⊗ y = x.unsqueeze(2) @ y.unsqueeze(1)
    示例：x=[nBatch, d1], y=[nBatch, d2] → 输出[nBatch, d1, d2]
    Args:
        x (torch.Tensor): 第一个批量张量（形状[nBatch, d1]）
        y (torch.Tensor): 第二个批量张量（形状[nBatch, d2]）
    Returns:
        torch.Tensor: 批量外积结果（形状[nBatch, d1, d2]）
    """
    return x.unsqueeze(2).bmm(y.unsqueeze(1))


def get_sizes(G, A=None):
    """
    解析约束矩阵的维度：提取不等式约束数、优化变量维度、等式约束数、批量数
    是LCP转化为QP的核心维度解析函数，适配批量/非批量张量
    Args:
        G (torch.Tensor): 不等式约束矩阵（形状：[nineq, nz] 或 [nBatch, nineq, nz]）
        A (torch.Tensor/None): 等式约束矩阵（形状：[neq, nz] 或 [nBatch, neq, nz]；None表示无等式约束）
    Returns:
        tuple: (nineq, nz, neq, nBatch)
            nineq: 不等式约束数
            nz: 优化变量维度（如系统总速度维度）
            neq: 等式约束数（无则为None）
            nBatch: 批量数（非批量时为1）
    """
    # 解析G矩阵的维度（必选，至少有一个约束）
    if G.dim() == 2:  # 非批量：[nineq, nz]
        nineq, nz = G.size()
        nBatch = 1
    elif G.dim() == 3:  # 批量：[nBatch, nineq, nz]
        nBatch, nineq, nz = G.size()
    # 解析A矩阵的维度（可选）
    if A is not None:
        neq = A.size(1) if A.nelement() > 0 else 0  # 空张量时neq=0
    else:
        neq = None
    return nineq, nz, neq, nBatch


def bdiag(d):
    """
    构建批量块对角矩阵（Batch Diagonal Matrix）：将向量转为批量对角矩阵
    示例：d=[nBatch, sz] → 输出[nBatch, sz, sz]，对角元素为d的对应值，其余为0
    Args:
        d (torch.Tensor): 批量向量（形状[nBatch, sz]）
    Returns:
        torch.Tensor: 批量块对角矩阵（形状[nBatch, sz, sz]）
    """
    nBatch, sz = d.size()
    # 初始化全零矩阵
    D = torch.zeros(nBatch, sz, sz).type_as(d)
    # 生成批量单位矩阵的bool掩码（仅对角为True）
    I = torch.eye(sz).repeat(nBatch, 1, 1).type_as(d).bool()
    # 填充对角元素：将d展平后赋值到对角位置
    D[I] = d.squeeze().view(-1)
    return D


def expandParam(X, nBatch, nDim):
    """
    扩展张量维度：将非批量张量扩展为批量张量，统一输入维度（适配批处理QP求解）
    核心逻辑：若张量维度不足（如2D→3D），添加batch维度并扩展到指定批量数
    Args:
        X (torch.Tensor): 待扩展的张量（如Q/p/G/h/A/b/F）
        nBatch (int): 目标批量数
        nDim (int): 目标维度数（如Q矩阵目标维度为3，p矩阵为2）
    Returns:
        tuple: (扩展后的张量, 是否扩展标记)
            扩展后的张量：维度统一为[nBatch, ...]
            是否扩展标记：True=已扩展，False=无需扩展（原维度已匹配）
    """
    # 边界情况：空张量/维度已匹配 → 无需扩展
    if X.ndimension() in (0, nDim) or X.nelement() == 0:
        return X, False
    # 维度不足：添加batch维度并扩展（如2D→3D：[d1,d2] → [nBatch, d1, d2]）
    elif X.ndimension() == nDim - 1:
        return X.unsqueeze(0).expand(*([nBatch] + list(X.size()))), True
    else:
        raise RuntimeError("Unexpected number of dimensions.")


def extract_nBatch(Q, p, G, h, A, b, F):
    """
    提取批量数：从QP问题的输入矩阵（Q/p/G/h/A/b/F）中自动识别批量数
    核心逻辑：遍历所有输入，找到第一个维度匹配目标的张量，提取其batch维度
    Args:
        Q/p/G/h/A/b/F: QP问题的输入矩阵（维度分别为3/2/3/2/3/2/3）
    Returns:
        int: 批量数（非批量时返回1）
    """
    dims = [3, 2, 3, 2, 3, 2, 3]  # 各输入的目标维度（Q=3D, p=2D, G=3D...）
    params = [Q, p, G, h, A, b, F]
    # 遍历所有输入，找到第一个维度匹配的张量，提取batch数
    for param, dim in zip(params, dims):
        if param.ndimension() == dim:
            return param.size(0)
    return 1  # 所有输入均为非批量，返回1


def extract_batch_size(Q, p, G, h, A, b):
    """
    简化版批量数提取：仅处理Q/p/G/h/A/b（无F矩阵），逻辑与extract_nBatch一致
    适配无互补修正矩阵F的QP场景
    """
    dims = [3, 2, 3, 2, 3, 2]
    params = [Q, p, G, h, A, b]
    for param, dim in zip(params, dims):
        if param.ndimension() == dim:
            return param.size(0)
    return 1


def efficient_btriunpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True):
    """
    高效批量LU分解解包：替代PyTorch原生btriunpack，提升GPU/批量场景的效率
    核心：手动解包LU分解的结果（置换矩阵P、下三角矩阵L、上三角矩阵U）
    来源：https://github.com/pytorch/pytorch/issues/15182
    Args:
        LU_data (torch.Tensor): LU分解的紧凑存储数据（形状[nBatch, sz, sz]）
        LU_pivots (torch.Tensor): LU分解的主元索引（形状[nBatch, sz]）
        unpack_data (bool): 是否解包L/U矩阵（默认True）
        unpack_pivots (bool): 是否解包置换矩阵P（默认True）
    Returns:
        tuple: (P, L, U)
            P: 置换矩阵（形状[nBatch, sz, sz]）
            L: 下三角矩阵（单位对角，形状[nBatch, sz, sz]）
            U: 上三角矩阵（形状[nBatch, sz, sz]）
    """
    nBatch, sz = LU_data.shape[:-1]

    # 解包L/U矩阵：从LU_data中分离下三角（L）和上三角（U）
    if unpack_data:
        # 生成上三角掩码（triu_：上三角为True，下三角为False）
        I_U = torch.ones(sz, sz, device=LU_data.device, dtype=torch.uint8).triu_().expand_as(LU_data)
        zero = torch.tensor(0.).type_as(LU_data)
        U = torch.where(I_U, LU_data, zero)  # 上三角保留LU_data，下三角置0 → U矩阵
        L = torch.where(I_U, zero, LU_data)  # 下三角保留LU_data，上三角置0 → L矩阵
        L.diagonal(dim1=-2, dim2=-1).fill_(1)  # L矩阵对角置1（单位下三角）
    else:
        L = U = None

    # 解包置换矩阵P：根据主元索引重构置换矩阵
    if unpack_pivots:
        # 初始化单位矩阵（批量）
        P = torch.eye(sz, device=LU_data.device, dtype=LU_data.dtype).unsqueeze(0).repeat(nBatch, 1, 1)
        LU_pivots = LU_pivots - 1  # 主元索引转0-based（PyTorch默认1-based）
        # 逐batch重构置换矩阵
        for i in range(nBatch):
            final_order = list(range(sz))  # 初始顺序
            # 根据主元索引交换行
            for k, j in enumerate(LU_pivots[i]):
                final_order[k], final_order[j] = final_order[j], final_order[k]
            P[i] = P[i][final_order]  # 应用行置换
        P = P.transpose(-2, -1)  # 转置：行置换→列置换（适配KKT系统求解）
    else:
        P = None

    return P, L, U


def orthogonal(v):
    """
    计算3D空间中与输入向量正交的任意向量：用于构建摩擦方向（垂直于接触法向）
    核心逻辑：找到输入向量中绝对值最小的维度，构建基向量后叉乘，确保正交性
    Args:
        v (torch.Tensor): 3D向量（如接触法向量，形状[3]）
    Returns:
        torch.Tensor: 与v正交的3D向量（形状[3]）
    """
    # 找到v中绝对值最小的维度（如v=[1,0,0] → min_index=1/2）
    min_index = torch.argmin(v.abs())
    
    # 构建基向量：最小维度置1，其余置0（确保与v正交）
    base_vector = torch.zeros_like(v)
    base_vector[min_index] = 1.0
    
    # 叉乘：base_vector × v → 结果与v正交
    return torch.cross(base_vector, v)