""" 这段代码是 PIN-WM可微分仿真与物理参数计算的底层核心，所有函数都围绕 “3D 刚体物理属性建模” 展开，直接支撑论文的核心创新（物理参数端到端优化），关键作用可概括为 3 类：
1. 物理核心参数计算（论文优化目标）
cal_MassProperties/get_ang_inertia：计算物体的体积、质心、惯性张量（论文中优化的核心物理参数之一），支持 PyTorch 自动微分，可直接融入模型训练流程；
两个 NumPy 版本（cal_MassProperties_np/inertia_diagonalize_np）：用于离线预处理（如数据集初始化时计算物体初始物理属性）或结果验证。
2. 仿真姿态与交互计算（仿真环境支撑）
cal_transform_matrix/get_mesh_world：更新物体在世界坐标系的姿态（位置 + 旋转），适配仿真中物体的运动状态更新；
create_sample_points/get_contact_info：生成接触点并转换到世界坐标系，为机器人与物体的交互力计算提供基础（如论文中的推 / 翻转操作）。
3. 惯性张量优化辅助（物理计算简化）
inertia_diagonalize：将惯性张量对角化，得到惯性主轴坐标系，简化 3D 刚体旋转动力学方程的求解（让仿真计算更高效、稳定）。 """
# 导入核心依赖库
import trimesh  # 3D网格处理核心库（加载/操作物体3D模型、计算网格属性）
import torch  # 深度学习框架（支持GPU计算、自动微分，适配模型训练）
import numpy as np  # 数值计算库（纯数值物理计算、数组处理）

def cal_MassProperties(mesh: trimesh.base.Trimesh, device):
    """
    【PyTorch版本】计算3D网格物体的质量特性（适配模型训练，支持自动微分）
    核心功能：基于网格的三角形信息，通过积分计算物体的体积、质心、单位密度惯性张量
    算法参考：http://www.geometrictools.com/Documentation/PolyhedralMassProperties.pdf（多面体质量特性计算标准文档）
    Args:
        mesh: trimesh网格对象（存储物体的3D三角形顶点、面信息）
        device: 计算设备（如torch.device("cuda")，支持GPU加速）
    Returns:
        volume: 物体体积（float，单位与网格坐标一致）
        center_mass: 质心坐标（3维Tensor，[x, y, z]，物体局部坐标系下）
        inertia_body_unit: 单位密度惯性张量（3x3对称Tensor，物体局部坐标系下，需乘以密度得到实际惯性张量）
    """
    # 提取网格的三角形顶点（Nx3x3：N个三角形，每个三角形3个顶点，每个顶点3个坐标），转换为PyTorch张量
    triangles = torch.tensor(mesh.triangles, dtype=torch.float32, device=device)
    # 提取三角形的叉积（trimesh预计算，用于快速计算积分，Nx3）
    crosses = torch.tensor(mesh.triangles_cross, dtype=torch.float32, device=device)

    # 积分子表达式f1：每个三角形三个顶点的坐标和（替代triangles.sum(axis=1)，提速7倍）
    f1 = triangles[:, 0, :] + triangles[:, 1, :] + triangles[:, 2, :]
    # 积分子表达式f2：二次项组合（用于计算体积、质心相关积分）
    f2 = (
        triangles[:, 0, :] ** 2  # 第一个顶点坐标的平方
        + triangles[:, 1, :] ** 2  # 第二个顶点坐标的平方
        + triangles[:, 0, :] * triangles[:, 1, :]  # 前两个顶点坐标的乘积
        + triangles[:, 2, :] * f1  # 第三个顶点坐标与f1的乘积
    )
    # 积分子表达式f3：三次项组合（用于计算惯性张量相关积分）
    f3 = (
        (triangles[:, 0, :] ** 3)  # 第一个顶点坐标的三次方
        + (triangles[:, 0, :] ** 2) * (triangles[:, 1, :])  # 第一个顶点平方×第二个顶点
        + (triangles[:, 0, :]) * (triangles[:, 1, :] ** 2)  # 第一个顶点×第二个顶点平方
        + (triangles[:, 1, :] ** 3)  # 第二个顶点坐标的三次方
        + (triangles[:, 2, :] * f2)  # 第三个顶点坐标与f2的乘积
    )
    # 积分子表达式g0/g1/g2：用于计算惯性张量的交叉项积分
    g0 = f2 + (triangles[:, 0, :] + f1) * triangles[:, 0, :]
    g1 = f2 + (triangles[:, 1, :] + f1) * triangles[:, 1, :]
    g2 = f2 + (triangles[:, 2, :] + f1) * triangles[:, 2, :]

    # 初始化积分结果矩阵（10个积分项 × N个三角形，10项对应体积、质心、惯性张量的不同分量）
    integral = torch.zeros((10, len(f1)), dtype=torch.float32, device=device)
    integral[0] = crosses[:, 0] * f1[:, 0]  # 第0项：体积相关积分
    integral[1:4] = (crosses * f2).T  # 第1-3项：质心相关积分（x/y/z方向）
    integral[4:7] = (crosses * f3).T  # 第4-6项：惯性张量对角项相关积分
    # 第7-9项：惯性张量交叉项相关积分（遍历x/y/z三个方向）
    for i in range(3):
        triangle_i = torch.fmod(torch.tensor(i + 1, device=device), 3)  # 循环索引（0→1,1→2,2→0）
        integral[i + 7] = crosses[:, i] * (
            (triangles[:, 0, triangle_i] * g0[:, i])
            + (triangles[:, 1, triangle_i] * g1[:, i])
            + (triangles[:, 2, triangle_i] * g2[:, i])
        )

    # 积分结果的归一化系数（根据参考文档的数学推导，10个积分项对应不同系数）
    coefficients = 1.0 / torch.tensor(
        [6, 24, 24, 24, 60, 60, 60, 120, 120, 120], dtype=torch.float32, device=device
    )
    integrated = integral.sum(axis=1) * coefficients  # 对所有三角形积分求和，乘以系数得到最终积分结果

    volume = integrated[0]  # 体积（第0项积分结果）
    center_mass = integrated[1:4] / volume  # 质心（第1-3项积分结果 ÷ 体积，满足质心定义）

    # 初始化单位密度惯性张量（3x3对称矩阵，存储物体抵抗旋转的能力）
    inertia_body_unit = torch.zeros((3, 3), dtype=torch.float32, device=device)
    # 惯性张量对角项（Ixx, Iyy, Izz）：绕x/y/z轴的转动惯量
    inertia_body_unit[0, 0] = (integrated[5] + integrated[6] - (volume * (center_mass[[1, 2]] ** 2).sum()))  # Ixx
    inertia_body_unit[1, 1] = (integrated[4] + integrated[6] - (volume * (center_mass[[0, 2]] ** 2).sum()))  # Iyy
    inertia_body_unit[2, 2] = (integrated[4] + integrated[5] - (volume * (center_mass[[0, 1]] ** 2).sum()))  # Izz
    # 惯性张量交叉项（Ixy, Iyz, Ixz）：对称性，Ixy=Iyx，Iyz=Izy，Ixz=Izx
    inertia_body_unit[0, 1] = -(integrated[7] - (volume * torch.prod(center_mass[[0, 1]])))  # Ixy
    inertia_body_unit[1, 2] = -(integrated[8] - (volume * torch.prod(center_mass[[1, 2]])))  # Iyz
    inertia_body_unit[0, 2] = -(integrated[9] - (volume * torch.prod(center_mass[[0, 2]])))  # Ixz
    inertia_body_unit[2, 0] = inertia_body_unit[0, 2]  # Izx = Ixz
    inertia_body_unit[2, 1] = inertia_body_unit[1, 2]  # Izy = Iyz
    inertia_body_unit[1, 0] = inertia_body_unit[0, 1]  # Iyx = Ixy

    return volume, center_mass, inertia_body_unit


def cal_MassProperties_np(mesh: trimesh.base.Trimesh):
    """
    【NumPy版本】计算3D网格物体的质量特性（纯数值计算，无自动微分，用于预处理/验证）
    功能、算法、返回值与cal_MassProperties完全一致，仅将PyTorch张量替换为NumPy数组
    适用场景：无需梯度的离线计算（如数据集预处理时提前计算物体初始物理属性）
    Args:
        mesh: trimesh网格对象
    Returns:
        volume: 体积（np.float32）
        center_mass: 质心坐标（3维np.array）
        inertia_body_unit: 单位密度惯性张量（3x3 np.array）
    """
    triangles = np.asanyarray(mesh.triangles, dtype=np.float32)  # 网格三角形顶点（NumPy数组）
    crosses = np.asanyarray(mesh.triangles_cross, dtype=np.float32)  # 三角形叉积（NumPy数组）

    # 与PyTorch版本完全一致的积分子表达式计算
    f1 = triangles[:, 0, :] + triangles[:, 1, :] + triangles[:, 2, :]
    f2 = (
        triangles[:, 0, :] ** 2
        + triangles[:, 1, :] ** 2
        + triangles[:, 0, :] * triangles[:, 1, :]
        + triangles[:, 2, :] * f1
    )
    f3 = (
        (triangles[:, 0, :] ** 3)
        + (triangles[:, 0, :] ** 2) * (triangles[:, 1, :])
        + (triangles[:, 0, :]) * (triangles[:, 1, :] ** 2)
        + (triangles[:, 1, :] ** 3)
        + (triangles[:, 2, :] * f2)
    )
    g0 = f2 + (triangles[:, 0, :] + f1) * triangles[:, 0, :]
    g1 = f2 + (triangles[:, 1, :] + f1) * triangles[:, 1, :]
    g2 = f2 + (triangles[:, 2, :] + f1) * triangles[:, 2, :]

    # 积分结果计算（NumPy版本）
    integral = np.zeros((10, len(f1)), dtype=np.float32)
    integral[0] = crosses[:, 0] * f1[:, 0]
    integral[1:4] = (crosses * f2).T
    integral[4:7] = (crosses * f3).T
    for i in range(3):
        triangle_i = np.mod(i + 1, 3)  # NumPy的取模运算
        integral[i + 7] = crosses[:, i] * (
            (triangles[:, 0, triangle_i] * g0[:, i])
            + (triangles[:, 1, triangle_i] * g1[:, i])
            + (triangles[:, 2, triangle_i] * g2[:, i])
        )

    # 归一化系数与积分求和
    coefficients = 1.0 / np.array(
        [6, 24, 24, 24, 60, 60, 60, 120, 120, 120], dtype=np.float32
    )
    integrated = integral.sum(axis=1) * coefficients

    # 体积、质心、惯性张量计算（与PyTorch版本逻辑一致）
    volume = integrated[0]
    center_mass = integrated[1:4] / volume

    inertia_body_unit = np.zeros((3, 3), dtype=np.float32)
    inertia_body_unit[0, 0] = (integrated[5] + integrated[6] - (volume * (center_mass[[1, 2]] ** 2).sum()))
    inertia_body_unit[1, 1] = (integrated[4] + integrated[6] - (volume * (center_mass[[0, 2]] ** 2).sum()))
    inertia_body_unit[2, 2] = (integrated[4] + integrated[5] - (volume * (center_mass[[0, 1]] ** 2).sum()))
    inertia_body_unit[0, 1] = -(integrated[7] - (volume * np.prod(center_mass[[0, 1]])))
    inertia_body_unit[1, 2] = -(integrated[8] - (volume * np.prod(center_mass[[1, 2]])))
    inertia_body_unit[0, 2] = -(integrated[9] - (volume * np.prod(center_mass[[0, 2]])))
    inertia_body_unit[2, 0] = inertia_body_unit[0, 2]
    inertia_body_unit[2, 1] = inertia_body_unit[1, 2]
    inertia_body_unit[1, 0] = inertia_body_unit[0, 1]

    return volume, center_mass, inertia_body_unit


def cal_transform_matrix(scale_factors, rotation_quaternion, translation_vectors):
    """
    计算3D物体的齐次变换矩阵（4x4），融合缩放、旋转、平移三种变换
    变换顺序：缩放 → 旋转 → 平移（矩阵乘法右乘，实际应用时按此顺序叠加）
    核心作用：将物体从局部坐标系转换到世界坐标系（适配仿真中的物体姿态更新）
    Args:
        scale_factors: 缩放因子（3维np.array，[sx, sy, sz]，1.0表示无缩放）
        rotation_quaternion: 旋转四元数（4维np.array，[x, y, z, w]，表示物体旋转姿态）
        translation_vectors: 平移向量（3维np.array，[tx, ty, tz]，物体在世界坐标系的位置）
    Returns:
        transform_matrix: 齐次变换矩阵（4x4 np.array，可直接用于trimesh的apply_transform）
    """
    # 1. 缩放矩阵（4x4，仅对角线元素为缩放因子，其他为0，最后一行/列为齐次项）
    scaling_matrix = np.eye(4, dtype=np.float32)
    scaling_matrix[0, 0] = scale_factors[0]  # x轴缩放
    scaling_matrix[1, 1] = scale_factors[1]  # y轴缩放
    scaling_matrix[2, 2] = scale_factors[2]  # z轴缩放

    # 2. 旋转矩阵（从四元数转换为3x3旋转矩阵，再补全为4x4齐次矩阵）
    from utils.quaternion_utils import xyzw_quaternion_to_rotmat  # 导入四元数→旋转矩阵工具函数
    rotation_matrix = xyzw_quaternion_to_rotmat(rotation_quaternion)  # 3x3旋转矩阵
    rotation_matrix = np.concatenate((rotation_matrix, [[0, 0, 0]]), axis=0)  # 补全为4x3
    rotation_matrix = np.concatenate((rotation_matrix, [[0], [0], [0], [1]]), axis=1)  # 补全为4x4

    # 3. 平移矩阵（4x4，对角线为1，平移量存储在第4列前3行）
    translation_matrix = np.eye(4, dtype=np.float32)
    translation_matrix[:3, 3] = translation_vectors  # x/y/z方向平移量

    # 4. 组合变换矩阵：平移 × 旋转 × 缩放（矩阵乘法顺序与变换顺序相反，右乘为实际应用顺序）
    transform_matrix = np.dot(translation_matrix, np.dot(rotation_matrix, scaling_matrix))
    return transform_matrix


def get_mesh_world(mesh, position, rotation):
    """
    生成世界坐标系下的3D网格副本（基于物体的当前位置和旋转）
    核心作用：在仿真中实时更新物体的网格姿态，用于接触检测、渲染等后续操作
    Args:
        mesh: 物体的局部坐标系网格（trimesh对象）
        position: 物体在世界坐标系的位置（3维np.array，平移向量）
        rotation: 物体的旋转四元数（4维np.array，[x, y, z, w]）
    Returns:
        mesh_copy: 世界坐标系下的网格副本（trimesh对象，已应用平移和旋转）
    """
    import copy  # 导入深拷贝模块（避免修改原始网格）
    mesh_copy = copy.deepcopy(mesh)  # 深拷贝原始网格（防止副作用）
    # 计算变换矩阵：无缩放（[1,1,1]），仅旋转和平移
    transform_matrix = cal_transform_matrix([1, 1, 1], rotation, position)
    mesh_copy.apply_transform(transform_matrix)  # 应用变换矩阵到网格
    return mesh_copy


def create_sample_points(sample_num, mesh_local):
    """
    在3D物体的局部坐标系网格表面采样点（用于接触检测：机器人与物体的接触位置选择）
    步骤：1. 切片去除上下表面 → 2. 表面均匀采样 → 3. 拼接位置、齐次项、法向量
    Args:
        sample_num: 采样点数量（如1000，采样点越多接触检测越精准）
        mesh_local: 物体的局部坐标系网格（trimesh对象）
    Returns:
        sample_points: 采样点数据（Nx8 np.array，N=sample_num，每行为[x,y,z,1,nx,ny,nz]）
            - x/y/z：采样点在局部坐标系的位置
            - 1：齐次项（用于矩阵变换）
            - nx/ny/nz：采样点的法向量（指向物体外部）
    """
    # 切片平面：去除物体上下表面的点（避免采样到无效接触点，如地面接触面）
    # 第一个切片：z > -0.0001（去除z轴负方向的下表面）
    # 第二个切片：z < 0.0001（去除z轴正方向的上表面）
    sample_mesh = (mesh_local.slice_plane(plane_normal=[0, 0, 1], plane_origin=[0, 0, -0.0001])
                   ).slice_plane(plane_normal=[0, 0, -1], plane_origin=[0, 0, 0.0001])
    
    # 在切片后的网格表面均匀采样（seed=0保证采样可复现）
    points_position, faces_index = trimesh.sample.sample_surface(
        sample_mesh, count=sample_num, face_weight=None, seed=0
    )
    # 获取每个采样点的法向量（通过采样点所属的面的法向量获取）
    points_normal = sample_mesh.face_normals[faces_index]
    
    # 拼接采样点数据：位置（Nx3）→ 齐次项（Nx1，全1）→ 法向量（Nx3）→ 最终Nx8
    sample_points = np.concatenate((
        points_position, 
        np.ones((sample_num, 1), dtype=np.float32),  # 齐次项
        points_normal
    ), axis=1)  
    return sample_points


def get_contact_info(action, sample_points, obj_position, obj_rotation):
    """
    计算世界坐标系下的接触点信息（机器人与物体的交互核心数据）
    核心作用：根据采样点索引（action），获取接触点的世界坐标和法向量，用于仿真中的力计算
    Args:
        action: 采样点索引（int，选择哪个采样点作为接触点）
        sample_points: 局部坐标系下的采样点数据（Nx8 np.array，来自create_sample_points）
        obj_position: 物体在世界坐标系的位置（3维np.array）
        obj_rotation: 物体的旋转四元数（4维np.array）
    Returns:
        contact_point_position: 接触点在世界坐标系的位置（3维np.array）
        contact_point_normal: 接触点在世界坐标系的法向量（3维np.array）
    """
    contact_point = sample_points[action]  # 选择指定索引的采样点（局部坐标系）
    # 计算物体的变换矩阵（无缩放，仅旋转和平移）
    transform_matrix = cal_transform_matrix([1, 1, 1], obj_rotation, obj_position)
    # 接触点位置转换到世界坐标系：变换矩阵 × 齐次坐标（前4个元素[x,y,z,1]）
    contact_point_position = np.dot(transform_matrix, contact_point[:4])[:3]
    # 接触点法向量转换到世界坐标系：仅用旋转矩阵（法向量不随平移变化）
    contact_point_normal = np.dot(transform_matrix[:3, :3], contact_point[4:])
    return contact_point_position, contact_point_normal


def comp_projection_integrals(verts, faces, A, B):
    """
    计算面投影积分（惯性张量计算的辅助函数）
    核心作用：将3D体积积分投影到2D平面，简化惯性张量的积分计算（参考OpenFOAM实现）
    Args:
        verts: 物体顶点坐标（Vx3 Tensor，V为顶点数）
        faces: 物体面索引（Fx3 Tensor，F为面数，每个元素是顶点索引）
        A/B: 投影轴索引（0=x,1=y,2=z，如A=0、B=1表示投影到xy平面）
    Returns:
        10个投影积分结果（P1, Pa, Paa, Paaa, Pb, Pbb, Pbbb, Pab, Paab, Pabb）
    """
    # 提取每个面的三个顶点在A/B轴的坐标（按面的循环顺序）
    a0 = verts[faces][torch.arange(faces.shape[0]), :, A]
    b0 = verts[faces][torch.arange(faces.shape[0]), :, B]
    a1 = verts[faces[:, [1, 2, 0]]][torch.arange(faces.shape[0]), :, A]  # 顶点顺序循环移位
    b1 = verts[faces[:, [1, 2, 0]]][torch.arange(faces.shape[0]), :, B]

    # 计算顶点坐标的差值和幂次项
    da = a1 - a0
    db = b1 - b0
    a0_2 = a0 * a0
    a0_3 = a0_2 * a0
    a0_4 = a0_3 * a0
    b0_2 = b0 * b0
    b0_3 = b0_2 * b0
    b0_4 = b0_3 * b0
    a1_2 = a1 * a1
    a1_3 = a1_2 * a1
    b1_2 = b1 * b1
    b1_3 = b1_2 * b1

    # 积分子表达式（用于简化投影积分计算）
    C1 = a1 + a0
    Ca = a1 * C1 + a0_2
    Caa = a1 * Ca + a0_3
    Caaa = a1 * Caa + a0_4
    Cb = b1 * (b1 + b0) + b0_2
    Cbb = b1 * Cb + b0_3
    Cbbb = b1 * Cbb + b0_4
    Cab = 3 * a1_2 + 2 * a1 * a0 + a0_2
    Kab = a1_2 + 2 * a1 * a0 + 3 * a0_2
    Caab = a0 * Cab + 4 * a1_3
    Kaab = a1 * Kab + 4 * a0_3
    Cabb = 4 * b1_3 + 3 * b1_2 * b0 + 2 * b1 * b0_2 + b0_3
    Kabb = b1_3 + 2 * b1_2 * b0 + 3 * b1 * b0_2 + 4 * b0_3

    # 计算10个投影积分项
    P1 = (db * C1).sum(dim=1) / 2.0
    Pa = (db * Ca).sum(dim=1) / 6.0
    Paa = (db * Caa).sum(dim=1) / 12.0
    Paaa = (db * Caaa).sum(dim=1) / 20.0
    Pb = (da * Cb).sum(dim=1) / -6.0
    Pbb = (da * Cbb).sum(dim=1) / -12.0
    Pbbb = (da * Cbbb).sum(dim=1) / -20.0
    Pab = (db * (b1 * Cab + b0 * Kab)).sum(dim=1) / 24.0
    Paab = (db * (b1 * Caab + b0 * Kaab)).sum(dim=1) / 60.0
    Pabb = (da * (a1 * Cabb + a0 * Kabb)).sum(dim=1) / -60.0

    return P1, Pa, Paa, Paaa, Pb, Pbb, Pbbb, Pab, Paab, Pabb


def comp_face_integrals(verts, faces, normals, w, A, B, C):
    """
    计算单个面的体积积分（惯性张量计算的核心辅助函数）
    核心作用：基于投影积分结果，结合面法向量，计算3D体积积分的各项分量
    Args:
        verts: 顶点坐标（Vx3 Tensor）
        faces: 面索引（Fx3 Tensor）
        normals: 面法向量（Fx3 Tensor，每个面的法向量）
        w: 平面方程参数（Fx1 Tensor，面满足n·x + w = 0）
        A/B/C: 坐标轴索引（0=x,1=y,2=z，C为法向量主方向）
    Returns:
        12个面积分结果（Fa, Fb, Fc, Faa, Fbb, Fcc, Faaa, Fbbb, Fccc, Faab, Fbbc, Fcca）
    """
    # 第一步：计算投影积分
    P1, Pa, Paa, Paaa, Pb, Pbb, Pbbb, Pab, Paab, Pabb = comp_projection_integrals(verts, faces, A, B)

    # 法向量主方向的系数（C为法向量绝对值最大的轴，即主方向）
    k1 = 1 / normals[torch.arange(normals.shape[0]), C]
    k2 = k1 * k1
    k3 = k2 * k1
    k4 = k3 * k1

    # 提取法向量在A/B轴的分量
    nA = normals[torch.arange(normals.shape[0]), A]
    nB = normals[torch.arange(normals.shape[0]), B]

    # 计算12个面积分项（对应体积、质心、惯性张量的不同分量）
    Fa = k1 * Pa
    Fb = k1 * Pb
    Fc = -k2 * (nA * Pa + nB * Pb + w * P1)
    Faa = k1 * Paa
    Fbb = k1 * Pbb
    Fcc = k3 * (nA * nA * Paa + 2 * nA * nB * Pab + nB * nB * Pbb + w * (2 * (nA * Pa + nB * Pb) + w * P1))
    Faaa = k1 * Paaa
    Fbbb = k1 * Pbbb
    Fccc = -k4 * (nA ** 3 * Paaa + 3 * nA * nA * nB * Paab + 3 * nA * nB * nB * Pabb + nB * nB * nB * Pbbb
                  + 3 * w * (nA * nA * Paa + 2 * nA * nB * Pab + nB * nB * Pbb)
                  + w * w * (3 * (nA * Pa + nB * Pb) + w * P1))
    Faab = k1 * Paab
    Fbbc = -k2 * (nA * Pabb + nB * Pbbb + w * Pbb)
    Fcca = k3 * (nA * nA * Paaa + 2 * nA * nB * Paab + nB * nB * Pabb + w * (2 * (nA * Paa + nB * Pab) + w * Pa))

    return Fa, Fb, Fc, Faa, Fbb, Fcc, Faaa, Fbbb, Fccc, Faab, Fbbc, Fcca


def comp_volume_integrals(verts, faces, normals, w):
    """
    计算整个物体的体积积分（惯性张量计算的汇总函数）
    核心作用：遍历所有面，求和面积分结果，得到物体整体的体积、质心、惯性张量相关积分
    Args:
        verts: 顶点坐标（Vx3 Tensor）
        faces: 面索引（Fx3 Tensor）
        normals: 面法向量（Fx3 Tensor）
        w: 平面方程参数（Fx1 Tensor）
    Returns:
        T0: 体积积分结果（float）
        T1: 质心相关积分结果（3维Tensor）
        T2: 惯性张量对角项相关积分结果（3维Tensor）
        TP: 惯性张量交叉项相关积分结果（3维Tensor）
    """
    # 选择每个面法向量的主方向（绝对值最大的轴，用于投影优化）
    C = torch.argmax(normals.abs(), dim=1)
    A = (C + 1) % 3  # 循环轴索引（C→A→B→C）
    B = (A + 1) % 3

    # 计算所有面的面积分
    Fa, Fb, Fc, Faa, Fbb, Fcc, Faaa, Fbbb, Fccc, Faab, Fbbc, Fcca = comp_face_integrals(
        verts, faces, normals, w, A, B, C
    )

    # 汇总体积积分（T0）
    T0 = verts.new_zeros(faces.shape[0])
    T0[A == 0] = normals[A == 0, 0] * Fa[A == 0]
    T0[B == 0] = normals[B == 0, 0] * Fb[B == 0]
    T0[C == 0] = normals[C == 0, 0] * Fc[C == 0]
    T0 = T0.sum()

    # 提取法向量在A/B/C轴的分量
    normA = normals[torch.arange(normals.shape[0]), A]
    normB = normals[torch.arange(normals.shape[0]), B]
    normC = normals[torch.arange(normals.shape[0]), C]

    # 汇总质心相关积分（T1）
    T1 = verts.new_zeros(faces.shape[0], 3)
    T1[torch.arange(faces.shape[0]), A] = normA * Faa
    T1[torch.arange(faces.shape[0]), B] = normB * Fbb
    T1[torch.arange(faces.shape[0]), C] = normC * Fcc
    T1 = T1.sum(dim=0) / 2

    # 汇总惯性张量对角项相关积分（T2）
    T2 = verts.new_zeros(faces.shape[0], 3)
    T2[torch.arange(faces.shape[0]), A] = normA * Faaa
    T2[torch.arange(faces.shape[0]), B] = normB * Fbbb
    T2[torch.arange(faces.shape[0]), C] = normC * Fccc
    T2 = T2.sum(dim=0) / 3

    # 汇总惯性张量交叉项相关积分（TP）
    TP = verts.new_zeros(faces.shape[0], 3)
    TP[torch.arange(faces.shape[0]), A] = normA * Faab
    TP[torch.arange(faces.shape[0]), B] = normB * Fbbc
    TP[torch.arange(faces.shape[0]), C] = normC * Fcca
    TP = TP.sum(dim=0) / 2

    return T0, T1, T2, TP


def get_ang_inertia(verts, faces, mass):
    """
    计算物体的惯性张量（PyTorch版本，支持自动微分）
    算法参考：OpenFOAM的体积积分实现（适用于任意多面体网格）
    核心作用：直接通过顶点和面信息，结合物体质量，计算实际惯性张量（无需额外密度参数）
    Args:
        verts: 顶点坐标（Vx3 Tensor）
        faces: 面索引（Fx3 Tensor）
        mass: 物体质量（float/Tensor，已知量）
    Returns:
        J: 惯性张量（3x3对称Tensor，物体局部坐标系下）
    """
    # 计算每个面的法向量（通过面的两个边向量叉乘得到）
    normals = torch.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]], dim=1)
    normals = normals / normals.norm(dim=1).unsqueeze(1)  # 法向量归一化（单位向量）

    # 计算每个面的平面方程参数w（满足n·x + w = 0，x为面内任意顶点）
    w = (-normals * verts[faces[:, 0]]).sum(dim=1)

    # 计算体积积分的四个核心结果
    T0, T1, T2, TP = comp_volume_integrals(verts, faces, normals, w)

    # 计算密度（密度 = 质量 / 体积）
    density = mass / T0

    # 构建惯性张量（3x3对称矩阵）
    J = torch.diag(density * (T2[[1, 2, 0]] + T2[[2, 0, 1]]))  # 对角项（Ixx, Iyy, Izz）
    J[0, 1] = J[1, 0] = -density * TP[0]  # 交叉项Ixy=Iyx
    J[1, 2] = J[2, 1] = -density * TP[1]  # 交叉项Iyz=Izy
    J[2, 0] = J[0, 2] = -density * TP[2]  # 交叉项Ixz=Izx

    return J


def inertia_diagonalize(inertia):
    """
    【PyTorch版本】将惯性张量对角化（找到惯性主轴坐标系）
    核心作用：惯性张量在主轴坐标系下为对角矩阵，简化旋转动力学计算（如论文中的3D刚体运动仿真）
    原理：通过特征值分解，特征值为绕主轴的转动惯量，特征向量为主轴方向
    Args:
        inertia: 原始惯性张量（3x3对称Tensor）
    Returns:
        I_prime: 对角化后的惯性张量（3x3对角Tensor，对角线为特征值）
    """
    # 特征值分解：eigenvalues=特征值（转动惯量），eigenvectors=特征向量（主轴方向）
    eigenvalues, eigenvectors = torch.linalg.eigh(inertia)

    # 提取特征值（I1, I2, I3为绕三个主轴的转动惯量）
    I1, I2, I3 = eigenvalues
    # 提取特征向量（v1, v2, v3为三个主轴方向的单位向量）
    v1, v2, v3 = eigenvectors[:, 0], eigenvectors[:, 1], eigenvectors[:, 2]

    # 构建主轴方向的旋转矩阵R（列向量为特征向量）
    R = torch.stack([v1, v2, v3], dim=1)

    # 对角化：I' = R^T × I × R（正交矩阵的逆=转置，简化计算）
    I_prime = R.T @ inertia @ R
    return I_prime


def inertia_diagonalize_np(inertia):
    """
    【NumPy版本】将惯性张量对角化（纯数值计算，无自动微分）
    功能、原理与PyTorch版本一致，仅适用于离线预处理/验证
    Args:
        inertia: 原始惯性张量（3x3 np.array）
    Returns:
        I_prime: 对角化后的惯性张量（3x3对角np.array）
    """
    eigenvalues, eigenvectors = np.linalg.eigh(inertia)  # NumPy特征值分解
    I1, I2, I3 = eigenvalues
    v1, v2, v3 = eigenvectors[:, 0], eigenvectors[:, 1], eigenvectors[:, 2]
    R = np.stack([v1, v2, v3], axis=1)  # 主轴旋转矩阵
    I_prime = R.T @ inertia @ R  # 对角化
    return I_prime