""" 这段代码是 PIN-WM3D 刚体旋转姿态处理的工具库，所有函数围绕 “四元数” 展开 —— 四元数是 3D 旋转的最优表示（无万向锁、存储高效、计算简洁），直接支撑论文的可微分仿真核心需求，关键作用可概括为 4 类：
1. 旋转核心运算（仿真姿态更新）
quaternion_multiply/multiply：组合多个旋转（如物体同时绕 X 轴和 Z 轴旋转）；
rotate_point_by_quaternion：计算物体上的点随姿态旋转后的坐标（如接触点、网格顶点）；
create_q_x/create_q_y/create_q_z：生成特定轴的旋转四元数（用于物体姿态初始化、仿真中的主动旋转）。
2. 旋转表示互转（多模块适配）
rotmat_to_quaternion/quaternion_to_rotmat：旋转矩阵↔四元数互转（适配相机外参计算、坐标系变换等需要矩阵的场景）；
xyzw_quaternion_to_rotmat：适配 xyzw 格式的四元数（兼容不同数据集 / 工具库的格式）。
3. 格式与标准化（参数一致性）
wxyz2xyzw/xyzw2wxyz及 NumPy 版本：解决不同模块的四元数格式冲突（有的用 wxyz，有的用 xyzw）；
quaternion_standardize：消除旋转歧义（q 和 - q 表示同一旋转），保证优化过程中参数的唯一性；
normalize：确保四元数为单位长度（旋转四元数的必要条件，避免缩放误差）。
4. 多精度支持（灵活适配场景）
PyTorch 版本：支持自动微分和 GPU 加速，融入模型训练流程（如物理参数优化中的姿态梯度计算）；
NumPy 版本：纯数值计算，用于离线预处理（如数据集初始化）或结果验证，无梯度计算需求。 """
# 导入核心依赖库
import torch  # 深度学习框架（支持张量计算、自动微分，适配模型训练）
import numpy as np  # 数值计算库（纯数值旋转计算、数组处理）
import math  # 数学工具库（角度转换、三角函数）

def quaternion_conjugate(q):
    """
    计算四元数的共轭（Conjugate）—— 单位四元数的逆等于其共轭
    四元数格式：q = [w, x, y, z]（w=实部，x/y/z=虚部，对应旋转的 scalar + vector 表示）
    共轭公式：q* = [w, -x, -y, -z]
    核心用途：四元数旋转点时需用到（旋转公式：q * p_q * q_conjugate）
    Args:
        q: 输入四元数（Tensor，形状[4]，wxyz格式）
    Returns:
        共轭四元数（Tensor，形状[4]，与输入同设备）
    """
    w, x, y, z = q  # 拆分实部和虚部
    # 按共轭公式生成结果，保持与输入相同的计算设备（避免CPU/GPU张量冲突）
    return torch.tensor(([w, -x, -y, -z]), device=q.device)

def rotate_point_by_quaternion(q, p):
    """
    用四元数旋转3D点——3D旋转的高效实现（无万向锁问题）
    核心原理：将3D点p转换为纯四元数p_q（实部=0），通过四元数乘法实现旋转：p_rot = q * p_q * q^-1
    适用场景：仿真中物体上的点随物体姿态（四元数q）旋转、接触点位置更新等
    Args:
        q: 旋转四元数（Tensor，形状[4]，wxyz格式，需为单位四元数）
        p: 3D点坐标（Tensor，形状[3]或[B,3]，B为批量数）
    Returns:
        rotated_p: 旋转后的3D点坐标（Tensor，形状与p一致）
    """
    # 步骤1：将3D点p转换为纯四元数p_q（实部=0，格式[0, x, y, z]）
    # torch.zeros_like(p[..., :1])：生成与p第一维同形状的0张量（实部）
    p_q = torch.cat((torch.zeros_like((p[..., :1]), device=p.device), p), dim=-1)

    # 步骤2：计算q的共轭（单位四元数的逆=共轭，简化计算）
    q_conjugate = torch.cat((q[..., 0:1], -q[..., 1:4]), dim=-1)

    # 步骤3：执行四元数乘法：temp = q * p_q，rotated_p_q = temp * q_conjugate
    temp = quaternion_multiply(q, p_q)
    rotated_p_q = quaternion_multiply(temp, q_conjugate)

    # 步骤4：提取旋转后纯四元数的虚部（即旋转后的3D点坐标）
    rotated_p = rotated_p_q[..., 1:]
    return rotated_p

def quaternion_multiply(q1, q2):
    """
    四元数乘法（Hamilton乘积，不满足交换律）—— 旋转组合的核心运算
    乘法公式（wxyz格式）：
    q1*q2 = [w1w2 - x1x2 - y1y2 - z1z2,
             w1x2 + x1w2 + y1z2 - z1y2,
             w1y2 - x1z2 + y1w2 + z1x2,
             w1z2 + x1y2 - y1x2 + z1w2]
    核心用途：组合两个旋转（q1后接q2旋转）、旋转点时的中间计算
    Args:
        q1: 第一个四元数（Tensor，形状[4]或[B,4]，wxyz格式）
        q2: 第二个四元数（Tensor，形状与q1一致，wxyz格式）
    Returns:
        q: 乘法结果四元数（Tensor，形状与q1一致）
    """
    # 拆分q1和q2的实部（w）和虚部（x/y/z）
    w1, x1, y1, z1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
    w2, x2, y2, z2 = q2[..., 0:1], q2[..., 1:2], q2[..., 2:3], q2[..., 3:4]

    # 按Hamilton乘积公式计算各分量
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2  # 实部
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2  # x虚部
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2  # y虚部
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2  # z虚部

    # 拼接各分量，返回结果四元数
    q = torch.cat((w, x, y, z), dim=-1)
    return q

def normalize(quaternion):
    """
    四元数归一化——旋转四元数必须为单位四元数（模长=1），否则会引入缩放
    归一化公式：q_norm = q / (||q|| + 1e-5)，加1e-5避免除零
    核心用途：旋转计算前的预处理（如优化过程中四元数参数偏离单位长度时）
    Args:
        quaternion: 输入四元数（Tensor，形状[4]，wxyz格式）
    Returns:
        归一化后的单位四元数（Tensor，形状[4]）
    """
    # 计算四元数的L2模长（||q||），加1e-5防止模长为0导致除零错误
    norm = quaternion.norm(p=2, dim=0) + 1e-5
    # 按模长归一化
    return quaternion / norm

def rotmat_to_quaternion(R):
    """
    3x3旋转矩阵转换为四元数（wxyz格式）—— 两种旋转表示的互转工具
    核心优势：四元数无万向锁问题，存储效率更高（4个参数vs9个参数）
    实现逻辑：分4种情况计算（基于旋转矩阵的迹和对角元大小），保证数值稳定性
    Args:
        R: 3x3旋转矩阵（Tensor，形状[3,3]或[B,3,3]，B为批量数）
    Returns:
        q: 对应的四元数（Tensor，形状[4]或[B,4]，wxyz格式，单位四元数）
    """
    # 验证输入旋转矩阵的形状（必须是3x3）
    assert R.shape[-2:] == (3, 3)
    
    # 计算旋转矩阵的迹（对角线元素和，用于判断计算分支）
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    
    # 初始化输出四元数（形状与输入旋转矩阵一致，仅最后两维变为4）
    q = torch.zeros(R.shape[:-2] + (4,), device=R.device, dtype=R.dtype)
    
    # 情况1：迹>0（数值最稳定，优先选择）
    trace_positive = trace > 0
    if trace_positive.any():
        s = torch.sqrt(trace[trace_positive] + 1.0) * 2  # s = 4*w（实部相关系数）
        q[trace_positive, 0] = 0.25 * s  # w = s/4
        q[trace_positive, 1] = (R[trace_positive, 2, 1] - R[trace_positive, 1, 2]) / s  # x
        q[trace_positive, 2] = (R[trace_positive, 0, 2] - R[trace_positive, 2, 0]) / s  # y
        q[trace_positive, 3] = (R[trace_positive, 1, 0] - R[trace_positive, 0, 1]) / s  # z
    
    # 情况2：迹≤0，且R[0,0]是对角元中最大的
    cond1 = (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2]) & (~trace_positive)
    if cond1.any():
        s = torch.sqrt(1.0 + R[cond1, 0, 0] - R[cond1, 1, 1] - R[cond1, 2, 2]) * 2  # s = 4*x
        q[cond1, 0] = (R[cond1, 2, 1] - R[cond1, 1, 2]) / s  # w
        q[cond1, 1] = 0.25 * s  # x
        q[cond1, 2] = (R[cond1, 0, 1] + R[cond1, 1, 0]) / s  # y
        q[cond1, 3] = (R[cond1, 0, 2] + R[cond1, 2, 0]) / s  # z
    
    # 情况3：迹≤0，且R[1,1]是对角元中最大的
    cond2 = (R[..., 1, 1] > R[..., 2, 2]) & (~cond1) & (~trace_positive)
    if cond2.any():
        s = torch.sqrt(1.0 + R[cond2, 1, 1] - R[cond2, 0, 0] - R[cond2, 2, 2]) * 2  # s = 4*y
        q[cond2, 0] = (R[cond2, 0, 2] - R[cond2, 2, 0]) / s  # w
        q[cond2, 1] = (R[cond2, 0, 1] + R[cond2, 1, 0]) / s  # x
        q[cond2, 2] = 0.25 * s  # y
        q[cond2, 3] = (R[cond2, 1, 2] + R[cond2, 2, 1]) / s  # z
    
    # 情况4：迹≤0，且R[2,2]是对角元中最大的
    cond3 = ~cond1 & ~cond2 & ~trace_positive
    if cond3.any():
        s = torch.sqrt(1.0 + R[cond3, 2, 2] - R[cond3, 0, 0] - R[cond3, 1, 1]) * 2  # s = 4*z
        q[cond3, 0] = (R[cond3, 1, 0] - R[cond3, 0, 1]) / s  # w
        q[cond3, 1] = (R[cond3, 0, 2] + R[cond3, 2, 0]) / s  # x
        q[cond3, 2] = (R[cond3, 1, 2] + R[cond3, 2, 1]) / s  # y
        q[cond3, 3] = 0.25 * s  # z

    return q    

def quaternion_to_rotmat(quaternion):
    """
    四元数转换为3x3旋转矩阵（wxyz格式）—— 适配需要矩阵运算的场景（如坐标系变换）
    转换公式（基于四元数的旋转矩阵展开，保证正交性）
    核心用途：相机外参旋转矩阵计算、物体姿态的矩阵表示（如仿真中的坐标变换）
    Args:
        quaternion: 输入四元数（Tensor，形状[4]，wxyz格式，单位四元数）
    Returns:
        rotmat: 对应的3x3旋转矩阵（Tensor，形状[3,3]，正交矩阵）
    """
    # 拆分四元数的实部和虚部
    r = quaternion[0]  # w（实部）
    i = quaternion[1]  # x（虚部x）
    j = quaternion[2]  # y（虚部y）
    k = quaternion[3]  # z（虚部z）
    
    # 初始化旋转矩阵（3x3，与输入同设备、同数据类型）
    rotmat = torch.zeros(3, 3, dtype=quaternion.dtype, device=quaternion.device)
    
    # 预计算重复项（减少计算量，提高效率）
    twoisq = 2 * i * i
    twojsq = 2 * j * j
    twoksq = 2 * k * k
    twoij = 2 * i * j
    twoik = 2 * i * k
    twojk = 2 * j * k
    twori = 2 * r * i
    tworj = 2 * r * j
    twork = 2 * r * k
    
    # 按转换公式填充旋转矩阵（保证正交性）
    rotmat[0, 0] = 1 - twojsq - twoksq
    rotmat[0, 1] = twoij - twork
    rotmat[0, 2] = twoik + tworj
    rotmat[1, 0] = twoij + twork
    rotmat[1, 1] = 1 - twoisq - twoksq
    rotmat[1, 2] = twojk - twori
    rotmat[2, 0] = twoik - tworj
    rotmat[2, 1] = twojk + twori
    rotmat[2, 2] = 1 - twoisq - twojsq
    
    return rotmat

def xyzw_quaternion_to_rotmat(quaternion):
    """
    四元数转换为3x3旋转矩阵（xyzw格式，与上面函数的格式区别）
    适配场景：部分数据集/工具库使用xyzw格式（x/y/z/w），需单独处理
    Args:
        quaternion: 输入四元数（np.array，形状[4]，xyzw格式，单位四元数）
    Returns:
        rotmat: 对应的3x3旋转矩阵（np.array，形状[3,3]，float32类型）
    """
    # 拆分xyzw格式的四元数（x/y/z为虚部，w为实部）
    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]
    
    # 按xyzw格式的转换公式生成旋转矩阵
    rotmat = np.array([
        [1 - 2 * (y ** 2) - 2 * (z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x ** 2) - 2 * (z ** 2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2) - 2 * (y ** 2)]
    ], dtype=np.float32)
    
    return rotmat

def multiply(q1, q2):
    """
    四元数乘法的另一种实现（向量-标量分离法，wxyz格式）
    原理：将四元数拆分为标量r和向量v，乘法公式：
    q1*q2 = [r1r2 - v1·v2, r1v2 + r2v1 + v1×v2]
    功能与quaternion_multiply一致，备用实现（适配不同计算场景）
    Args:
        q1: 第一个四元数（Tensor，形状[4]，wxyz格式）
        q2: 第二个四元数（Tensor，形状[4]，wxyz格式）
    Returns:
        乘法结果四元数（Tensor，形状[4]，wxyz格式）
    """
    r1 = q1[0]  # q1的标量部分（w）
    v1 = q1[1:]  # q1的向量部分（x/y/z）
    r2 = q2[0]  # q2的标量部分（w）
    v2 = q2[1:]  # q2的向量部分（x/y/z）
    
    # 按向量-标量公式计算，拼接标量和向量部分
    return torch.cat(
        (
            # 标量部分：r1r2 - v1·v2（向量点积）
            r1 * r2 - torch.matmul(v1.view(1, 3), v2.view(3, 1)).view(-1),
            # 向量部分：r1v2 + r2v1 + v1×v2（向量叉积）
            r1 * v2 + r2 * v1 + torch.cross(v1, v2),
        ),
        dim=0,
    )

def wxyz2xyzw(q):
    """
    四元数格式转换：wxyz → xyzw（适配不同模块的格式要求）
    转换规则：[w, x, y, z] → [x, y, z, w]
    Args:
        q: 输入四元数（Tensor，形状[4]，wxyz格式）
    Returns:
        转换后的四元数（Tensor，形状[4]，xyzw格式，与输入同设备）
    """
    return torch.tensor(([q[1], q[2], q[3], q[0]]), device=q.device)

def xyzw2wxyz(q):
    """
    四元数格式转换：xyzw → wxyz（适配不同模块的格式要求）
    转换规则：[x, y, z, w] → [w, x, y, z]
    Args:
        q: 输入四元数（Tensor，形状[4]，xyzw格式）
    Returns:
        转换后的四元数（Tensor，形状[4]，wxyz格式，与输入同设备）
    """
    return torch.tensor(([q[3], q[0], q[1], q[2]]), device=q.device)

def wxyz2xyzw_np(q):
    """
    NumPy版本：wxyz → xyzw格式转换（纯数值计算场景）
    Args:
        q: 输入四元数（np.array，形状[4]，wxyz格式）
    Returns:
        转换后的四元数（np.array，形状[4]，xyzw格式，float32类型）
    """
    return np.array(([q[1], q[2], q[3], q[0]]), dtype=np.float32)

def xyzw2wxyz_np(q):
    """
    NumPy版本：xyzw → wxyz格式转换（纯数值计算场景）
    Args:
        q: 输入四元数（np.array，形状[4]，xyzw格式）
    Returns:
        转换后的四元数（np.array，形状[4]，wxyz格式，float32类型）
    """
    return np.array(([q[3], q[0], q[1], q[2]]), dtype=np.float32)

def create_q_x(theta):
    """
    生成绕X轴旋转的单位四元数（wxyz格式）
    旋转公式：q = [cos(θ/2), sin(θ/2), 0, 0]，θ为旋转角度（度）
    核心用途：仿真中物体绕X轴的旋转（如翻转）、姿态初始化
    Args:
        theta: 旋转角度（float，单位：度）
    Returns:
        q_x: 绕X轴旋转的单位四元数（Tensor，形状[4]，float32类型）
    """
    theta_rad = math.radians(theta)  # 角度转弧度（三角函数需弧度输入）
    half_theta = theta_rad / 2  # 旋转角度的一半（四元数旋转的核心）
    cos_half_theta = np.cos(half_theta)  # 余弦值（实部）
    sin_half_theta = np.sin(half_theta)  # 正弦值（X轴虚部）
    # 生成四元数（wxyz格式：[cos(θ/2), sin(θ/2), 0, 0]）
    q_x = torch.tensor([cos_half_theta, sin_half_theta, 0, 0], dtype=torch.float32)
    return q_x

def create_q_z(theta):
    """
    生成绕Z轴旋转的单位四元数（wxyz格式）
    旋转公式：q = [cos(θ/2), 0, 0, sin(θ/2)]，θ为旋转角度（度）
    核心用途：仿真中物体绕Z轴的旋转（如旋转）、姿态初始化
    Args:
        theta: 旋转角度（float，单位：度）
    Returns:
        q_z: 绕Z轴旋转的单位四元数（Tensor，形状[4]，float32类型）
    """
    theta_rad = math.radians(theta)  # 角度转弧度
    half_theta = theta_rad / 2
    cos_half_theta = np.cos(half_theta)  # 实部
    sin_half_theta = np.sin(half_theta)  # Z轴虚部
    # 生成四元数（wxyz格式：[cos(θ/2), 0, 0, sin(θ/2)]）
    q_z = torch.tensor([cos_half_theta, 0, 0, sin_half_theta], dtype=torch.float32)
    return q_z

def create_q_y(theta):
    """
    生成绕Y轴旋转的单位四元数（wxyz格式）
    旋转公式：q = [cos(θ/2), 0, sin(θ/2), 0]，θ为旋转角度（度）
    核心用途：仿真中物体绕Y轴的旋转（如俯仰）、姿态初始化
    Args:
        theta: 旋转角度（float，单位：度）
    Returns:
        q_y: 绕Y轴旋转的单位四元数（Tensor，形状[4]，float32类型）
    """
    theta_rad = math.radians(theta)  # 角度转弧度
    half_theta = theta_rad / 2
    cos_half_theta = np.cos(half_theta)  # 实部
    sin_half_theta = np.sin(half_theta)  # Y轴虚部
    # 生成四元数（wxyz格式：[cos(θ/2), 0, sin(θ/2), 0]）
    q_y = torch.tensor([cos_half_theta, 0, sin_half_theta, 0], dtype=torch.float32)
    return q_y

def quaternion_standardize(q):
    """
    四元数标准化——消除旋转表示的歧义（q和-q表示同一个旋转）
    标准化规则：若四元数最后一个虚部（z分量）<0，则取反（保证z≥0）
    核心用途：优化过程中统一四元数表示，避免同一旋转对应两个不同参数
    Args:
        q: 输入四元数（Tensor，形状[4]，wxyz格式）
    Returns:
        标准化后的四元数（Tensor，形状[4]，z分量非负）
    """
    # torch.where：条件满足则取-q，否则取q
    return torch.where(q[-1] < 0, -q, q)

def quaternion_standardize_np(q):
    """
    NumPy版本：四元数标准化（纯数值计算场景）
    Args:
        q: 输入四元数（np.array，形状[4]，wxyz格式）
    Returns:
        标准化后的四元数（np.array，形状[4]，z分量非负）
    """
    return np.where(q[-1] < 0, -q, q)

def quaternion_multiply_np(q1, q2):
    """
    NumPy版本：四元数乘法（xyzw格式，Hamilton乘积）
    功能与quaternion_multiply一致，适配纯数值计算场景（无自动微分需求）
    Args:
        q1: 第一个四元数（np.array，形状[4]，xyzw格式）
        q2: 第二个四元数（np.array，形状[4]，xyzw格式）
    Returns:
        乘法结果四元数（np.array，形状[4]，xyzw格式）
    """
    # 拆分xyzw格式的四元数（x/y/z为虚部，w为实部）
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    # 按Hamilton乘积公式计算（xyzw格式）
    x3 = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y3 = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z3 = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w3 = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return np.array([x3, y3, z3, w3])