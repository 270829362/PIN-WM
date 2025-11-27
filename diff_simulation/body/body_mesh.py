# Body_Mesh类是论文《PIN-WM: Learning Physics-INformed World Models for Non-Prehensile Manipulation》中网格类型刚体的具体实现类，继承自抽象基类Body，专门处理基于三角网格（trimesh）的 3D 刚体。其核心功能是将网格几何与刚体物理属性、运动状态绑定，实现网格在世界坐标系下的姿态变换、视觉几何提取及 PyBullet 仿真姿态同步，为 PIN-WM 的3D 刚体动力学模拟、视觉渲染对齐（如 2D Gaussian Splatting）及 Sim2Real 仿真提供底层网格支持。
from .base import Body  # 继承抽象刚体基类Body
import torch
import trimesh  # 用于处理三角网格数据（3D几何体表示）
import numpy as np
import copy

from diff_simulation.physical_material import Physical_Materials  # 物理材料类（存储质量、摩擦等参数）

import pybullet as p  # PyBullet物理引擎，用于网格姿态同步与仿真
class Body_Mesh(Body):
    """
    网格类型刚体的具体实现类，处理基于trimesh的3D刚体，支持物理属性计算、姿态变换与PyBullet同步
    对应论文中3D刚体的几何表示，为动力学模拟和视觉渲染提供网格支持
    """

    def __init__(self, mesh: trimesh.base.Trimesh, physical_materials: Physical_Materials, urdf,
                 requires_grad, device, world_position=None, world_rotation=None):
        """
        初始化网格刚体
        Args:
            mesh: trimesh三角网格对象（刚体的几何形状）
            physical_materials: 物理材料实例（存储质量、摩擦系数等参数，对应论文θ）
            urdf: URDF文件路径（用于PyBullet加载网格，None则不加载）
            requires_grad: 是否开启姿态参数（位置/旋转）的自动微分（适配PIN-WM端到端优化）
            device: 计算设备（CPU/GPU）
            world_position: 初始世界坐标位置（None则默认[0,0,0]）
            world_rotation: 初始世界坐标旋转（四元数wxyz，None则默认[1,0,0,0]，即无旋转）
        """
        # 导入网格工具函数，计算网格的质量属性（体积、单位转动惯量）——对应论文中刚体惯性参数θ^M的初始化
        from utils.mesh_utils import cal_MassProperties, get_ang_inertia, inertia_diagonalize
        # 计算网格的体积、质心（未使用）、本体坐标系下单位质量的转动惯量（单位惯量×质量=实际转动惯量）
        volume, _, inertia_body_unit = cal_MassProperties(mesh, device)

        # 初始化世界坐标位置为可训练参数（支持自动微分，用于PIN-WM的可微优化）
        if world_position is None:
            # 默认位置[0,0,0]，设为nn.Parameter以支持梯度传播
            world_position = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0], device=device), requires_grad)
        else:
            # 若传入位置，直接封装为可训练参数
            world_position = torch.nn.Parameter(world_position, requires_grad)

        # 初始化世界坐标旋转（四元数wxyz）为可训练参数（支持自动微分）
        if world_rotation is None:
            # 默认无旋转（四元数[1,0,0,0]），设为nn.Parameter
            world_rotation = torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device), requires_grad)
        else:
            # 若传入旋转，直接封装为可训练参数
            world_rotation = torch.nn.Parameter(world_rotation, requires_grad)
            
        # 保存本地坐标系下的原始网格（视觉几何的基准）
        trimesh_mesh_local = mesh
        
        # 调用父类Body的构造函数：碰撞几何暂设为None（可后续扩展），视觉几何为本地网格，传入物理参数、URDF、姿态等
        super().__init__(None, trimesh_mesh_local, physical_materials, urdf,
                         world_position, world_rotation, device)

    def sample_face(self, mesh):
        """
        在网格的三角形面上采样点（用于碰撞检测、表面特征提取或2D Gaussian Splatting初始化）
        Args:
            mesh: 输入网格（此处代码注释放置为kaolin Mesh，推测支持trimesh/kaolin两种网格格式）
        Returns:
            sample_points: 采样点集合，shape=(面数, 每个面采样数, 3)（3为x/y/z坐标）
        """
        # 获取网格每个面的3个顶点坐标（假设为kaolin格式，cuda加速）
        face_vertices = mesh.face_vertices.cuda()  # shape=(face_num, 3, 3)
        sample_num = 100  # 每个三角形面采样100个点
        # 调用工具函数在三角形面上随机采样（代码中未给出实现，推测为均匀采样）
        sample_points = self.random_point_on_triangle(face_vertices, sample_num)  # shape=(face_num, sample_num, 3)
        return sample_points

    def get_visual_geom_world(self):
        """
        获取世界坐标系下的视觉网格（用于渲染、可视化或与真实观测对齐，对应论文1-47段2D Gaussian Splatting的姿态同步）
        Returns:
            mesh_copy: 变换到世界坐标系后的网格拷贝（避免修改原始本地网格）
        """
        # 深拷贝本地视觉网格（防止原始网格被污染）
        mesh_copy = copy.deepcopy(self.visual_geom)
        # 导入坐标变换与四元数工具函数
        from utils.mesh_utils import cal_transform_matrix
        from utils.quaternion_utils import wxyz2xyzw
        # 获取当前刚体的姿态（位置+旋转），转移到CPU（避免GPU内存占用）
        obj_position, obj_rotation = self.get_pose_cpu()
        # 四元数格式转换：wxyz→xyzw（PyBullet/变换矩阵计算的常用格式）
        rotation = wxyz2xyzw(obj_rotation).numpy()
        position = obj_position.numpy()
        # 计算姿态变换矩阵：缩放（此处为1,1,1即无缩放）→旋转→平移
        transform_matrix = cal_transform_matrix(
            scale=[1, 1, 1],  # 无缩放
            rotation=rotation,  # 世界坐标系旋转
            translation=position  # 世界坐标系位置
        )
        # 将本地网格应用变换矩阵，转换到世界坐标系
        mesh_copy.apply_transform(transform_matrix)
        return mesh_copy

    def get_vertices_world(self):
        """
        单独获取世界坐标系下的网格顶点（用于碰撞检测、质心更新或顶点级动力学计算）
        Returns:
            vertices_world: 世界坐标系下的顶点集合，shape=(顶点数, 3)
        """
        # 获取本地坐标系下的原始顶点
        vertices_loacl = self.visual_geom.vertices
        # 导入坐标变换与四元数工具函数
        from utils.mesh_utils import cal_transform_matrix
        from utils.quaternion_utils import wxyz2xyzw
        # 获取当前刚体的姿态（位置+旋转），转移到CPU
        obj_position, obj_rotation = self.get_pose_cpu()
        # 四元数格式转换：wxyz→xyzw
        rotation = wxyz2xyzw(obj_rotation).numpy()
        position = obj_position.numpy()
        # 计算姿态变换矩阵（缩放→旋转→平移）
        transform_matrix = cal_transform_matrix(
            scale=[1, 1, 1],
            rotation=rotation,
            translation=position
        )
        # 顶点转换为齐次坐标（添加w=1，方便矩阵乘法）：shape=(顶点数, 4)
        vertices_h = np.hstack((vertices_loacl, np.ones((vertices_loacl.shape[0], 1))))
        # 应用变换矩阵：(4×4矩阵) × (4×顶点数) → 转置后取前3列（去除w=1），得到世界坐标系顶点
        vertices_world = (transform_matrix @ vertices_h.T).T[:, :3]
        return vertices_world
    
    def get_visual_geom_local(self):
        """
        获取本地坐标系下的视觉网格（用于初始姿态保存、本地几何分析或参数初始化）
        Returns:
            本地坐标系网格的深拷贝（避免修改原始网格）
        """
        return copy.deepcopy(self.visual_geom)

    def update_collision_geom(self):
        """
        更新PyBullet中的碰撞几何姿态（确保代码中的动力学计算与PyBullet仿真姿态一致，对应论文1-82段集成Bullet引擎）
        父类Body中为抽象方法，此处针对网格刚体实现PyBullet同步逻辑
        """
        from utils.quaternion_utils import wxyz2xyzw
        # 若PyBullet几何ID存在（即初始化时传入了URDF）
        if self.pybullet_geom_id is not None:
            # 调用PyBullet接口重置刚体基态姿态：位置+旋转（四元数wxyz→xyzw）
            p.resetBasePositionAndOrientation(
                bodyUniqueId=self.pybullet_geom_id,  # PyBullet中的刚体ID
                posObj=self.world_position.detach().cpu().numpy(),  # 世界坐标位置（detach避免梯度传播）
                ornObj=wxyz2xyzw(self.world_rotation).detach().cpu().numpy()  # 世界坐标旋转（格式转换）
            )