# 该代码是论文《PIN-WM: Learning Physics-INformed World Models for Non-Prehensile Manipulation》中3D 刚体动力学模拟的核心底层代码，定义了抽象基类Body（刚体），封装了刚体的物理属性、运动状态及核心动力学计算逻辑，结合 PyBullet 实现几何加载与可视化，为论文中的物理感知世界模型（PIN-WM）提供刚体运动模拟支持，支撑非抓握操作（推、翻等）的物理参数识别与政策学习。
from abc import ABCMeta, abstractmethod
import torch
import copy

# 导入自定义的力模型、物理材料类、四元数工具
from diff_simulation.force.constant_force import Constant_Force
from diff_simulation.physical_material import Physical_Materials
from utils.quaternion_utils import multiply, rotate_point_by_quaternion, quaternion_to_rotmat
from utils.quaternion_utils import wxyz2xyzw

# 导入PyBullet用于物理模拟的几何加载与可视化
import pybullet as p

# 定义刚体抽象基类（Abstract Base Class），需子类实现抽象方法
class Body(metaclass=ABCMeta):
    def __init__(self, collision_geom, visual_geom, physical_materials: Physical_Materials,
                 urdf, world_position, world_rotation, device):
        """
        初始化刚体实例
        Args:
            collision_geom: 碰撞几何（用于ODE碰撞检测）
            visual_geom: 视觉几何（用于渲染可视化）
            physical_materials: 物理材料实例，存储质量、摩擦系数等参数（对应论文θ参数）
            urdf: URDF文件路径（用于PyBullet加载几何模型，None则不加载）
            world_position: 世界坐标系下初始位置 (torch.Tensor, shape=[3])
            world_rotation: 世界坐标系下初始旋转（四元数，wxyz格式，torch.Tensor, shape=[4]）
            device: 计算设备（CPU/GPU）
        """
        self.id = None  # 刚体唯一标识ID（后续由外部设置）
        self.collision_geom = collision_geom  # 碰撞几何（ODE碰撞检测用）
        
        # 加载PyBullet几何模型（用于可视化和物理模拟联动）
        if urdf is not None:
            self.urdf = urdf
            # 转换四元数格式（wxyz→xyzw）并加载URDF，获取PyBullet几何ID
            self.pybullet_geom_id = p.loadURDF(
                urdf,
                world_position.detach().cpu().numpy(),
                wxyz2xyzw(world_rotation).detach().cpu().numpy()
            )
        else:
            self.pybullet_geom_id = None  # 无URDF时不加载PyBullet几何
        
        self.visual_geom = visual_geom  # 视觉几何（渲染用）
        self.device = device  # 计算设备
        self.physical_materials = physical_materials  # 物理材料实例
        
        # 从物理材料中提取核心物理参数（对应论文中的θ^M、θ^μ、θ^k）
        self.mass = self.physical_materials.get_material("mass")  # 质量（θ^M）
        self.friction_coefficient = self.physical_materials.get_material("friction_coefficient")  # 摩擦系数（θ^μ）
        self.restitution = self.physical_materials.get_material("restitution")  # 恢复系数（θ^k）
        self.inertia_body = self.physical_materials.get_material("inertia") * self.mass  # 本体坐标系转动惯量（= 单位惯量×质量）
        
        # 刚体运动状态（世界坐标系）
        self.world_position = world_position  # 位置 (x,y,z)
        self.world_rotation = world_rotation  # 旋转（四元数wxyz）
        self.linear_momentum = torch.zeros((3), device=self.device)  # 线动量
        self.angular_momentum = torch.zeros((3), device=self.device)  # 角动量
        self.linear_velocity = torch.zeros((3), device=self.device)  # 线速度
        self.angular_velocity = torch.zeros((3), device=self.device)  # 角速度
        
        self.forces = []  # 作用在刚体上的外力列表
        self.apply_positions = []  # 外力作用点列表（世界坐标系，None则作用于质心）
        self.add_gravity()  # 初始化时添加重力
        
        # 存储初始状态（用于重置）
        self.init_world_position = self.world_position.clone()
        self.init_world_rotation = self.world_rotation.clone()

    def get_physical_materials(self):
        """获取当前刚体的物理材料实例"""
        return self.physical_materials
    
    def set_physical_materials(self, physical_materials):
        """更新物理材料，并同步更新相关物理参数（支持论文中的参数优化）"""
        self.physical_materials = physical_materials
        self.mass = self.physical_materials.get_material("mass")
        self.inertia_body = self.physical_materials.get_material("inertia") * self.mass  # 重新计算转动惯量
        self.friction_coefficient = self.physical_materials.get_material("friction_coefficient")
        self.restitution = self.physical_materials.get_material("restitution")

    def set_id(self, id):
        """设置刚体唯一标识ID，并同步到物理材料"""
        self.id = id
        self.physical_materials.body_id = self.id
        
    def reset(self):
        """重置刚体状态到初始值（用于模拟episode重启）"""
        # 重置物理参数
        self.mass = self.physical_materials.get_material("mass")
        self.friction_coefficient = self.physical_materials.get_material("friction_coefficient")
        self.restitution = self.physical_materials.get_material("restitution")        
        self.inertia_body = self.physical_materials.get_material("inertia") * self.mass 

        # 重置运动状态
        self.world_position = self.init_world_position.clone()
        self.world_rotation = self.init_world_rotation.clone()
        self.linear_momentum = torch.zeros((3), device=self.device)
        self.angular_momentum = torch.zeros((3), device=self.device)
        self.linear_velocity = torch.zeros((3), device=self.device)
        self.angular_velocity = torch.zeros((3), device=self.device)
        
        # 重置外力（仅保留重力）
        self.forces = []
        self.apply_positions = []
        self.add_gravity()

    def add_gravity(self):
        """添加重力（恒定力，方向沿z轴负方向）"""
        gravity = Constant_Force(
            magnitude=9.8 * self.mass,  # 重力大小=mg（g=9.8m/s²）
            direction=torch.tensor(([0.0, 0.0, -1.0]), device=self.device),  # 方向：向下（z负方向）
        )
        self.add_external_force(gravity)  # 将重力加入外力列表

    def get_pose_clone(self):
        """获取当前姿态的深拷贝（避免引用冲突）"""
        return self.world_position.clone(), self.world_rotation.clone()

    def get_pose_cpu(self):
        """获取当前姿态（位置+旋转），转移到CPU并detach（用于后续处理）"""
        return self.world_position.detach().cpu(), self.world_rotation.detach().cpu()

    def get_pose_np(self):
        """获取当前姿态，转换为numpy数组（用于PyBullet更新或可视化）"""
        return self.world_position.detach().cpu().numpy(), self.world_rotation.detach().cpu().numpy()
    
    def get_pose(self):
        """获取当前姿态（原始tensor，带梯度）"""
        return self.world_position, self.world_rotation
    
    def get_body_vel_np(self):
        """获取当前速度（线速度+角速度），转换为numpy数组"""
        return self.linear_velocity.detach().cpu().numpy(), self.angular_velocity.detach().cpu().numpy()
    
    def change_pose(self, world_position=None, world_rotation=None):
        """修改刚体姿态（位置/旋转），并更新碰撞几何"""
        if world_position is not None:
            self.world_position = world_position
        if world_rotation is not None:
            self.world_rotation = world_rotation
        self.update_collision_geom()  # 子类需实现：同步碰撞几何到新姿态

    def change_vel(self, linear_velocity=None, angular_velocity=None):
        """直接修改刚体速度（线速度/角速度）"""
        if linear_velocity is not None:
            self.linear_velocity = linear_velocity
        if angular_velocity is not None:
            self.angular_velocity = angular_velocity

    def compute_inertia_world_inv(self):
        """计算世界坐标系下的转动惯量逆矩阵（用于角动量→角速度转换）"""
        # 1. 将四元数转换为旋转矩阵（本体→世界）
        rotmat = quaternion_to_rotmat(self.world_rotation)
        # 2. 转动惯量逆的坐标变换：J_world_inv = R * J_body_inv * R^T
        return torch.matmul(torch.matmul(rotmat, self.inertia_body_inv), rotmat.transpose(0, 1))
    
    def compute_inertia_world(self):
        """计算世界坐标系下的转动惯量矩阵（用于动力学计算）"""
        # 1. 将四元数转换为旋转矩阵（本体→世界）
        rotmat = quaternion_to_rotmat(self.world_rotation)
        # 2. 转动惯量的坐标变换：J_world = R * J_body * R^T
        return torch.matmul(torch.matmul(rotmat, self.inertia_body), rotmat.transpose(0, 1))
    
    @staticmethod
    def compute_angular_velocity_from_angular_momentum(inertia_world_inv, angular_momentum):
        """从角动量计算角速度（ω = J_world_inv · L）"""
        return torch.matmul(inertia_world_inv, angular_momentum.view(-1, 1)).squeeze(-1)
    
    @staticmethod
    def compute_linear_velocity_from_linear_momentum(linear_momentum, mass):
        """从线动量计算线速度（v = p/m）"""
        return linear_momentum / mass
    
    @staticmethod
    def compute_center_of_mass(vertices, masses):
        """计算质心（用于非均质刚体，vertices为顶点坐标，masses为顶点质量）"""
        return (masses.view(-1, 1) * vertices).sum(0) / masses.sum()

    def compute_state_derivatives(self, cur_time):
        """计算状态导数（用于数值积分，如论文中的半隐式欧拉法）"""
        dposition = self.linear_velocity  # 位置导数 = 线速度
        # 旋转导数：四元数微分公式（ṙ = 0.5 * ω_quat × r）
        angular_velocity_quat = torch.zeros((4), device=self.device)
        angular_velocity_quat[1:] = self.angular_velocity  # ω_quat = [0, ω_x, ω_y, ω_z]
        drotation = 0.5 * multiply(angular_velocity_quat, self.world_rotation)  # 四元数乘法
        # 动量导数 = 外力/外力矩（由apply_external_forces计算）
        dlinear_momentum, dangular_momentum = self.apply_external_forces(cur_time)
        return dposition, drotation, dlinear_momentum, dangular_momentum
    
    def add_external_force(self, force, apply_pos=None):
        """添加外部力到刚体"""
        self.forces.append(force)  # 力实例（需实现apply方法返回力向量）
        self.apply_positions.append(apply_pos)  # 作用点（None→质心）

    def clear_force(self):
        """清空所有外部力（不含重力，重力需通过reset重新添加）"""
        self.forces = []
        self.apply_positions = []

    def apply_external_forces(self, cur_time):
        """计算所有外部力的合力与合力矩（用于动量更新）"""
        total_force = torch.zeros((3), device=self.device)  # 合力（线动量导数）
        total_torque = torch.zeros((3), device=self.device)  # 合力矩（角动量导数）
        for force, apply_pos in zip(self.forces, self.apply_positions):
            force_vector = force.apply(cur_time)  # 获取当前时刻的力向量
            total_force += force_vector
            # 若指定作用点，计算力矩（τ = r × F，r为作用点到质心的向量）
            if apply_pos is not None:
                torque_vector = torch.cross(apply_pos - self.world_position, force_vector)
                total_torque += torque_vector
        # 返回格式：[合力矩, 合力]（对应论文中角动量、线动量的更新顺序）
        return torch.cat([total_torque, total_force])
    
    def apply_impulse(self, contact_pos, j):
        """应用冲量（用于碰撞响应，对应论文中的LCP求解结果）"""
        # 冲量j对动量的影响：Δp = j，ΔL = r × j（r为接触点到质心的向量）
        self.linear_momentum = self.linear_momentum + j
        self.angular_momentum = self.angular_momentum + torch.cross(contact_pos - self.world_position, j)
        # 从动量更新速度
        self.linear_velocity = self.linear_momentum / self.mass
        self.angular_velocity = torch.matmul(self.compute_inertia_world_inv(), self.angular_momentum)    
    
    def get_M_world(self):
        """获取世界坐标系下的质量矩阵（6×6，对应[角动量; 线动量]的系数矩阵）"""
        M = torch.zeros((6, 6), device=self.device)
        M[:3, :3] = self.compute_inertia_world()  # 上3×3：转动惯量矩阵
        M[3:, 3:] = torch.eye((3), device=self.device) * self.mass  # 下3×3：质量对角矩阵
        return M
    
    def get_vel_vec(self):
        """获取速度向量（6维，格式：[角速度x, 角速度y, 角速度z, 线速度x, 线速度y, 线速度z]）"""
        v = torch.cat((self.angular_velocity, self.linear_velocity))
        return v
    
    @abstractmethod
    def get_collision_geom_world(self):
        """抽象方法：获取世界坐标系下的碰撞几何（需子类实现具体几何类型）"""
        pass
    
    @abstractmethod
    def get_collision_geom_local(self):
        """抽象方法：获取本体坐标系下的碰撞几何（需子类实现）"""
        pass
    
    @abstractmethod
    def get_visual_geom_world(self):
        """抽象方法：获取世界坐标系下的视觉几何（需子类实现）"""
        pass
    
    @abstractmethod
    def get_visual_geom_local(self):
        """抽象方法：获取本体坐标系下的视觉几何（需子类实现）"""
        pass

    @abstractmethod
    def update_collision_geom(self):
        """抽象方法：更新碰撞几何到当前姿态（需子类实现）"""
        pass