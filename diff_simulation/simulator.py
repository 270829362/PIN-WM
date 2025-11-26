""" 该代码实现了一个可微分物理仿真环境，核心功能包括：
刚体与约束管理：支持添加刚体（Body）和关节约束（Constraint），为每个刚体和约束分配唯一 ID。
碰撞检测：通过PyBullet 引擎和自定义平面检测两种方式，识别刚体间的接触（法向量、接触点、穿透深度等信息）。
物理求解：基于LCP（线性互补问题）求解器处理碰撞约束，计算刚体的速度更新。
位姿积分：根据求解得到的速度，积分更新刚体的位置和姿态（四元数表示）。
可视化与参数管理：支持场景可视化（刚体、接触点），以及物理材料参数（恢复系数、摩擦系数、质量等）的加载、设置与优化。
该仿真环境可用于可微分物理参数识别（如识别物体的摩擦、质量等物理属性）、强化学习环境（基于物理的智能体交互）等场景，通过结合自动微分（PyTorch）实现物理过程的端到端优化。 """
import torch
from typing import List
import trimesh
import numpy as np

import pybullet as p
from diff_simulation.solver.lcp_solver import LCP_Solver
from diff_simulation.body.base import Body
from diff_simulation.constraints.base import Constraint
from diff_simulation.physical_material import Physical_Materials
from utils.quaternion_utils import multiply, normalize


class Simulator(object):
    """可微分物理仿真环境类：用于管理刚体、约束、碰撞检测与物理求解，支持可微分物理参数优化"""

    def __init__(self, dtime, device, vis):
        """
        初始化物理仿真环境
        
        Args:
            dtime (float): 仿真时间步长
            device (torch.device): 计算设备（CPU/GPU）
            vis (bool): 是否启用可视化（PyBullet GUI或DIRECT模式）
        """
        self.device = device
        self.dtime = dtime  # 仿真时间步长
        self.cur_time = 0.0  # 当前仿真时间
        self.solver = LCP_Solver(self)  # LCP约束求解器实例
        self.velocity_dim = 6  # 刚体速度维度（3维线速度+3维角速度）
        self.fric_dirs = 8  # 摩擦方向数量（用于多方向摩擦建模）
        
        self.bodies: List[Body] = []  # 刚体列表
        self.joints: List[Constraint] = []  # 约束（关节）列表
        self.body_id_counter = 0  # 刚体ID计数器
        self.joint_id_counter = 0  # 关节ID计数器

        # 初始化PyBullet物理引擎连接
        self.physicsClient = p.connect(p.GUI if vis else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # 关闭PyBullet GUI界面
        # 设置调试相机视角
        p.resetDebugVisualizerCamera(cameraYaw=45.0, cameraPitch=-30, cameraDistance=0.5, cameraTargetPosition=[0.0, 0.0, 0.0])

    def add_body(self, body: Body):
        """
        向仿真环境添加刚体
        
        Args:
            body (Body): 刚体实例
            
        Returns:
            int: 刚体ID
        """
        body_id = self.body_id_counter
        self.body_id_counter += 1
        body.set_id(body_id)
        self.bodies.append(body)
        return body_id

    def add_joint(self, joint):
        """
        向仿真环境添加关节（约束）
        
        Args:
            joint (Constraint): 约束实例
            
        Returns:
            int: 关节ID
        """
        joint_id = self.joint_id_counter
        self.joint_id_counter += 1
        joint.set_id(joint_id)
        self.joints.append(joint)
        return joint_id
    
    def get_body(self, body_id):
        """根据ID获取刚体实例"""
        for body in self.bodies:
            if body.id == body_id:
                return body
        return None
    
    def get_body_from_pybullet_id(self, pybullet_id):
        """根据PyBullet的几何ID获取刚体实例"""
        for body in self.bodies:
            if body.pybullet_geom_id == pybullet_id:
                return body
        return None

    def get_body_list_index(self, body_id):
        """根据ID获取刚体在列表中的索引"""
        for index, body in enumerate(self.bodies):
            if body.id == body_id:
                return index
        return None
    
    def get_joint(self, joint_id):
        """根据ID获取关节实例"""
        for joint in self.joints:
            if joint.id == joint_id:
                return joint
        return None        
    
    def get_joint_list_index(self, joint_id):
        """根据ID获取关节在列表中的索引"""
        for index, joint in enumerate(self.joints):
            if joint.id == joint_id:
                return index
        return None
    

    def collision_detection_pybullet(self):
        """
        基于PyBullet的碰撞检测
        
        Returns:
            tuple: (接触信息列表, 可视化接触点列表)
            - 接触信息：包含法向量、接触点相对位置、穿透深度、刚体ID对
            - 可视化接触点：用于Trimesh可视化的接触点与法向量
        """
        for body in self.bodies:
            body.update_collision_geom()  # 更新刚体的碰撞几何
        p.performCollisionDetection()  # 执行PyBullet碰撞检测
        pybullet_contact_infos = p.getContactPoints()  # 获取所有接触点
        contact_infos = []
        vis_contact_points = []
        for pybullet_contact_info in pybullet_contact_infos:
            # 存储可视化用的接触点和法向量
            vis_contact_points.append(pybullet_contact_info[5] + pybullet_contact_info[7])
            # 映射PyBullet的几何ID到自定义刚体实例
            body_a = self.get_body_from_pybullet_id(pybullet_contact_info[1])
            body_b = self.get_body_from_pybullet_id(pybullet_contact_info[2])
            # 转换接触信息为Tensor
            normal = torch.tensor(pybullet_contact_info[7], device=self.device)
            p_a = torch.tensor(pybullet_contact_info[5], device=self.device) - body_a.world_position
            p_b = torch.tensor(pybullet_contact_info[6], device=self.device) - body_b.world_position
            penetration = torch.tensor(pybullet_contact_info[8], device=self.device)
            # 存储接触信息（法向量、相对位置、穿透深度、刚体ID对）
            contact_infos.append(((normal, p_a[:3], p_b[:3], penetration),
                                  body_a.id, body_b.id))
        return contact_infos, vis_contact_points
    
    def collision_detection_plane(self):
        """
        平面碰撞检测（针对地面平面刚体）
        
        Returns:
            list: 平面与刚体的接触信息列表
        """
        contact_infos = []
        plane_world_position = self.bodies[0].world_position  # 假设第一个刚体是地面平面
        plane_id = 0 
        normal = torch.tensor([0.0, 0.0, -1.0], device=self.device)  # 平面法向量（垂直向下）
        penetration = torch.tensor([0.0], device=self.device, dtype=torch.float32)
        # 检测其他刚体与平面的碰撞
        for body in self.bodies[1:]:
            vertices_world = body.get_vertices_world()  # 获取刚体的世界坐标系顶点
            # 筛选z坐标小于平面z坐标的顶点（潜在接触点）
            index = vertices_world[...,-1] < (plane_world_position[2] + 5.0 + 0.0001).detach().cpu().numpy()
            contact_points = torch.tensor(vertices_world[index], dtype=torch.float32, device=self.device)

            if contact_points.any():
                p_a_batch = contact_points - plane_world_position
                p_b_batch = contact_points - body.world_position    
                # 批量添加接触信息
                contact_infos += [((normal, p_a, p_b, penetration), plane_id, body.id) for p_a, p_b in zip(p_a_batch, p_b_batch)]
        return contact_infos


    def get_sum_constraint_dim(self):
        """计算所有关节的约束维度总和"""
        sum_constraint_dim = 0
        for joint in self.joints:
            sum_constraint_dim += joint.constraint_dim
        return sum_constraint_dim
    
    def apply_external_forces(self, cur_time):
        """收集所有刚体的外力（用于物理求解）"""
        return torch.cat([b.apply_external_forces(cur_time) for b in self.bodies])
    
    def get_vel_vec(self):
        """收集所有刚体的速度向量（线速度+角速度）"""
        return torch.cat([b.get_vel_vec() for b in self.bodies])

    def integrate_transform(self, new_velocitys):
        """
        积分更新刚体的位姿（位置和姿态）
        
        Args:
            new_velocitys (torch.Tensor): 求解得到的新速度向量
        """
        for index, body in enumerate(self.bodies):
            # 提取当前刚体的速度向量
            new_velocity = new_velocitys[index * self.velocity_dim : index * self.velocity_dim + self.velocity_dim]
            body.angular_velocity = new_velocity[:3]  # 更新角速度
            body.linear_velocity = new_velocity[3:]   # 更新线速度
            # 积分更新位置
            body.world_position = body.world_position + body.linear_velocity * self.dtime
            # 积分更新姿态（四元数）
            body.world_rotation = body.world_rotation + multiply(
                torch.cat([torch.tensor(([0]), device=self.device), 0.5*self.dtime*body.angular_velocity], dim=0),
                body.world_rotation
            )
            body.world_rotation = normalize(body.world_rotation)  # 归一化四元数

    def vis_scene(self, vis_contact_points):
        """
        可视化仿真场景（刚体、接触点与法向量）
        
        Args:
            vis_contact_points (list): 用于可视化的接触点和法向量
        """
        scene = trimesh.Scene()
        vis_contact_points = np.array(vis_contact_points)
        contact_point_positions = vis_contact_points[..., :3]
        contact_point_normals = vis_contact_points[..., 3:]
        # 可视化接触点
        contact_points = trimesh.PointCloud(contact_point_positions)
        scene.add_geometry(contact_points)
        # 可视化接触法向量
        lines = []
        for contact_point, contact_point_normal in zip(contact_point_positions, contact_point_normals):
            start_point = contact_point
            end_point = start_point + contact_point_normal * 0.003 
            lines.append([start_point, end_point])
        normal_lines = trimesh.load_path(lines)
        scene.add_geometry(normal_lines)
        # 可视化所有刚体
        for body in self.bodies[1:]:
            vis_geom = body.get_visual_geom_world()
            scene.add_geometry(vis_geom)
        scene.show()  # 显示场景

    def get_body_pose(self, body_id):
        """获取刚体的位姿（位置和姿态）"""
        body = self.get_body(body_id)
        return body.get_pose()

    def get_body_pose_clone(self, body_id):
        """获取刚体位姿的深拷贝"""
        body = self.get_body(body_id)
        return body.get_pose_clone()
    
    def get_body_pose_cpu(self, body_id):
        """获取刚体位姿并转移到CPU"""
        body = self.get_body(body_id)
        return body.get_pose_cpu()
    
    def get_body_pose_np(self, body_id):
        """获取刚体位姿的NumPy数组"""
        body = self.get_body(body_id)
        return body.get_pose_np()
    
    def get_body_vel_np(self, body_id):
        """获取刚体速度的NumPy数组"""
        body = self.get_body(body_id)
        return body.get_body_vel_np()
        

    def change_body_pose(self, body_id, world_position=None, world_rotation=None):
        """修改刚体的位姿"""
        body = self.get_body(body_id)
        body.change_pose(world_position, world_rotation)

    def change_body_vel(self, body_id, linear_velocity=None, angular_velocity=None):
        """修改刚体的速度"""
        body = self.get_body(body_id)
        body.change_vel(linear_velocity, angular_velocity)
        
    def create_mesh_body(self, mesh, physical_materials, urdf, requires_grad, world_position=None, world_rotation=None):
        """
        创建网格刚体
        
        Args:
            mesh: 网格模型
            physical_materials (Physical_Materials): 物理材料参数
            urdf: URDF文件路径
            requires_grad (bool): 是否启用自动微分
            world_position: 初始世界位置
            world_rotation: 初始世界姿态（四元数）
            
        Returns:
            int: 刚体ID
        """
        from diff_simulation.body.body_mesh import Body_Mesh
        body = Body_Mesh(mesh, physical_materials, urdf, requires_grad, self.device, world_position, world_rotation)
        body_id = self.add_body(body)
        return body_id


    def create_joint(self, body_id, joint_type):
        """
        创建关节（约束）
        
        Args:
            body_id (int): 刚体ID
            joint_type: 关节类型（如固定约束、旋转约束等）
            
        Returns:
            int: 关节ID
        """
        from diff_simulation.constraints.base import Joint_Type
        if joint_type == Joint_Type.FIX_CONSTRAINT:
            from diff_simulation.constraints.fix_constraint import Fix_Constraint
            constraint = Fix_Constraint(body_id)
        elif joint_type == Joint_Type.NO_ROT_CONSTRATNT:
            from diff_simulation.constraints.rot_constraint import Rot_Constraint
            constraint = Rot_Constraint(body_id)
        elif joint_type == Joint_Type.NO_TRANS_Z_CONSTRATNT:
            from diff_simulation.constraints.trans_constraint import TransZ_Constraint
            constraint = TransZ_Constraint(body_id)
        joint_id = self.add_joint(constraint)
        return joint_id
            
    def get_body_physical_materials(self, body_id):
        """获取刚体的物理材料参数"""
        body = self.get_body(body_id)
        return body.get_physical_materials()
    
    def get_all_physical_materials(self):
        """获取所有刚体的物理材料参数"""
        all_physical_materials = []
        for body in self.bodies:
            all_physical_materials.append(body.get_physical_materials())
        return all_physical_materials

    def load_all_physical_materials(self, json_path):
        """从JSON文件加载物理材料参数"""
        import json
        with open(json_path, 'r') as json_file:
            all_physical_materials_original_json = json.load(json_file)["activate"]
        for physical_materials_original in all_physical_materials_original_json:
            body = self.get_body(physical_materials_original["body_id"])
            physical_materials = body.physical_materials
            for key, value in physical_materials_original.items():
                if key in physical_materials.all:
                    physical_materials.set_material(key, value)
            body.set_physical_materials(physical_materials)

    def set_physical_materials(self, body_id, physical_materials):
        """设置刚体的物理材料参数"""
        body = self.get_body(body_id)
        return body.set_physical_materials(physical_materials)

    def step(self):
        """执行单步仿真：碰撞检测→约束求解→位姿积分→时间更新"""
        # 执行碰撞检测
        contact_infos1, vis_contact_points = self.collision_detection_pybullet()
        contact_infos2 = self.collision_detection_plane()
        contact_infos = contact_infos1 + contact_infos2

        # 约束求解（LCP）
        x = self.solver.solve_constraint(contact_infos)
        # 提取刚体速度
        new_velocitys = x[:self.velocity_dim * len(self.bodies)].squeeze(0)
        # 积分更新刚体位姿
        self.integrate_transform(new_velocitys)
        # 更新当前仿真时间
        self.cur_time = self.cur_time + self.dtime

    def reset(self):
        """重置仿真环境（时间、刚体状态）"""
        self.cur_time = 0.0
        for body in self.bodies:
            body.reset()

    def close(self):
        """关闭PyBullet连接"""
        p.disconnect(self.physicsClient)