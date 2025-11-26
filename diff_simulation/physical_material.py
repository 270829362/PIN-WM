import torch
import random
import numpy as np

class Physical_Materials:
    """
    物理材料参数管理类：用于封装、优化和管理刚体的物理属性（恢复系数、摩擦系数、质量、转动惯量）
    支持自动微分（通过torch.nn.Parameter）、数值范围约束（如sigmoid映射、clip裁剪）和参数序列化
    """
    def __init__(self, requires_grad, device):
        """
        初始化物理材料参数，将所有参数封装为可微分的torch.nn.Parameter
        
        Args:
            requires_grad (bool): 是否对参数启用自动微分（用于优化时设为True）
            device (torch.device): 参数存储的设备（CPU/GPU）
        """
        self.requires_grad = requires_grad  # 标记是否需要梯度计算
        self.body_id = None                 # 刚体ID（可选，用于多刚体场景的标识）
        self.device = device                 # 参数所在的计算设备
        self.all = {}                        # 存储所有物理参数的字典
        
        # 初始化恢复系数（restitution）：通过logit变换初始化，后续用sigmoid映射到[0,1]
        self.all["restitution"] = torch.nn.Parameter(
            torch.tensor(random.uniform(-5, -3), device=self.device),
            requires_grad=self.requires_grad
        )
        # 初始化摩擦系数（friction_coefficient）：同理logit变换+后续sigmoid映射到[0,1]
        self.all["friction_coefficient"] = torch.nn.Parameter(
            torch.tensor(random.uniform(-5, 5), device=self.device),
            requires_grad=self.requires_grad
        )
        # 初始化质量（mass）：直接初始化在[0.001, 1]区间，后续用clip限制最小值
        self.all["mass"] = torch.nn.Parameter(
            torch.tensor(random.uniform(0.001, 1), device=self.device),
            requires_grad=self.requires_grad
        )
        # 初始化转动惯量（inertia）：三维对角矩阵，初始值为0.1，后续clip限制最小值
        self.all["inertia"] = torch.nn.Parameter(
            0.1 * torch.ones((3), device=self.device),
            requires_grad=self.requires_grad
        )

    def set_material(self, material_name, value):
        """
        设置物理参数的原始值（反向应用数值变换，将物理有界值转换为优化用的无界值）
        
        Args:
            material_name (str): 参数名称（"restitution"、"friction_coefficient"、"mass"、"inertia"）
            value: 物理意义上的参数值（需符合对应参数的范围约束）
        """
        if material_name == "restitution":
            # 恢复系数：logit变换（将[0,1]映射到实数域）
            self.all[material_name].data = torch.tensor(
                np.log(value / (1 - value)), 
                device=self.device
            )
        elif material_name == "friction_coefficient":
            # 摩擦系数：同理logit变换
            self.all[material_name].data = torch.tensor(
                np.log(value / (1 - value)), 
                device=self.device
            )
        elif material_name == "mass":
            # 质量：直接赋值（后续用clip限制范围）
            self.all[material_name].data = torch.tensor(value, device=self.device)
        elif material_name == "inertia":
            # 转动惯量：取对角元素（假设输入是3x3对角矩阵）
            self.all[material_name].data = torch.tensor(
                [value[0][0], value[1][1], value[2][2]], 
                device=self.device
            )

    def get_material(self, material_name):
        """
        获取物理意义上的参数值（应用数值变换，将优化用的无界值映射回物理有界范围）
        
        Args:
            material_name (str): 参数名称
        
        Returns:
            torch.Tensor: 物理意义上的参数值（符合对应范围约束）
        """
        if material_name == "restitution":
            # 恢复系数：sigmoid映射到[0,1]
            return torch.sigmoid(self.all[material_name])
        elif material_name == "friction_coefficient":
            # 摩擦系数：sigmoid映射到[0,1]
            return torch.sigmoid(self.all[material_name]) 
        elif material_name == "mass":
            # 质量：clip到最小值0.1（避免质量为0或负数）
            return torch.clip(self.all[material_name], 0.1)
        elif material_name == 'inertia':
            # 转动惯量：构造3x3对角矩阵，每个对角元素clip到1e-3（避免为0或负数）
            inertia = torch.zeros((3, 3), device=self.device)
            inertia[0, 0] = torch.clip(self.all[material_name][0], 1e-3)
            inertia[1, 1] = torch.clip(self.all[material_name][1], 1e-3)
            inertia[2, 2] = torch.clip(self.all[material_name][2], 1e-3)
            return inertia

    def get_material_num(self):
        """返回物理参数的数量"""
        return len(self.all)
    
    def get_material_names(self):
        """返回所有物理参数的名称列表"""
        return self.all.keys()
    
    def get_original_json_dict(self):
        """
        导出原始参数的JSON字典（未经过物理范围映射，用于存储优化变量的原始值）
        """
        json_dict = {"body_id": self.body_id}
        for key in self.all.keys():
            json_dict[key] = self.all[key].tolist()
        return json_dict
    
    def get_activate_json_dict(self):
        """
        导出激活后参数的JSON字典（经过物理范围映射，用于物理引擎加载）
        """
        json_dict = {"body_id": self.body_id}
        for key in self.all.keys():
            json_dict[key] = self.get_material(key).tolist()
        return json_dict
    
    def no_optimize(self, material_name):
        """关闭指定参数的自动微分（用于固定某些参数不参与优化）"""
        self.all[material_name].requires_grad = False

    def add_noise(self):    
        """为所有参数添加随机噪声（用于探索式优化或数据增强）"""
        noise = (2 * random.random() - 1)
        self.all["mass"] += noise
        noise = (2 * random.random() - 1)
        self.all["friction_coefficient"] += noise
        noise = (2 * random.random() - 1)
        self.all["restitution"] += noise