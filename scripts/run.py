# 导入核心依赖库
import mmcv  # 计算机视觉工具库（未直接使用，预留可视化/数据处理扩展）
import torch  # 深度学习框架（张量计算、自动微分，核心优化工具）
import numpy as np  # 数值计算（数组处理、参数初始化）
import random  # 随机数生成（实验可重复性控制）
import os, sys  # 文件路径、系统路径操作
import json  # 物理参数保存（JSON格式，便于后续加载分析）
from tqdm import tqdm  # 训练进度条（直观显示训练过程）
import trimesh  # 3D网格加载/处理（仿真场景中物体模型加载）
import time  # 时间戳（用于输出目录命名，区分多次训练）
#
# 设置当前工作目录并添加到系统路径（确保模块导入正常）
cur_work_path = os.getcwd()
sys.path.append(cur_work_path)

# 导入自定义模块（PIN-WM论文核心组件）
from builder import Builder  # 组件构建器（统一创建仿真器、渲染器、优化器等）
from utils.sys_utils import prepare_output_and_logger  # 输出目录+日志初始化（之前分析过的工具）
from utils.data_utils import load_pybullet_dynamic_dataset  # 加载PyBullet动态数据集（真实观测数据）
from utils.metric_utils import mean_psnr  # 平均PSNR计算（图像相似度评估，之前分析过）
from utils.quaternion_utils import wxyz2xyzw, rotmat_to_quaternion  # 四元数格式转换（之前分析过）

# 高斯渲染相关模块（2D高斯溅射渲染，生成仿真场景图像）
from diff_rendering.gaussian_splatting_2d.utils.camera_utils import cameraList_from_camInfos  # 从相机信息创建相机列表
from diff_rendering.gaussian_splatting_2d.utils.render_utils import save_img_u8  # 保存8位RGB图像
from diff_rendering.gaussian_splatting_2d.scene.gaussian_model import GaussianModel  # 高斯模型（渲染核心）
from diff_rendering.gaussian_splatting_2d.utils.loss_utils import rendering_loss_batch, pose_loss_batch  # 渲染损失、姿态损失

# 可微分仿真相关模块（物理引擎，支持梯度反向传播）
from diff_simulation.simulator import Simulator  # 可微分仿真器（核心：模拟物理运动）
from diff_simulation.physical_material import Physical_Materials  # 物理材质类（存储/优化物理参数）
from diff_simulation.constraints.base import Joint_Type  # 关节约束类型（限制物体运动自由度）

# ============================ 实验参数配置 ============================
# 系统参数（实验基础配置）
sys_args = dict(
    seed=0,  # 随机种子（保证实验可复现）
    output_path='./output/sim_push_t'  # 实验输出根目录
)

# 数据参数（数据集相关配置）
data_args = dict(
    white_background=True,  # 渲染背景为白色
    n_frames=32,  # 每个序列的帧数（仿真/真实数据均为32帧）
    test_num=0,  # 测试集数量（当前仅训练，不测试）
    train_num=1,  # 训练集数量（1个训练样本）
    H=800,  # 图像高度（800像素）
    W=800,  # 图像宽度（800像素）
    data_path='dataset/sim_push_t2'  # 数据集路径（存储真实观测图像、相机参数、姿态数据）
)

# 渲染参数（高斯渲染相关配置）
render_args = dict(
    sh_degree=3,  # 球谐函数阶数（控制高斯渲染的光照细节，3阶为常用值）
)

# 仿真参数（可微分仿真相关配置）
sim_args = dict(
    dtime=1.0 / 240,  # 仿真步长（240Hz，高频仿真保证物理精度）
    frame_dt=1.0 / 24,  # 帧间隔（24Hz，每帧包含多个仿真子步）
    train_iteration=100,  # 总训练迭代次数
    save_iteration=10,  # 保存间隔（每10次迭代保存渲染图像和物理参数）
    opt_interval_num=8  # 优化间隔数量（将32帧分为8段，逐段优化）
)

# 策略参数（当前未使用，预留末端执行器控制策略扩展）
policy_args = dict()


def train_physical_materials(output_path, builder, obj_id, ee_id, dataset, simulator: Simulator,
                             gaussians: GaussianModel, gaussian_renderer, random_init_index,
                             optimizer, logger=None):
    """
    核心训练函数：优化物理参数，使仿真的物体运动+渲染图像匹配真实数据
    核心逻辑：可微分仿真（模拟物体运动）→ 高斯渲染（生成仿真图像）→ 损失计算（图像+姿态误差）→ 反向传播（优化物理参数）
    Args:
        output_path: 当前训练样本的输出路径（保存图像、物理参数）
        builder: 组件构建器（统一获取实验参数）
        obj_id: 待优化物体的仿真ID（立方体ID）
        ee_id: 末端执行器的仿真ID（推物体的工具ID）
        dataset: 单样本训练数据（含真实图像、相机参数、物体姿态、末端执行器目标位置）
        simulator: 可微分仿真器实例（模拟物理运动）
        gaussians: 高斯模型（用于渲染仿真场景图像）
        gaussian_renderer: 高斯渲染器（生成仿真图像）
        random_init_index: 随机初始化索引（多轮初始化时区分不同实验）
        optimizer: 优化器（更新物理参数）
        logger: 日志器（记录损失、PSNR等指标）
    Returns:
        loss: 最终训练损失（图像匹配损失）
    """
    # 计算每帧包含的仿真子步数（帧间隔 / 仿真步长 = 24Hz / 240Hz = 10个子步）
    n_substeps = round(builder.sim_args.frame_dt / builder.sim_args.dtime)
    f_max = builder.data_args.n_frames  # 总帧数（32帧）
    
    # 步骤1：从数据集创建训练相机（匹配真实观测的相机参数）
    dynamic_train_camera = cameraList_from_camInfos(dataset["cam_infos"], 1.0, builder.gaussian_args.dataset)
    # 步骤2：加载真实图像（gt_images）并移至GPU
    gt_images = []
    for cam in dynamic_train_camera:
        gt_images.append(cam.original_image.cuda())  # 真实图像（Ground Truth）
    
    # 步骤3：加载真实物体姿态（gt_poses）和末端执行器目标位置（goal_pos）
    gt_poses = torch.tensor(dataset["pose_datas"], device=simulator.device).cuda()  # 真实物体姿态（位置+四元数）
    goal_pos = torch.tensor(dataset["ee_goal_position"], device=simulator.device)  # 末端执行器的目标位置（推物体的终点）
    
    # 训练配置参数读取
    train_iteration = builder.sim_args.train_iteration  # 总迭代次数（100）
    opt_interval_num = builder.sim_args.opt_interval_num  # 优化间隔数量（8）
    opt_interval = int(train_iteration / opt_interval_num)  # 每段优化的迭代次数（100/8=12.5→12？实际代码按整数处理）
    assert f_max % opt_interval_num == 0  # 确保帧数能被优化间隔数量整除（32/8=4，每段4帧）
    frame_interval = int(f_max / opt_interval_num)  # 每段优化的帧数（4帧）
    start_idx = 0  # 优化起始帧索引（逐段优化时更新）
    max_idx = f_max - frame_interval  # 优化终止帧索引（32-4=28）
    
    # 创建训练进度条（直观显示迭代进度）
    progress_bar = tqdm(range(0, train_iteration), desc="Dynamic Train Progress")
    
    # 核心训练循环（迭代100次，每次迭代完整仿真32帧）
    for iter in range(0, train_iteration + 1):
        # 步骤1：重置仿真器（每次迭代重新开始仿真，避免历史状态干扰）
        simulator.reset()
        images = []  # 存储当前迭代的仿真渲染图像
        poses = []   # 存储当前迭代的仿真物体姿态
        
        # 步骤2：逐帧仿真+渲染（32帧）
        for f in range(f_max):
            # 从第2帧开始，控制末端执行器向目标位置运动
            if f > 0:
                # 获取末端执行器当前世界坐标（位置+姿态）
                ee_world_position, _ = simulator.get_body_pose_clone(ee_id)
                # 计算末端执行器线速度（比例控制：与目标位置的误差×5，加速收敛）
                ee_linear_velocity = (goal_pos - ee_world_position[:2]) * 5  # 仅控制XY平面（Z轴固定）
                ee_linear_velocity = torch.cat((ee_linear_velocity, torch.tensor([0], device=simulator.device)))  # Z轴速度为0
                # 更新末端执行器速度（角速度为0，仅平移）
                simulator.change_body_vel(ee_id, linear_velocity=ee_linear_velocity, angular_velocity=torch.zeros((3), device=simulator.device))
                # 执行仿真子步（每帧10个子步，保证物理运动平滑）
                for i in range(n_substeps):
                    simulator.step()

            # 步骤3：获取物体当前姿态（位置+旋转四元数）
            obj_position, obj_rotation = simulator.get_body_pose(obj_id)
            # 更新高斯模型的位置和旋转（将高斯模型绑定到仿真物体，随物体运动）
            gaussians.reset_position_rotation(obj_position, obj_rotation)
            # 步骤4：高斯渲染（生成当前帧的仿真图像）
            render_pkg = gaussian_renderer.render(dynamic_train_camera[0], gaussians)
            image = render_pkg['render']  # 提取渲染图像（[C, H, W]，0~1范围）
            
            # 存储当前帧的渲染图像和物体姿态
            images.append(image)
            # 姿态拼接：位置（3维）+ 四元数（wxyz→xyzw格式，4维）→ 7维姿态向量
            poses.append(torch.cat((obj_position.clone(), wxyz2xyzw(obj_rotation.clone())), dim=-1))

        # 步骤5：更新优化起始帧（逐段优化，每次迭代优化后续4帧）
        if iter % opt_interval == 0 and iter > 0:
            start_idx += frame_interval
            if start_idx > max_idx:
                start_idx = max_idx  # 避免索引超出范围

        # 步骤6：计算损失（仅用前12帧计算渲染损失，平衡训练效率和精度）
        loss = rendering_loss_batch(images[:12], gt_images[:12])  # 批量渲染损失（MSE+SSIM等，论文定义）

        # 步骤7：反向传播（计算物理参数的梯度）
        loss.backward(retain_graph=True)  # retain_graph=True：保留计算图，支持后续可能的多损失计算
        
        # 步骤8：计算PSNR（评估仿真图像与真实图像的相似度，值越高越好）
        psnr = mean_psnr(images, gt_images)  # 调用之前分析的平均PSNR函数

        # 步骤9：记录日志（损失、PSNR）
        all_physical_materials = simulator.get_all_physical_materials()  # 获取所有可优化的物理材质
        if logger is not None:
            logger.record("metric/loss" + str(random_init_index), loss)  # 记录损失
            logger.record("metric/psnr" + str(random_init_index), psnr)  # 记录PSNR
            logger.dump(iter)  # 刷新日志

        # 步骤10：更新进度条（显示当前损失和PSNR）
        info_dict = {"loss": loss.item(), "psnr": psnr.item()}
        progress_bar.set_postfix(info_dict)
        progress_bar.update(1)

        # 步骤11：定期保存结果（每10次迭代保存渲染图像和物理参数）
        if (iter % builder.sim_args.save_iteration) == 0:
            save_path = os.path.join(output_path, "iteration_{}".format(iter))  # 当前迭代的保存路径
            os.makedirs(save_path, exist_ok=True)
            image_path = os.path.join(save_path, "images")  # 渲染图像保存路径
            os.makedirs(image_path, exist_ok=True)
            
            # 保存每帧的仿真渲染图像
            for index, image in enumerate(images):
                # 图像格式转换：[C, H, W]→[H, W, C] → CPU → NumPy → 0~255 uint8
                save_img_u8(image.permute(1, 2, 0).cpu().detach().numpy(), os.path.join(image_path, 'sim_{}.png'.format(index)))
            
            # 保存物理参数（原始配置+当前激活的可优化参数）
            physical_materials_original_json_list = []
            physical_materials_activate_json_list = []
            for physical_materials in all_physical_materials:
                physical_materials_original_json_list.append(physical_materials.get_original_json_dict())  # 原始参数
                physical_materials_activate_json_list.append(physical_materials.get_activate_json_dict())  # 可优化参数
            # 写入JSON文件
            with open(os.path.join(save_path, 'physical_materials_iter.json'), 'w') as json_file:
                json.dump({
                    "activate": physical_materials_activate_json_list}, fp=json_file, indent=4)

        # 步骤12：更新优化器（应用梯度，更新物理参数）
        optimizer.step()
        # 步骤13：清空梯度（避免梯度累积）
        optimizer.zero_grad()
    
    # 训练结束，关闭进度条
    progress_bar.close()

    return loss


def create_push_t_scene(builder, simulator: Simulator):
    """
    创建仿真场景：加载3D模型（平面、立方体、末端执行器），初始化物理参数和运动约束
    场景描述：平面固定在场景底部，立方体放置在平面上，末端执行器可在XY平面平移（Z轴固定，无旋转）
    Args:
        builder: 组件构建器（获取实验参数）
        simulator: 可微分仿真器实例（场景将添加到该仿真器）
    Returns:
        obj_id: 立方体的仿真ID（待优化物理参数的物体）
        ee_id: 末端执行器的仿真ID（推立方体的工具）
    """
    # 3D模型和URDF文件路径（URDF：统一机器人描述格式，定义物体的物理属性和约束）
    plane_mesh_path = "./envs/asset/plane/plane_collision.obj"  # 平面网格路径（碰撞检测用）
    obj_mesh_path = "./envs/asset/cube_t/cube_t.obj"  # 立方体网格路径（待推物体）
    ee_mesh_path = "./envs/asset/ee/ee.obj"  # 末端执行器网格路径（推物体的工具）
    plane_urdf_path = "./envs/asset/plane/plane.urdf"  # 平面URDF路径
    obj_urdf_path = "./envs/asset/cube_t/cube_t_mesh.urdf"  # 立方体URDF路径
    ee_urdf_path = "./envs/asset/ee/ee.urdf"  # 末端执行器URDF路径

    # 步骤1：创建平面（场景地面，固定不动）
    plane_mesh = trimesh.load(plane_mesh_path)  # 加载平面网格
    plane_mesh.apply_scale([30, 30, 10])  # 缩放平面（30×30×10，确保足够大）
    plane_physical_material = Physical_Materials(requires_grad=True, device=simulator.device)  # 平面物理材质（支持梯度）
    plane_physical_material.no_optimize("mass")  # 固定平面质量（不优化）
    plane_physical_material.no_optimize('inertia')  # 固定平面惯性张量（不优化）
    # 创建平面物体（添加到仿真器）
    plane_id = simulator.create_mesh_body(
        plane_mesh, plane_physical_material, requires_grad=True,
        urdf=plane_urdf_path,
        world_position=torch.tensor([0.0, 0.0, -5.0], device=simulator.device),  # 平面位置（Z=-5，作为地面）
        world_rotation=torch.tensor([1.0, 0.0, 0.0, 0.0], device=simulator.device)  # 平面旋转（无旋转）
    )
    # 给平面添加固定约束（禁止平移和旋转，固定在场景中）
    simulator.create_joint(plane_id, Joint_Type.FIX_CONSTRAINT)

    # 步骤2：创建立方体（待推物体，优化其物理参数）
    obj_mesh = trimesh.load(obj_mesh_path)  # 加载立方体网格
    obj_physical_material = Physical_Materials(requires_grad=True, device=simulator.device)  # 立方体物理材质（可优化）
    # 初始化立方体惯性张量（3×3对角矩阵，控制旋转惯性）
    obj_physical_material.set_material("inertia", [
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.01]
    ])
    # 创建立方体物体（添加到仿真器）
    obj_id = simulator.create_mesh_body(
        obj_mesh, obj_physical_material, requires_grad=True,
        urdf=obj_urdf_path,
        world_position=torch.tensor([0.0, 0.0, 0.025], device=simulator.device),  # 立方体位置（Z=0.025，放在平面上）
        world_rotation=torch.tensor([1.0, 0.0, 0.0, 0.0], device=simulator.device)  # 无初始旋转
    )

    # 步骤3：创建末端执行器（推物体的工具，物理参数固定）
    ee_mesh = trimesh.load(ee_mesh_path)  # 加载末端执行器网格
    ee_physical_material = Physical_Materials(requires_grad=True, device=simulator.device)  # 末端执行器物理材质
    ee_physical_material.set_material("mass", 1000.0)  # 设置末端执行器质量（1000.0，足够重以推动立方体）
    ee_physical_material.no_optimize("inertia")  # 固定惯性张量（不优化）
    ee_physical_material.no_optimize("mass")  # 固定质量（不优化）
    # 创建末端执行器物体（添加到仿真器）
    ee_id = simulator.create_mesh_body(
        ee_mesh, ee_physical_material, requires_grad=True,
        urdf=ee_urdf_path,
        world_position=torch.tensor([0.05, 0.05, 0.05], device=simulator.device),  # 初始位置（XY平面偏移，Z轴略高于立方体）
        world_rotation=torch.tensor([1.0, 0.0, 0.0, 0.0], device=simulator.device)  # 无初始旋转
    )
    # 给末端执行器添加约束：Z轴固定（不能上下移动）
    simulator.create_joint(ee_id, Joint_Type.NO_TRANS_Z_CONSTRATNT)
    # 给末端执行器添加约束：禁止旋转（仅平移）
    simulator.create_joint(ee_id, Joint_Type.NO_ROT_CONSTRATNT)

    # 返回立方体和末端执行器的仿真ID（用于后续控制和姿态获取）
    return obj_id, ee_id


def test_dynamic_train_multbody_push_t():
    """
    主训练流程：初始化实验环境→加载数据集→创建仿真场景→启动物理参数优化
    整体流程：参数配置→输出目录+日志初始化→数据集加载→场景创建→训练→结果保存
    """
    # 步骤1：整合所有实验参数
    all_args = {
        'sys_args': sys_args,
        'data_args': data_args,
        'render_args': render_args,
        'sim_args': sim_args,
        'policy_args': policy_args
    }
    
    # 步骤2：初始化输出目录和日志器（调用之前分析的prepare_output_and_logger函数）
    all_args, loggers = prepare_output_and_logger(all_args, need_logger=True)
    # 步骤3：创建组件构建器（统一创建仿真器、渲染器、优化器等核心组件）
    builder = Builder(all_args)
    
    # 步骤4：加载训练数据集（动态数据集，含真实图像、姿态、相机参数）
    dynamic_train_datasets, _ = load_pybullet_dynamic_dataset(builder.data_args)

    # 步骤5：创建训练输出目录（按时间戳命名，区分多次训练）
    train_path = os.path.join(builder.sys_args.output_path, "dynamic", "train", str(int(time.time())))
    os.makedirs(train_path, exist_ok=True)

    # 步骤6：多轮随机初始化训练（当前仅1轮，可扩展为多轮取最优结果）
    random_init_num = 1
    for random_init_index in range(random_init_num):
        # 创建当前随机初始化的输出目录
        train_sub_path = os.path.join(train_path, "random_init" + str(random_init_index))
        os.makedirs(train_sub_path, exist_ok=True)
        loss_mean = 0.0  # 记录所有训练样本的平均损失
        
        # 步骤7：遍历每个训练样本（当前仅1个样本）
        for dataset_index, dynamic_train_dataset in enumerate(dynamic_train_datasets):
            # 创建当前样本的输出目录
            output_path = os.path.join(train_sub_path, "dataset" + str(dataset_index))
            os.makedirs(output_path, exist_ok=True)
            print("Save folder:", train_path)
            
            # 步骤8：初始化可微分仿真器（GPU加速，开启可视化）
            sim_device = "cuda"  # 仿真设备（GPU）
            vis = True  # 开启仿真可视化（实时查看物体运动）
            simulator = Simulator(builder.sim_args.dtime, device=sim_device, vis=vis)
            
            # 步骤9：创建仿真场景（平面+立方体+末端执行器）
            obj_id, ee_id = create_push_t_scene(builder, simulator)
            
            # 步骤10：获取立方体初始姿态（位置+旋转）
            obj_position, obj_rotation = simulator.get_body_pose_clone(obj_id)
            # 步骤11：构建高斯模型（2D高斯溅射渲染核心）
            gaussians = builder.build_static_2dgs()
            # 将高斯模型绑定到立方体的局部坐标系（随立方体运动）
            gaussians.translate2localframe(obj_position, obj_rotation)
            # 步骤12：构建高斯渲染器（生成仿真图像）
            gaussian_renderer = builder.build_renderer()
            # 步骤13：构建优化器（用于更新物理参数）
            optimizer = builder.build_optimizer(simulator)
            # 调整优化器学习率（针对特定参数组，精细调优）
            optimizer.param_groups[4]['lr'] = 4e-2
            optimizer.param_groups[5]['lr'] = 1e-4
            
            # 步骤14：启动核心训练（优化物理参数）
            loss = train_physical_materials(
                output_path, builder, obj_id, ee_id, dynamic_train_dataset,
                simulator, gaussians, gaussian_renderer, random_init_index,
                optimizer, loggers["dynamic"][dataset_index]
            )
            
            # 步骤15：关闭仿真器（释放资源）
            simulator.close()
            # 累加损失，计算平均损失
            loss_mean += loss / len(dynamic_train_datasets)

        # 输出所有样本的平均损失
        print("loss_mean", loss_mean)


# 主函数入口（启动训练）
if __name__ == '__main__':
    test_dynamic_train_multbody_push_t()