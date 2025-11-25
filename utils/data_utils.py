""" 这段代码是 PIN-WM 模型的数据集预处理核心模块，专门处理 Pybullet 仿真生成的动态观测数据，为后续的高斯渲染、物理参数优化提供 “干净、统一格式” 的输入，核心作用可概括为 3 点：
数据读取与解析：从 JSON 文件中提取关键信息 —— 相机内外参（用于渲染视角匹配）、机器人末端执行器（EE）的初始 / 目标位置（用于非抓握操作任务定义）、图像路径（用于加载观测图像）。
数据预处理：
图像处理：融合透明通道与背景（避免透明区域影响观测损失计算），统一图像格式为 RGBA；
相机参数校准：调整相机坐标系轴方向，转换 c2w→w2c 矩阵，计算视场角（FoV），确保相机参数符合高斯渲染器的输入要求；
时序排序：按帧号排序相机数据，保证物体运动的时序一致性。
数据集组织：按训练 / 测试样本数量拆分数据集，支持多相机系统（可筛选指定相机的时序数据），最终输出渲染器 / 仿真器可直接使用的格式（CameraInfo 列表、结构化样本字典）。 """
# 导入核心依赖库
import torch  # 深度学习框架（后续可能用于张量转换）
import numpy as np  # 数值计算、数组处理（核心数据格式）
import imageio  # 图像读写（备用，此处未直接使用）
from tqdm import tqdm  # 进度条显示（备用，此处未直接使用）
import os, json, sys  # 系统路径、JSON文件、系统配置
import cv2  # OpenCV库（图像保存功能）
from PIL import Image  # PIL库（图像读取、格式转换）
from typing import NamedTuple  # 结构化数据类型（适配相机信息类）

# 导入论文专用模块（对应高斯溅射渲染和相机工具）
from diff_rendering.gaussian_splatting_2d.scene.dataset_readers import (
    SceneInfo, BasicPointCloud, SH2RGB, storePly, fetchPly,
    CameraInfo, fov2focal, focal2fov, getNerfppNorm
)
# CameraInfo：相机信息结构化类（存储外参、内参、图像等）
# fov2focal/focal2fov：视场角（FoV）与焦距的转换函数（相机内参计算）
from diff_rendering.gaussian_splatting_2d.utils.camera_utils import cameraList_from_camInfos
# 相机列表转换函数：将CameraInfo列表转换为渲染器可直接使用的相机格式

# 导入可微分仿真模块的力类（备用，此处未直接使用）
from diff_simulation.force.constant_force import Constant_Force
# 导入旋转处理库（备用，此处未直接使用）
from scipy.spatial.transform import Rotation as RR

def readDataFromJson(path, json_file, white_background, extension):
    """
    从JSON文件读取单样本数据（核心数据读取函数）
    功能：解析相机参数、机器人末端执行器（EE）位置、图像路径，处理图像背景和相机内外参
    Args:
        path: 数据集根路径（存储JSON文件和图像的目录）
        json_file: 单样本的JSON配置文件（如dynamic_train0.json）
        white_background: 是否使用白色背景（True=白色，False=黑色，适配渲染器背景）
        extension: 图像文件后缀（如".png"，用于截取图像文件名）
    Returns:
        cam_infos: CameraInfo列表（每个元素是一帧的相机信息，含内外参、图像）
        ee_init_position: 机器人末端执行器初始位置（3D坐标，np.array）
        ee_goal_position: 机器人末端执行器目标位置（3D坐标，np.array）
        pose_datas: 物体姿态数据（可选，存储物体的3D姿态序列，如旋转矩阵/四元数）
    """
    # 打开并读取JSON文件（存储样本的所有配置信息）
    with open(os.path.join(path, json_file)) as json_file:
        contents = json.load(json_file)

    # 读取机器人末端执行器（EE）的初始位置和目标位置（用于非抓握操作任务，如推/拨）
    ee_init_position = np.array(contents["ee_init_position"], dtype=np.float32)  # 初始位置
    ee_goal_position = np.array(contents["ee_goal_position"], dtype=np.float32)  # 目标位置

    # 初始化相机信息列表（存储每帧的相机数据）
    cam_infos = []
    # 读取JSON中的图像数据列表（每个元素对应一帧的图像路径和相机参数）
    image_datas = contents["image_datas"]
    for entry in image_datas:
        # 解析图像路径中的相机ID（cam_id）和帧号（frame_id）
        # 图像路径格式假设为：r_cam0_0.png → 拆分后cam_id=0，frame_id=0
        image_path = entry["file_path"]
        cam_id, frame_id = [int(i) for i in image_path.split("/")[-1].rstrip(extension).lstrip("r_").split("_")]
        # 跳过无效帧（frame_id<0的帧）
        if frame_id < 0:
            continue
        
        # 读取相机外参：c2w（camera-to-world，相机到世界的变换矩阵，4x3→补全为4x4）
        c2w = entry["c2w"]  # 原始c2w是4x3矩阵（3x3旋转+3x1平移）
        c2w.append([0.0, 0.0, 0.0, 1.0])  # 补全为4x4齐次矩阵（符合变换矩阵规范）
        c2w = np.array(c2w)
        
        # 关键：调整相机坐标系轴方向（适配渲染器/仿真器的坐标系约定，避免图像翻转）
        # 翻转y和z轴（因为不同系统的坐标系轴方向可能不一致，需统一）
        c2w[:3, 1:3] *= -1
        
        # 计算w2c（world-to-camera，世界到相机的变换矩阵）= c2w的逆矩阵
        w2c = np.linalg.inv(c2w)
        # 提取相机外参：旋转矩阵R（3x3）和平移向量T（3x1）
        # w2c的旋转部分转置（因为相机外参R定义为世界→相机的旋转，需满足渲染器要求）
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]  # 平移向量（相机在世界坐标系中的位置）
        
        # 修正图像路径：将"r"替换为"m"（假设数据集图像命名规则：r_xxx是原始图，m_xxx是处理后图）
        # 例：r_cam0_0.png → m_cam0_0.png（m可能代表mask或processed）
        image_path = os.path.join(path, 'm'.join(image_path.rsplit('r', 1)))
        # 提取图像文件名（不含后缀），用于后续存储/索引
        image_name = os.path.basename(image_path).split(".")[0]
        
        # 读取图像（支持RGBA格式，含透明通道）
        image = Image.open(image_path)
        # 转换为numpy数组（RGBA：4通道，R/G/B/Alpha，Alpha=透明度）
        im_data = np.array(image.convert("RGBA"))
        
        # 配置背景色（与渲染器背景一致，避免观测损失计算偏差）
        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])  # 白/黑背景

        # 图像预处理：融合透明通道与背景（关键！避免透明区域影响模型训练）
        norm_data = im_data / 255.0  # 像素值归一化到[0,1]
        # 公式：RGB = 图像RGB * Alpha（透明通道） + 背景 * (1 - Alpha)
        # 逻辑：Alpha=1（不透明）→ 保留原图；Alpha=0（完全透明）→ 显示背景
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        # 拼接Alpha通道（保持4通道格式，便于后续处理）
        arr = np.concatenate((arr, norm_data[:, :, 3][..., np.newaxis]), axis=-1)
        # 转换回PIL图像（uint8格式，0-255）
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGBA")

        # 读取相机内参（intrinsic：3x3矩阵，如[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]）
        intrinsic = entry["intrinsic"]
        # 计算水平视场角（FovX）：通过内参fx和图像宽度推导
        # fx = (图像宽度/2) / tan(FovX/2) → 反解FovX
        fovx = np.arctan(0.5 * image.size[0] / intrinsic[0][0]) * 2
        # 计算垂直视场角（FovY）：通过FovX和图像宽高比转换（利用fov2focal/focal2fov函数）
        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        # 存储视场角（适配渲染器的相机内参配置）
        FovY = fovy 
        FovX = fovx            
        
        # 将当前帧的所有信息封装为CameraInfo对象，添加到列表
        cam_infos.append(
            CameraInfo(
                uid=cam_id,  # 相机ID（多相机系统时区分不同相机）
                frameid=frame_id,  # 帧号（按时间顺序排序）
                R=R,  # 相机外参旋转矩阵（3x3）
                T=T,  # 相机外参平移向量（3x1）
                FovY=FovY,  # 垂直视场角
                FovX=FovX,  # 水平视场角
                image=image,  # 预处理后的PIL图像
                image_path=image_path,  # 图像文件路径
                image_name=image_name,  # 图像文件名（不含后缀）
                width=image.size[0],  # 图像宽度
                height=image.size[1]  # 图像高度
            )
        )
    
    # 读取物体姿态数据（若JSON中存在，存储物体的3D姿态序列，如旋转矩阵）
    if "pose_datas" in contents:
        pose_datas = np.array(contents["pose_datas"])
    else:
        pose_datas = None  # 无姿态数据时返回None

    return cam_infos, ee_init_position, ee_goal_position, pose_datas


def load_pybullet_dynamic_dataset(data_args):
    """
    加载Pybullet生成的动态数据集（核心数据集加载函数）
    功能：按训练/测试样本数量，批量读取JSON文件，生成训练集和测试集
    Args:
        data_args: 数据集配置参数（含data_path、train_num、test_num、white_background等）
    Returns:
        train_datasets: 训练集列表（每个元素是一个样本的字典，含cam_infos、EE位置等）
        test_datasets: 测试集列表（结构与训练集一致）
    """
    # 初始化数据集列表（存储所有训练+测试样本）
    dynamic_datasets = []
    # 遍历所有样本（训练样本数 + 测试样本数）
    for index in range(data_args.train_num + data_args.test_num):
        # 区分训练集和测试集的JSON文件名（命名规则：dynamic_trainX.json / dynamic_testX.json）
        if index < data_args.train_num:
            json_name = "dynamic_train{}.json".format(index)  # 训练样本（如dynamic_train0.json）
        else:
            # 测试样本：索引偏移（index - 训练样本数）→ 从0开始编号
            json_name = "dynamic_test{}.json".format(index - data_args.train_num)
        
        # 调用readDataFromJson读取当前样本数据
        dynamic_cam_infos, ee_init_position, ee_goal_position, pose_datas = readDataFromJson(
            data_args.data_path,  # 数据集根路径
            json_name,  # 当前样本的JSON文件
            data_args.white_background,  # 背景色配置
            extension=".png"  # 图像后缀（固定为png，可根据数据集调整）
        )
        
        # 将当前样本数据封装为字典，添加到数据集列表
        dynamic_datasets.append(
            {
                "cam_infos": dynamic_cam_infos,  # 相机信息列表
                'ee_init_position': ee_init_position,  # EE初始位置
                "ee_goal_position": ee_goal_position,  # EE目标位置
                "pose_datas": pose_datas  # 物体姿态数据（可选）
            }
        )
    
    # 拆分训练集和测试集：前train_num个为训练集，后续test_num个为测试集
    return dynamic_datasets[:data_args.train_num], dynamic_datasets[data_args.train_num:data_args.train_num + data_args.test_num]


def get_all_frames_train_camera(cam_id, all_train_cam_infos, dataset):
    """
    从训练集相机信息中，筛选指定相机ID的所有帧并排序
    功能：多相机系统中提取单一相机的时序数据（保证视角一致性）
    Args:
        cam_id: 目标相机ID（如0，筛选该相机拍摄的所有帧）
        all_train_cam_infos: 所有训练集的相机信息列表（混合多个相机、多个样本）
        dataset: 数据集配置（备用，适配相机列表转换函数）
    Returns:
        all_frames_train_camera: 渲染器可直接使用的相机列表（按帧号排序）
    """
    # 筛选指定相机ID的所有相机信息
    all_frames_train_camera_infos = []
    for cam in all_train_cam_infos:
        if cam.uid == cam_id:  # 匹配相机ID
            all_frames_train_camera_infos.append(cam)
    
    # 按帧号（frameid）排序（保证时序顺序，符合物体运动逻辑）
    all_frames_train_camera_infos = sorted(all_frames_train_camera_infos, key=lambda x: x.frameid)
    
    # 将CameraInfo列表转换为渲染器可直接使用的相机格式（缩放因子1.0，保持图像原始尺寸）
    all_frames_train_camera = cameraList_from_camInfos(all_frames_train_camera_infos, 1.0, dataset)
    return all_frames_train_camera

def save_images(images, image_paths):
    """
    批量保存图像（辅助函数）
    功能：将渲染结果或处理后的图像保存到指定路径
    Args:
        images: 图像列表（numpy数组格式，如[H, W, 3]或[H, W, 4]）
        image_paths: 图像保存路径列表（与images一一对应）
    """
    # 遍历图像和路径，逐个保存（使用OpenCV保存）
    for image, image_path in zip(images, image_paths):
        cv2.imwrite(image_path, image)