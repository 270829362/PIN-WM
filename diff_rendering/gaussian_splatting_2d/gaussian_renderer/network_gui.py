# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import torch  # 用于处理相机矩阵等张量数据（适配后续渲染模块）
import traceback  # 用于捕获并打印异常堆栈（便于调试通信错误）
import socket  # 用于建立TCP Socket通信（核心通信模块）
import json  # 用于解析JSON格式的通信数据（外部发送的参数多为JSON格式）
# 导入论文中定义的轻量级相机类（用于封装相机内外参，供渲染函数使用）
from diff_rendering.gaussian_splatting_2d.scene.cameras import MiniCam

# 1. 全局通信参数：默认Socket连接的IP和端口（可通过init函数修改）
host = "127.0.0.1"  # 默认本地回环地址（适合同机程序通信，如机器人控制程序与渲染程序同机）
port = 6009          # 默认通信端口（需与外部程序端口一致，避免冲突）

# 2. 全局通信对象：存储Socket连接和客户端地址（后续通信复用）
conn = None   # 已建立的Socket连接对象（None表示未连接）
addr = None   # 客户端地址（如外部程序的IP:端口）

# 3. 初始化Socket监听器：创建TCP类型的Socket（用于监听外部连接请求）
listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def init(wish_host, wish_port):
    """
    初始化Socket监听器：绑定指定的IP和端口，开启监听（仅需调用一次）
    
    Args:
        wish_host: 期望绑定的IP地址（如"192.168.1.100"，跨设备通信需指定网卡IP）
        wish_port: 期望绑定的端口号（需与外部程序约定一致，如6009）
    """
    global host, port, listener  # 引用全局变量（修改全局的IP、端口和监听器）
    host = wish_host
    port = wish_port
    listener.bind((host, port))  # 将监听器绑定到指定IP和端口
    listener.listen()            # 开启监听模式（等待外部连接请求）
    listener.settimeout(0)       # 设置监听超时为0（非阻塞模式，避免程序卡在监听步骤）


def try_connect():
    """
    尝试建立Socket连接：非阻塞式检查是否有外部连接请求，成功则创建连接对象
    
    说明：因listener设为非阻塞，调用时若无连接请求会捕获异常并忽略，不阻塞主程序
    """
    global conn, addr, listener  # 引用全局连接对象和地址
    try:
        # 接收外部连接请求：返回（连接对象，客户端地址）
        conn, addr = listener.accept()
        print(f"\nConnected by {addr}")  # 打印连接成功的客户端地址（如('127.0.0.1', 54321)）
        conn.settimeout(None)  # 取消连接的超时限制（建立连接后需稳定通信，避免超时断开）
    except Exception as inst:
        pass  # 无连接请求时捕获异常，不处理（仅忽略，不影响主程序流程）


def read():
    """
    从已建立的Socket连接中读取数据：按“4字节长度+JSON数据”格式解析
    
    通信协议约定：
    1. 外部发送数据前，先发送4字节的“数据长度”（小端字节序）
    2. 再发送对应长度的JSON格式字符串（存储相机参数和控制指令）
    
    Returns:
        dict: 解析后的JSON数据（键值对形式，如{"resolution_x": 640, "fov_y": 30.0}）
    """
    global conn  # 引用全局连接对象（已通过try_connect建立）
    # 第一步：读取4字节的消息长度（小端字节序，如b'\x00\x00\x01\x00'表示256字节）
    messageLength = conn.recv(4)
    messageLength = int.from_bytes(messageLength, 'little')  # 转换为整数长度
    # 第二步：读取对应长度的JSON消息（按UTF-8解码为字符串）
    message = conn.recv(messageLength)
    return json.loads(message.decode("utf-8"))  # 解析JSON字符串为Python字典


def send(message_bytes, verify):
    """
    向外部发送数据：支持发送二进制数据和验证字符串，确保外部接收完整
    
    Args:
        message_bytes: 待发送的二进制数据（如渲染结果的图像字节流，可为None）
        verify: 验证字符串（如"OK"、"ERROR"，用于告知外部接收状态）
    """
    global conn  # 引用全局连接对象
    if message_bytes != None:
        conn.sendall(message_bytes)  # 发送二进制数据（若有）
    # 发送验证信息：先发送4字节的验证字符串长度，再发送验证字符串本身
    conn.sendall(len(verify).to_bytes(4, 'little'))  # 验证字符串长度（4字节小端）
    conn.sendall(bytes(verify, 'ascii'))  # 验证字符串（ASCII编码）


def receive():
    """
    核心数据接收与解析函数：读取外部发送的参数，封装为MiniCam相机实例并返回控制指令
    
    Returns:
        tuple: (custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier)
        - custom_cam: MiniCam实例（封装相机内外参，供渲染函数使用）；若分辨率为0则为None
        - do_training: bool（是否启用训练模式，True=优化2DGS参数）
        - do_shs_python: bool（是否在Python端计算球谐函数→颜色，论文中默认False）
        - do_rot_scale_python: bool（是否在Python端计算旋转/缩放，论文中默认False）
        - keep_alive: bool（是否保持Socket连接，True=不关闭连接，持续通信）
        - scaling_modifier: float（高斯点全局缩放系数，传递给渲染函数）
    """
    # 第一步：读取外部发送的JSON数据（相机参数+控制指令）
    message = read()

    # 提取图像分辨率（外部发送的相机采集分辨率，如640x480）
    width = message["resolution_x"]  # 图像宽度
    height = message["resolution_y"]  # 图像高度

    if width != 0 and height != 0:
        # 分辨率非0：正常解析相机参数并创建MiniCam实例
        try:
            # 解析控制指令（布尔型）
            do_training = bool(message["train"])  # 是否训练（优化2DGS参数）
            do_shs_python = bool(message["shs_python"])  # 是否Python端计算SH→颜色
            do_rot_scale_python = bool(message["rot_scale_python"])  # 是否Python端计算旋转/缩放
            keep_alive = bool(message["keep_alive"])  # 是否保持连接
            
            # 解析相机内参（渲染必需参数）
            fovy = message["fov_y"]  # 垂直视场角（单位：度）
            fovx = message["fov_x"]  # 水平视场角（单位：度）
            znear = message["z_near"]  # 相机近裁剪面（距离相机最近的可见距离）
            zfar = message["z_far"]    # 相机远裁剪面（距离相机最远的可见距离）
            scaling_modifier = message["scaling_modifier"]  # 高斯点缩放系数
            
            # 解析相机外参（视图矩阵、投影矩阵）并转换为PyTorch张量（GPU上）
            # 视图矩阵（world→view）：将世界坐标系的点转换到相机坐标系
            world_view_transform = torch.reshape(
                torch.tensor(message["view_matrix"]),  # JSON中的列表→张量
                (4, 4)  # 重塑为4x4矩阵（齐次坐标变换）
            ).cuda()  # 转移到GPU（适配后续渲染模块的GPU计算）
            # 坐标系适配：翻转y轴和z轴（因外部程序（如Unity）的坐标系可能与代码不一致，需统一）
            world_view_transform[:, 1] = -world_view_transform[:, 1]
            world_view_transform[:, 2] = -world_view_transform[:, 2]
            
            # 投影矩阵（view→proj）：将相机坐标系的点转换到裁剪坐标系
            full_proj_transform = torch.reshape(
                torch.tensor(message["view_projection_matrix"]),  # JSON中的列表→张量
                (4, 4)  # 重塑为4x4矩阵
            ).cuda()  # 转移到GPU
            # 坐标系适配：翻转y轴（与视图矩阵保持一致的坐标系）
            full_proj_transform[:, 1] = -full_proj_transform[:, 1]
            
            # 封装为MiniCam实例（论文中定义的轻量级相机类，供render函数调用）
            custom_cam = MiniCam(
                width, height, fovy, fovx, znear, zfar,
                world_view_transform, full_proj_transform
            )
        except Exception as e:
            # 捕获解析异常：打印堆栈信息（便于定位错误，如参数缺失、格式错误）
            print("")
            traceback.print_exc()
            raise e  # 重新抛出异常（让上层程序处理，避免静默失败）
        # 返回解析后的相机实例和控制指令
        return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier
    else:
        # 分辨率为0：外部告知无有效相机数据，返回全None
        return None, None, None, None, None, None