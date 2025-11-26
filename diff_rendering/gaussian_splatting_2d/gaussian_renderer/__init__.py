# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import torch
import math
# 导入2D高斯光栅化相关模块（底层渲染核心）和高斯模型类
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_rendering.gaussian_splatting_2d.scene.gaussian_model import GaussianModel
# 导入球谐函数（SH）计算工具（用于颜色/光照建模）和深度转法向量工具
from diff_rendering.gaussian_splatting_2d.utils.sh_utils import eval_sh
from diff_rendering.gaussian_splatting_2d.utils.point_utils import depth_to_normal

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, only_rgb_alpha=False):
    """
    2D高斯Splatting渲染核心函数：根据相机视角、高斯模型和流水线配置，生成渲染结果及辅助数据
    
    Args:
        viewpoint_camera: 相机实例（包含内参如视场角、分辨率，外参如世界-视图变换矩阵）
        pc: GaussianModel实例（存储2D高斯点的3D位置、缩放、旋转、透明度、球谐特征等参数）
        pipe: PipelineParams实例（渲染流水线配置，如是否用Python计算协方差、深度比率等）
        bg_color: 背景色张量（必须在GPU上，论文中默认白色背景）
        scaling_modifier: 高斯点缩放系数（全局调整高斯点大小，默认1.0）
        override_color: 自定义颜色（若不为None，直接用该颜色覆盖高斯点的计算颜色，用于调试）
        only_rgb_alpha: 是否仅返回RGB和Alpha图（True时不计算深度、法向量等正则化数据，加速调试）
    
    Returns:
        dict: 渲染结果字典，包含渲染图、屏幕空间点、可见性筛选、深度图、法向量图等
    """
 
    # 1. 创建屏幕空间点张量：用于保留2D（屏幕坐标系）高斯均值的梯度
    # 目的：后续通过渲染损失反向传播时，能更新高斯点的屏幕位置，实现视觉对齐
    screenspace_points = torch.zeros_like(
        pc.get_xyz,  # 与高斯点3D位置张量形状一致（num_gaussians, 3）
        dtype=pc.get_xyz.dtype, 
        requires_grad=True,  # 开启梯度计算
        device="cuda"
    ) + 0  # 初始值设为0
    try:
        screenspace_points.retain_grad()  # 强制保留梯度（避免被自动求导优化掉）
    except:
        pass  # 若保留失败（如计算图中断），不报错（兼容特殊场景）

    # 2. 配置光栅化参数：定义2D高斯点如何渲染到图像平面
    # 计算相机视场角的正切值（水平FoVx、垂直FoVy，用于将3D点投影到2D屏幕）
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),  # 渲染图像高度（与相机采集图像一致）
        image_width=int(viewpoint_camera.image_width),    # 渲染图像宽度（与相机采集图像一致）
        tanfovx=tanfovx,                                  # 水平视场角正切值（投影用）
        tanfovy=tanfovy,                                  # 垂直视场角正切值（投影用）
        bg=bg_color,                                      # 背景色（论文中默认白色）
        scale_modifier=scaling_modifier,                  # 高斯点全局缩放系数
        viewmatrix=viewpoint_camera.world_view_transform, # 世界→视图变换矩阵（将3D点转到相机坐标系）
        projmatrix=viewpoint_camera.full_proj_transform,  # 完整投影矩阵（视图→裁剪→屏幕坐标系）
        sh_degree=pc.active_sh_degree,                    # 球谐函数阶数（用于颜色/光照计算，论文中默认3）
        campos=viewpoint_camera.camera_center,            # 相机在世界坐标系中的位置（用于SH光照计算）
        prefiltered=False,                                # 是否预过滤高斯点（关闭以保留细节）
        debug=False,                                       # 是否开启调试模式（关闭以提升速度）
        # pipe.debug  # 可选：用流水线配置的debug开关（论文中默认关闭）
    )

    # 初始化光栅化器：根据配置实例化渲染核心（底层基于CUDA加速）
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 3. 提取高斯模型核心参数：准备渲染所需的高斯点属性
    means3D = pc.get_xyz               # 高斯点的3D位置（世界坐标系，shape: [num_gaussians, 3]）
    means2D = screenspace_points       # 高斯点的2D屏幕位置（用于梯度传播）
    opacity = pc.get_opacity           # 高斯点的透明度（shape: [num_gaussians, 1]，控制可见性）

    # 4. 处理高斯点的3D协方差：两种模式（Python计算/光栅器计算）
    # 协方差描述高斯点的形状（椭圆），影响渲染时的模糊程度
    scales = None          # 高斯点的缩放参数（若用光栅器计算协方差则需提供）
    rotations = None       # 高斯点的旋转参数（若用光栅器计算协方差则需提供）
    cov3D_precomp = None   # 预计算的3D协方差矩阵（若用Python计算则赋值）
    if pipe.compute_cov3D_python:
        # 模式1：用Python计算协方差（不支持法向量一致性损失，论文中默认不启用）
        # 计算高斯点→世界坐标系的协方差变换矩阵
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height  # 图像宽高
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar            # 相机近/远裁剪面
        # 构建NDC（规范化设备坐标）→像素坐标的变换矩阵
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],  # 水平方向缩放+偏移
            [0, H / 2, 0, (H-1) / 2],  # 垂直方向缩放+偏移
            [0, 0, far-near, near],     # 深度方向映射（近→0，远→1）
            [0, 0, 0, 1]
        ]).float().cuda().T  # 转置为列优先矩阵（符合图形学惯例）
        # 计算世界坐标系→像素坐标系的变换矩阵
        world2pix = viewpoint_camera.full_proj_transform @ ndc2pix
        # 预计算协方差矩阵（投影到像素空间，shape: [num_gaussians, 9]，展开为3x3矩阵）
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9)
    else:
        # 模式2：用光栅器计算协方差（论文中默认启用，效率更高）
        scales = pc.get_scaling    # 高斯点的3D缩放（x/y/z方向，控制椭圆大小）
        rotations = pc.get_rotation # 高斯点的3D旋转（四元数，控制椭圆朝向）
    
    # 5. 处理高斯点的颜色：两种模式（球谐函数计算/预定义颜色）
    pipe.convert_SHs_python = False  # 强制关闭Python端SH转换（用光栅器加速，论文中默认设置）
    shs = None              # 球谐特征（若用光栅器计算颜色则提供）
    colors_precomp = None   # 预计算颜色（若用Python计算或自定义颜色则赋值）
    if override_color is None:
        # 子模式1：用高斯模型的球谐特征计算颜色（论文中核心颜色生成方式）
        if pipe.convert_SHs_python:
            # （关闭状态）Python端计算SH→RGB：将球谐特征转换为颜色（用于调试）
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)  # 相机→高斯点的单位方向向量
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)  # SH转换为RGB
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)  # 调整亮度（SH输出范围[-0.5,0.5]）
        else:
            # （默认）光栅器端计算SH→RGB：直接传递球谐特征给光栅器（效率更高）
            shs = pc.get_features
    else:
        # 子模式2：用自定义颜色（override_color不为None，用于调试或固定颜色场景）
        colors_precomp = override_color
    
    # 6. 执行光栅化：核心渲染步骤，将2D高斯点渲染为图像
    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,          # 高斯点3D位置
        means2D = means2D,          # 高斯点2D屏幕位置（用于梯度）
        shs = shs,                  # 球谐特征（颜色计算用）
        colors_precomp = colors_precomp,  # 预计算颜色（若有则用）
        opacities = opacity,        # 高斯点透明度
        scales = scales,            # 高斯点缩放（协方差计算用）
        rotations = rotations,      # 高斯点旋转（协方差计算用）
        cov3D_precomp = cov3D_precomp  # 预计算协方差（若有则用）
    )
    
    # 7. 构建基础返回结果：包含渲染图和关键筛选信息
    rets =  {
        "render": rendered_image,   # 最终渲染的RGB图像（shape: [3, H, W]）
        "viewspace_points": means2D, # 高斯点的2D屏幕位置（带梯度，用于反向优化）
        "visibility_filter": radii > 0,  # 可见性筛选（半径>0的高斯点为可见，用于后续优化）
        "radii": radii,             # 高斯点在屏幕上的渲染半径（用于可见性判断）
    }
    
    # 8. 计算额外正则化数据（仅在不启用only_rgb_alpha时执行，用于渲染损失计算）
    render_alpha = allmap[1:2]  # 从allmap中提取Alpha透明度图（shape: [1, H, W]，0=透明，1=不透明）
    if only_rgb_alpha:
        # 若仅需RGB和Alpha，补充Alpha后返回（加速调试）
        rets.update({'rend_alpha': render_alpha})
        return rets
    
    # 8.1 处理法向量图：视图空间→世界空间（用于法向量一致性损失Ln，论文公式6）
    render_normal = allmap[2:5]  # 提取视图空间法向量图（shape: [3, H, W]）
    # 转换为世界空间：用相机的世界→视图变换矩阵的逆（转置，因正交矩阵逆=转置）
    render_normal = (render_normal.permute(1,2,0) @ viewpoint_camera.world_view_transform[:3,:3].T).permute(2,0,1)
    
    # 8.2 处理深度图：两种深度（中位数深度、期望深度，用于不同场景）
    render_depth_median = allmap[5:6]  # 提取中位数深度图（shape: [1, H, W]，抗噪性好，适合有界场景）
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)  # 替换NaN为0（避免计算错误）

    render_depth_expected = allmap[0:1]  # 提取期望深度图（shape: [1, H, W]，精度高，适合无界场景）
    render_depth_expected = (render_depth_expected / render_alpha)  # 除以Alpha（去除透明区域的无效深度）
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)  # 替换NaN为0
    
    # 8.3 处理深度畸变图：用于深度畸变损失Ld（论文公式6，约束深度一致性）
    render_dist = allmap[6:7]  # 提取深度畸变图（shape: [1, H, W]，描述深度分布的畸变程度）

    # 8.4 计算表面属性：生成“伪表面”的深度和法向量（用于正则化，提升渲染稳定性）
    # 表面深度：结合期望深度和中位数深度（由pipe.depth_ratio控制权重，论文中根据场景调整）
    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + pipe.depth_ratio * render_depth_median
    
    # 表面法向量：从表面深度图生成（用深度梯度计算，用于法向量一致性正则化）
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)  # 深度→法向量（基于相机内参）
    surf_normal = surf_normal.permute(2,0,1)  # 调整维度为[3, H, W]（符合PyTorch通道优先惯例）
    # 乘以Alpha图（仅保留非透明区域的法向量，透明区域设为0）
    surf_normal = surf_normal * (render_alpha).detach()  # detach()：不反向更新Alpha，仅用其作为掩码

    # 8.5 补充正则化数据到返回结果
    rets.update({
        'rend_alpha': render_alpha,    # Alpha透明度图（用于图像合成）
        'rend_normal': render_normal,  # 世界空间法向量图（用于Ln损失）
        'rend_dist': render_dist,      # 深度畸变图（用于Ld损失）
        'surf_depth': surf_depth,      # 表面深度图（用于正则化）
        'surf_normal': surf_normal     # 表面法向量图（用于正则化）
    })

    return rets