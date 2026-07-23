# %%
import os
import math
import json
import logging
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import seaborn as sns
import networkx as nx
from scipy import stats as scipystats
from scipy.stats import entropy, gaussian_kde
from scipy.spatial.distance import jensenshannon
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from matplotlib import patches, transforms, cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.transforms import Affine2D
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

from shapely import Polygon
from shapely.geometry import Polygon as ShapelyPolygon, LineString
from shapely.affinity import rotate, scale
from shapely.ops import unary_union, nearest_points
from shapely import speedups
from matplotlib.path import Path as MplPath
from ntp_manager import parcellate, SliceMeta, NTPRegion
from correction_para import CorrectionPara

import cv2
# Enable Shapely speedups if available
if speedups.available:
    speedups.enable()
import sys
sys.executable

# %%
ntp_path = {}
ntp_path = list(Path('/mnt/inner-data/ntp/macaque/Mq277-CQW-layer-ROI/').glob('*.ntp'))

ntp_file_path = {
    p.name.split('-')[1]:p
    for p in ntp_path
} 
ntp_file_path
# %%
chips = {'T912','T915','T925','T926'}

chips
# %%
for i, chip in enumerate(chips):
    print(i, chip)
# %%
transfer_path = Path('/mnt/90-connectome/Personal/CQW/motor_snRNA_new/allmotor_cellbins/trans_results/obs/')
file_paths = []  # 创建一个空列表来存储路径
data_dict = {}

for chip in chips:
    file_path = f'{transfer_path}/{chip}.parquet'
    df = pd.read_parquet(file_path)
    print(df.head())
    file_paths.append(file_path)  # 将每个路径添加到列表中
    # dataframes.append(df)
    data_dict[chip] = df
print(data_dict)

# %%

def process_region_and_plot(
        chip_name,
        data_dict,
        # annotation='L3_4_5.1',
        celltype='L3_4_5.1',
        region_need={'F1'},
        *,
        scale: float | None = None,
        target_width: int = 2100
    ):
    """
    处理特定芯片的区域数据并绘制细胞分布图,使用OpenCV绘制。

    如果传入 **scale** (像素/坐标单位)，所有调用都会使用同一缩放系数 -> 全部图片保持一致尺⼦；
    若 **scale** 为 None，则保持旧逻辑：画布宽度固定为 ``target_width`` 并按各自坐标范围自适应缩放。这样单图尺寸一致，但跨图比例可能不同。
    
    Parameters:
    -----------
    chip_name : str
        芯片名称 (如 'T915')
    data_dict : dict
        包含所有芯片数据的字典
    annotation : str
        要分析的细胞类型
    region_need : set
        需要分析的区域集合
        
    Returns:
    --------
    dict
        包含转换后区域的字典
    """
    # 1. 初始化参数
    cp = CorrectionPara.select_one(chip_name)
    if cp is None:
        return
    img_cp = cp

    w, h = cp.w, cp.h
    is_first_monkey = int(chip_name[1:]) < 167
    if is_first_monkey:
        w, h = h, w

    # 2. 获取区域分区数据
    ntp_sm_path = f'/mnt/inner-data/ntp/macaque/Mq277-CQW-layer-ROI/Mq277-{chip_name}-P0.ntp'
    # ntp_sm_path = f'/mnt/inner-data/ntp/macaque/MQ286-motor-ROI/MQ286-{chip_name}-P0.ntp'
    # ntp_sm_path = f'/mnt/inner-data/ntp/macaque/Mq286-motor/Mq286-{chip_name}-P0.ntp'
    sm = parcellate(str(ntp_sm_path), um_per_pixel=0.5, bin_size=1, return_bytes=False, w=w, h=h)
    df = data_dict[chip_name]
    df['x'] = df['spatial_x']
    df['y'] = df['spatial_y']
    df['rx'] = df['spatial_rx']
    df['ry'] = df['spatial_ry']
    # df['x'] = df['spatial_rx']
    # df['y'] = df['spatial_ry']

    # 3. 过滤细胞类型并转换坐标
    # 获取目标细胞类型的坐标
    # df_target = df[df['annotation'] == annotation]
    df_target = df[df['celltype'] == celltype]
    coords_target = np.vstack([df_target['x'].values, df_target['y'].values]).T
    if is_first_monkey:
        coords_target = coords_target[:, [1, 0]]
    transformed_coords_target = cp.warp_point(coords_target, with_offset=True)
    x_transformed_target, y_transformed_target = transformed_coords_target[:, 0], transformed_coords_target[:, 1]

    # 获取其他细胞类型的坐标
    # df_others = df[df['annotation'] != annotation]
    df_others = df[df['celltype'] != celltype]
    coords_others = np.vstack([df_others['x'].values, df_others['y'].values]).T
    if is_first_monkey:
        coords_others = coords_others[:, [1, 0]]
    transformed_coords_others = cp.warp_point(coords_others, with_offset=True)
    x_transformed_others, y_transformed_others = transformed_coords_others[:, 0], transformed_coords_others[:, 1]

    # 4. 区域变换
    transformed_regions = []
    transformed_regions_dict = {}
    
    for region in sm.regions:
        if hasattr(region.polygon, 'geoms'):
            polygons = list(region.polygon.geoms)
        else:
            polygons = [region.polygon]
        
        for poly in polygons:
            points = np.array(poly.exterior.coords)
            if is_first_monkey:
                points = points[:, [1, 0]]
            warped_points = cp.warp_point(points, with_offset=False)
            new_polygon = Polygon(warped_points)
            transformed_regions.append(new_polygon)
            transformed_regions_dict[region.label.name] = new_polygon

    # 5. 提取需要的区域
    region_poly = {}
    for i, r in enumerate(transformed_regions_dict):
        r_name = r.split('-')[1]
        if r_name in region_need:
            region_poly[i] = transformed_regions_dict[r]

    # 6. 获取多边形坐标并判断点是否在区域内
    polygon_coords = []
    paths = []
    for poly in region_poly.values():
        coords = np.array(poly.exterior.coords)
        polygon_coords.append(coords)
        paths.append(MplPath(coords))

    x_transformed_target = df_target['rx'].values
    y_transformed_target = df_target['ry'].values
    x_transformed_others = df_others['rx'].values
    y_transformed_others = df_others['ry'].values
    # x_transformed_target, y_transformed_target = df_target[['rx']].values, df_target[['ry']].values
    # 判断目标细胞类型点是否在区域内
    points_target = np.column_stack((x_transformed_target, y_transformed_target))
    mask_target = np.zeros(len(points_target), dtype=bool)
    for path in paths:
        mask_target = mask_target | path.contains_points(points_target)

    # 判断其他细胞类型点是否在区域内
    points_others = np.column_stack((x_transformed_others, y_transformed_others))
    mask_others = np.zeros(len(points_others), dtype=bool)
    for path in paths:
        mask_others = mask_others | path.contains_points(points_others)

    # 7. 使用OpenCV绘图
    # 计算坐标范围以确定图像大小
    # 检查是否有点在区域内
    if np.sum(mask_target) == 0 and np.sum(mask_others) == 0:
        print(f"No cells found in region {region_need} for chip {chip_name} and celltype {celltype}")
        return {}
    
    # 合并有效的坐标点
    valid_coords = []
    if np.sum(mask_target) > 0:
        valid_coords.append(points_target[mask_target])
    if np.sum(mask_others) > 0:
        valid_coords.append(points_others[mask_others])
    
    all_coords = np.vstack(valid_coords)

    min_x, min_y = np.min(all_coords, axis=0)
    max_x, max_y = np.max(all_coords, axis=0)
    
    width  = int(math.ceil((max_x - min_x) * scale)) + 100
    height = int(math.ceil((max_y - min_y) * scale)) + 100
    
    # 创建透明背景图像
    img = np.zeros((height, width, 4), dtype=np.uint8)  # 4通道RGBA图像
    
    # 坐标转换函数
    def transform_coords(x, y):
        return (int((x - min_x) * scale), int((y - min_y) * scale))
    
    # 绘制目标细胞类型点（红色，不透明）
    for x, y in points_target[mask_target]:
        px, py = transform_coords(x, y)
        cv2.circle(img, (px, py), 30, (0, 0, 255, 255), -1)  # alpha=255表示完全不透明
    
    # 绘制区域边界（黑色，不透明）
    for coords in polygon_coords:
        points = []
        for x, y in coords:
            px, py = transform_coords(x, y)
            points.append([px, py])
        points = np.array(points, dtype=np.int32)
        cv2.polylines(img, [points], True, (0, 0, 0, 255), 5)

    # 保存图像（使用PNG格式以支持透明度）
    save_path = f'/mnt/90-connectome/Personal/CQW/motor_snRNA_new/article/fig2/transfer_fig/mq277_fig2/{chip_name}_{celltype}_{region_need}.png'


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)
    plt.imshow(img)
    plt.show()


# %%
# celltype = 'L3_5-IT.05'
celltype = 'Oli.03'
# celltype = 'L6-IT(Car3).02'

args = [
    ('T915', {'F1'}),
    ('T915', {'F2'}),
    ('T912', {'F3'}),
    ('T926', {'F4'}),
    ('T926', {'F5'}),
    ('T912', {'F6'}),
    ('T915', {'F7'}),
]

for chip, region_need in args:
    p = process_region_and_plot(chip, data_dict, celltype=celltype, region_need=region_need, scale=0.5)

