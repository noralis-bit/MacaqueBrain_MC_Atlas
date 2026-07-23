# %%
import s2a
import os,re
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
from tqdm.auto import tqdm

import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Iterable
import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import s2a
import scanpy as sc
import scipy
import squidpy as sq
from sklearn import metrics
from tqdm import tqdm

# %%

adatas = s2a.process_stereo_folder(
    '/mnt/inner-data/sde/total_gene_2D/macaque-20251023-motor-new/',
    obs_src = 'cell_label',
    # spot_bin= 'cell_label', # cell_label or spot_bin
    # spot_bin_size= 200, # if you are using spot_bin, spot_bin_size is required
    save_to='/data/data0-1/transfer/motor/stereo/macaque-20251017-allmotor/',
    workers = 1,
    verbose=True
)
print(adatas)

# %%
data_dir = "/data/data0-1/transfer/motor/stereo/macaque-20251017-allmotor/"

def process_h5ad_file(file_path, output_dir=None):
    """
    处理单个h5ad文件 - 只添加spatial坐标到obs中
    
    Parameters:
    -----------
    file_path : str
        h5ad文件的路径
    output_dir : str, optional
        输出目录，如果为None则使用原目录
    """
    print(f"正在处理文件: {os.path.basename(file_path)}")
    
    # 读取h5ad文件
    adata = sc.read_h5ad(file_path)

    spatial_coords = adata.obsm['spatial']
    spatial_coords_r = adata.obsm['spatial_r']

    adata.obs['spatial_x'] = spatial_coords[:, 0]
    adata.obs['spatial_y'] = spatial_coords[:, 1]
    adata.obs['spatial_rx'] = spatial_coords_r[:, 0]
    adata.obs['spatial_ry'] = spatial_coords_r[:, 1]
    print(f"已将spatial坐标添加到obs中: spatial_x, spatial_y, spatial_rx, spatial_ry")

    
    # 保存处理后的数据
    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件名
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}.h5ad")
    
    # 保存处理后的文件
    adata.write(output_file)
    print(f"处理完成，保存到: {output_file}")
    
    return adata

def batch_process_h5ad_files(data_dir, output_dir=None, pattern="*.h5ad"):
    """
    批量处理目录中的所有h5ad文件
    
    Parameters:
    -----------
    data_dir : str
        包含h5ad文件的目录路径
    output_dir : str, optional
        输出目录，如果为None则使用原目录
    pattern : str
        文件匹配模式
    """
    data_path = Path(data_dir)
    
    # 查找所有h5ad文件
    h5ad_files = list(data_path.glob(pattern))
    
    if not h5ad_files:
        print(f"在目录 {data_dir} 中没有找到匹配 {pattern} 的文件")
        return
    
    print(f"找到 {len(h5ad_files)} 个h5ad文件")
    
    # 批量处理文件
    processed_files = []
    for file_path in tqdm(h5ad_files, desc="处理h5ad文件"):
        try:
            adata = process_h5ad_file(str(file_path), output_dir)
            processed_files.append(str(file_path))
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue
    
    print(f"成功处理了 {len(processed_files)} 个文件")
    return processed_files

# 执行批量处理
if __name__ == "__main__":
    # 设置输出目录（可选）
    output_directory = "/data/data0-1/transfer/motor/stereo/macaque-20251017-allmotor/processed/"

    # 创建输出目录
    os.makedirs(output_directory, exist_ok=True)
    
    # 批量处理所有h5ad文件
    processed_files = batch_process_h5ad_files(data_dir, output_directory)
    
    print("所有文件处理完成！")
