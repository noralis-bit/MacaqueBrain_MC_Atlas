# %%
import scanpy as sc
import pandas as pd
# import squidpy as sq
import matplotlib.pyplot as plt
import math
import os
import polars as pl
# import cv2
from ntp_manager import parcellate, SliceMeta
from pathlib import Path 
from matplotlib.transforms import Affine2D
import matplotlib.patches as patches
import json
import concurrent.futures
import logging

from matplotlib import patches, transforms
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, scale
import json
from typing import List, Tuple, Optional
from shapely import speedups
from correction_para import CorrectionPara
import tqdm
from tqdm import tqdm

# %%

csv_files = {}
csv_files = list(Path('/mnt/90-connectome/Personal/CQW/motor_snRNA_new/allmotor_cellbins/trans_results/all_trans_results/').glob('*.csv'))
csv_files_map = {
    p.name.split('_')[0].split('-')[1]: p
    for p in csv_files
}

h5ad_files = list(Path('/mnt/90-connectome/Personal/CQW/motor_snRNA_new/allmotor_cellbins/').glob('*.h5ad'))
h5ad_files_map = {
    p.name.split('_')[2]: p
    for p in h5ad_files
}
chip_names = list(h5ad_files_map.keys())
h5ad_files_map


# %%
regions_to_highlight = [
    'F1-l1', 'F1-l2', 'F1-l3', 'F1-l5', 'F1-l6',
    'F2-l1', 'F2-l2', 'F2-l3', 'F2-l5', 'F2-l6',
    'F3-l1', 'F3-l2', 'F3-l3', 'F3-l5', 'F3-l6',
    'F4-l1', 'F4-l2', 'F4-l3', 'F4-l5', 'F4-l6',
    'F5-l1', 'F5-l2', 'F5-l3', 'F5-l5', 'F5-l6',
    'F6-l1', 'F6-l2', 'F6-l3', 'F6-l5', 'F6-l6',
    'F7-l1', 'F7-l2', 'F7-l3', 'F7-l5', 'F7-l6',
    'F1-s','F2-s','F3-s','F4-s','F5-s','F6-s','F7-s',
    'RF1-s','RF2-s','RF3-s','RF4-s','RF5-s','RF6-s','RF7-s'
]

for chip_name in tqdm(chip_names):
 
    adata = sc.read_h5ad(h5ad_files_map[chip_name])
    df = pd.read_csv(csv_files_map[chip_name],index_col='cell')
    df['celltype'] = df['celltype_pred']
    df['celltype'] = df['celltype'].str.replace('\/', '_', regex=True)
    df['subclass'] = df['celltype'].str.split('.').str[0]

# 创建一个新的 'class' 列，默认值为 'Other'
    # df['class'] = 'Other'
    adata.obs['celltype'] = df['celltype']
    adata.obs['subclass'] = df['subclass']
    
    # mask = adata.obs['region_name'].isin(regions_to_highlight)
    # adata_highlighted = adata[mask].copy()
    obs_save_dir = Path('/mnt/90-connectome/Personal/CQW/motor_snRNA_new/allmotor_cellbins/trans_results/obs/')
    adata_save_dir = Path('/mnt/90-connectome/Personal/CQW/motor_snRNA_new/allmotor_cellbins/trans_results/datasets/')
    obs_save_dir.mkdir(parents=True, exist_ok=True)
    adata_save_dir.mkdir(parents=True, exist_ok=True)
    
    obs_save_path = obs_save_dir/ f"{chip_name}.parquet"
    adata_save_path = adata_save_dir/ f"{chip_name}.h5ad"
    adata.obs.to_parquet(obs_save_path, index=True)
    adata.write_h5ad(adata_save_path)

# %%
