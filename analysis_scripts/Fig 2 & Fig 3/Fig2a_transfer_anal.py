# %%
from cProfile import label
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import json
import logging
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Optional

# Data processing
import polars as pl
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from scipy import stats as scipystats
from scipy.stats import entropy, chi2_contingency, fisher_exact
from scipy.spatial.distance import jensenshannon
from statsmodels.stats.multitest import multipletests

# Visualization
from matplotlib import patches, transforms, cm
from matplotlib.transforms import Affine2D
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

# Spatial analysis
from ntp_manager import parcellate, SliceMeta
from shapely.geometry import Polygon
from shapely.affinity import rotate, scale
from shapely import speedups
import networkx as nx
import matplotlib.colors as mcolors
# Custom modules
from correction_para import CorrectionPara

import itertools
# %%
class_mapping = {
    'Excitory_neuron': ['L2_3_IT', 'L3_5_IT','L4-IT(like)', 'L5_IT', 'L5_ET(PT)', 'L5_6_NP', 'L6_IT', 'L6_CT', 'L6_IT(Car3)', 'L6b',
                        'L2_3', 'L3_4', 'L5_6', 'L2_3_4', 'L4', 'L2', 'L4_5_6', 'L4_5', 'L6', 'L3_4_5','L5_ET',
                        'L2', 'L2/3', 'L2/3/4', 'L3/4', 'L3/4/5', 'L4','L4/5', 'L4/5/6', 'L5/6', 'L6'
                        ],
    'Inhibitory_neuron': ['PV', 
    'SST', 'VIP', 'SNCG', 'LAMP5', 'PV_CHC',
                         'PVALB', 'VIP', 'LAMP5', 'SST', 'RELN', 'VIP_RELN', 'PV_CHC'],
    'Non_neuron': ['OPC', 'OLI', 'VLMC_EC', 'MIC', 'AST', 'VLMC',
                   'OPC', 'OLG', 'ASC', 'VLMC', 'MG','Mic','Oli','Ast','']
}

# %%
regions = [
    'L-F1-l1', 'L-F1-l2', 'L-F1-l3', 'L-F1-l5', 'L-F1-l6',
    'L-F2-l1', 'L-F2-l2', 'L-F2-l3', 'L-F2-l5', 'L-F2-l6',
    'L-F3-l1', 'L-F3-l2', 'L-F3-l3', 'L-F3-l5', 'L-F3-l6',
    'L-F4-l1', 'L-F4-l2', 'L-F4-l3', 'L-F4-l5', 'L-F4-l6',
    'L-F5-l1', 'L-F5-l2', 'L-F5-l3', 'L-F5-l5', 'L-F5-l6',
    'L-F6-l1', 'L-F6-l2', 'L-F6-l3', 'L-F6-l5', 'L-F6-l6',
    'L-F7-l1', 'L-F7-l2', 'L-F7-l3', 'L-F7-l5', 'L-F7-l6',
    'R-F1-l1', 'R-F1-l2', 'R-F1-l3', 'R-F1-l5', 'R-F1-l6',
    'R-F2-l1', 'R-F2-l2', 'R-F2-l3', 'R-F2-l5', 'R-F2-l6',
    'R-F3-l1', 'R-F3-l2', 'R-F3-l3', 'R-F3-l5', 'R-F3-l6',
    'R-F4-l1', 'R-F4-l2', 'R-F4-l3', 'R-F4-l5', 'R-F4-l6',
    'R-F5-l1', 'R-F5-l2', 'R-F5-l3', 'R-F5-l5', 'R-F5-l6',
    'R-F6-l1', 'R-F6-l2', 'R-F6-l3', 'R-F6-l5', 'R-F6-l6',
    'R-F7-l1', 'R-F7-l2', 'R-F7-l3', 'R-F7-l5', 'R-F7-l6',
    'R-F1-s', 'R-F2-s', 'R-F3-s', 'R-F4-s', 'R-F5-s', 'R-F6-s', 'R-F7-s',
    'RF1-s','RF2-s','RF3-s','RF4-s','RF5-s','RF6-s','RF7-s'
]

region_order = ['F1','F2','F3','F4','F5','F6','F7']
subclass_order = [
    "L2",
    "L2_3",
    "L2_3_4",
    "L3_4",
    "L3_4_5",
    "L4",
    "L4_5",
    "L4_5_6",
    "L5_6",
    "L6",
    "LAMP5",
    "RELN",
    "VIP_RELN",
    "VIP",
    "PV_CHC",
    "PVALB",
    "SST",
    "ASC",
    "OPC",
    "OLG",
    "MG",
    "EC",
    "VLMC"
] 
layers_order = ['l6','l5','l3','l2','l1']
region_colors = {
    'F1': '#1F77B4',
    'F2': '#FF7F0E',
    'F3': '#2CA02C',
    'F4': '#D62728',
    'F5': '#9467BD',
    'F6': '#8C564B',
    'F7': '#E377C2'
}

# %%
df_info = pd.read_excel('/mnt/90-connectome/Personal/CQW/motor_snRNA_new/motor_cortex_libinfo.xlsx',sheet_name='st-rna.matrix')
# df_info = df_info[df_info['Animal'] != 'MQC286']
# df_info = df_info[df_info['Animal'] != 'MQ277']
# df_info = df_info[df_info['Animal'] != 'MQC121']
# df_info = df_info[df_info['Animal'] != 'MQC179']
chips = df_info['Chipid'].tolist()
chips = [chip for chip in chips if chip != 'T916']
chips = [chip for chip in chips if chip != 'T908']

# %%
### 生成areasize
raw_dirs = [
    # '/mnt/inner-data/ntp/macaque/Mq179-motor/region-mask/',
    # '/mnt/inner-data/ntp/macaque/Mq286-CQW/region-mask/',
    # '/mnt/inner-data/ntp/macaque/Mq121-motor/region-mask/',
    # '/mnt/inner-data/ntp/macaque/Mq277-CQW-newlayer/region-mask/'
    '/mnt/inner-data/ntp/macaque/Mq1234_all_motor/region-mask/'

]

# 假设 chips 是一个包含多个 chip 名称的列表，例如：
# chips = ['Mq179', 'Mq286', 'Mq121', 'Mq277']
# 请确保你在运行这段代码前定义了 chips

files = []
for dir in raw_dirs:
    curr_files = list(Path(dir).glob('*.csv'))
    curr_files_obj = [{
        'name': p.name.split('-')[0],  # 以 "-" 拆分文件名，取第一个部分作为名称
        'path': str(p)                 # 路径转为字符串以便存储
    } for p in curr_files]
    files.extend(curr_files_obj)

# 创建 DataFrame
df = pd.DataFrame(files)
# 根据 chips 列表筛选
df_filt = df[df['name'].isin(chips)]

# 按每个 chip 逐个处理
areasize = pd.DataFrame()

for chip in chips:
    chip_data = df_filt[df_filt['name'] == chip]
    
    if chip_data.empty:
        print(f'[Warning] No CSV file found for chip: {chip}')
        continue  # 跳过当前循环
    
    print(f'Processing chip: {chip}')
    
    # 读取 CSV 数据
    try:
        data = pd.read_csv(chip_data['path'].values[0])
        data = data[data['origin_name'].isin(regions)][['origin_name', 'area_size']]
        data.columns = ['region', 'area_size']
        data['area_size'] = data['area_size']*0.25*10**-6
        data['chip'] = chip  # 可选：加上 chip 列便于识别来源
        areasize = pd.concat([areasize, data], ignore_index=True)
    except Exception as e:
        print(f'[Error] Failed to read/process {chip_data["path"].values[0]}: {e}')
# %%
areasize.to_csv('/mnt/90-connectome/Personal/CQW/motor_snRNA_new/article/fig2/areasize_region_layer_chip_star_20260705.csv')

# %%
### areasize：每张芯片的每个gene_area的areasize
### areasize_stat：每张芯片的每个region的areasize
### areasize_sum：每个region的areasize
### areasize_filt：每个region的areasize，去掉矢状面芯片

areasize['motor_cortex_region'] = areasize['region'].str.split('-').str[1]
areasize['layer'] = areasize['region'].str.split('-').str[2]

areasize_filt = areasize[areasize['layer'] != 's']

areasize_stat = areasize.groupby(['motor_cortex_region', 'chip'])['area_size'].sum().reset_index()
areasize_stat.columns = ['origin_name', 'chip', 'area_size']
areasize_sum = areasize_stat.groupby(['origin_name'])['area_size'].sum().reset_index()


areasize_layer = areasize.groupby(['motor_cortex_region','layer'])['area_size'].sum().reset_index()
areasize_layer = areasize_layer[areasize_layer['layer'] != 's']
areasize_layer['region_name'] = 'L-'+ areasize_layer['motor_cortex_region'] + '-' + areasize_layer['layer']
# areasize.to_csv('/mnt/90-connectome/Personal/CQW/motor_snRNA_new/article/fig2/areasize_region_layer_chip_stat.csv')
# areasize_stat.to_csv('/mnt/90-connectome/Personal/CQW/motor_snRNA_new/article/fig2/areasize_region_chip_stat.csv')
# # %%
# ### subclass from zzy
# ##############################################################################################################
# ##############################################################################################################
# ##############################################################################################################
# ##############################################################################################################
# ##############################################################################################################


# %%
# del list  # 删除你定义的同名变量，恢复内置的 list 函数
csv_files = {}
# csv_files_1 = list(Path('/mnt/90-connectome/Personal/CQW/motor_snRNA_new/cellpaper_h5ad_spatial/obs/').glob('*.parquet'))

# csv_files_2 = list(Path('/mnt/90-connectome/Personal/CQW/motor_snRNA_new/mq277_cellbins/tran_result/macaque4_Motor_266c_CTXsn_csv/obs/').glob('*.parquet'))

# csv_files1 = list(Path('/mnt/90-connectome/Personal/CQW/motor_snRNA_new/mq179_cellbins/tran_result/macaque1_Motor_107c_csv/obs/').glob('*.parquet'))
# csv_files2 = list(Path('/mnt/90-connectome/Personal/CQW/motor_snRNA_new/mq286_cellbins/tran_result/macaque2_Motor_107c_csv/obs/').glob('*.parquet'))
# csv_files3 = list(Path('/mnt/90-connectome/Personal/CQW/motor_snRNA_new/mq121_cellbins/tran_result/macaque3_Motor_107c_csv/obs/').glob('*.parquet'))
# csv_files4 = list(Path('/mnt/90-connectome/Personal/CQW/motor_snRNA_new/mq277_cellbins/tran_result/macaque4_Motor_107c_csv/obs/').glob('*.parquet'))
csv_files = list(Path('/mnt/90-connectome/Personal/CQW/motor_snRNA_new/allmotor_cellbins/trans_results/obs/').glob('*.parquet'))

# 合并两个路径下的文件
# csv_files = csv_files1 + csv_files2 + csv_files3 + csv_files4

csv_files_map = {
    p.name.split('.')[0]: p
    for p in csv_files
}
csv_files_map

# %%
data_dict = {}
for chip_name in chips:
    df = pd.read_parquet(csv_files_map[chip_name])
    data_dict[chip_name] = df

# %%
class_mapping = {
    'Excitory_neuron': ['L2_3-IT', 'L3_5-IT', 'L5-IT', 'L5-ET', 'L5_6-NP', 'L6-IT', 'L6-CT', 'L6-IT(Car3)', 'L6b',
                        'L2_3', 'L3_4', 'L5_6', 'L2_3_4', 'L4', 'L2', 'L4_5_6', 'L4_5', 'L6', 'L3_4_5','L5-ET(PT)',
                        'L2', 'L2/3', 'L2/3/4', 'L3/4', 'L3/4/5', 'L4','L4/5', 'L4/5/6', 'L5/6', 'L6'
                        ],
    'Inhibitory_neuron': ['PV', 'SST', 'VIP', 'SNCG', 'LAMP5', 'PV_CHC',
                         'PVALB', 'VIP', 'LAMP5', 'SST', 'RELN', 'VIP_RELN', 'PV_CHC'],
    'Non_neuron': ['OPC', 'OLI', 'VLMC_EC', 'MIC', 'AST', 'VLMC',
                   'OPC', 'OLG', 'ASC', 'VLMC', 'MG','Mic','Oli','Ast','']
}
# %%
subclass_order = [
    "L2_3-IT",
    "L3_5-IT",
    "L4-IT(like)",
    "L5-IT",
    "L5-ET(PT)",
    "L5_6-NP",
    "L6-IT",
    "L6-CT",
    "L6-IT(Car3)",
    "L6b",
    "PV",
    "RELN",
    "SST",
    "VIP",
    "SNCG",
    "LAMP5",
    "PV_CHC",
    "OPC",
    "OLI",
    "VLMC_EC",
    "MIC",
    "AST"
]

# %%
# Create reverse mapping from subclass to class
subclass_to_class = {}
for class_name, subclasses in class_mapping.items():
    for subclass in subclasses:
        subclass_to_class[subclass] = class_name

for chip_name in chips:
    df = data_dict[chip_name]
    df['SubClass'] = df['subclass']
    df['class'] = df['subclass'].map(subclass_to_class)
    df['gene_area'] = df['region_name']
    data_dict[chip_name] = df

areasize_stat.columns = ['region', 'chip', 'area_size']
areasize_sum.columns = ['region', 'area_size']

# %%
### proportion
# 首先计算每个chip中每个region内各celltype的比例
chip_region_celltype_proportions = {}
for chip, df in data_dict.items():
        # 确保数据框有必要的列
    df['region'] = df['gene_area'].str.split('-').str[1]
    df['layer'] = df['gene_area'].str.split('-').str[2]
    df['region_name'] = df['region'] + '-' + df['layer']
    # 计算每个region中每个celltype的计数
    region_celltype_counts = df.groupby(['region', 'celltype']).size().reset_index(name='count')
    # 计算每个region的总细胞数
    region_totals = region_celltype_counts.groupby('region')['count'].sum().reset_index(name='total')
    # 合并以计算比例
    region_celltype_counts = region_celltype_counts.merge(region_totals, on='region')
    chip_area = areasize_stat.loc[areasize_stat['chip'] == chip]
    region_celltype_counts = pd.merge(region_celltype_counts, chip_area, on='region')
    region_celltype_counts['density'] = region_celltype_counts['count'] / region_celltype_counts['area_size']
    region_celltype_counts['proportion'] = region_celltype_counts['count'] / region_celltype_counts['total']
    # region_celltype_counts = region_celltype_counts[region_celltype_counts['count'] > 10]
    # 存储到字典中
    if chip not in chip_region_celltype_proportions:   
        chip_region_celltype_proportions[chip] = []
    chip_region_celltype_proportions[chip].append(region_celltype_counts)
    print(chip)
# %%
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

### density计算，每个region的每个celltype的密度

density_region = pd.DataFrame()
for i in chip_region_celltype_proportions.keys():
    df = chip_region_celltype_proportions[i][0]
    density_region = pd.concat([density_region, df], ignore_index=True)

# density_region.to_csv('/mnt/90-connectome/Personal/CQW/motor_snRNA_new/article/fig2/mq1234_density_region.csv')

# %%
# 绘制L5-ET(PT).02(Betz-Like)细胞类型的density图
celltype_sub = density_region.loc[density_region['celltype'] == 'L5-ET(PT).02(Betz-Like)'].copy()

# celltype_sub.to_csv('/mnt/90-connectome/Personal/CQW/motor_snRNA_new/article/fig2/Betz-Like-density.csv')

# %%
### 过滤counts总数小于1000的celltype
### density
density_region_filt = density_region.copy()
density_region_sum = density_region_filt.groupby(['region', 'celltype'])['count'].sum().reset_index()
density_region_sum1 = density_region_sum.merge(areasize_sum, on='region')
density_region_sum1['density'] = density_region_sum1['count'] / density_region_sum1['area_size']
# Calculate maximum count and density for each celltype across regions
density_region_sum1['max_density'] = density_region_sum1.groupby('celltype')['density'].transform('max')
density_region_sum1['max_count'] = density_region_sum1.groupby('celltype')['count'].transform('max')
# density_region_sum1 = density_region_sum1.loc[density_region_sum1['max_count'] > 1000]

# Convert density_region_sum1 to DataFrame if it's not already
density_region_sum1 = pd.DataFrame(density_region_sum1)

median_densities = {}
for celltype in density_region_sum1['celltype'].unique():
    group = density_region_sum1[density_region_sum1['celltype'] == celltype]
    densities = group['density'].values
    if len(densities) > 1:  # 确保有多于一个值
        densities_without_max = densities[densities != max(densities)]
        if len(densities_without_max) > 0:  # 确保去除最大值后还有值
            median_densities[celltype] = np.median(densities_without_max)
        else:
            median_densities[celltype] = np.nan  # 或者适当的默认值
    else:
        median_densities[celltype] = np.nan  # 或者适当的默认值

# 将计算的中位数赋值给新列
density_region_sum1['median_density'] = density_region_sum1['celltype'].map(median_densities)

# density_region_sum1['max_fold_to_min_density'] = density_region_sum1.groupby('celltype')['fold_to_min_density'].transform('max')
density_region_sum1['fold_to_median_density'] = density_region_sum1['density'] / density_region_sum1['median_density']
density_region_sum1['max_fold_to_min_density'] = density_region_sum1.groupby('celltype')['fold_to_median_density'].transform('max')
desnsity_filt = density_region_sum1[density_region_sum1['max_fold_to_min_density'] > 1.3]
# desnsity_filt = density_region_sum1.loc[density_region_sum1['max_density'] > 3]
desnsity_filt = desnsity_filt.loc[desnsity_filt['max_density'] > 3]

# %%
# 修改celltype名称
desnsity_filt.loc[desnsity_filt['celltype'] == 'L5-ET(PT).02(Betz-Like)', 'celltype'] = 'L5-ET(PT).02'
desnsity_filt.loc[desnsity_filt['celltype'] == 'L3_5-IT.03', 'celltype'] = 'L4-IT(like).00'
desnsity_filt.loc[desnsity_filt['celltype'] == 'L3_5-IT.04', 'celltype'] = 'L4-IT(like).01'
desnsity_filt.loc[desnsity_filt['celltype'] == 'L3_5-IT.05', 'celltype'] = 'L4-IT(like).02'

desnsity_filt
############################################################################################################

# %%
# 创建数据透视表，行为region，列为celltype，值为density
pivot_data = desnsity_filt.pivot(index='region', columns='celltype', values='density')

# subclass_order
# 按照celltype名称中'.'分割后的第一部分，根据subclass_order进行排序
celltype_subclass_map = {col: col.split('.')[0] for col in pivot_data.columns}
sorted_celltypes = sorted(pivot_data.columns, key=lambda x: subclass_order.index(celltype_subclass_map[x]) if celltype_subclass_map[x] in subclass_order else len(subclass_order))
# 按排序后的celltype顺序重新排列列
pivot_data = pivot_data.reindex(columns=sorted_celltypes)

# pivot_data = proportion_filt.pivot(index='region', columns='celltype', values='fold_to_median_proportion')
# 对每列进行z-score标准化
zscore_data = pivot_data.apply(lambda x: (x - x.mean()) / x.std())
zscore_data = zscore_data.fillna(0)
# 创建蓝-白-红的渐变
colors = ['blue', 'white', 'red'] 
custom_cmap = mcolors.LinearSegmentedColormap.from_list('blue_red', colors, N=100)

# 设置字体为Arial和字体大小
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['pdf.fonttype'] = 42  # 设置PDF中的字体为可编辑的TrueType字体
plt.rcParams['ps.fonttype'] = 42   # 设置PostScript中的字体为可编辑的TrueType字体

fig = plt.figure(figsize=(25, 4))
gs = fig.add_gridspec(1, 2, width_ratios=[40, 0.5], wspace=0.05)

ax = fig.add_subplot(gs[0, 0])
cbar_ax = fig.add_subplot(gs[0, 1]) 

sns.heatmap(
    zscore_data,
    cmap=custom_cmap,
    center=0,
    annot=False,
    fmt='.2f',
    vmin=-2.5,
    vmax=2.5,
    ax=ax,
    cbar_ax=cbar_ax,
    cbar_kws={'label': 'Z-score of Density'}
)

ax.set_xlabel('Cell Types',fontsize=15, fontfamily='Arial')
ax.set_ylabel('Regions',fontsize=15, fontfamily='Arial')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontfamily='Arial')
plt.setp(ax.get_yticklabels(), fontfamily='Arial')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# plt.tight_layout()
# plt.savefig('/mnt/90-connectome/Personal/CQW/motor_snRNA_new/article/fig2/density_region_heatmap-newtrans.pdf',dpi=300)
plt.show()
# %%
