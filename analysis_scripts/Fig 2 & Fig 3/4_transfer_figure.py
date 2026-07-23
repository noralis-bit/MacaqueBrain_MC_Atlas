# %%
import argparse
from dataclasses import dataclass, field
import json
import csv
import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
from ntp_manager import parcellate
from pathlib import Path
from correction_para import CorrectionPara
from typing import List, Optional, Dict, Any
try:
    import yaml  # optional
except Exception:
    yaml = None
import math
import re

# %%

base_colors = {
    'F1': '#1f77b4',
    'F2': '#ff7f0e',
    'F3': '#2ca02c',
    'F4': '#d62728',
    'F5': '#9467bd',
    'F6': '#8c564b',
    'F7': '#e377c2',
}

gradient_factors = {
    'l1': 0.2,
    'l2': 0.3,
    'l3': 0.4,
    'l4': 0.5,
    'l5': 0.6,
    'l6': 0.7,
    's': 0.8,
}


@dataclass
class Config:
    species: str = 'macaque'
    animal_id: str = 'MQ277'  # for metadata filtering
    monkey_id: str = 'Mq277'  # for ntp path naming
    animal_ids: Optional[List[str]] = None  # support multiple animals
    monkey_ids: Optional[List[str]] = None  # optional 1:1 with animal_ids
    delete_chip_list: List[str] = field(default_factory=lambda: ['T572', 'T26'])
    csv_files_dir: str = '/mnt/90-connectome/Personal/CQW/motor_snRNA_new/allmotor_cellbins/tran_result/macaque4_Motor_107c_csv/obs/'
    libinfo_xlsx: str = '/mnt/90-connectome/Personal/CQW/motor_snRNA_new/motor_cortex_libinfo.xlsx'
    libinfo_sheet: str = 'st-rna.matrix'
    um_per_pixel: float = 0.5
    ntp_version: str = 'Mq1234_all_motor'
    ntp_root_dir: str = '/mnt/inner-data/ntp'
    edge: float = 0.02
    save_dir: str = '/mnt/90-connectome/Personal/CQW/motor_snRNA_new/article/fig2/cor_trans/'
    scalebar_um: int = 1000
    lock_global_extent: bool = True
    max_cols: int = 9
    target_region_regex: str = r'^L-F[1-7]-l[1-7]$'
    dpi: int = 300
    celltypes: Optional[List[str]] = None  # if None, auto-detect
    log_progress: bool = True
    precompute_workers: int = 0  # 0/1 means no parallel; >1 use threads
    # benchmark controls
    limit_chips: Optional[int] = None
    limit_celltypes: Optional[int] = None
    skip_if_exists: bool = True
    save_formats: Optional[List[str]] = field(default_factory=lambda: ['pdf', 'png'])


def _build_ntp_path(cfg: Config, chip_name: str) -> str:
    return f"{cfg.ntp_root_dir}/{cfg.species}/{cfg.ntp_version}/{cfg.monkey_id}-{chip_name}-P0.ntp"


def _load_chip_names(cfg: Config) -> List[str]:
    df_info = pd.read_excel(cfg.libinfo_xlsx, sheet_name=cfg.libinfo_sheet)
    df_info = df_info[df_info['Animal'] == cfg.animal_id]
    chip_names = df_info['Chipid'].tolist()
    exclude = set(cfg.delete_chip_list or [])
    chip_names = [chip for chip in chip_names if chip not in exclude]
    return chip_names


def _derive_monkey_id_from_animal(animal_id: str) -> str:
    # Convert like MQ277 -> Mq277 if needed
    if animal_id and len(animal_id) >= 3 and animal_id[:2].upper() == 'MQ':
        return 'Mq' + animal_id[2:]
    return animal_id


def _load_data_dict(cfg: Config, chip_names: List[str]) -> dict:
    base_dir = Path(cfg.csv_files_dir)
    if not base_dir.exists():
        if cfg.log_progress:
            print(f"[data] directory not found: {base_dir}")
        return {}
    # Support both parquet and csv; prefer parquet when both exist
    parquet_files = list(base_dir.glob('*.parquet'))
    csv_files = list(base_dir.glob('*.csv'))
    file_map = {p.stem: p for p in parquet_files}
    for p in csv_files:
        if p.stem not in file_map:
            file_map[p.stem] = p
    if cfg.log_progress:
        print(f"[data] discovered files: parquet={len(parquet_files)} csv={len(csv_files)} unique_bases={len(file_map)}")
    data_dict = {}
    missing = []
    for chip_name in chip_names:
        p = file_map.get(chip_name)
        if p is None:
            missing.append(chip_name)
            continue
        if p.suffix.lower() == '.parquet':
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
        data_dict[chip_name] = df
    if cfg.log_progress:
        print(f"[data] loaded chips: {len(data_dict)}/{len(chip_names)}")
        if missing:
            print(f"[data] missing {len(missing)} chips without data files (e.g., {missing[:5]})")
    return data_dict


def _select_xy_columns(df: pd.DataFrame) -> Optional[tuple]:
    candidates = [
        ('spatial_rx', 'spatial_ry'),
        ('rx', 'ry'),
        ('spatial_x', 'spatial_y'),
        ('x', 'y'),
    ]
    cols = set(df.columns)
    for x_col, y_col in candidates:
        if x_col in cols and y_col in cols:
            return x_col, y_col
    return None


def _load_config_file(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    suffix = p.suffix.lower()
    # if suffix == '.json':
    #     return json.loads(p.read_text())
    if suffix in ('.yaml', '.yml'):
        if yaml is None:
            raise RuntimeError('pyyaml is not installed but a YAML config was provided')
        return yaml.safe_load(p.read_text())
    raise ValueError(f"Unsupported config format: {suffix}")


def _normalize_value(key: str, value: Any) -> Any:
    if value is None:
        return None
    # Lists
    if key in ('delete_chip_list', 'celltypes', 'animal_ids', 'monkey_ids', 'save_formats'):
        def _clean_list(vals: List[Any]) -> Optional[List[str]]:
            cleaned = [str(v).strip() for v in vals if str(v).strip() not in ('', 'None', 'none', 'NULL', 'null')]
            return cleaned or None
        if isinstance(value, list):
            return _clean_list(value)
        if isinstance(value, str):
            s = value.strip()
            if s.lower() in ('', 'none', 'null', 'all'):
                return None
            return _clean_list(value.split(','))
    # Ints
    if key in ('scalebar_um', 'max_cols', 'dpi'):
        return int(value)
    # Floats
    if key in ('um_per_pixel', 'edge'):
        return float(value)
    # Bools
    if key in ('lock_global_extent',):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    # Strings
    return value


def _merge_config(default_cfg: Config, file_cfg: Optional[Dict[str, Any]], args: argparse.Namespace) -> Config:
    merged: Dict[str, Any] = dict(default_cfg.__dict__)
    # Apply file config first
    if file_cfg:
        for k, v in file_cfg.items():
            if k in merged:
                merged[k] = _normalize_value(k, v)
    # Then apply CLI overrides (only if provided)
    for field_name in merged.keys():
        if not hasattr(args, field_name):
            continue
        cli_val = getattr(args, field_name)
        if cli_val is not None:
            merged[field_name] = _normalize_value(field_name, cli_val)
    # Special handling: delete_chips alias -> delete_chip_list
    if hasattr(args, 'delete_chips') and args.delete_chips is not None:
        merged['delete_chip_list'] = _normalize_value('delete_chip_list', args.delete_chips)
    # Derive monkey_ids if multiple animals provided without monkeys
    if merged.get('animal_ids') and not merged.get('monkey_ids'):
        merged['monkey_ids'] = [_derive_monkey_id_from_animal(a) for a in merged['animal_ids']]
    # Ensure single vs multiple coherence
    if merged.get('animal_ids'):
        # prefer multiple mode; ignore single animal_id/monkey_id in this case
        pass
    else:
        # ensure single fields present
        if not merged.get('monkey_id') and merged.get('animal_id'):
            merged['monkey_id'] = _derive_monkey_id_from_animal(merged['animal_id'])
    # Return Config
    return Config(**merged)

def parse_label(label):
    """
    解析标签，提取 F* 和 l* 部分。

    Args:
        label (str): 标签字符串，例如 'L-F1-l1' 或 'R-F2-s'.

    Returns:
        tuple: (F_label, sub_label) 例如 ('F1', 'l1') 或 ('F2', 's')
    """
    parts = label.split('-')
    F_label = None
    sub_label = None
    for part in parts:
        if part.startswith('F') and part[1:].isdigit():
            F_label = part
        elif (part.startswith('l') and part[1:].isdigit()) or part == 's':
            sub_label = part


# %%
#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################

### runtime will be configured via CLI; defaults are in Config

# %%

# 计算每个 chip 在矫正坐标系中的统一边界，确保所有子图使用相同的坐标范围
def _compute_chip_warp_bounds(cfg: Config, chip_name: str):
    try:
        cp = CorrectionPara.select_one(chip_name)
        if cp is None:
            return None
        w, h = cp.w, cp.h
        is_first_monkey = int(chip_name[1:]) < 167
        if is_first_monkey:
            w, h = h, w

        ntp_sm_path = _build_ntp_path(cfg, chip_name)
        sm = parcellate(str(ntp_sm_path), um_per_pixel=cfg.um_per_pixel, bin_size=1, return_bytes=False, w=w, h=h)

        xmin, xmax, ymin, ymax = float('inf'), -float('inf'), float('inf'), -float('inf')
        for region in sm.regions:
            # 支持 Polygon 与 MultiPolygon
            poly_obj = region.polygon
            if isinstance(poly_obj, MultiPolygon):
                polygons = list(poly_obj.geoms)
            elif isinstance(poly_obj, ShapelyPolygon):
                polygons = [poly_obj]
            else:
                continue
            for poly in polygons:
                if not hasattr(poly, 'exterior'):
                    continue
                points = np.array(poly.exterior.coords)
                if is_first_monkey:
                    points = points[:, [1, 0]]
                warped_points = cp.warp_point(points, with_offset=False)
                if warped_points is None or len(warped_points) == 0:
                    continue
                x = warped_points[:, 0]
                y = warped_points[:, 1]
                xmin = min(xmin, float(x.min()))
                xmax = max(xmax, float(x.max()))
                ymin = min(ymin, float(y.min()))
                ymax = max(ymax, float(y.max()))

        if xmin == float('inf') or ymin == float('inf'):
            return None

        # 适当加一点边距
        pad_x = (xmax - xmin) * cfg.edge
        pad_y = (ymax - ymin) * cfg.edge
        return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y
    except Exception:
        return None


def _precompute_single_chip(cfg: Config, chip_name: str):
    cp = CorrectionPara.select_one(chip_name)
    if cp is None:
        return None
    w, h = cp.w, cp.h
    is_first_monkey = int(chip_name[1:]) < 167
    if is_first_monkey:
        w, h = h, w

    ntp_sm_path = _build_ntp_path(cfg, chip_name)
    sm = parcellate(str(ntp_sm_path), um_per_pixel=cfg.um_per_pixel, bin_size=1, return_bytes=False, w=w, h=h)

    xmin, xmax, ymin, ymax = float('inf'), -float('inf'), float('inf'), -float('inf')
    for region in sm.regions:
        poly_obj = region.polygon
        if isinstance(poly_obj, MultiPolygon):
            poly = list(poly_obj.geoms)[0]
        elif isinstance(poly_obj, ShapelyPolygon):
            poly = poly_obj
        else:
            continue
        points = np.array(poly.exterior.coords)
        if is_first_monkey:
            points = points[:, [1, 0]]
        warped_points = cp.warp_point(points, with_offset=False)
        region.polygon = ShapelyPolygon(warped_points)
        x = warped_points[:, 0]
        y = warped_points[:, 1]
        xmin = min(xmin, float(x.min()))
        xmax = max(xmax, float(x.max()))
        ymin = min(ymin, float(y.min()))
        ymax = max(ymax, float(y.max()))

    if xmin == float('inf') or ymin == float('inf'):
        return None

    pad_x = (xmax - xmin) * cfg.edge
    pad_y = (ymax - ymin) * cfg.edge
    bounds = (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)

    # Precompute region color map once
    region_color_map = {}
    target_region_regex = cfg.target_region_regex
    for region in sm.regions:
        if re.search(target_region_regex, region.label.name):
            region_color_map[region.label.name] = 'black'
        else:
            region_color_map[region.label.name] = 'gray'

    return {
        'sm': sm,
        'bounds': bounds,
        'region_color_map': region_color_map,
    }


def _precompute_all_chips(cfg: Config, chip_names: List[str]) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    if cfg.precompute_workers and cfg.precompute_workers > 1:
        if cfg.log_progress:
            print(f"[prep] precomputing chips in parallel workers={cfg.precompute_workers}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.precompute_workers) as executor:
            future_map = {executor.submit(_precompute_single_chip, cfg, chip): chip for chip in chip_names}
            for idx, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
                chip = future_map[future]
                try:
                    res = future.result()
                except Exception:
                    res = None
                if res is not None:
                    results[chip] = res
                if cfg.log_progress and (idx == 1 or idx % 10 == 0 or idx == len(future_map)):
                    print(f"  - precomputed {idx}/{len(future_map)}")
    else:
        if cfg.log_progress:
            print("[prep] precomputing chips sequentially")
        for idx, chip in enumerate(chip_names, start=1):
            res = _precompute_single_chip(cfg, chip)
            if res is not None:
                results[chip] = res
            if cfg.log_progress and (idx == 1 or idx % 10 == 0 or idx == len(chip_names)):
                print(f"  - precomputed {idx}/{len(chip_names)}")
    return results

def _draw_scalebar(ax, bounds, um_length: int = 1000,
                   um_per_pixel: float = 0.5, bar_height_px: int = 8,
                   pad_frac: float = 0.03, color: str = 'white', text_color: str = 'white',
                   fontsize: int = 10):
    xmin, xmax, ymin, ymax = bounds
    span_x = xmax - xmin
    span_y = ymax - ymin
    pad_x = span_x * pad_frac
    pad_y = span_y * pad_frac

    # 坐标轴单位是像素坐标（经配准后的同一坐标系）；将微米换算为像素长度
    bar_len_px = um_length / um_per_pixel

    # 放在左下角
    x0 = xmin + pad_x
    y0 = ymin + pad_y

    rect = patches.Rectangle((x0, y0), bar_len_px, bar_height_px, color=color, lw=0, zorder=10)
    ax.add_patch(rect)
    ax.text(x0 + bar_len_px / 2.0, y0 + bar_height_px + (span_y * 0.005), f"{int(um_length)} μm",
            ha='center', va='bottom', color=text_color, fontsize=fontsize, weight='bold', zorder=11)
# %%
def plot_celltype_distribution(
    cfg: Config,
    data_dict: dict,
    chip_names: List[str],
):
    """
    为每个celltype绘制在所有芯片上的空间分布图
    
    Args:
        data_dict: 包含每个芯片数据的字典
        chip_names: 芯片名称列表
    """
    # 获取所有唯一的celltype或使用配置中的过滤
    # Always derive defaults from actual data_dict; cfg.celltypes only filters if provided
    all_celltypes_set = set()
    for chip in chip_names:
        if chip in data_dict:
            if 'celltype' in data_dict[chip].columns:
                all_celltypes_set.update(data_dict[chip]['celltype'].dropna().unique())
    all_celltypes_all = sorted(list(all_celltypes_set))
    all_celltypes = all_celltypes_all if not cfg.celltypes else [ct for ct in all_celltypes_all if ct in set(cfg.celltypes)]
    if cfg.limit_celltypes is not None:
        all_celltypes = all_celltypes[: cfg.limit_celltypes]
    if cfg.log_progress:
        print(f"  celltypes detected: {len(all_celltypes_all)} selected: {len(all_celltypes)}")
        if not all_celltypes:
            print("[warn] no celltypes selected; check data or --celltypes filter")
    
    # 预先计算所有 chip：bounds、region map、以及形状映射
    precomputed = _precompute_all_chips(cfg, chip_names)
    bounds_per_chip = {c: precomputed[c]['bounds'] for c in precomputed}
    global_bounds = None  # (xmin, xmax, ymin, ymax)
    for c, b in bounds_per_chip.items():
            if global_bounds is None:
                global_bounds = list(b)
            else:
                global_bounds[0] = min(global_bounds[0], b[0])
                global_bounds[1] = max(global_bounds[1], b[1])
                global_bounds[2] = min(global_bounds[2], b[2])
                global_bounds[3] = max(global_bounds[3], b[3])
    if global_bounds is not None:
        global_bounds = tuple(global_bounds)
    

    # 为每个celltype创建一个图
    for celltype in all_celltypes:
        if cfg.log_progress:
            print(f"[plot] celltype={celltype} chips={len(chip_names)}")
        # 计算行列数
        ncols = min(cfg.max_cols, len(chip_names))
        nrows = math.ceil(len(chip_names) / ncols)
        
        # 创建子图
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, 4 * nrows), constrained_layout=True)
        axes = axes.flatten() if nrows * ncols > 1 else [axes]
        
        # 遍历每个芯片
        chip_iter = chip_names[: cfg.limit_chips] if cfg.limit_chips is not None else chip_names
        for i, chip_name in enumerate(chip_iter):
            ax = axes[i]
            
            if chip_name not in data_dict:
                ax.set_axis_off()
                continue
                
            # 获取CorrectionPara对象
            cp = CorrectionPara.select_one(chip_name)
            if cp is None:
                print(f"无法检索 {chip_name}")
                ax.set_axis_off()
                continue
            
            w, h = cp.w, cp.h
            is_first_monkey = int(chip_name[1:]) < 167
            if is_first_monkey:
                w, h = h, w
            
            # 获取区域数据（使用预计算）
            pre = precomputed.get(chip_name)
            if pre is None:
                ax.set_axis_off()
                if cfg.log_progress:
                    print(f"  - skip {chip_name}: precompute missing")
                continue
            sm = pre['sm']
            if cfg.log_progress and (i == 0 or (i + 1) % 10 == 0 or (i + 1) == len(chip_names)):
                print(f"  - chip {i+1}/{len(chip_names)}: {chip_name}")

            # if chip_name == 'T926':
            #     ntp_sm_path = '/mnt/inner-data/ntp/macaque/Mq277-CQW-newlayer/Mq277-T926-P0.ntp'
            #     sm = parcellate(str(ntp_sm_path), um_per_pixel=0.5, bin_size=1, return_bytes=False, w=w, h=h)
            
            # 筛选特定celltype的数据
            df_celltype = data_dict[chip_name][data_dict[chip_name]['celltype'] == celltype]
            
            if len(df_celltype) == 0:
                ax.text(0.5, 0.5, f"No {celltype} cells", 
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
                continue
            
            # 获取细胞坐标（多候选列名回退）
            xy_cols = _select_xy_columns(df_celltype)
            if xy_cols is None:
                if cfg.log_progress:
                    print(f"[plot] {chip_name} missing coordinate columns; available: {list(df_celltype.columns)[:10]} ...")
                ax.text(0.5, 0.5, f"No XY columns", 
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
                continue
            x_col, y_col = xy_cols
            x_coords = df_celltype[x_col].values
            y_coords = df_celltype[y_col].values
            # x_coords, y_coords = y_coords, x_coords
            # 坐标转换
            # transformed_coords = [cp.warp_point(np.array([[x, y]]), with_offset=True)[0]
            #                     for x, y in zip(x_coords, y_coords)]
            # transformed_coords = np.array(transformed_coords)
            # x_t, y_t = transformed_coords[:, 0], transformed_coords[:, 1]
            
            # 形状已在预计算阶段变换过，此处无需重复
            
            # 绘制散点图
            ax.scatter(
                # x_t, 
                # y_t,
                x_coords, 
                y_coords,
                c='red',
                s=0.5,  # 根据需要调整点的大小
                alpha=1,  # 调整透明度
                label=celltype,
                marker='o',
                edgecolors='none'
            )

            target_region_regex = cfg.target_region_regex
            region_color_map = pre['region_color_map']


            sm.show_in_plt(ax, cell_alpha=0.0, cell_size=1.0, line_alpha=0.1, line_width=0.1,
                          show_cells=False, show_raw_lines=False,
                          unmatch_region_alpha=0.05, show_region_labels=False, 
                          target_region_regex=target_region_regex,
                          region_color_map=region_color_map)
            ax.set_title(f'{chip_name}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal')

            # 统一坐标范围，确保相同 scale，并绘制 scalebar
            if cfg.lock_global_extent and global_bounds is not None:
                xmin, xmax, ymin, ymax = global_bounds
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                _draw_scalebar(
                    ax,
                    global_bounds,
                    um_length=cfg.scalebar_um,
                    um_per_pixel=cfg.um_per_pixel,
                    bar_height_px=8,
                    pad_frac=0.03,
                    color='white',
                    text_color='white',
                    fontsize=8,
                )
            else:
                # 使用当前子图范围绘制标尺（比例一致，但视窗不锁定）
                current_bounds = (
                    float(ax.get_xlim()[0]), float(ax.get_xlim()[1]),
                    float(ax.get_ylim()[0]), float(ax.get_ylim()[1])
                )
                _draw_scalebar(
                    ax,
                    current_bounds,
                    um_length=cfg.scalebar_um,
                    um_per_pixel=cfg.um_per_pixel,
                    bar_height_px=8,
                    pad_frac=0.03,
                    color='white',
                    text_color='white',
                    fontsize=8,
                )
        
        # 隐藏未使用的子图
        for j in range(i + 1, nrows * ncols):
            axes[j].set_axis_off()  
        
        # 添加总标题    
        # fig.suptitle(f'Spatial Distribution of {celltype}', fontsize=16)
        
        # 保存图像
        save_path = Path(cfg.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        # 保存为配置中指定的所有格式
        formats = cfg.save_formats or ['pdf']
        base_filename = f'{cfg.animal_id}_{celltype}_distribution_all_chips'
        for fmt in formats:
            fmt_clean = str(fmt).strip().lower()
            if fmt_clean == '':
                continue
            out_file = save_path / f"{base_filename}.{fmt_clean}"
            if cfg.skip_if_exists and out_file.exists():
                if cfg.log_progress:
                    print(f"[skip] exists: {out_file.name}")
                continue
            fig.savefig(out_file, dpi=cfg.dpi, bbox_inches='tight')
            if cfg.log_progress:
                print(f"[save] {out_file.name}")
        plt.close(fig)
# %%
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot spatial distribution of cell types across chips')
    parser.add_argument('--config', type=str, default=None, help='Path to JSON/YAML/CSV config file')
    parser.add_argument('--species', type=str, default=None)
    parser.add_argument('--animal_id', type=str, default=None)
    parser.add_argument('--animal_ids', type=str, default=None, help='Comma-separated list for multiple animals')
    parser.add_argument('--monkey_id', type=str, default=None)
    parser.add_argument('--monkey_ids', type=str, default=None, help='Comma-separated list aligned with --animal_ids')
    parser.add_argument('--csv_files_dir', type=str, default=None)
    parser.add_argument('--libinfo_xlsx', type=str, default=None)
    parser.add_argument('--libinfo_sheet', type=str, default=None)
    parser.add_argument('--delete_chips', type=str, default=None, help='Comma-separated chip IDs to exclude')
    parser.add_argument('--um_per_pixel', type=float, default=None)
    parser.add_argument('--ntp_version', type=str, default=None)
    parser.add_argument('--ntp_root_dir', type=str, default=None)
    parser.add_argument('--edge', type=float, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--scalebar_um', type=int, default=None)
    parser.add_argument('--lock_global_extent', dest='lock_global_extent', action='store_true')
    parser.add_argument('--no-lock_global_extent', dest='lock_global_extent', action='store_false')
    parser.set_defaults(lock_global_extent=None)
    parser.add_argument('--max_cols', type=int, default=None)
    parser.add_argument('--target_region_regex', type=str, default=None)
    parser.add_argument('--dpi', type=int, default=None)
    parser.add_argument('--celltypes', type=str, default=None, help='Comma-separated celltypes to plot; if omitted, plot all')
    parser.add_argument('--save_formats', type=str, default=None, help='Comma-separated formats to save figures, e.g., pdf,png')
    parser.add_argument('--progress', dest='log_progress', action='store_true')
    parser.add_argument('--no-progress', dest='log_progress', action='store_false')
    parser.set_defaults(log_progress=None)
    parser.add_argument('--precompute_workers', type=int, default=None, help='Threads for precomputing regions (0/1=off)')
    parser.add_argument('--limit_chips', type=int, default=None, help='Limit number of chips for benchmarking')
    parser.add_argument('--limit_celltypes', type=int, default=None, help='Limit number of celltypes for benchmarking')
    parser.add_argument('--skip_if_exists', dest='skip_if_exists', action='store_true')
    parser.add_argument('--no-skip_if_exists', dest='skip_if_exists', action='store_false')
    parser.set_defaults(skip_if_exists=None)

    return parser.parse_args()


def main():
    args = _parse_args()
    default_cfg = Config()
    file_cfg = _load_config_file(args.config) if args.config else None
    cfg = _merge_config(default_cfg, file_cfg, args)

    # multiple animals mode
    if cfg.animal_ids:
        animals = cfg.animal_ids
        monkeys = cfg.monkey_ids or [None] * len(animals)
        for idx, animal in enumerate(animals):
            cur_cfg = Config(**{**cfg.__dict__})
            cur_cfg.animal_id = animal
            cur_cfg.monkey_id = monkeys[idx] or _derive_monkey_id_from_animal(animal)
            if cfg.log_progress:
                print(f"[animal] {idx+1}/{len(animals)}: animal_id={cur_cfg.animal_id} monkey_id={cur_cfg.monkey_id}")
            chip_names = _load_chip_names(cur_cfg)
            if cfg.log_progress:
                print(f"  chips selected: {len(chip_names)}")
            data_dict = _load_data_dict(cur_cfg, chip_names)
            plot_celltype_distribution(cur_cfg, data_dict, chip_names)
    else:
        if cfg.log_progress:
            print(f"[animal] animal_id={cfg.animal_id} monkey_id={cfg.monkey_id}")
        chip_names = _load_chip_names(cfg)
        if cfg.log_progress:
            print(f"  chips selected: {len(chip_names)}")
        data_dict = _load_data_dict(cfg, chip_names)
        plot_celltype_distribution(cfg, data_dict, chip_names)


if __name__ == '__main__':
    main()