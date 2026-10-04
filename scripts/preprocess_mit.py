#!/usr/bin/env python3
"""
Exported from MIT完整流程(内存处理）.ipynb.
Historical research code for the Transformer-based seizure prediction system.
Original test data are unavailable; provide your own public/approved data paths.
"""

import os
import mne
import numpy as np
from collections import defaultdict
from datetime import datetime
from pykalman import KalmanFilter
from mne.preprocessing import ICA
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from scipy import signal, fftpack
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split
import random
import math
from tqdm import tqdm
import re
import argparse
import shutil
import sys

# 禁用matplotlib交互显示
plt.ioff()
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# 新增：检查磁盘空间函数
def check_disk_space(path, min_space_gb=5):
    """检查指定路径所在磁盘的剩余空间，如果小于最小要求则终止程序"""
    try:
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024 ** 3)  # 转换为GB
        print(f"当前磁盘剩余空间: {free_gb:.2f} GB")
        
        if free_gb < min_space_gb:
            print(f"警告: 磁盘空间不足！剩余 {free_gb:.2f} GB，最低要求 {min_space_gb} GB")
            print("程序终止以防止磁盘空间耗尽")
            sys.exit(1)
        return True
    except Exception as e:
        print(f"检查磁盘空间时出错: {e}")
        return False

# 新增：删除文件夹中的FIF文件
def delete_fif_files(folder_path):
    """删除指定文件夹中的所有.fif文件"""
    fif_files = []
    size_deleted = 0
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.fif'):
                file_path = os.path.join(root, file)
                try:
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    fif_files.append(file)
                    size_deleted += file_size
                    print(f"已删除: {file_path}")
                except Exception as e:
                    print(f"删除文件 {file_path} 时出错: {e}")
    
    size_deleted_mb = size_deleted / (1024 * 1024)  # 转换为MB
    print(f"已删除 {len(fif_files)} 个FIF文件，释放 {size_deleted_mb:.2f} MB空间")

# 步骤1: 解析摘要文件获取癫痫发作信息，为EDF文件添加发作注释
def parse_seizure_summary(summary_path):
    """解析摘要文件获取癫痫发作信息"""
    seizure_info = defaultdict(list)
    current_file = None

    with open(summary_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('File Name:'):
                file_name = line.split(': ')[1].split('.edf')[0] + '.edf'
                current_file = file_name
            elif line.startswith('Seizure Start Time:') and current_file:
                start_sec = float(line.split(': ')[1].split(' seconds')[0])
            elif line.startswith('Seizure End Time:') and current_file:
                end_sec = float(line.split(': ')[1].split(' seconds')[0])
                seizure_info[current_file].append((start_sec, end_sec))

    return seizure_info

# 新增一个函数来自动查找摘要文件
def find_summary_file(directory):
    """在指定目录中查找摘要文件（*.txt）"""
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            return os.path.join(directory, file)
    return None

# 修改步骤1，处理单个文件夹
def step1_process_seizure_files(folder_path):
    """为有发作的EDF文件添加发作注释并保存为FIF文件"""
    summary_path = find_summary_file(folder_path)
    if not summary_path:
        print(f"警告: 在 {folder_path} 中未找到摘要文件，跳过此目录")
        return
        
    print(f"\n处理目录: {folder_path}")
    print(f"找到摘要文件: {os.path.basename(summary_path)}")
    
    # 检查磁盘空间
    check_disk_space(folder_path)
    
    seizure_info = parse_seizure_summary(summary_path)
    
    # 打印发作信息摘要
    if seizure_info:
        print(f"成功解析摘要文件，找到 {len(seizure_info)} 个文件的发作信息:")
        for file_name, seizures in seizure_info.items():
            print(f"  {file_name}: {len(seizures)} 个发作")
    else:
        print("警告: 未从摘要文件中解析到任何发作信息")

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.edf'):
                edf_path = os.path.join(root, file)
                edf_name = file
                seizures = seizure_info.get(edf_name, [])

                if not seizures:
                    print(f"跳过 {edf_name}：无发作信息")
                    continue

                try:
                    # 读取 EDF 文件（preload=True 确保注释正确关联时间）
                    raw = mne.io.read_raw_edf(edf_path, preload=True)
                    
                    # 处理 datetime 类型的 meas_date
                    meas_date = raw.info.get('meas_date')
                    if isinstance(meas_date, datetime):
                        # 将 datetime 对象转换为 Unix 时间戳（整数秒）
                        meas_date = int(meas_date.timestamp())
                        print(f"检测到 datetime 时间戳，转换为 {meas_date}")
                    elif isinstance(meas_date, (float, int)):
                        meas_date = int(meas_date)  # 确保为整数
                    else:
                        meas_date = 0  # 其他情况设为0

                    # 校验时间戳是否在有效范围内
                    if not (-2147483648 <= meas_date <= 2147483647):
                        print(f"重置无效时间戳 {meas_date} 为 0")
                        meas_date = 0
                    raw.set_meas_date(meas_date)

                    # 创建注释对象（使用相对时间，基于文件开头的秒数）
                    onsets = [start for start, end in seizures]
                    durations = [end - start for start, end in seizures]
                    descriptions = ['seizure'] * len(seizures)
                    annotations = mne.Annotations(
                        onset=onsets,
                        duration=durations,
                        description=descriptions,
                        orig_time=meas_date  # 使用转换后的整数时间戳
                    )

                    # 添加注释
                    raw.set_annotations(annotations)
                    print(f"成功为 {edf_name} 添加 {len(seizures)} 条发作注释")

                    # 保存为符合 MNE 规范的文件名
                    fif_name = edf_name.replace('.edf', '_raw.fif')
                    fif_path = os.path.join(root, fif_name)
                    
                    # 检查磁盘空间
                    check_disk_space(root)
                    
                    raw.save(fif_path, overwrite=True)
                    print(f"已保存为 {os.path.basename(fif_path)}")

                except Exception as e:
                    print(f"处理 {edf_name} 时出错: {e}")

# 步骤2: 将无发作的EDF文件转换为FIF文件
def step2_process_non_seizure_files(folder_path):
    """将无发作的EDF文件转换为FIF文件"""
    summary_path = find_summary_file(folder_path)
    if not summary_path:
        print(f"警告: 在 {folder_path} 中未找到摘要文件，跳过此目录")
        return
        
    print(f"\n处理目录: {folder_path}")
    print(f"找到摘要文件: {os.path.basename(summary_path)}")
    
    # 检查磁盘空间
    check_disk_space(folder_path)
    
    seizure_info = parse_seizure_summary(summary_path)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.edf'):
                edf_path = os.path.join(root, file)
                edf_name = file
                seizures = seizure_info.get(edf_name, [])

                if seizures:
                    print(f"跳过 {edf_name}：有发作信息")
                    continue

                try:
                    # 读取 EDF 文件
                    raw = mne.io.read_raw_edf(edf_path, preload=True)

                    # 处理 datetime 类型的 meas_date
                    meas_date = raw.info.get('meas_date')
                    if isinstance(meas_date, datetime):
                        meas_date = int(meas_date.timestamp())
                        print(f"检测到 datetime 时间戳，转换为 {meas_date}")
                    elif isinstance(meas_date, (float, int)):
                        meas_date = int(meas_date)  # 确保为整数
                    else:
                        meas_date = 0  # 其他情况设为0

                    # 校验时间戳是否在有效范围内
                    if not (-2147483648 <= meas_date <= 2147483647):
                        print(f"重置无效时间戳 {meas_date} 为 0")
                        meas_date = 0
                    raw.set_meas_date(meas_date)

                    # 保存为符合 MNE 规范的文件名
                    fif_name = edf_name.replace('.edf', '_raw.fif')
                    fif_path = os.path.join(root, fif_name)
                    
                    # 检查磁盘空间
                    check_disk_space(root)
                    
                    raw.save(fif_path, overwrite=True)
                    print(f"已将 {edf_name} 转换并保存为 {os.path.basename(fif_path)}")

                except Exception as e:
                    print(f"处理 {edf_name} 时出错: {e}")

# 步骤3: 删除FIF文件中的SSP投影器
def step3_remove_ssp_projectors(folder_path):
    """删除FIF文件中的SSP投影器"""
    print(f"\n处理目录: {folder_path}")
    
    # 检查磁盘空间
    check_disk_space(folder_path)
    
    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.fif'):
                # 构建文件的完整路径
                fif_path = os.path.join(root, file)
                try:
                    # 读取 fif 文件
                    raw = mne.io.read_raw_fif(fif_path, preload=True)
                    # 检查 'projs' 是否存在于 raw.info 中
                    if "projs" in raw.info:
                        ssp_projectors = raw.info["projs"]
                    else:
                        ssp_projectors = []
                        print(f"{file} 中无 SSP 投影器")

                    # 删除投影器
                    raw.del_proj()

                    # 保存修改后的文件
                    new_fif_path = fif_path.replace('.fif', '_no_proj.fif')
                    
                    # 检查磁盘空间
                    check_disk_space(root)
                    
                    raw.save(new_fif_path, overwrite=True)

                    print(f"已成功删除 {file} 中的 SSP 投影器，并保存为 {os.path.basename(new_fif_path)}")

                except Exception as e:
                    print(f"处理 {file} 时出错: {e}")

# 步骤4: 应用高通滤波和陷波滤波
def step4_apply_filters(folder_path):
    """对无投影器FIF文件应用高通滤波和陷波滤波处理电源线干扰"""
    # 截止频率
    highpass_cutoff = 0.2
    
    print(f"\n处理目录: {folder_path}")
    
    # 检查磁盘空间
    check_disk_space(folder_path)
    
    # 获取文件夹下所有以 _raw_no_proj.fif 结尾的文件
    fif_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith("_raw_no_proj.fif"):
                fif_files.append(os.path.join(root, file))
    
    print(f"找到 {len(fif_files)} 个待处理的FIF文件")
    
    for file_path in fif_files:
        try:
            # 读取原始数据
            raw = mne.io.read_raw_fif(file_path, preload=True)
            
            # 动态获取采样频率
            sfreq = raw.info['sfreq']
            
            # 生成电源线干扰频率及其谐波，最大不超过采样频率的一半
            powerline_freqs = tuple([i * 60 for i in range(1, int(sfreq / 120) + 1)])
            
            # 选择所有数据通道
            all_picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads')
            
            # 进行高通滤波
            raw_highpass = raw.copy().filter(l_freq=highpass_cutoff, h_freq=None, picks=all_picks)
            
            # 进行陷波滤波
            raw_notch = raw_highpass.copy().notch_filter(freqs=powerline_freqs, picks=all_picks)
            
            # 保存处理后的数据为FIF文件
            output_file_path = file_path.replace("_raw_no_proj.fif", "_processed.fif")
            
            # 检查磁盘空间
            check_disk_space(os.path.dirname(output_file_path))
            
            raw_notch.save(output_file_path, overwrite=True)
            print(f"处理后的数据已保存到 {os.path.basename(output_file_path)}")
            
        except Exception as e:
            print(f"处理文件 {os.path.basename(file_path)} 时出现错误: {e}")

# 步骤5: 高级信号处理函数
def apply_kalman_filter(data, R=0.01):
    """卡尔曼滤波基线校正"""
    kf = KalmanFilter(
        initial_state_mean=0,
        initial_state_covariance=1,
        transition_matrices=1,
        observation_matrices=1,
        observation_covariance=R,
        transition_covariance=0.001
    )
    state_means, _ = kf.filter(data)
    return data - state_means.flatten()

def apply_zscore(raw_obj):
    """对连续数据（Raw对象）进行Z-分数归一化（按通道）"""
    data = raw_obj.get_data()
    mean = np.mean(data, axis=1, keepdims=True)  # 按通道计算均值
    std = np.std(data, axis=1, keepdims=True)     # 按通道计算标准差
    std[std == 0] = 1.0
    normalized_data = (data - mean) / std
    raw_obj._data = normalized_data
    return raw_obj

def get_seizure_annotations(raw):
    """提取癫痫发作的注释，返回所有发作的开始和结束时间"""
    seizures = []
    if hasattr(raw, 'annotations') and len(raw.annotations) > 0:
        for annot in raw.annotations:
            description = annot['description'].lower() if isinstance(annot['description'], str) else str(annot['description']).lower()
            # 查找可能的癫痫发作注释关键词
            if any(keyword in description for keyword in ['seizure', 'seiz', 'ictal', '发作', '癫痫']):
                onset = annot['onset']
                duration = annot['duration']
                seizures.append((onset, onset + duration))
                print(f"发现癫痫注释: {description}, 开始时间: {onset}秒, 持续时间: {duration}秒")
    return seizures

def create_epochs_with_seizure_info(raw, regular_duration, seizures, pre_time, post_time):
    """创建考虑癫痫发作的时间窗口分段"""
    sfreq = raw.info['sfreq']
    data_duration = raw.times[-1]
    
    # 初始化事件列表和标签
    events = []
    event_labels = []  # 0=常规, 1=癫痫前, 2=癫痫发作中, 3=癫痫后
    
    # 如果没有发作标记，则使用常规分段
    if not seizures:
        print("没有发现癫痫发作注释，使用常规分段")
        return create_nonoverlapping_epochs(raw, regular_duration), None
    
    # 当有癫痫发作注释时，创建特殊的epochs
    # 1. 标记所有的发作前期、发作中和发作后期的区间
    marked_regions = []
    for seizure_start, seizure_end in seizures:
        # 确保不超出数据范围
        pre_start = max(0, seizure_start - pre_time)
        post_end = min(data_duration, seizure_end + post_time)
        
        # 添加到标记区域
        marked_regions.append((pre_start, seizure_start, "pre-seizure"))  # 发作前
        marked_regions.append((seizure_start, seizure_end, "seizure"))   # 发作中
        marked_regions.append((seizure_end, post_end, "post-seizure"))   # 发作后
    
    # 2. 对于发作相关区域，创建特定的epoch
    for start, end, label in marked_regions:
        if end > start:  # 确保区间有效
            # 转换为样本点
            start_sample = int(start * sfreq)
            
            # 创建事件
            event_type = {"pre-seizure": 1, "seizure": 2, "post-seizure": 3}[label]
            events.append([start_sample, 0, event_type])
            event_labels.append(label)
            
            print(f"添加 {label} 事件: {start}-{end}秒 (标签={event_type})")
    
    # 3. 对于未被标记的区域，使用常规分段，但限制为最多5个epoch
    # 首先找出所有已标记区域
    all_marked = []
    for start, end, _ in marked_regions:
        all_marked.append((start, end))
    
    # 合并重叠区间
    all_marked.sort()
    merged = []
    for interval in all_marked:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
    
    # 找出未标记区域
    unmarked = []
    last_end = 0
    for start, end in merged:
        if start > last_end:
            unmarked.append((last_end, start))
        last_end = end
    if last_end < data_duration:
        unmarked.append((last_end, data_duration))
    
    # 从未标记区域中创建常规epoch候选列表
    regular_events_candidates = []
    for start, end in unmarked:
        if end - start >= regular_duration:  # 只有当区间足够长时才分段
            # 计算该区间内可以创建的常规epochs数量
            n_epochs = int((end - start) // regular_duration)
            for i in range(n_epochs):
                epoch_start = start + i * regular_duration
                start_sample = int(epoch_start * sfreq)
                regular_events_candidates.append([start_sample, 0, 0, epoch_start, epoch_start+regular_duration])  # 添加时间信息用于打印
    
    # 如果常规epoch候选数量超过5个，随机选择5个
    max_regular_epochs = 5
    if len(regular_events_candidates) > max_regular_epochs:
        print(f"常规区域可创建 {len(regular_events_candidates)} 个epochs，随机选择 {max_regular_epochs} 个")
        selected_indices = random.sample(range(len(regular_events_candidates)), max_regular_epochs)
        selected_regular_events = [regular_events_candidates[i] for i in selected_indices]
    else:
        print(f"常规区域可创建 {len(regular_events_candidates)} 个epochs，全部保留")
        selected_regular_events = regular_events_candidates
    
    # 添加选中的常规events
    for event in selected_regular_events:
        events.append(event[:3])  # 只保留event格式的前三个元素
        event_labels.append("regular")
        print(f"添加常规事件: {event[3]}-{event[4]}秒 (标签=0)")
    
    # 确保events是numpy数组
    events = np.array(events, dtype=int)
    
    # 创建epochs
    if len(events) > 0:
        # 对事件按时间排序
        sort_idx = np.argsort(events[:, 0])
        events = events[sort_idx]
        event_labels = [event_labels[i] for i in sort_idx]
        
        # 创建epochs字典，指定不同事件类型的时间区间
        event_dict = {'regular': 0, 'pre-seizure': 1, 'seizure': 2, 'post-seizure': 3}
        
        # 分别创建和保存不同类型的epochs，不尝试合并它们
        epochs_dict = {}
        
        # 创建常规事件epochs
        regular_events = events[events[:, 2] == 0]
        if len(regular_events) > 0:
            regular_epochs = mne.Epochs(
                raw,
                regular_events,
                event_id={'regular': 0},
                tmin=0,
                tmax=regular_duration,
                baseline=None,
                preload=True,
                verbose=False
            )
            epochs_dict['regular'] = regular_epochs
            print(f"创建了 {len(regular_epochs)} 个常规epochs，每个时长 {regular_duration} 秒")
        
        # 创建发作前epochs
        pre_events = events[events[:, 2] == 1]
        if len(pre_events) > 0:
            pre_epochs = mne.Epochs(
                raw,
                pre_events,
                event_id={'pre-seizure': 1},
                tmin=0,
                tmax=pre_time,
                baseline=None,
                preload=True,
                verbose=False
            )
            epochs_dict['pre-seizure'] = pre_epochs
            print(f"创建了 {len(pre_epochs)} 个发作前epochs，每个时长 {pre_time} 秒")
        
        # 对于发作事件，为每个发作创建单独的epoch
        seizure_epochs_list = []
        for i, (start, end) in enumerate(seizures):
            seizure_duration = end - start
            seizure_event = np.array([[int(start * sfreq), 0, 2]])  # 手动创建事件
            seizure_epoch = mne.Epochs(
                raw,
                seizure_event,
                event_id={'seizure': 2},
                tmin=0,
                tmax=seizure_duration,
                baseline=None,
                preload=True,
                verbose=False
            )
            if len(seizure_epoch) > 0:
                seizure_epochs_list.append(seizure_epoch)
                print(f"创建了发作epoch: {start}-{end}秒，持续时间: {seizure_duration}秒")
        
        if seizure_epochs_list:
            epochs_dict['seizure'] = seizure_epochs_list
        
        # 创建发作后epochs
        post_events = events[events[:, 2] == 3]
        if len(post_events) > 0:
            post_epochs = mne.Epochs(
                raw,
                post_events,
                event_id={'post-seizure': 3},
                tmin=0,
                tmax=post_time,
                baseline=None,
                preload=True,
                verbose=False
            )
            epochs_dict['post-seizure'] = post_epochs
            print(f"创建了 {len(post_epochs)} 个发作后epochs，每个时长 {post_time} 秒")
        
        # 使用常规epochs作为主要对象返回，加上类型字典
        if 'regular' in epochs_dict:
            return epochs_dict['regular'], epochs_dict
        elif len(epochs_dict) > 0:
            # 返回找到的第一种类型
            key = list(epochs_dict.keys())[0]
            if key == 'seizure':
                return seizure_epochs_list[0], epochs_dict
            return epochs_dict[key], epochs_dict
        else:
            print("未能创建有效的epochs，退回到常规分段方法")
            return create_nonoverlapping_epochs(raw, regular_duration), None
    else:
        print("未检测到有效事件，退回到常规分段方法")
        return create_nonoverlapping_epochs(raw, regular_duration), None

def create_nonoverlapping_epochs(raw, duration):
    """创建不重叠的时间窗口分段（用于无癫痫注释的情况）"""
    # 计算可以创建的epochs数量（向下取整）
    data_duration = raw.times[-1]
    n_epochs = int(data_duration // duration)
    
    # 限制最大epoch数量为5
    max_epochs = 5
    if n_epochs > max_epochs:
        print(f"可创建 {n_epochs} 个epochs，随机选择 {max_epochs} 个")
        selected_indices = random.sample(range(n_epochs), max_epochs)
        selected_indices.sort()  # 排序以保持时间顺序
    else:
        print(f"可创建 {n_epochs} 个epochs，全部保留")
        selected_indices = range(n_epochs)
    
    # 创建人工事件，根据选择的索引
    events = np.zeros((len(selected_indices), 3), dtype=int)
    for i, original_idx in enumerate(selected_indices):
        # 事件样本点位置、前一个事件的ID（0）、当前事件的ID（1）
        events[i, 0] = int(original_idx * duration * raw.info['sfreq'])
        events[i, 2] = 0  # 使用0表示常规区域
    
    # 创建epochs
    epochs = mne.Epochs(
        raw, 
        events, 
        event_id={'regular': 0},
        tmin=0, 
        tmax=duration, 
        baseline=None, 
        preload=True,
        reject=None,
        verbose=False
    )
    return epochs

def stockwell_transform(data, fs, fmin=0, fmax=80, n_freq=40):
    """对信号进行Stockwell变换"""
    n_channels, n_samples = data.shape
    
    # 生成频率点 (对数均匀分布)
    freq_bins = np.logspace(np.log10(max(0.5, fmin)), np.log10(fmax), n_freq)
    
    def st_channel(x):
        # 单通道的Stockwell变换实现
        X = fftpack.fft(x)  # 计算FFT
        n = len(x)
        
        # 预计算所有频率点的指数项
        k = np.arange(n)
        factor = 2 * np.pi * k / n
        
        # 初始化结果数组
        st_channel_data = np.zeros((n_freq, n), dtype=complex)
        
        # 使用贝利分解定理，快速计算变换
        for i, freq in enumerate(freq_bins):
            # 计算标准差 - 高斯窗函数宽度随频率自适应
            sigma = freq / fs
            # 构建高斯窗
            gaussian_window = np.exp(-2 * (np.pi * sigma * k / n) ** 2)
            
            # 使用矩阵操作一次性完成所有时间点的计算
            Xw = X * gaussian_window
            st_channel_data[i, :] = fftpack.ifft(Xw)
            
        return st_channel_data
    
    # 并行处理每个通道
    st_results = Parallel(n_jobs=-1, prefer="threads")(delayed(st_channel)(data[i, :]) for i in range(n_channels))
    st_data = np.array(st_results)
    
    return st_data

def apply_stockwell_to_epoch(epoch_data, sfreq):
    """对一个epoch应用Stockwell变换"""
    # 获取epoch数据
    data = epoch_data.get_data()  # (epochs, channels, samples)
    n_epochs, n_channels, n_samples = data.shape
    
    # 初始化结果数组
    st_results = []
    
    # 对每个epoch进行Stockwell变换
    for i in range(n_epochs):
        # 应用Stockwell变换
        st_data = stockwell_transform(data[i], fs=sfreq, fmin=0, fmax=80, n_freq=40)
        
        # 计算幅度
        st_amplitude = np.abs(st_data)
        st_results.append(st_amplitude)
    
    return np.array(st_results)  # (epochs, channels, frequencies, samples)

# 修改步骤5，处理单个文件夹
def step5_advanced_processing(folder_path, target_sfreq=128, epoch_duration=60.0, 
                             pre_seizure_time=60.0, post_seizure_time=60.0, 
                             consider_annotations=True):
    """高级信号处理与Stockwell变换应用"""
    # 低通滤波截止频率
    lowpass_freq = 80
    
    print(f"\n处理目录: {folder_path}")
    
    # 检查磁盘空间
    check_disk_space(folder_path)
    
    # 遍历文件夹中的processed.fif文件
    processed_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith("_processed.fif"):
                processed_files.append(os.path.join(root, file))
    
    print(f"找到 {len(processed_files)} 个_processed.fif文件")
    
    for file_path in processed_files:
        try:
            # 检查磁盘空间
            check_disk_space(os.path.dirname(file_path))
            
            # 1. 读取原始数据
            raw = mne.io.read_raw_fif(file_path, preload=True)
            print(f"处理文件: {os.path.basename(file_path)}")
            
            # 检查是否有癫痫发作注释
            seizures = []
            if consider_annotations:
                seizures = get_seizure_annotations(raw)
                if seizures:
                    print(f"发现 {len(seizures)} 个癫痫发作注释")
                else:
                    print("未发现癫痫发作注释")

            # 2. 低通滤波
            raw_filtered = raw.copy().filter(l_freq=None, h_freq=lowpass_freq, method="fir")

            # 3. 下采样
            raw_downsampled = raw_filtered.copy().resample(
                sfreq=target_sfreq, method="polyphase", n_jobs=-1, verbose=False
            )
            print(f"完成下采样至 {target_sfreq} Hz")

            # 4. 使用卡尔曼滤波进行基线校正
            data = raw_downsampled.get_data()
            num_channels = data.shape[0]
            processed_channels = Parallel(n_jobs=-1)(
                delayed(apply_kalman_filter)(data[i, :]) for i in range(num_channels)
            )
            data = np.array(processed_channels)
            raw_downsampled._data = data
            print("完成卡尔曼滤波基线校正")

            # 5. 使用Picard算法进行ICA（在连续数据上）
            try:
                n_components = 0.99  # 保留99%的方差
                ica = ICA(
                    n_components=n_components,
                    method='picard',
                    random_state=42,
                    max_iter=500
                )
                ica.fit(raw_downsampled)
                print("完成ICA拟合")

                # 分析ICA成分，识别伪迹成分
                components = ica.get_sources(raw_downsampled).get_data()
                for i, component in enumerate(components):
                    # 计算功率谱密度
                    n = len(component)
                    fft_result = np.fft.fft(component)
                    psd = np.abs(fft_result) ** 2 / n
                    freqs = np.fft.fftfreq(n, 1 / target_sfreq)
                    
                    # 眼电伪迹判断
                    eog_like = np.sum(psd[(freqs >= 0) & (freqs < 5)]) / np.sum(psd) > 0.5
                    # 肌电伪迹判断
                    emg_like = np.sum(psd[(freqs >= 50)]) / np.sum(psd) > 0.3
                    # 心电伪迹判断
                    ecg_like = np.sum(psd[(freqs >= 45) & (freqs <= 65)]) / np.sum(psd) > 0.1
                    
                    if eog_like or emg_like or ecg_like:
                        ica.exclude.append(i)

                print(f"检测到 {len(ica.exclude)} 个伪迹相关ICA成分")

                # 应用ICA校正到连续数据
                raw_cleaned = ica.apply(raw_downsampled.copy())
                print("完成ICA伪迹移除")
            except Exception as e:
                print(f"ICA处理失败，使用未经ICA处理的数据继续: {e}")
                raw_cleaned = raw_downsampled.copy()  # 使用未经ICA处理的数据继续

            # 对连续数据进行Z-分数归一化（按通道）
            raw_normalized = apply_zscore(raw_cleaned)
            print("完成Z-分数归一化")

            # 保存文件前检查磁盘空间
            check_disk_space(os.path.dirname(file_path))
            
            # 保存连续数据结果
            output_file = file_path.replace("_processed.fif", f"_processed_full_{target_sfreq}Hz.fif")
            raw_normalized.save(output_file, overwrite=True)
            print(f"保存连续数据至: {os.path.basename(output_file)}")
            
            # 创建考虑癫痫发作的epochs
            epochs_all_types = {}
            if consider_annotations and seizures:
                main_epochs, epochs_dict = create_epochs_with_seizure_info(
                    raw_normalized, epoch_duration, seizures, pre_seizure_time, post_seizure_time
                )
                if epochs_dict:
                    epochs_all_types = epochs_dict
                    print(f"成功创建不同类型的epochs")
                    
                    # 保存各种类型的epochs到单独的文件
                    for epoch_type, epoch_obj in epochs_dict.items():
                        if epoch_type != 'seizure':  # 常规处理非发作epochs
                            type_output_file = file_path.replace(
                                "_processed.fif", 
                                f"_epoched_{epoch_type}_{int(epoch_duration)}s_{target_sfreq}Hz-epo.fif"
                            )
                            epoch_obj.save(type_output_file, overwrite=True)
                            print(f"保存{epoch_type} epochs数据至: {os.path.basename(type_output_file)}")
                            
                            # 对这种类型应用Stockwell变换
                            print(f"对{epoch_type} epochs应用Stockwell变换...")
                            st_results = apply_stockwell_to_epoch(epoch_obj, target_sfreq)
                            print(f"完成{epoch_type} Stockwell变换，结果形状: {st_results.shape}")
                            
                            # 归一化
                            st_mean = np.mean(st_results, axis=(2, 3), keepdims=True)
                            st_std = np.std(st_results, axis=(2, 3), keepdims=True)
                            st_std[st_std == 0] = 1.0
                            st_normalized = (st_results - st_mean) / st_std
                            
                            # 保存
                            st_output_file = file_path.replace(
                                "_processed.fif", 
                                f"_stockwell_{epoch_type}_{int(epoch_duration)}s_{target_sfreq}Hz.npy"
                            )
                            np.save(st_output_file, st_normalized)
                            print(f"保存{epoch_type} Stockwell变换结果至: {os.path.basename(st_output_file)}")
                            
                            # 保存标签
                            labels = np.ones(len(epoch_obj)) * (0 if epoch_type == 'regular' else 
                                        1 if epoch_type == 'pre-seizure' else 
                                        3 if epoch_type == 'post-seizure' else 2)
                            labels_file = file_path.replace(
                                "_processed.fif", 
                                f"_labels_{epoch_type}_{int(epoch_duration)}s.npy"
                            )
                            np.save(labels_file, labels)
                        else:  # 特殊处理发作epochs列表
                            for i, seizure_epoch in enumerate(epoch_obj):
                                seizure_output_file = file_path.replace(
                                    "_processed.fif", 
                                    f"_epoched_seizure_{i+1}_{target_sfreq}Hz-epo.fif"
                                )
                                seizure_epoch.save(seizure_output_file, overwrite=True)
                                print(f"保存发作{i+1} epochs数据至: {os.path.basename(seizure_output_file)}")
                                
                                # 对发作应用Stockwell变换
                                print(f"对发作{i+1} epochs应用Stockwell变换...")
                                st_results = apply_stockwell_to_epoch(seizure_epoch, target_sfreq)
                                print(f"完成发作{i+1} Stockwell变换，结果形状: {st_results.shape}")
                                
                                # 归一化
                                st_mean = np.mean(st_results, axis=(2, 3), keepdims=True)
                                st_std = np.std(st_results, axis=(2, 3), keepdims=True)
                                st_std[st_std == 0] = 1.0
                                st_normalized = (st_results - st_mean) / st_std
                                
                                # 保存
                                st_output_file = file_path.replace(
                                    "_processed.fif", 
                                    f"_stockwell_seizure_{i+1}_{target_sfreq}Hz.npy"
                                )
                                np.save(st_output_file, st_normalized)
                                print(f"保存发作{i+1} Stockwell变换结果至: {os.path.basename(st_output_file)}")
                                
                                # 保存标签
                                labels = np.ones(len(seizure_epoch)) * 2  # 标签2表示发作
                                labels_file = file_path.replace(
                                    "_processed.fif", 
                                    f"_labels_seizure_{i+1}.npy"
                                )
                                np.save(labels_file, labels)
                else:
                    # 如果没有成功创建分类的epochs，使用main_epochs继续
                    epochs = main_epochs
                    print(f"创建了 {len(epochs)} 个epochs")
                    
                    # 保存epochs数据
                    epochs_output_file = file_path.replace(
                        "_processed.fif", 
                        f"_epoched_{int(epoch_duration)}s_{target_sfreq}Hz-epo.fif"
                    )
                    epochs.save(epochs_output_file, overwrite=True)
                    print(f"保存epoched数据至: {os.path.basename(epochs_output_file)}")
                    
                    # 应用Stockwell变换
                    print("开始对每个epoch计算Stockwell变换...")
                    st_results = apply_stockwell_to_epoch(epochs, target_sfreq)
                    print(f"完成Stockwell变换，结果形状: {st_results.shape}")
                    
                    # 对Stockwell变换结果进行归一化
                    st_mean = np.mean(st_results, axis=(2, 3), keepdims=True)
                    st_std = np.std(st_results, axis=(2, 3), keepdims=True)
                    st_std[st_std == 0] = 1.0
                    st_normalized = (st_results - st_mean) / st_std
                    
                    # 保存Stockwell变换结果
                    st_output_file = file_path.replace(
                        "_processed.fif", 
                        f"_stockwell_{int(epoch_duration)}s_{target_sfreq}Hz.npy"
                    )
                    np.save(st_output_file, st_normalized)
                    print(f"保存Stockwell变换结果至: {os.path.basename(st_output_file)}")
                    
                    # 保存标签 (全部为0表示常规)
                    labels = np.zeros(len(epochs))
                    labels_file = file_path.replace(
                        "_processed.fif", 
                        f"_labels_regular_{int(epoch_duration)}s.npy"
                    )
                    np.save(labels_file, labels)

        except Exception as e:
            print(f"处理失败: {os.path.basename(file_path)}，错误信息: {e}")
            import traceback
            traceback.print_exc()  # 打印完整错误堆栈

# 步骤6: 通道选择相关函数
class QPSOParticle:
    """Quantum-inspired Particle Swarm Optimization particle"""

    def __init__(self, n_channels, n_select=8):
        self.position = np.zeros(n_channels)
        # Initialize with some random channels selected
        selected_indices = np.random.choice(n_channels, n_select, replace=False)
        self.position[selected_indices] = 1
        self.pbest = self.position.copy()
        self.pbest_fitness = -float('inf')

    def update_position(self, gbest, beta):
        # Quantum-inspired position update
        for i in range(len(self.position)):
            # Calculate quantum probability
            p = random.random()
            if p < 0.5:
                self.position[i] = 1 if random.random() < abs(beta * self.pbest[i] + (1 - beta) * gbest[i]) else 0
            else:
                self.position[i] = 1 if random.random() < abs(beta * gbest[i] + (1 - beta) * self.pbest[i]) else 0

        # Ensure exactly n_select channels are chosen
        indices = np.argsort(self.position)[::-1]
        self.position = np.zeros_like(self.position)
        self.position[indices[:8]] = 1  # n_select changed to 8

def load_stockwell_data(file_path):
    """Load Stockwell transformed EEG data"""
    try:
        data = np.load(file_path)
        print(f"Loaded data with shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def find_label_file(stockwell_file):
    """Find the corresponding label file for a Stockwell file"""
    base_dir = os.path.dirname(stockwell_file)
    base_name = os.path.basename(stockwell_file)

    # Extract the pattern from the stockwell filename
    # Expected pattern: _stockwell_[type]_[duration]s_[freq]Hz.npy
    match = re.search(r'_stockwell_(.+?)_(\d+)s_(\d+)Hz\.npy', base_name)
    if not match:
        return None

    segment_type = match.group(1)
    duration = match.group(2)

    # Try to find matching label file
    # Expected pattern: _labels_[type]_[duration]s.npy
    label_pattern = f"_labels_{segment_type}_{duration}s.npy"

    # For seizure files which might have special naming
    if "seizure" in segment_type and segment_type != "seizure":
        # Handle pre-seizure, post-seizure
        label_pattern = f"_labels_{segment_type}_{duration}s.npy"
    elif "seizure" in segment_type:
        # Extract seizure number if present
        seizure_match = re.search(r'seizure_(\d+)', segment_type)
        if seizure_match:
            seizure_num = seizure_match.group(1)
            label_pattern = f"_labels_seizure_{seizure_num}.npy"
        else:
            label_pattern = f"_labels_seizure.npy"

    # Look for the label file in the same directory
    for file in os.listdir(base_dir):
        if label_pattern in file:
            return os.path.join(base_dir, file)

    # If no specific match found, try a more general approach
    for file in os.listdir(base_dir):
        if f"_labels_{segment_type}" in file:
            return os.path.join(base_dir, file)

    return None

def compute_functional_connectivity(data, method='correlation'):
    """Compute functional connectivity matrix between channels"""
    n_epochs, n_channels, n_freq, n_times = data.shape

    # Reshape data for easier processing
    reshaped_data = np.mean(data, axis=2)  # Average across frequencies
    reshaped_data = reshaped_data.reshape(n_epochs, n_channels, -1)

    # Calculate connectivity matrix
    connectivity = np.zeros((n_channels, n_channels))

    for ch1 in range(n_channels):
        for ch2 in range(n_channels):
            if ch1 == ch2:
                connectivity[ch1, ch2] = 1.0
                continue

            # Correlation across all epochs
            correlations = []
            for epoch in range(n_epochs):
                if method == 'correlation':
                    corr = np.corrcoef(reshaped_data[epoch, ch1], reshaped_data[epoch, ch2])[0, 1]
                    correlations.append(corr)
                elif method == 'mutual_info':
                    # Discretize for mutual information
                    bins = 10
                    c_xy = np.histogram2d(reshaped_data[epoch, ch1], reshaped_data[epoch, ch2], bins)[0]
                    mi = mutual_info_score(None, None, contingency=c_xy)
                    correlations.append(mi)

            connectivity[ch1, ch2] = np.mean(correlations)

    return connectivity

def compute_channel_importance(data):
    """计算通道重要性，替代Granger因果关系分析"""
    n_epochs, n_channels, n_freq, n_times = data.shape

    # 初始化通道重要性得分
    importance_scores = np.zeros(n_channels)

    # 计算每个通道的平均能量
    energy = np.zeros(n_channels)
    for ch in range(n_channels):
        # 平均所有epochs和频率
        ch_data = np.mean(np.abs(data[:, ch, :, :]), axis=(0, 1))
        energy[ch] = np.sum(ch_data ** 2) / len(ch_data)

    # 计算每个通道的频域熵
    entropy = np.zeros(n_channels)
    for ch in range(n_channels):
        # 取平均频谱
        avg_spectrum = np.mean(np.mean(data[:, ch, :, :], axis=0), axis=1)
        # 计算归一化功率谱
        power = np.abs(avg_spectrum) ** 2
        power_norm = power / (np.sum(power) + 1e-10)
        # 计算熵
        power_norm = power_norm[power_norm > 0]  # 避免log(0)
        entropy[ch] = -np.sum(power_norm * np.log2(power_norm))

    # 计算通道间的相关性 - 高相关性表示信息冗余
    redundancy = np.zeros(n_channels)
    avg_data = np.mean(data, axis=(0, 2))  # 平均所有epochs和频率
    for ch1 in range(n_channels):
        ch1_correlations = []
        for ch2 in range(n_channels):
            if ch1 != ch2:
                corr = np.abs(np.corrcoef(avg_data[ch1], avg_data[ch2])[0, 1])
                ch1_correlations.append(corr)
        redundancy[ch1] = np.mean(ch1_correlations)

    # 计算最终重要性得分：高能量、高熵、低冗余的通道更重要
    importance_scores = 0.4 * (energy / np.max(energy)) + \
                        0.4 * (entropy / np.max(entropy)) - \
                        0.2 * (redundancy / np.max(redundancy))

    return importance_scores

def compute_betweenness_centrality(connectivity):
    """Compute betweenness centrality from connectivity matrix"""
    n = connectivity.shape[0]

    # Threshold and binarize connectivity
    threshold = np.percentile(connectivity, 70)  # Top 30% connections
    adj_matrix = (connectivity > threshold).astype(float)
    np.fill_diagonal(adj_matrix, 0)  # No self-connections

    # Initialize centrality values
    betweenness = np.zeros(n)

    # Simple implementation of betweenness centrality
    # For each node pair, find shortest paths and count node occurrences
    for s in range(n):
        for t in range(s + 1, n):
            # Find shortest paths from s to t using breadth-first search
            visited = np.zeros(n, dtype=bool)
            distance = np.ones(n) * np.inf
            distance[s] = 0
            queue = [s]
            predecessors = [[] for _ in range(n)]

            while queue:
                node = queue.pop(0)
                if node == t:
                    break

                if visited[node]:
                    continue

                visited[node] = True
                neighbors = np.where(adj_matrix[node] > 0)[0]

                for neighbor in neighbors:
                    if distance[neighbor] > distance[node] + 1:
                        distance[neighbor] = distance[node] + 1
                        predecessors[neighbor] = [node]
                        queue.append(neighbor)
                    elif distance[neighbor] == distance[node] + 1:
                        predecessors[neighbor].append(node)
                        if neighbor not in queue:
                            queue.append(neighbor)

            # Count paths through each node
            if distance[t] != np.inf:
                # Count shortest paths
                n_paths = np.zeros(n)
                n_paths[s] = 1
                nodes_by_distance = sorted([(distance[i], i) for i in range(n) if distance[i] < np.inf])

                for _, node in nodes_by_distance:
                    for pred in predecessors[node]:
                        n_paths[node] += n_paths[pred]

                # Calculate dependency
                dependency = np.zeros(n)
                nodes_by_reverse_distance = sorted([(distance[i], i) for i in range(n) if distance[i] < np.inf],
                                                   reverse=True)

                for _, node in nodes_by_reverse_distance:
                    if node != s and node != t:
                        for pred in predecessors[node]:
                            if n_paths[node] > 0:
                                dependency[pred] += (n_paths[pred] / n_paths[node]) * (1 + dependency[node])

                betweenness += dependency

    return betweenness

def compute_spectral_entropy(data):
    """计算光谱熵 - 单独函数方便重用"""
    spectral_entropy = 0
    n_epochs, n_channels = data.shape[0], data.shape[1]

    for epoch in range(min(n_epochs, 5)):  # Limit to 5 epochs for speed
        for ch in range(n_channels):
            # Average over frequencies
            signal_avg = np.mean(data[epoch, ch], axis=0)
            ps = np.abs(np.fft.fft(signal_avg)) ** 2
            ps_norm = ps / (np.sum(ps) + 1e-10)
            ps_norm = ps_norm[ps_norm > 0]  # Remove zeros
            if len(ps_norm) > 0:  # Ensure there are non-zero values
                spectral_entropy += -np.sum(ps_norm * np.log2(ps_norm)) / np.log2(len(ps_norm))

    spectral_entropy /= (min(n_epochs, 5) * n_channels)
    return spectral_entropy

def evaluate_channel_subset(data, channel_mask, labels=None):
    """Evaluate a subset of channels based on signal characteristics"""
    selected_data = data[:, channel_mask.astype(bool), :, :]

    # If labels are available, use classification accuracy
    if labels is not None and len(labels) == data.shape[0]:
        # 修复：处理样本数量为1的情况
        if len(labels) == 1:
            # 如果只有一个样本，使用信号特征进行评估
            variances = np.var(np.mean(selected_data, axis=2), axis=2).mean()
            spectral_entropy = compute_spectral_entropy(selected_data)
            channel_count_score = 1 - abs(np.sum(channel_mask) - 8) / 8  # n_select changed to 8

            # Combined score
            score = 0.4 * variances + 0.4 * spectral_entropy + 0.2 * channel_count_score
            return score

        # 正常情况下的分类评估
        variances = np.var(np.mean(selected_data, axis=2), axis=2)  # Variance across time
        X_train, X_test, y_train, y_test = train_test_split(variances, labels, test_size=0.3, random_state=42)

        # Simple threshold classifier
        best_accuracy = 0
        best_threshold = 0
        for threshold in np.linspace(np.min(X_train), np.max(X_train), 20):
            predictions = (X_train.mean(axis=1) > threshold).astype(int)
            accuracy = np.mean(predictions == y_train)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        # Evaluate on test set
        test_predictions = (X_test.mean(axis=1) > best_threshold).astype(int)
        accuracy = np.mean(test_predictions == y_test)
        return accuracy

    # Without labels, use signal characteristics
    # 1. Signal variance (higher is better)
    variances = np.var(np.mean(selected_data, axis=2), axis=2).mean()

    # 2. Spectral entropy
    spectral_entropy = compute_spectral_entropy(selected_data)

    # 3. Optimal number of channels (closer to n_select is better)
    channel_count_score = 1 - abs(np.sum(channel_mask) - 8) / 8  # n_select changed to 8

    # Combined score
    score = 0.4 * variances + 0.4 * spectral_entropy + 0.2 * channel_count_score
    return score

def quantum_pso_channel_selection(data, centrality_scores, importance_scores, n_select=8, n_particles=10, iterations=20,
                                  labels=None):
    """Select channels using Quantum PSO"""
    n_channels = data.shape[1]

    # Initialize particles
    particles = [QPSOParticle(n_channels, n_select) for _ in range(n_particles)]
    gbest = np.zeros(n_channels)
    gbest_fitness = -float('inf')

    # Initialize particles with some bias from centrality and importance scores
    for i, particle in enumerate(particles):
        # Add bias toward high centrality and importance channels
        combined_score = 0.5 * centrality_scores / (centrality_scores.max() + 1e-10) + 0.5 * importance_scores / (
                    importance_scores.max() + 1e-10)
        top_indices = np.argsort(combined_score)[-n_select:]
        particle.position = np.zeros(n_channels)
        particle.position[top_indices] = 1

        # Evaluate initial position
        fitness = evaluate_channel_subset(data, particle.position, labels)
        particle.pbest_fitness = fitness

        if fitness > gbest_fitness:
            gbest_fitness = fitness
            gbest = particle.position.copy()

    # Main QPSO loop
    history = []
    for t in range(iterations):
        beta = 1.0 - 0.5 * t / iterations  # Linear decreasing beta

        for particle in particles:
            # Update position
            particle.update_position(gbest, beta)

            # Evaluate fitness
            fitness = evaluate_channel_subset(data, particle.position, labels)

            # Update personal best
            if fitness > particle.pbest_fitness:
                particle.pbest_fitness = fitness
                particle.pbest = particle.position.copy()

            # Update global best
            if fitness > gbest_fitness:
                gbest_fitness = fitness
                gbest = particle.position.copy()

        history.append(gbest_fitness)
        print(f"Iteration {t + 1}/{iterations}, Best fitness: {gbest_fitness:.4f}, Selected channels: {np.sum(gbest)}")

    # Get final selected channels
    selected_channels = np.where(gbest == 1)[0]

    # If not exactly n_select channels, force it
    if len(selected_channels) != n_select:
        combined_score = 0.5 * centrality_scores / (centrality_scores.max() + 1e-10) + 0.5 * importance_scores / (
                    importance_scores.max() + 1e-10)
        if len(selected_channels) < n_select:
            # Add more channels
            remaining = list(set(range(n_channels)) - set(selected_channels))
            sorted_remaining = sorted(remaining, key=lambda x: combined_score[x], reverse=True)
            selected_channels = np.append(selected_channels, sorted_remaining[:n_select - len(selected_channels)])
        else:
            # Remove excess channels
            sorted_channels = sorted(selected_channels, key=lambda x: combined_score[x])
            selected_channels = np.array(sorted_channels[len(selected_channels) - n_select:])

    # Calculate final channel rankings
    rankings = np.zeros(n_channels)
    rankings[selected_channels] = range(n_select, 0, -1)  # Higher rank is better (n_select is best)

    return selected_channels, rankings, history

def process_and_select_channels(input_file, n_select=8):
    """Process Stockwell data and select the best channels"""
    print(f"\nProcessing file: {os.path.basename(input_file)}")
    
    # 检查磁盘空间
    check_disk_space(os.path.dirname(input_file))
    
    # 设置输出文件名
    base_name = os.path.basename(input_file)
    dir_name = os.path.dirname(input_file)
    output_file = os.path.join(dir_name, base_name.replace('.npy', f'_selected{n_select}ch.npy'))

    # Load data
    data = load_stockwell_data(input_file)
    if data is None:
        return

    # Load labels if available
    label_file = find_label_file(input_file)
    labels = None
    if label_file and os.path.exists(label_file):
        try:
            labels = np.load(label_file)
            print(f"Loaded {len(labels)} labels (from {os.path.basename(label_file)})")
        except Exception as e:
            print(f"Error loading labels: {e}")

    n_epochs, n_channels, n_freq, n_times = data.shape
    print(f"Processing data: {n_epochs} epochs, {n_channels} channels")

    # 1. Compute functional connectivity
    print("Computing functional connectivity...")
    connectivity = compute_functional_connectivity(data)

    # 2. Compute centrality measures
    print("Computing betweenness centrality...")
    centrality = compute_betweenness_centrality(connectivity)

    # 3. Compute channel importance 
    print("Computing channel importance...")
    importance_scores = compute_channel_importance(data)

    # 4. Select channels using QPSO
    print("Selecting channels using QPSO...")
    selected_channels, channel_rankings, history = quantum_pso_channel_selection(
        data, centrality, importance_scores, n_select=n_select, labels=labels
    )

    # Print information about selected channels and rankings
    print("\n----- CHANNEL RANKINGS -----")
    # Sort channels by ranking for display
    channel_ranking_pairs = [(i, channel_rankings[i]) for i in range(n_channels)]
    sorted_channels = sorted(channel_ranking_pairs, key=lambda x: x[1], reverse=True)

    # Print all channels sorted by importance
    print(f"{'Channel':<8}{'Rank':<8}{'Centrality':<12}{'Importance':<12}{'Selected':<8}")
    print("-" * 48)

    for ch_idx, rank in sorted_channels:
        is_selected = "Yes" if ch_idx in selected_channels else "No"
        print(
            f"{ch_idx:<8}{int(rank) if rank > 0 else '-':<8}{centrality[ch_idx]:<12.4f}{importance_scores[ch_idx]:<12.4f}{is_selected:<8}")

    print("\n----- SELECTED CHANNELS SUMMARY -----")
    print(f"Selected {len(selected_channels)} channels: {', '.join(map(str, selected_channels))}")

    # Extract selected channel data
    selected_data = data[:, selected_channels, :, :]

    # Save selected data
    np.save(output_file, selected_data)
    print(f"Selected channel data saved to {os.path.basename(output_file)}")
    
    return output_file

# 修改步骤6，处理单个文件夹
def step6_channel_selection(data_dir, n_select=8):
    """Process all Stockwell files in a directory and select important channels"""
    processed_files = []
    
    print(f"\n处理目录: {data_dir}")
    print(f"Searching for Stockwell transform files in {data_dir}...")
    
    # 检查磁盘空间
    check_disk_space(data_dir)

    # Find all Stockwell transformed files (exclude already selected ones)
    stockwell_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if "_stockwell_" in file and file.endswith(".npy") and "selected" not in file:
                stockwell_files.append(os.path.join(root, file))

    print(f"Found {len(stockwell_files)} Stockwell transform files")

    # Process each file
    for i, file in enumerate(stockwell_files):
        print(f"\nProcessing file {i + 1}/{len(stockwell_files)}: {os.path.basename(file)}")
        try:
            # 检查磁盘空间
            check_disk_space(os.path.dirname(file))
            
            output_file = process_and_select_channels(file, n_select=n_select)
            if output_file:
                processed_files.append(output_file)
        except Exception as e:
            print(f"Error processing file {os.path.basename(file)}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nChannel selection complete! Processed {len(processed_files)} files")
    return processed_files

# 新增：完整处理一个文件夹的所有步骤
def process_folder_complete(folder_path, args):
    """完整处理一个文件夹的所有预处理步骤"""
    print(f"\n===========================================================")
    print(f"开始处理文件夹: {folder_path}")
    print(f"===========================================================")
    
    try:
        # 步骤1: 处理有发作的EDF文件
        print("\n步骤1: 处理有发作的EDF文件...")
        step1_process_seizure_files(folder_path)
        
        # 步骤2: 处理无发作的EDF文件
        print("\n步骤2: 处理无发作的EDF文件...")
        step2_process_non_seizure_files(folder_path)
        
        # 步骤3: 删除SSP投影器
        print("\n步骤3: 删除SSP投影器...")
        step3_remove_ssp_projectors(folder_path)
        
        # 步骤4: 应用高通和陷波滤波
        print("\n步骤4: 应用高通和陷波滤波...")
        step4_apply_filters(folder_path)
        
        # 步骤5: 高级信号处理
        print("\n步骤5: 高级信号处理与Stockwell变换...")
        step5_advanced_processing(folder_path, 
                                  target_sfreq=args.target_sfreq,
                                  epoch_duration=args.epoch_duration,
                                  pre_seizure_time=args.pre_seizure_time,
                                  post_seizure_time=args.post_seizure_time,
                                  consider_annotations=args.consider_annotations)
        
        # 步骤6: 通道选择
        print("\n步骤6: 通道选择...")
        step6_channel_selection(folder_path, n_select=args.n_select)
        
        print(f"\n文件夹 {folder_path} 处理完成!")
        
        # 删除所有FIF文件，释放空间
        print(f"\n清理FIF文件以释放空间...")
        if os.getenv("EEG_ENABLE_CLEANUP") == "1":
            delete_fif_files(folder_path)
        else:
            print("Preserving FIF intermediates; set EEG_ENABLE_CLEANUP=1 to remove them.")
        
        return True
    
    except Exception as e:
        print(f"处理文件夹 {folder_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

# 修改主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="EEG数据预处理与通道选择流水线")
    parser.add_argument("--data_dirs", nargs='+', type=str, required=True, help="一个或多个EDF文件所在目录")
    parser.add_argument("--target_sfreq", type=float, default=128, help="目标采样率")
    parser.add_argument("--epoch_duration", type=float, default=60.0, help="时间窗口大小(秒)")
    parser.add_argument("--pre_seizure_time", type=float, default=60.0, help="发作前时间(秒)")
    parser.add_argument("--post_seizure_time", type=float, default=60.0, help="发作后时间(秒)")
    parser.add_argument("--n_select", type=int, default=8, help="要选择的通道数量")
    parser.add_argument("--no_vis", action="store_true", help="禁用可视化")
    parser.add_argument("--consider_annotations", action="store_true", default=True, help="考虑癫痫发作注释")
    parser.add_argument("--min_disk_space", type=float, default=5.0, help="最小所需磁盘空间(GB)")
    
    args = parser.parse_args()
    
    # 修改为一个文件夹一个文件夹地完整处理
    print("开始EEG数据预处理流水线...")
    print(f"待处理 {len(args.data_dirs)} 个数据目录: {', '.join(args.data_dirs)}")
    
    # 检查初始磁盘空间
    if not check_disk_space(args.data_dirs[0], min_space_gb=args.min_disk_space):
        print("初始磁盘空间检查失败，程序终止")
        return
    
    # 逐个处理文件夹
    successful_folders = 0
    for folder_path in args.data_dirs:
        success = process_folder_complete(folder_path, args)
        if success:
            successful_folders += 1
    
    print(f"\n预处理流水线完成! 成功处理 {successful_folders}/{len(args.data_dirs)} 个文件夹")
    
if __name__ == "__main__":
    main() 