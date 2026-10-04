#!/usr/bin/env python3
"""
Exported from 正常人处理流程.ipynb.
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
import logging

# 配置日志级别，减少输出
mne.set_log_level('WARNING')  # 减少MNE日志输出
logging.basicConfig(level=logging.WARNING)  # 设置基本日志级别

# 禁用matplotlib交互显示
plt.ioff()
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# 检查磁盘空间是否足够
def check_disk_space(path, min_gb=5):
    """检查指定路径所在磁盘的可用空间是否足够"""
    free_bytes = shutil.disk_usage(path).free
    free_gb = free_bytes / (1024 ** 3)

    if free_gb < min_gb:
        print(f"警告: 磁盘空间不足! 剩余 {free_gb:.2f} GB, 至少需要 {min_gb} GB")
        return False
    return True

# 卡尔曼滤波基线校正
def apply_kalman_filter(data, R=0.01):
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

# 对数据进行Z-分数归一化
def apply_zscore(data):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (data - mean) / std

# Stockwell变换
def stockwell_transform(data, fs, fmin=0, fmax=None, n_freq=40, use_gpu=False, chunk_size=10000):
    """对信号进行Stockwell变换，可选GPU加速"""
    n_channels, n_samples = data.shape
    
    # 自动设置最大频率为Nyquist频率的80%
    if fmax is None or fmax > 0.8 * (fs / 2):
        fmax = 0.8 * (fs / 2)
    
    # 生成频率点 (对数均匀分布)
    freq_bins = np.logspace(np.log10(max(0.5, fmin)), np.log10(fmax), n_freq)
    
    # 如果启用GPU加速，尝试导入相关库
    if use_gpu:
        try:
            import cupy as cp
            import cupyx.scipy.fftpack as cufft
            
            # 分块处理以减少内存使用
            def process_channel_gpu(channel_data):
                # 计算需要处理的块数
                n_chunks = int(np.ceil(len(channel_data) / chunk_size))
                result = np.zeros((n_freq, len(channel_data)), dtype=complex)
                
                for i in range(n_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, len(channel_data))
                    chunk = channel_data[start_idx:end_idx]
                    
                    # 处理当前块
                    X = cufft.fft(cp.asarray(chunk))
                    chunk_len = len(chunk)
                    k = cp.arange(chunk_len)
                    
                    for j, freq in enumerate(freq_bins):
                        sigma = freq / fs
                        gaussian_window = cp.exp(-2 * (cp.pi * sigma * k / chunk_len) ** 2)
                        Xw = X * gaussian_window
                        result[j, start_idx:end_idx] = cp.asnumpy(cufft.ifft(Xw))
                
                return result
            
            # 并行处理每个通道
            st_results = Parallel(n_jobs=-1, prefer="threads")(
                delayed(process_channel_gpu)(data[i, :]) for i in range(n_channels)
            )
            
            print("使用GPU加速完成Stockwell变换")
            return np.array(st_results)
            
        except ImportError:
            print("未找到GPU加速库(CuPy)，使用CPU计算")
            use_gpu = False
    
    # CPU版本的Stockwell变换，分块处理
    def process_channel_cpu(channel_data):
        # 计算需要处理的块数
        n_chunks = int(np.ceil(len(channel_data) / chunk_size))
        result = np.zeros((n_freq, len(channel_data)), dtype=complex)
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(channel_data))
            chunk = channel_data[start_idx:end_idx]
            
            # 处理当前块
            X = fftpack.fft(chunk)
            chunk_len = len(chunk)
            k = np.arange(chunk_len)
            
            for j, freq in enumerate(freq_bins):
                sigma = freq / fs
                gaussian_window = np.exp(-2 * (np.pi * sigma * k / chunk_len) ** 2)
                Xw = X * gaussian_window
                result[j, start_idx:end_idx] = fftpack.ifft(Xw)
        
        return result
    
    # 并行处理每个通道
    print(f"使用CPU处理Stockwell变换，分块大小: {chunk_size}采样点")
    st_results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(process_channel_cpu)(data[i, :]) for i in range(n_channels)
    )
    
    return np.array(st_results)

# 合并文件夹中的所有EDF文件
def merge_subject_edf_files(folder_path, target_sfreq=128, use_gpu=False, verbose=False):
    """将文件夹中每个EDF文件单独处理并生成对应的NPY文件"""
    print(f"处理受试者文件夹: {os.path.basename(folder_path)}")
    
    # 检查磁盘空间
    if not check_disk_space(folder_path):
        print(f"磁盘空间不足，跳过文件夹: {folder_path}")
        return None
    
    # 获取文件夹下所有EDF文件
    edf_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.edf'):
                edf_files.append(os.path.join(root, file))
    
    if not edf_files:
        print(f"未在 {folder_path} 中找到EDF文件")
        return None
    
    print(f"找到 {len(edf_files)} 个EDF文件")
    
    # 存储处理后文件的路径
    processed_files = []
    
    # 使用tqdm显示进度
    for file_idx, file_path in enumerate(tqdm(edf_files, desc="处理EDF文件", ncols=100, leave=False)):
        try:
            # 为每个文件创建唯一的输出文件名
            file_basename = os.path.basename(file_path).replace('.edf', '')
            output_file = os.path.join(folder_path, f"{file_basename}_stockwell.npy")
            
            # 读取EDF文件
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=verbose)
            
            # 获取采样率
            sfreq = raw.info['sfreq']
            
            # 计算60秒的采样点数
            samples_60sec = int(60 * sfreq)
            
            # 如果文件长度小于60秒，使用整个文件
            file_length = raw.n_times
            if file_length < samples_60sec:
                print(f"文件 {os.path.basename(file_path)} 长度小于60秒，使用全部数据")
                cropped_raw = raw
            else:
                # 截取前60秒
                cropped_raw = raw.copy().crop(tmin=0, tmax=60, include_tmax=False)
            
            # 计算Nyquist频率的80%作为低通滤波器截止频率
            lowpass_freq = min(80, 0.8 * (sfreq / 2))
            
            # 选择所有数据通道
            all_picks = mne.pick_types(cropped_raw.info, eeg=True, meg=False, exclude='bads')
            
            # 高通滤波
            raw_highpass = cropped_raw.copy().filter(l_freq=0.2, h_freq=None, picks=all_picks, verbose=verbose)
            
            # 低通滤波
            raw_filtered = raw_highpass.copy().filter(l_freq=None, h_freq=lowpass_freq, method="fir", picks=all_picks, verbose=verbose)
            
            # 下采样
            raw_downsampled = raw_filtered.copy().resample(
                sfreq=target_sfreq, method="polyphase", n_jobs=-1, verbose=verbose
            )
            
            # 电源线干扰去除
            new_sfreq = raw_downsampled.info['sfreq']
            powerline_freqs = tuple([i * 60 for i in range(1, int(new_sfreq / 120) + 1)])
            raw_notch = raw_downsampled.copy().notch_filter(freqs=powerline_freqs, picks=all_picks, verbose=verbose)
            
            # 获取处理后的数据
            data = raw_notch.get_data()
            
            # 卡尔曼滤波进行基线校正
            num_channels = data.shape[0]
            processed_channels = Parallel(n_jobs=-1)(
                delayed(apply_kalman_filter)(data[i, :]) for i in range(num_channels)
            )
            processed_data = np.array(processed_channels)
            
            # 应用Z-分数归一化
            normalized_data = apply_zscore(processed_data)
            
            # 创建虚拟Raw对象用于ICA
            ch_names = [f'EEG {i+1}' for i in range(normalized_data.shape[0])]
            if raw_notch.info:
                # 如果有参考信息，尝试使用原始通道名
                if len(raw_notch.info['ch_names']) == normalized_data.shape[0]:
                    ch_names = raw_notch.info['ch_names']
            
            # 创建用于ICA的raw对象
            info = mne.create_info(ch_names=ch_names, sfreq=target_sfreq, ch_types='eeg')
            raw = mne.io.RawArray(normalized_data, info)
            
            # 应用ICA去除伪迹
            try:
                ica = ICA(n_components=0.99, method='picard', random_state=42, max_iter=500, verbose=verbose)
                ica.fit(raw, verbose=verbose)
                
                # 识别伪迹成分
                components = ica.get_sources(raw).get_data()
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
                
                if ica.exclude:
                    print(f"排除了 {len(ica.exclude)} 个伪迹ICA成分")
                    cleaned_raw = ica.apply(raw.copy(), verbose=verbose)
                    normalized_data = cleaned_raw.get_data()
                else:
                    print("未检测到明显伪迹")
                    
            except Exception as e:
                print(f"ICA处理失败: {e}")
            
            # 应用Stockwell变换
            print(f"对文件 {os.path.basename(file_path)} 应用Stockwell变换...")
            
            # 使用小块处理以减少内存使用
            chunk_size = 2000  # 设置更小的块大小，减少内存使用
            
            st_data = stockwell_transform(normalized_data, fs=target_sfreq, use_gpu=use_gpu, chunk_size=chunk_size)
            
            # 计算幅度
            st_amplitude = np.abs(st_data)
            
            # 重塑为(1, channels, frequencies, samples)格式
            st_amplitude = st_amplitude.reshape(1, st_amplitude.shape[0], st_amplitude.shape[1], st_amplitude.shape[2])
            
            # 归一化
            st_mean = np.mean(st_amplitude, axis=(2, 3), keepdims=True)
            st_std = np.std(st_amplitude, axis=(2, 3), keepdims=True)
            st_std[st_std == 0] = 1.0
            st_normalized = (st_amplitude - st_mean) / st_std
            
            # 保存Stockwell变换结果
            np.save(output_file, st_normalized)
            print(f"已保存Stockwell变换结果: {os.path.basename(output_file)}")
            print(f"数据形状: {st_normalized.shape}")
            
            processed_files.append(output_file)
            
        except Exception as e:
            print(f"处理文件 {os.path.basename(file_path)} 时出错: {e}")
            continue
    
    if not processed_files:
        print("所有文件处理失败")
        return None
    
    print(f"成功处理了 {len(processed_files)}/{len(edf_files)} 个EDF文件")
    return processed_files

# QPSOParticle定义保持不变
class QPSOParticle:
    """Quantum-inspired Particle Swarm Optimization particle"""
    def __init__(self, n_channels, n_select=8):
        self.position = np.zeros(n_channels)
        selected_indices = np.random.choice(n_channels, n_select, replace=False)
        self.position[selected_indices] = 1
        self.pbest = self.position.copy()
        self.pbest_fitness = -float('inf')

    def update_position(self, gbest, beta):
        for i in range(len(self.position)):
            p = random.random()
            if p < 0.5:
                self.position[i] = 1 if random.random() < abs(beta * self.pbest[i] + (1 - beta) * gbest[i]) else 0
            else:
                self.position[i] = 1 if random.random() < abs(beta * gbest[i] + (1 - beta) * self.pbest[i]) else 0

        indices = np.argsort(self.position)[::-1]
        self.position = np.zeros_like(self.position)
        self.position[indices[:8]] = 1

# 功能连接计算，保持代码，但简化日志输出
def compute_functional_connectivity(data, method='correlation'):
    n_epochs, n_channels, n_freq, n_times = data.shape
    reshaped_data = np.mean(data, axis=2)
    reshaped_data = reshaped_data.reshape(n_epochs, n_channels, -1)
    connectivity = np.zeros((n_channels, n_channels))

    for ch1 in range(n_channels):
        for ch2 in range(n_channels):
            if ch1 == ch2:
                connectivity[ch1, ch2] = 1.0
                continue

            correlations = []
            for epoch in range(n_epochs):
                if method == 'correlation':
                    corr = np.corrcoef(reshaped_data[epoch, ch1], reshaped_data[epoch, ch2])[0, 1]
                    correlations.append(corr)
                elif method == 'mutual_info':
                    bins = 10
                    c_xy = np.histogram2d(reshaped_data[epoch, ch1], reshaped_data[epoch, ch2], bins)[0]
                    mi = mutual_info_score(None, None, contingency=c_xy)
                    correlations.append(mi)

            connectivity[ch1, ch2] = np.mean(correlations)

    return connectivity

def compute_channel_importance(data):
    n_epochs, n_channels, n_freq, n_times = data.shape
    importance_scores = np.zeros(n_channels)

    # 计算每个通道的平均能量
    energy = np.zeros(n_channels)
    for ch in range(n_channels):
        ch_data = np.mean(np.abs(data[:, ch, :, :]), axis=(0, 1))
        energy[ch] = np.sum(ch_data ** 2) / len(ch_data)

    # 计算每个通道的频域熵
    entropy = np.zeros(n_channels)
    for ch in range(n_channels):
        avg_spectrum = np.mean(np.mean(data[:, ch, :, :], axis=0), axis=1)
        power = np.abs(avg_spectrum) ** 2
        power_norm = power / (np.sum(power) + 1e-10)
        power_norm = power_norm[power_norm > 0]
        entropy[ch] = -np.sum(power_norm * np.log2(power_norm))

    # 计算通道间的相关性
    redundancy = np.zeros(n_channels)
    avg_data = np.mean(data, axis=(0, 2))
    for ch1 in range(n_channels):
        ch1_correlations = []
        for ch2 in range(n_channels):
            if ch1 != ch2:
                corr = np.abs(np.corrcoef(avg_data[ch1], avg_data[ch2])[0, 1])
                ch1_correlations.append(corr)
        redundancy[ch1] = np.mean(ch1_correlations)

    # 计算最终重要性得分
    importance_scores = 0.4 * (energy / np.max(energy)) + \
                        0.4 * (entropy / np.max(entropy)) - \
                        0.2 * (redundancy / np.max(redundancy))

    return importance_scores

def compute_betweenness_centrality(connectivity):
    n = connectivity.shape[0]
    threshold = np.percentile(connectivity, 70)
    adj_matrix = (connectivity > threshold).astype(float)
    np.fill_diagonal(adj_matrix, 0)
    betweenness = np.zeros(n)

    for s in range(n):
        for t in range(s + 1, n):
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

            if distance[t] != np.inf:
                n_paths = np.zeros(n)
                n_paths[s] = 1
                nodes_by_distance = sorted([(distance[i], i) for i in range(n) if distance[i] < np.inf])

                for _, node in nodes_by_distance:
                    for pred in predecessors[node]:
                        n_paths[node] += n_paths[pred]

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
    spectral_entropy = 0
    n_epochs, n_channels = data.shape[0], data.shape[1]

    for epoch in range(min(n_epochs, 5)):
        for ch in range(n_channels):
            signal_avg = np.mean(data[epoch, ch], axis=0)
            ps = np.abs(np.fft.fft(signal_avg)) ** 2
            ps_norm = ps / (np.sum(ps) + 1e-10)
            ps_norm = ps_norm[ps_norm > 0]
            if len(ps_norm) > 0:
                spectral_entropy += -np.sum(ps_norm * np.log2(ps_norm)) / np.log2(len(ps_norm))

    spectral_entropy /= (min(n_epochs, 5) * n_channels)
    return spectral_entropy

def evaluate_channel_subset(data, channel_mask, labels=None):
    selected_data = data[:, channel_mask.astype(bool), :, :]

    if labels is not None and len(labels) == data.shape[0]:
        if len(labels) == 1:
            variances = np.var(np.mean(selected_data, axis=2), axis=2).mean()
            spectral_entropy = compute_spectral_entropy(selected_data)
            channel_count_score = 1 - abs(np.sum(channel_mask) - 8) / 8
            score = 0.4 * variances + 0.4 * spectral_entropy + 0.2 * channel_count_score
            return score

        variances = np.var(np.mean(selected_data, axis=2), axis=2)
        X_train, X_test, y_train, y_test = train_test_split(variances, labels, test_size=0.3, random_state=42)

        best_accuracy = 0
        best_threshold = 0
        for threshold in np.linspace(np.min(X_train), np.max(X_train), 20):
            predictions = (X_train.mean(axis=1) > threshold).astype(int)
            accuracy = np.mean(predictions == y_train)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        test_predictions = (X_test.mean(axis=1) > best_threshold).astype(int)
        accuracy = np.mean(test_predictions == y_test)
        return accuracy

    variances = np.var(np.mean(selected_data, axis=2), axis=2).mean()
    spectral_entropy = compute_spectral_entropy(selected_data)
    channel_count_score = 1 - abs(np.sum(channel_mask) - 8) / 8
    score = 0.4 * variances + 0.4 * spectral_entropy + 0.2 * channel_count_score
    return score

def quantum_pso_channel_selection(data, centrality_scores, importance_scores, n_select=8, n_particles=10, iterations=20,
                                  labels=None):
    n_channels = data.shape[1]
    particles = [QPSOParticle(n_channels, n_select) for _ in range(n_particles)]
    gbest = np.zeros(n_channels)
    gbest_fitness = -float('inf')

    # 初始化粒子
    for i, particle in enumerate(particles):
        combined_score = 0.5 * centrality_scores / (centrality_scores.max() + 1e-10) + 0.5 * importance_scores / (
                importance_scores.max() + 1e-10)
        top_indices = np.argsort(combined_score)[-n_select:]
        particle.position = np.zeros(n_channels)
        particle.position[top_indices] = 1

        fitness = evaluate_channel_subset(data, particle.position, labels)
        particle.pbest_fitness = fitness

        if fitness > gbest_fitness:
            gbest_fitness = fitness
            gbest = particle.position.copy()

    # 主循环
    history = []
    for t in range(iterations):
        beta = 1.0 - 0.5 * t / iterations

        for particle in particles:
            particle.update_position(gbest, beta)
            fitness = evaluate_channel_subset(data, particle.position, labels)

            if fitness > particle.pbest_fitness:
                particle.pbest_fitness = fitness
                particle.pbest = particle.position.copy()

            if fitness > gbest_fitness:
                gbest_fitness = fitness
                gbest = particle.position.copy()

        history.append(gbest_fitness)
        if (t + 1) % 5 == 0 or t == 0 or t == iterations - 1:  # 减少日志输出
            print(f"迭代 {t + 1}/{iterations}, 最佳适应度: {gbest_fitness:.4f}")

    # 获取最终选择的通道
    selected_channels = np.where(gbest == 1)[0]

    # 确保选择的通道数量正确
    if len(selected_channels) != n_select:
        combined_score = 0.5 * centrality_scores / (centrality_scores.max() + 1e-10) + 0.5 * importance_scores / (
                importance_scores.max() + 1e-10)
        if len(selected_channels) < n_select:
            remaining = list(set(range(n_channels)) - set(selected_channels))
            sorted_remaining = sorted(remaining, key=lambda x: combined_score[x], reverse=True)
            selected_channels = np.append(selected_channels, sorted_remaining[:n_select - len(selected_channels)])
        else:
            sorted_channels = sorted(selected_channels, key=lambda x: combined_score[x])
            selected_channels = np.array(sorted_channels[len(selected_channels) - n_select:])

    # 计算最终通道排名
    rankings = np.zeros(n_channels)
    rankings[selected_channels] = range(n_select, 0, -1)

    return selected_channels, rankings, history

# 处理单个Stockwell数据文件并选择通道
def process_and_select_channels(input_file, n_select=8):
    """处理Stockwell数据并选择最佳通道"""
    print(f"处理文件: {os.path.basename(input_file)}")

    # 设置输出文件名
    base_name = os.path.basename(input_file)
    dir_name = os.path.dirname(input_file)
    output_file = os.path.join(dir_name, base_name.replace('.npy', f'_selected{n_select}ch.npy'))

    # 加载数据
    data = np.load(input_file)
    print(f"数据形状: {data.shape}")

    n_epochs, n_channels, n_freq, n_times = data.shape

    # 计算功能连接矩阵
    connectivity = compute_functional_connectivity(data)

    # 计算中心性测量
    centrality = compute_betweenness_centrality(connectivity)

    # 计算通道重要性
    importance_scores = compute_channel_importance(data)

    # 使用QPSO选择通道
    print("使用QPSO选择通道...")
    selected_channels, channel_rankings, history = quantum_pso_channel_selection(
        data, centrality, importance_scores, n_select=n_select
    )

    # 打印通道选择摘要
    print(f"选择了 {len(selected_channels)} 个通道: {', '.join(map(str, selected_channels))}")

    # 提取选定通道数据
    selected_data = data[:, selected_channels, :, :]

    # 保存选定通道数据
    np.save(output_file, selected_data)
    print(f"已保存选定通道数据: {os.path.basename(output_file)}")
    
    # 保存通道索引
    channels_file = os.path.join(dir_name, base_name.replace('.npy', f'_channel_indices.npy'))
    np.save(channels_file, selected_channels)

    return output_file

# 处理正常人EEG数据
def process_normal_eeg_data(data_dirs, target_sfreq=128, n_select=8, use_gpu=False, verbose=False):
    """处理多个正常人EEG数据文件夹"""
    print(f"开始处理 {len(data_dirs)} 个数据目录...")
    
    results = []
    
    for folder_path in data_dirs:
        print(f"\n==== 处理受试者: {os.path.basename(folder_path)} ====")
        
        # 检查磁盘空间
        if not check_disk_space(folder_path):
            print(f"磁盘空间不足，跳过受试者 {folder_path}")
            continue
        
        try:
            # 步骤1: 处理EDF文件生成单独的NPY文件
            print("步骤1: 处理EDF文件并生成NPY文件...")
            stockwell_files = merge_subject_edf_files(folder_path, target_sfreq, use_gpu, verbose)
            
            if stockwell_files is None or len(stockwell_files) == 0:
                print(f"无法处理受试者 {folder_path}，跳过")
                continue
            
            # 步骤2: 为每个NPY文件选择重要通道
            print(f"步骤2: 为每个NPY文件选择重要通道，共{len(stockwell_files)}个文件...")
            
            selected_files = []
            for i, stockwell_file in enumerate(stockwell_files):
                print(f"\n处理文件 {i+1}/{len(stockwell_files)}: {os.path.basename(stockwell_file)}")
                try:
                    selected_file = process_and_select_channels(stockwell_file, n_select)
                    selected_files.append(selected_file)
                except Exception as e:
                    print(f"处理文件 {stockwell_file} 时出错: {e}")
            
            results.extend(selected_files)
            print(f"受试者 {os.path.basename(folder_path)} 处理完成，成功生成 {len(selected_files)} 个选定通道NPY文件")
            
        except Exception as e:
            print(f"处理受试者 {folder_path} 时出错: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n预处理完成，成功处理 {len(results)} 个NPY文件")
    return results

# 修改主函数
def main():
    parser = argparse.ArgumentParser(description="正常人EEG数据预处理流水线")
    parser.add_argument("--data_dirs", nargs='+', type=str, required=True, help="正常人EEG数据目录列表")
    parser.add_argument("--target_sfreq", type=float, default=128, help="目标采样率")
    parser.add_argument("--n_select", type=int, default=8, help="要选择的通道数量")
    parser.add_argument("--min_space", type=float, default=5.0, help="最小所需磁盘空间(GB)")
    parser.add_argument("--use_gpu", action="store_true", help="启用GPU加速")
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")

    args = parser.parse_args()

    # 如果启用GPU加速，检查是否可用
    if args.use_gpu:
        try:
            import cupy
            print("找到GPU加速库CuPy，将使用GPU加速")
        except ImportError:
            print("警告: 未找到CuPy库，无法使用GPU加速。将使用CPU计算。")
            print("可以通过安装CuPy启用GPU加速: pip install cupy-cuda11x (根据您的CUDA版本选择)")
            args.use_gpu = False

    # 处理正常人EEG数据
    process_normal_eeg_data(
        args.data_dirs,
        target_sfreq=args.target_sfreq,
        n_select=args.n_select,
        use_gpu=args.use_gpu,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main() 




