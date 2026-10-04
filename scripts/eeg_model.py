#!/usr/bin/env python3
"""
Exported from 深度学习模型.ipynb.
Historical research code for the Transformer-based seizure prediction system.
Original test data are unavailable; provide your own public/approved data paths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import math
import os
import argparse
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

try:
    from torchdiffeq import odeint_adjoint as odeint
except ImportError:
    print("Warning: torchdiffeq not installed. Using simplified ODE implementation.")
    odeint = None


# Memory-efficient attention mechanism
class EfficientAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # x: [batch_size, seq_len, embed_dim]
        batch_size, seq_len, _ = x.shape

        # Get query, key, value projections
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2), qkv)

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply masks if provided
        if attn_mask is not None:
            attn = attn + attn_mask
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Apply softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Get output
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)

        return out


# Advanced Rotary Position Embedding with fixed seq_len_cached initialization
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        # Initialize seq_len_cached to 0 instead of None to avoid comparison errors
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]

        if seq_len > self.seq_len_cached or self.cos_cached is None or self.sin_cached is None:
            self.seq_len_cached = seq_len
            t = torch.arange(self.seq_len_cached, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset=0):
    q_len = q.shape[1]
    k_len = k.shape[1]

    # Handle offset
    cos_q = cos[offset:offset + q_len]
    sin_q = sin[offset:offset + q_len]
    cos_k = cos[offset:offset + k_len]
    sin_k = sin[offset:offset + k_len]

    # Apply rotary embeddings
    q = (q * cos_q) + (rotate_half(q) * sin_q)
    k = (k * cos_k) + (rotate_half(k) * sin_k)

    return q, k


# Improved TransformerEncoderLayer
class EnhancedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=4096, dropout=0.1, activation=F.gelu):
        super().__init__()
        self.self_attn = EfficientAttention(d_model, nhead, dropout)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer norms and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, src, cos=None, sin=None, src_mask=None, src_key_padding_mask=None):
        # Self attention block with rotary embeddings
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)

        # Feed-forward block
        src2 = self.norm2(src)
        src2 = self.linear1(src2)
        src2 = self.activation(src2)
        src2 = self.dropout(src2)
        src2 = self.linear2(src2)
        src = src + self.dropout2(src2)

        return src


# Improved TransformerEncoder
class AdvancedTransformerEncoder(nn.Module):
    def __init__(self, feature_dim=1024, num_layers=6, num_heads=8, ff_dim=4096, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.rotary_emb = RotaryEmbedding(feature_dim // num_heads)

        self.layers = nn.ModuleList([
            EnhancedTransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x, mask=None):
        # Input shape: [batch_size, sequence_length, feature_dim]
        seq_len = x.shape[1]

        # Get rotary embeddings
        cos, sin = self.rotary_emb(x, seq_len=seq_len)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, cos, sin, src_mask=mask)

        # Final layer norm
        x = self.norm(x)

        return x


# Simplified Neural ODE function for efficiency
class SimpleODEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.Tanh()
        )

    def forward(self, t, states):
        h, x = states
        combined = torch.cat([h, x], dim=1)
        dh = self.net(combined)
        return dh, torch.zeros_like(x)


# Enhanced LNN Module with fallback for systems without torchdiffeq
class EnhancedLNNModule(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=512, time_delta=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.time_delta = time_delta

        # ODE function
        self.ode_func = SimpleODEFunc(input_dim, hidden_dim)

        # Initial hidden state projector
        self.h0_projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )

        # Final output layer
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # Input shape: [batch_size, sequence_length, feature_dim]
        batch_size, seq_len, feature_dim = x.shape
        device = x.device

        # Initial hidden state
        h0 = self.h0_projector(x[:, 0, :])
        h_final = h0
        final_input = x[:, -1, :]  # Save last input for residual connection

        # Process a reduced number of time steps to save memory
        steps = min(10, seq_len)
        step_size = max(1, seq_len // steps)

        # Simple Euler integration for ODE
        for t in range(0, seq_len, step_size):
            idx = min(t, seq_len - 1)
            combined = torch.cat([h_final, x[:, idx, :]], dim=1)
            dh = self.ode_func.net(combined)
            h_final = h_final + self.time_delta * dh

        # Apply final output projection with residual connection
        output = self.output_proj(torch.cat([h_final, final_input], dim=1))

        return output


# Fixed CNN model to correctly handle input dimensions
class AdvancedParallelCNNs(nn.Module):
    def __init__(self, num_channels, num_frequencies=40, output_dim=1024):
        super().__init__()
        self.output_dim = output_dim
        self.num_channels = num_channels
        self.num_frequencies = num_frequencies

        # Use the provided num_channels parameter to ensure it matches input
        self.conv_block = nn.Sequential(
            # First conv layer must match the number of EEG channels
            nn.Conv2d(num_channels, output_dim // 2, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(output_dim // 2),
            nn.GELU(),
            nn.Conv2d(output_dim // 2, output_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
        )

        # Frequency dimension projection
        self.freq_projection = nn.Conv2d(output_dim, output_dim, kernel_size=(num_frequencies, 1))

    def forward(self, x):
        # Handle dimension issues
        if x.dim() == 5:  # [batch, extra_dim, channels, freq, time]
            # Reshape 5D input to 4D: [batch*extra_dim, channels, freq, time]
            batch_size = x.shape[0]
            extra_dim = x.shape[1]
            x = x.reshape(batch_size * extra_dim, self.num_channels, self.num_frequencies, x.shape[-1])

        # Apply convolution
        x = self.conv_block(x)

        # Apply frequency dimension projection
        x = self.freq_projection(x)

        # Remove singleton dimensions
        x = x.squeeze(2)

        return x


# Fixed EEG Seizure Model - addressing dimension mismatch
class EEGSeizureModel(nn.Module):
    def __init__(self, num_channels, num_frequencies=40, hidden_dim=512, num_classes=1,
                 transformer_layers=2, transformer_heads=4, dropout=0.1, use_checkpointing=True):
        super().__init__()
        self.use_checkpointing = use_checkpointing and torch.cuda.is_available()
        self.num_channels = num_channels
        self.num_frequencies = num_frequencies

        # CNN module for raw EEG processing
        self.cnn = AdvancedParallelCNNs(num_channels=num_channels, num_frequencies=num_frequencies)

        # Projection layer to convert CNN output to transformer input
        self.projection = nn.Linear(1024, 1024)

        # Transformer module for sequence processing
        self.transformer = AdvancedTransformerEncoder(
            feature_dim=1024,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            dropout=dropout
        )

        # Global pooling layer to combine sequence features
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # LNN module for dynamic modeling
        self.lnn = EnhancedLNNModule(input_dim=1024, hidden_dim=hidden_dim, output_dim=hidden_dim)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _forward_cnn(self, x):
        return self.cnn(x)

    def _forward_transformer(self, x):
        return self.transformer(x)

    def forward(self, x):
        batch_size = x.shape[0]

        # Handle input shape issues
        if x.dim() == 4:  # [batch, channels, freq, time]
            # Ensure we have the correct number of channels
            if x.shape[1] != self.num_channels:
                print(f"Warning: Input has {x.shape[1]} channels, expected {self.num_channels}. Reshaping...")
                # Create a tensor with the right number of channels
                corrected_x = torch.zeros((x.shape[0], self.num_channels, x.shape[2], x.shape[3]), 
                                         device=x.device, dtype=x.dtype)
                # Copy available channels
                channels_to_copy = min(x.shape[1], self.num_channels)
                corrected_x[:, :channels_to_copy] = x[:, :channels_to_copy]
                x = corrected_x

        # CNN feature extraction
        if self.use_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(self._forward_cnn, x, use_reentrant=False)
        else:
            x = self.cnn(x)  # [batch_size, 1024, time_steps]

        # Handle CNN output dimensions
        if x.dim() == 4:  # If it's a 4D tensor [batch, channels, freq, time]
            # Convert to 3D by averaging the frequency dimension
            x = x.mean(dim=2)  # [batch, channels, time]

        # Ensure we have a 3D tensor before permute
        if x.dim() == 3:  # [batch, channels, time]
            x = x.permute(0, 2, 1)  # [batch, time, channels]
        else:
            # Log error dimension and raise exception
            print(f"Unexpected tensor dimension: {x.dim()}, shape: {x.shape}")
            raise ValueError(f"Expected 3D tensor, got {x.dim()}D")

        # Apply projection layer for dimension consistency
        x = self.projection(x)

        # Apply transformer
        if self.use_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(self._forward_transformer, x, use_reentrant=False)
        else:
            x = self.transformer(x)  # [batch_size, time_steps, 1024]

        # Global pooling to get a single representation per sequence
        x_pooled = x.mean(dim=1)  # [batch_size, 1024]

        # Apply LNN for temporal dynamic modeling
        x_lnn = self.lnn(x_pooled.unsqueeze(1))  # [batch_size, hidden_dim]

        # Apply classifier
        x = self.dropout(x_lnn)
        logits = self.classifier(x)  # [batch_size, num_classes]

        return logits


# Modified EEGDataset class
class EEGDataset(Dataset):
    def __init__(self, patients_dir, controls_dir, mode='train', transform=None, downsample_factor=8, max_length=None):
        self.mode = mode
        self.transform = transform
        self.downsample_factor = downsample_factor
        self.max_length = max_length
        
        # 存储数据文件路径、标签和受试者ID
        self.data_files = []
        self.labels = []
        self.subject_ids = []
        
        # 加载患者数据(标签为1)
        if os.path.exists(patients_dir):
            for subject_folder in os.listdir(patients_dir):
                subject_path = os.path.join(patients_dir, subject_folder)
                if os.path.isdir(subject_path):
                    for file in os.listdir(subject_path):
                        if file.endswith('.npy') and 'stockwell' in file:
                            # 确保是8通道的Stockwell数据文件，而不是标签文件
                            if 'selected8ch' in file or 'selected_8ch' in file:
                                # 添加数据文件路径
                                self.data_files.append(os.path.join(subject_path, file))
                                # 记录受试者ID
                                self.subject_ids.append(subject_folder)
                                # 患者标签为1
                                self.labels.append(1)
                            elif 'labels' not in file:  # 排除标签文件
                                # 检查通道数
                                try:
                                    data = np.load(os.path.join(subject_path, file))
                                    if data.shape[1] == 8:  # 确保是8通道
                                        # 添加数据文件路径
                                        self.data_files.append(os.path.join(subject_path, file))
                                        # 记录受试者ID
                                        self.subject_ids.append(subject_folder)
                                        # 患者标签为1
                                        self.labels.append(1)
                                except Exception as e:
                                    print(f"无法加载文件 {file}: {e}")
        
        # 加载正常人数据(标签为0)
        if os.path.exists(controls_dir):
            for subject_folder in os.listdir(controls_dir):
                subject_path = os.path.join(controls_dir, subject_folder)
                if os.path.isdir(subject_path):
                    for file in os.listdir(subject_path):
                        if file.endswith('.npy') and 'stockwell' in file:
                            # 确保是8通道的Stockwell数据文件，而不是标签文件
                            if 'selected8ch' in file or 'selected_8ch' in file:
                                # 添加数据文件路径
                                self.data_files.append(os.path.join(subject_path, file))
                                # 记录受试者ID
                                self.subject_ids.append(subject_folder)
                                # 正常人标签为0
                                self.labels.append(0)
                            elif 'labels' not in file:  # 排除标签文件
                                # 检查通道数
                                try:
                                    data = np.load(os.path.join(subject_path, file))
                                    if data.shape[1] == 8:  # 确保是8通道
                                        # 添加数据文件路径
                                        self.data_files.append(os.path.join(subject_path, file))
                                        # 记录受试者ID
                                        self.subject_ids.append(subject_folder)
                                        # 正常人标签为0
                                        self.labels.append(0)
                                except Exception as e:
                                    print(f"无法加载文件 {file}: {e}")
        
        # 检查是否找到任何数据文件
        if len(self.data_files) == 0:
            print(f"警告: 在指定的目录中没有找到任何匹配的数据文件!")
            print(f"请确保 {patients_dir} 和 {controls_dir} 目录中包含'.npy'文件, 并且文件名中包含'stockwell'。")
            if not os.path.exists(patients_dir):
                print(f"目录不存在: {patients_dir}")
            if not os.path.exists(controls_dir):
                print(f"目录不存在: {controls_dir}")
            # 设置默认值防止后续错误
            self.max_length = 1000 if self.max_length is None else self.max_length
        else:
            # 处理维度信息
            try:
                sample_data = np.load(self.data_files[0])
                # 减少调试信息，只打印重要信息
                print(f"样本数据形状: {sample_data.shape}")
                
                # 确保样本数据是8通道
                if sample_data.shape[1] != 8:
                    print(f"警告: 样本数据不是8通道！形状: {sample_data.shape}")
                
                # 确定最大长度
                if self.max_length is None:
                    self.max_length = sample_data.shape[-1] // self.downsample_factor
            except Exception as e:
                print(f"警告: 加载样本数据文件时出错: {e}")
                # 设置默认值防止错误
                self.max_length = 1000 if self.max_length is None else self.max_length

        print(f"Found {len(self.data_files)} files:")
        print(f"  Patients: {self.labels.count(1) if len(self.labels) > 0 else 0} files from {len(set([id for id, label in zip(self.subject_ids, self.labels) if label == 1]))} subjects")
        print(f"  Controls: {self.labels.count(0) if len(self.labels) > 0 else 0} files from {len(set([id for id, label in zip(self.subject_ids, self.labels) if label == 0]))} subjects")
        print(f"Using max sequence length: {self.max_length}")
        
    def __len__(self):
        return len(self.data_files)
        
    def __getitem__(self, idx):
        # 加载数据
        try:
            data = np.load(self.data_files[idx])
            # 确保data是8通道的
            if data.shape[1] != 8:
                print(f"警告: 文件 {self.data_files[idx]} 不是8通道！形状: {data.shape}")
                # 如果通道数不是8，创建一个8通道的零数组
                if len(data.shape) == 4:  # [epochs, channels, freq, time]
                    zero_data = np.zeros((data.shape[0], 8, data.shape[2], data.shape[3]))
                    # 复制现有通道
                    channels_to_copy = min(data.shape[1], 8)
                    zero_data[:, :channels_to_copy] = data[:, :channels_to_copy]
                    data = zero_data
                elif len(data.shape) == 3:  # [channels, freq, time]
                    zero_data = np.zeros((8, data.shape[1], data.shape[2]))
                    # 复制现有通道
                    channels_to_copy = min(data.shape[0], 8)
                    zero_data[:channels_to_copy] = data[:channels_to_copy]
                    data = zero_data
        except Exception as e:
            print(f"错误: 无法加载文件 {self.data_files[idx]}: {e}")
            # 创建替代数据（全零），防止程序崩溃
            data = np.zeros((8, 40, self.max_length))  # 使用默认形状
            
        label = self.labels[idx]
        
        # 减少调试信息
        # print(f"Original data shape before processing: {data.shape}")  # 注释掉过多调试信息
        
        # 标准化数据维度 (保持与原代码一致)
        if data.ndim == 4:  # 如果数据是 [batch?, channels, freq, time]
            if data.shape[0] == 1:  # 如果batch维度为1
                data = data[0]  # 移除batch维度 -> [channels, freq, time]
            elif data.shape[0] > 1:  # 如果第一维度 > 1 (可能是多个epoch)
                # 我们取第一个epoch，确保正确的维度顺序
                data = data[0]  # 取第一个epoch -> [channels, freq, time]
        
        # 下采样时间维度
        if self.downsample_factor > 1:
            data = data[..., ::self.downsample_factor]
            
        # 确保最大长度
        if self.max_length is not None:
            if data.shape[-1] > self.max_length:
                data = data[..., :self.max_length]
            elif data.shape[-1] < self.max_length:
                # 如果太短则填充
                pad_width = [(0, 0)] * (data.ndim - 1) + [(0, self.max_length - data.shape[-1])]
                data = np.pad(data, pad_width, mode='constant')
        
        # 确保数据维度是 [channels, freq, time]，channels=8
        if data.shape[0] != 8:
            print(f"警告: 处理后数据通道数不是8: {data.shape}")
            # 如果处理后通道数仍然不是8，创建一个8通道的零数组
            zero_data = np.zeros((8, data.shape[1], data.shape[2]))
            # 复制现有通道
            channels_to_copy = min(data.shape[0], 8)
            zero_data[:channels_to_copy] = data[:channels_to_copy]
            data = zero_data
        
        # 转换为tensor
        data = torch.from_numpy(data).float()
        label = torch.tensor(float(label), dtype=torch.float32)
        
        return data, label


# Improved custom_collate function that handles tensors with different dimensions
def custom_collate(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Check shapes of all items - 减少调试信息
    shapes = []
    for i, d in enumerate(data):
        shape = d.shape
        shapes.append(shape)

    # Ensure all tensors have the same number of dimensions
    max_dims = max(len(shape) for shape in shapes)
    normalized_data = []

    for i, tensor in enumerate(data):
        dims = len(tensor.shape)
        if dims < max_dims:
            # Add dimensions to match max_dims
            for _ in range(max_dims - dims):
                tensor = tensor.unsqueeze(0)
        normalized_data.append(tensor)

    # Now check if all normalized shapes are the same
    norm_shapes = [d.shape for d in normalized_data]
    same_shape = all(s == norm_shapes[0] for s in norm_shapes)

    if same_shape:
        # Stack directly if all same shape
        data_tensor = torch.stack(normalized_data)
    else:
        # Find max dimensions for each axis
        max_shape = []
        for dim in range(max_dims):
            max_size = max(s[dim] if dim < len(s) else 1 for s in shapes)
            max_shape.append(max_size)

        padded_data = []

        for item in normalized_data:
            # Calculate padding needed for each dimension
            padding = []
            for dim in range(max_dims):
                size_diff = max_shape[dim] - item.shape[dim]
                padding.append(0)  # Start padding
                padding.append(size_diff)  # End padding

            # Reverse padding for PyTorch convention
            padding.reverse()

            if any(p > 0 for p in padding):
                padded = F.pad(item, padding)
                padded_data.append(padded)
            else:
                padded_data.append(item)

        data_tensor = torch.stack(padded_data)

    # Convert labels to tensor - ensure they're floating point for BCE loss
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    return data_tensor, labels_tensor


# Fixed Training function to handle tensor shape issues
def train_epoch(model, train_loader, optimizer, criterion, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, targets in tqdm(train_loader, desc="Training"):
        # Move data to device
        data = data.to(device)
        targets = targets.to(device)

        # Debug print - use for troubleshooting only
        # print(f"Input data shape: {data.shape}")
        
        # Ensure data has expected shape
        if len(data.shape) == 4 and data.shape[1] != 8 and data.shape[2] == 40:
            # This is likely the [batch, channels, freq, time] format but with wrong channel count
            print(f"Warning: Input has {data.shape[1]} channels, expected 8. Reshaping...")
            # Create a zero tensor with the right shape
            correct_data = torch.zeros((data.shape[0], 8, data.shape[2], data.shape[3]), device=device)
            # Copy available channels
            channels_to_copy = min(data.shape[1], 8)
            correct_data[:, :channels_to_copy] = data[:, :channels_to_copy]
            data = correct_data

        # Zero gradients
        optimizer.zero_grad()

        if scaler:  # Use mixed precision if scaler is provided
            with torch.amp.autocast('cuda'):  # Updated to new API format
                outputs = model(data)
                # Ensure dimensions match for loss calculation
                outputs = outputs.view(-1)
                targets = targets.view(-1)
                loss = criterion(outputs, targets)

            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass
            outputs = model(data)
            # Ensure dimensions match for loss calculation
            outputs = outputs.view(-1)
            targets = targets.view(-1)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

        # Statistics
        running_loss += loss.item() * data.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# Evaluation function
def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_targets = []
    total_samples = 0  # Add sample counter

    with torch.no_grad():
        for data, targets in tqdm(val_loader, desc="Evaluating"):
            # Move data to device
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass - use autocast if available
            if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
                with torch.amp.autocast('cuda'):
                    outputs = model(data)
                    outputs = outputs.view(-1)
                    targets = targets.view(-1)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(data)
                outputs = outputs.view(-1)
                targets = targets.view(-1)
                loss = criterion(outputs, targets)

            # Update sample count and loss accumulation
            batch_size = data.size(0)
            total_samples += batch_size
            running_loss += loss.item() * batch_size

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    # Calculate loss using actual sample count
    val_loss = running_loss / total_samples
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    return val_loss, accuracy, precision, recall, f1, all_probs


# Predict function
def predict(model, data_loader, device):
    model.eval()  # Ensure model is in evaluation mode
    all_preds = []
    all_probs = []

    with torch.no_grad():  # Ensure gradient calculation is off
        for data, _ in tqdm(data_loader, desc="Predicting"):
            # Move data to device
            data = data.to(device)
            
            # Use autocast if available
            if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
                with torch.amp.autocast('cuda'):
                    outputs = model(data)
                    outputs = outputs.view(-1)
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
            else:
                outputs = model(data)
                outputs = outputs.view(-1)  
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

    return all_preds, all_probs


# Add function to split dataset by subject
def split_by_subject(dataset, train_ratio=0.8):
    """Split dataset based on subject IDs to ensure data from the same subject doesn't appear in both training and validation sets"""
    import random
    
    unique_subjects = list(set(dataset.subject_ids))
    
    # Shuffle subject order
    random.shuffle(unique_subjects)
    
    # Calculate number of subjects in training set
    n_train_subjects = max(1, int(len(unique_subjects) * train_ratio))
    
    # Split subjects
    train_subjects = unique_subjects[:n_train_subjects]
    val_subjects = unique_subjects[n_train_subjects:]
    
    # Filter indices based on subject IDs
    train_indices = [i for i, subj in enumerate(dataset.subject_ids) if subj in train_subjects]
    val_indices = [i for i, subj in enumerate(dataset.subject_ids) if subj in val_subjects]
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    print(f"Split dataset: {len(train_indices)} training samples, {len(val_indices)} validation samples")
    print(f"Training: {len(train_subjects)} subjects, Validation: {len(val_subjects)} subjects")
    
    return train_dataset, val_dataset


# Modified training function to support new data loading method
def train_model(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # GPU内存信息
    if torch.cuda.is_available():
        try:
            # 获取GPU内存信息
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # 转换为GB
            free_memory = torch.cuda.memory_reserved(0) / (1024**3)
            # 根据可用内存动态调整批处理大小
            suggested_batch_size = max(1, min(args.batch_size, int(free_memory / 0.5)))  # 假设每个样本需要0.5GB
            if suggested_batch_size < args.batch_size:
                print(f"警告: GPU内存可能不足. 总内存: {gpu_memory:.2f}GB, 当前保留: {free_memory:.2f}GB")
                print(f"建议将批处理大小从{args.batch_size}减少到{suggested_batch_size}")
                # 这里我们只提出建议而不强制更改，保持原始逻辑不变
        except Exception as e:
            print(f"尝试检查GPU内存时出错: {e}")
    
    # Memory optimization parameters
    downsample_factor = 8  # Reduce time dimension
    max_seq_len = 240  # Further reduce sequence length from 960

    # Create dataset - now accepts two directory parameters
    dataset = EEGDataset(
        patients_dir=args.patients_dir, 
        controls_dir=args.controls_dir,
        downsample_factor=downsample_factor,
        max_length=max_seq_len
    )
    
    # 处理没有数据的情况
    if len(dataset) == 0:
        raise ValueError("没有找到任何数据。请检查patients_dir和controls_dir路径是否正确，并且包含有效的数据文件。")
    
    # Split dataset based on subject IDs
    train_dataset, val_dataset = split_by_subject(dataset, train_ratio=0.8)
    
    # Use custom collate and batch size
    batch_size = args.batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Don't use multiprocessing
        collate_fn=custom_collate,
        pin_memory=False  # Prevent memory leaks
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=custom_collate,
        pin_memory=False
    )

    # Load sample to determine input shape
    sample_data, _ = dataset[0]
    num_channels = sample_data.shape[0]
    num_frequencies = sample_data.shape[1] if sample_data.dim() > 1 else 40

    print(f"Data shape after optimization: {sample_data.shape}")

    # Create model
    model = EEGSeizureModel(
        num_channels=num_channels,
        num_frequencies=num_frequencies,
        hidden_dim=256,  # Smaller hidden dimension
        transformer_layers=1,  # Fewer transformer layers
        transformer_heads=4,
        dropout=args.dropout,
        use_checkpointing=args.use_checkpointing
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params / 1e6:.2f}M parameters")

    # Set up optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if args.mixed_precision and torch.cuda.is_available() else None

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }

    # Add early stopping
    early_stop_count = 0
    early_stop_patience = 10

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)

        # Free up memory before validation
        torch.cuda.empty_cache()

        # Evaluate
        val_loss, val_acc, val_precision, val_recall, val_f1, _ = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'mixed_precision': args.mixed_precision,
                'dropout': args.dropout,
                'use_checkpointing': args.use_checkpointing,
                'model_config': {
                    'num_channels': num_channels,
                    'num_frequencies': num_frequencies,
                    'hidden_dim': 256,
                    'transformer_layers': 1,
                    'transformer_heads': 4
                }
            }, best_model_path)
            print(f"Saved best model with validation loss: {val_loss:.4f}")
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= early_stop_patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

        # Free up memory after each epoch
        torch.cuda.empty_cache()

    # Plot training history
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(2, 2, 3)
    plt.plot(history['val_precision'], label='Precision')
    plt.plot(history['val_recall'], label='Recall')
    plt.plot(history['val_f1'], label='F1 Score')
    plt.legend()
    plt.title('Metrics')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_history.png'))

    return model, history


# Modified evaluation function to support new data loading method
def evaluate_model(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Load checkpoint
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Extract model configuration, if not provided use default values
        num_channels = None
        num_frequencies = None
        hidden_dim = args.hidden_dim if hasattr(args, 'hidden_dim') else 256
        transformer_layers = 1
        transformer_heads = 4
        dropout = args.dropout if hasattr(args, 'dropout') else 0.3
        
        # Try to get more detailed configuration from checkpoint (if available)
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            num_channels = model_config.get('num_channels', num_channels)
            num_frequencies = model_config.get('num_frequencies', num_frequencies)
            hidden_dim = model_config.get('hidden_dim', hidden_dim)
            transformer_layers = model_config.get('transformer_layers', transformer_layers)
            transformer_heads = model_config.get('transformer_heads', transformer_heads)
        
        # Try to get dropout (if saved in checkpoint)
        if 'dropout' in checkpoint:
            dropout = checkpoint['dropout']
            
        # Create dataset - use new interface
        dataset = EEGDataset(
            patients_dir=args.patients_dir,
            controls_dir=args.controls_dir,
            downsample_factor=8,
            max_length=240
        )

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=custom_collate,
            pin_memory=False
        )

        # Infer missing parameters from sample
        if num_channels is None or num_frequencies is None:
            sample_data, _ = dataset[0]
            num_channels = sample_data.shape[0] if num_channels is None else num_channels
            num_frequencies = sample_data.shape[1] if sample_data.dim() > 1 and num_frequencies is None else num_frequencies
            if num_frequencies is None:
                num_frequencies = 40
        
        print(f"Using model configuration - channels: {num_channels}, frequencies: {num_frequencies}, "
              f"hidden_dim: {hidden_dim}, transformer_layers: {transformer_layers}, "
              f"transformer_heads: {transformer_heads}, dropout: {dropout}")
        
        # Create model
        model = EEGSeizureModel(
            num_channels=num_channels,
            num_frequencies=num_frequencies,
            hidden_dim=hidden_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            dropout=dropout,
            use_checkpointing=False  # Evaluation doesn't need checkpointing
        )

        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        # Check if using mixed precision
        use_mixed_precision = checkpoint.get('mixed_precision', False)
        criterion = nn.BCEWithLogitsLoss()
        
        # Decide whether to use mixed precision based on model training settings
        if use_mixed_precision and torch.cuda.is_available():
            print("Using mixed precision for evaluation")
            with torch.amp.autocast('cuda'):
                val_loss, accuracy, precision, recall, f1, probabilities = evaluate(model, dataloader, criterion, device)
        else:
            val_loss, accuracy, precision, recall, f1, probabilities = evaluate(model, dataloader, criterion, device)

        # Print evaluation results
        print("\nEvaluation Results:")
        print(f"Loss: {val_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return {
            'loss': val_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'probabilities': probabilities
        }
    
    except Exception as e:
        print(f"Error loading or evaluating model: {e}")
        import traceback
        traceback.print_exc()
        raise


# Modified predict function to support new data loading method
def predict_seizure(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Load checkpoint
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Extract model configuration, if not provided use default values
        num_channels = None
        num_frequencies = None
        hidden_dim = args.hidden_dim if hasattr(args, 'hidden_dim') else 256
        transformer_layers = 1
        transformer_heads = 4
        dropout = args.dropout if hasattr(args, 'dropout') else 0.3
        
        # Try to get more detailed configuration from checkpoint (if available)
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            num_channels = model_config.get('num_channels', num_channels)
            num_frequencies = model_config.get('num_frequencies', num_frequencies)
            hidden_dim = model_config.get('hidden_dim', hidden_dim)
            transformer_layers = model_config.get('transformer_layers', transformer_layers)
            transformer_heads = model_config.get('transformer_heads', transformer_heads)
        
        # Try to get dropout (if saved in checkpoint)
        if 'dropout' in checkpoint:
            dropout = checkpoint['dropout']
            
        # Create dataset - use new interface
        dataset = EEGDataset(
            patients_dir=args.patients_dir,
            controls_dir=args.controls_dir,
            downsample_factor=8,
            max_length=240
        )
        
        # 处理没有数据的情况
        if len(dataset) == 0:
            raise ValueError("没有找到任何数据。请检查patients_dir和controls_dir路径是否正确，并且包含有效的数据文件。")

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=custom_collate,
            pin_memory=False
        )

        # Infer missing parameters from sample
        if num_channels is None or num_frequencies is None:
            sample_data, _ = dataset[0]
            num_channels = sample_data.shape[0] if num_channels is None else num_channels
            num_frequencies = sample_data.shape[1] if sample_data.dim() > 1 and num_frequencies is None else num_frequencies
            if num_frequencies is None:
                num_frequencies = 40
        
        print(f"Using model configuration - channels: {num_channels}, frequencies: {num_frequencies}, "
              f"hidden_dim: {hidden_dim}, transformer_layers: {transformer_layers}, "
              f"transformer_heads: {transformer_heads}, dropout: {dropout}")
        
        # Create model
        model = EEGSeizureModel(
            num_channels=num_channels,
            num_frequencies=num_frequencies,
            hidden_dim=hidden_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            dropout=dropout,
            use_checkpointing=False  # Prediction doesn't need checkpointing
        )

        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        # Check if using mixed precision
        use_mixed_precision = checkpoint.get('mixed_precision', False)
        
        # Decide whether to use mixed precision based on model training settings
        if use_mixed_precision and torch.cuda.is_available():
            print("Using mixed precision for prediction")
            with torch.amp.autocast('cuda'):
                predictions, probabilities = predict(model, dataloader, device)
        else:
            predictions, probabilities = predict(model, dataloader, device)

        # 使用用户指定的阈值而不是默认的0.5
        threshold = args.threshold if hasattr(args, 'threshold') else 0.5
        if threshold != 0.5:
            print(f"使用自定义阈值: {threshold} (默认值: 0.5)")
            # 根据阈值重新计算预测结果
            predictions = [1.0 if prob >= threshold else 0.0 for prob in probabilities]

        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)

        # Save predictions to file
        results = {
            'file_names': [os.path.basename(f) for f in dataset.data_files],
            'subject_ids': dataset.subject_ids,  # Add subject ID information
            'predictions': predictions,
            'probabilities': probabilities
        }

        # 确保输出目录存在且可写
        try:
            np.save(os.path.join(args.output_dir, 'predictions.npy'), results)
            print(f"预测结果已保存至 {os.path.join(args.output_dir, 'predictions.npy')}")
        except Exception as e:
            print(f"错误: 无法保存预测结果到 {os.path.join(args.output_dir, 'predictions.npy')}: {e}")
            print("请确保目录存在且有写入权限")

        # Print summary
        print("\nPrediction Results:")
        print(f"Total files processed: {len(dataset.data_files)}")
        print(f"Positive predictions: {sum(predictions)}")
        
        # 新增: 按受试者ID分组并计算多数投票和平均概率
        print("\nCalculating per-subject results...")
        subject_results = {}
        
        # 遍历所有预测结果并按受试者ID分组
        for file_name, subject_id, pred, prob in zip(
                [os.path.basename(f) for f in dataset.data_files], 
                dataset.subject_ids, 
                predictions, 
                probabilities):
            if subject_id not in subject_results:
                subject_results[subject_id] = {'file_names': [], 'preds': [], 'probs': []}
            subject_results[subject_id]['file_names'].append(file_name)
            subject_results[subject_id]['preds'].append(float(pred))
            subject_results[subject_id]['probs'].append(float(prob))
        
        # 计算每个受试者的最终结果
        final_results = {}
        patient_count_vote = 0
        patient_count_prob = 0
        
        # 准备CSV输出
        csv_path = os.path.join(args.output_dir, 'detailed_predictions.csv')
        
        # 检查目录权限和写入权限
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            
            with open(csv_path, 'w') as csv_file:
                csv_file.write("Subject_ID,File_Name,Prediction,Probability\n")
                
                # 遍历每个受试者
                for subject_id, data in subject_results.items():
                    # 使用阈值进行多数投票
                    vote_result = 1 if sum(data['preds']) > len(data['preds'])/2 else 0
                    # 平均概率
                    avg_prob = sum(data['probs']) / len(data['probs'])
                    # 基于阈值的平均概率判断
                    prob_result = 1 if avg_prob >= threshold else 0
                    
                    # 统计患者数量
                    if vote_result == 1:
                        patient_count_vote += 1
                    if prob_result == 1:
                        patient_count_prob += 1
                        
                    # 保存结果
                    final_results[subject_id] = {
                        'segments_count': len(data['preds']),
                        'positive_segments': sum(data['preds']),
                        'vote_result': vote_result,
                        'avg_probability': avg_prob,
                        'prob_result': prob_result,
                        'threshold': threshold,
                        'individual_predictions': data['preds'],
                        'individual_probabilities': data['probs'],
                        'file_names': data['file_names']
                    }
                    
                    # 打印详细的个体片段预测结果
                    print(f"\nSubject: {subject_id}")
                    print(f"Final diagnosis (voting): {'Epilepsy Patient' if vote_result == 1 else 'Normal'} ({sum(data['preds'])}/{len(data['preds'])} segments)")
                    print(f"Final diagnosis (probability): {'Epilepsy Patient' if prob_result == 1 else 'Normal'} (Avg prob: {avg_prob:.4f}, Threshold: {threshold})")
                    print("Individual EEG segments:")
                    print("-" * 80)
                    print(f"{'File Name':<30} | {'Prediction':<15} | {'Probability':<10}")
                    print("-" * 80)
                    
                    # 遍历该受试者的所有片段
                    for i, (file_name, pred, prob) in enumerate(zip(data['file_names'], data['preds'], data['probs'])):
                        pred_text = "Epilepsy" if pred == 1 else "Normal"
                        print(f"{file_name:<30} | {pred_text:<15} | {prob:.4f}")
                        
                        # 写入CSV文件
                        csv_file.write(f"{subject_id},{file_name},{int(pred)},{prob:.6f}\n")
                    
                    print("-" * 80)
                    print("")
            
            # 保存按受试者整合的结果
            subject_results_path = os.path.join(args.output_dir, 'subject_predictions.npy')
            try:
                np.save(subject_results_path, final_results)
                print(f"受试者级别结果已保存至 {subject_results_path}")
            except Exception as e:
                print(f"错误: 无法保存受试者级别结果到 {subject_results_path}: {e}")
            
            # 打印汇总信息
            print(f"\nPer-subject Results Summary:")
            print(f"Total subjects: {len(subject_results)}")
            print(f"Patients (by voting): {patient_count_vote}")
            print(f"Patients (by avg probability): {patient_count_prob}")
            print(f"Detailed predictions saved to {csv_path}")
        
        except Exception as e:
            print(f"错误: 无法创建或写入CSV文件 {csv_path}: {e}")
            print("请确保目录存在且有写入权限")

        return results
    
    except Exception as e:
        print(f"Error loading or predicting with model: {e}")
        import traceback
        traceback.print_exc()
        raise


# Updated command line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='EEG Seizure Detection Model')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--patients_dir', type=str, required=True, help='Directory containing patient folders')
    train_parser.add_argument('--controls_dir', type=str, required=True, help='Directory containing control subject folders')
    train_parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model and results')
    train_parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    train_parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    train_parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    train_parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    train_parser.add_argument('--use_checkpointing', action='store_true',
                             help='Use gradient checkpointing to save memory')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--patients_dir', type=str, required=True, help='Directory containing patient folders')
    eval_parser.add_argument('--controls_dir', type=str, required=True, help='Directory containing control subject folders')
    eval_parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    eval_parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    eval_parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    eval_parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for classification (default: 0.5)')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions with the model')
    predict_parser.add_argument('--patients_dir', type=str, required=True, help='Directory containing patient folders')
    predict_parser.add_argument('--controls_dir', type=str, required=True, help='Directory containing control subject folders')
    predict_parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    predict_parser.add_argument('--output_dir', type=str, required=True, help='Directory to save predictions')
    predict_parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    predict_parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    predict_parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for classification (default: 0.5)')

    args = parser.parse_args()

    # Add default values for missing arguments
    if not hasattr(args, 'weight_decay'):
        args.weight_decay = 1e-5

    return args


# Run the appropriate function based on the command
def main():
    args = parse_args()

    # Set environment variable to avoid memory fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    if args.command == 'train':
        model, history = train_model(args)
    elif args.command == 'evaluate':
        results = evaluate_model(args)
    elif args.command == 'predict':
        results = predict_seizure(args)
    else:
        print("Please specify a command: train, evaluate, or predict")


if __name__ == '__main__':
    main()
