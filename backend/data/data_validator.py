#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据验证和可视化模块
用于检查预处理后的数据质量和特征分布
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
import argparse
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DataValidator:
    """
    数据验证器
    """
    
    def __init__(self, data_path: str):
        """
        初始化验证器
        
        Args:
            data_path: 预处理后的数据文件路径
        """
        self.data_path = data_path
        self.df = None
        self.load_data()
    
    def load_data(self) -> None:
        """
        加载数据
        """
        try:
            self.df = pd.read_csv(self.data_path)
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            print(f"成功加载数据: {self.data_path}")
            print(f"数据形状: {self.df.shape}")
        except Exception as e:
            print(f"加载数据失败: {e}")
            raise
    
    def basic_info(self) -> None:
        """
        显示基本信息
        """
        print("\n=== 数据基本信息 ===")
        print(f"数据形状: {self.df.shape}")
        print(f"时间范围: {self.df['datetime'].min()} 到 {self.df['datetime'].max()}")
        print(f"缺失值总数: {self.df.isnull().sum().sum()}")
        
        # 特征分类
        scaled_features = [col for col in self.df.columns if '_scaled' in col]
        time_features = [col for col in self.df.columns if '_sin' in col or '_cos' in col]
        
        print(f"\n缩放特征数量: {len(scaled_features)}")
        print(f"时间特征数量: {len(time_features)}")
        print(f"总特征数量: {len(scaled_features) + len(time_features)}")
        
        print("\n缩放特征列表:")
        for i, col in enumerate(scaled_features, 1):
            print(f"  {i:2d}. {col}")
        
        print("\n时间特征列表:")
        for i, col in enumerate(time_features, 1):
            print(f"  {i:2d}. {col}")
    
    def validate_feature_ranges(self) -> None:
        """
        验证特征范围
        """
        print("\n=== 特征范围验证 ===")
        
        # 检查缩放特征是否在[-1,1]范围内
        scaled_features = [col for col in self.df.columns if '_scaled' in col]
        
        print("缩放特征范围检查 (应该在[-1,1]之间):")
        for col in scaled_features:
            min_val = self.df[col].min()
            max_val = self.df[col].max()
            mean_val = self.df[col].mean()
            std_val = self.df[col].std()
            
            status = "✓" if -1.0 <= min_val <= max_val <= 1.0 else "✗"
            print(f"  {status} {col:25s}: [{min_val:8.4f}, {max_val:8.4f}] μ={mean_val:7.4f} σ={std_val:6.4f}")
        
        # 检查时间特征
        time_features = [col for col in self.df.columns if '_sin' in col or '_cos' in col]
        print("\n时间特征范围检查 (应该在[-1,1]之间):")
        for col in time_features:
            min_val = self.df[col].min()
            max_val = self.df[col].max()
            mean_val = self.df[col].mean()
            
            status = "✓" if -1.0 <= min_val <= max_val <= 1.0 else "✗"
            print(f"  {status} {col:15s}: [{min_val:8.4f}, {max_val:8.4f}] μ={mean_val:7.4f}")
    
    def check_data_continuity(self) -> None:
        """
        检查数据连续性
        """
        print("\n=== 数据连续性检查 ===")
        
        # 计算时间间隔
        time_diffs = self.df['datetime'].diff().dropna()
        expected_interval = pd.Timedelta(minutes=15)  # 15分钟K线
        
        # 统计时间间隔
        normal_intervals = (time_diffs == expected_interval).sum()
        total_intervals = len(time_diffs)
        
        print(f"总时间间隔数: {total_intervals}")
        print(f"正常间隔数 (15分钟): {normal_intervals}")
        print(f"异常间隔数: {total_intervals - normal_intervals}")
        print(f"连续性: {normal_intervals/total_intervals*100:.2f}%")
        
        # 显示异常间隔
        abnormal_intervals = time_diffs[time_diffs != expected_interval]
        if len(abnormal_intervals) > 0:
            print(f"\n异常间隔统计 (前10个):")
            for i, (idx, interval) in enumerate(abnormal_intervals.head(10).items()):
                datetime_val = self.df.loc[idx, 'datetime']
                print(f"  {i+1:2d}. {datetime_val}: {interval}")
    
    def plot_feature_distributions(self, save_path: Optional[str] = None) -> None:
        """
        绘制特征分布图
        
        Args:
            save_path: 保存路径
        """
        print("\n=== 绘制特征分布图 ===")
        
        scaled_features = [col for col in self.df.columns if '_scaled' in col]
        
        # 创建子图
        n_features = len(scaled_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(scaled_features):
            ax = axes[i]
            
            # 绘制直方图
            ax.hist(self.df[col], bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
            
            # 添加统计信息
            mean_val = self.df[col].mean()
            std_val = self.df[col].std()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7, label=f'+1σ: {mean_val+std_val:.3f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7, label=f'-1σ: {mean_val-std_val:.3f}')
            
            ax.set_title(f'{col}', fontsize=10)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-1.1, 1.1)
        
        # 隐藏多余的子图
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('特征分布图', fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征分布图已保存: {save_path}")
        
        plt.show()
    
    def plot_time_series(self, features: List[str] = None, 
                        sample_days: int = 7, save_path: Optional[str] = None) -> None:
        """
        绘制时间序列图
        
        Args:
            features: 要绘制的特征列表
            sample_days: 采样天数
            save_path: 保存路径
        """
        print(f"\n=== 绘制时间序列图 (最近{sample_days}天) ===")
        
        if features is None:
            features = ['log_ret_scaled', 'vol_z_scaled', 'hl_range_scaled']
        
        # 获取最近几天的数据
        end_date = self.df['datetime'].max()
        start_date = end_date - pd.Timedelta(days=sample_days)
        sample_df = self.df[self.df['datetime'] >= start_date].copy()
        
        print(f"采样数据: {len(sample_df)} 行")
        print(f"时间范围: {sample_df['datetime'].min()} 到 {sample_df['datetime'].max()}")
        
        # 创建子图
        fig, axes = plt.subplots(len(features) + 1, 1, figsize=(15, 3*(len(features)+1)))
        
        # 绘制价格
        axes[0].plot(sample_df['datetime'], sample_df['close'], color='black', linewidth=1)
        axes[0].set_title('BTCUSDT 收盘价', fontsize=12)
        axes[0].set_ylabel('Price (USDT)')
        axes[0].grid(True, alpha=0.3)
        
        # 绘制特征
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        for i, feature in enumerate(features):
            if feature in sample_df.columns:
                color = colors[i % len(colors)]
                axes[i+1].plot(sample_df['datetime'], sample_df[feature], 
                             color=color, linewidth=1, alpha=0.8)
                axes[i+1].set_title(f'{feature}', fontsize=12)
                axes[i+1].set_ylabel('Value')
                axes[i+1].set_ylim(-1.1, 1.1)
                axes[i+1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                axes[i+1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f'时间序列图 (最近{sample_days}天)', fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"时间序列图已保存: {save_path}")
        
        plt.show()
    
    def correlation_analysis(self, save_path: Optional[str] = None) -> None:
        """
        相关性分析
        
        Args:
            save_path: 保存路径
        """
        print("\n=== 特征相关性分析 ===")
        
        scaled_features = [col for col in self.df.columns if '_scaled' in col]
        
        # 计算相关性矩阵
        corr_matrix = self.df[scaled_features].corr()
        
        # 绘制热力图
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
        plt.title('特征相关性矩阵', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"相关性矩阵已保存: {save_path}")
        
        plt.show()
        
        # 显示高相关性特征对
        print("\n高相关性特征对 (|r| > 0.7):")
        high_corr_pairs = []
        for i in range(len(scaled_features)):
            for j in range(i+1, len(scaled_features)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((scaled_features[i], scaled_features[j], corr_val))
        
        if high_corr_pairs:
            for feat1, feat2, corr_val in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                print(f"  {feat1:25s} <-> {feat2:25s}: {corr_val:6.3f}")
        else:
            print("  未发现高相关性特征对")
    
    def generate_report(self, output_dir: str = "./reports") -> None:
        """
        生成完整的数据验证报告
        
        Args:
            output_dir: 输出目录
        """
        print(f"\n=== 生成数据验证报告 ===")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成图表
        dist_path = os.path.join(output_dir, "feature_distributions.png")
        ts_path = os.path.join(output_dir, "time_series.png")
        corr_path = os.path.join(output_dir, "correlation_matrix.png")
        
        self.plot_feature_distributions(save_path=dist_path)
        self.plot_time_series(save_path=ts_path)
        self.correlation_analysis(save_path=corr_path)
        
        print(f"\n报告已生成到目录: {output_dir}")
        print("包含文件:")
        print(f"  - {dist_path}")
        print(f"  - {ts_path}")
        print(f"  - {corr_path}")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='数据验证和可视化')
    parser.add_argument('--data_path', type=str, required=True,
                       help='预处理后的数据文件路径')
    parser.add_argument('--output_dir', type=str, default='./reports',
                       help='输出目录 (默认: ./reports)')
    parser.add_argument('--sample_days', type=int, default=7,
                       help='时间序列采样天数 (默认: 7)')
    
    args = parser.parse_args()
    
    # 创建验证器
    validator = DataValidator(args.data_path)
    
    # 执行验证
    validator.basic_info()
    validator.validate_feature_ranges()
    validator.check_data_continuity()
    
    # 生成报告
    validator.generate_report(args.output_dir)

if __name__ == "__main__":
    main()