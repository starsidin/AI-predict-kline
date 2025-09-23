#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理模块
对K线数据进行特征工程和归一化处理

特征包括：
- 对数收益率 (log_ret)
- 高低价范围 (hl_range) 
- 成交量Z-score (vol_z)
- 技术指标 (ATR, rolling_std, EMA, RSI, MACD)
- 时间特征 (sin/cos编码)

归一化方法：对数收益 + 波动缩放 + tanh压缩
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import List, Tuple, Optional
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class KlinePreprocessor:
    """
    K线数据预处理器
    """
    
    def __init__(self, window_size: int = 20, atr_period: int = 14, 
                 ema_period: int = 12, rsi_period: int = 14):
        """
        初始化预处理器
        
        Args:
            window_size: 滚动窗口大小
            atr_period: ATR周期
            ema_period: EMA周期
            rsi_period: RSI周期
        """
        self.window_size = window_size
        self.atr_period = atr_period
        self.ema_period = ema_period
        self.rsi_period = rsi_period
        
    def load_kline_data(self, file_path: str) -> pd.DataFrame:
        """
        加载K线数据
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            DataFrame: 标准化的K线数据
        """
        try:
            df = pd.read_csv(file_path, header=None)
            
            # 币安K线数据列名
            columns = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'count', 'taker_buy_volume', 
                'taker_buy_quote_volume', 'ignore'
            ]
            
            df.columns = columns
            
            # 转换时间戳
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 选择需要的列，排除成交额(quote_volume)
            df = df[['datetime', 'timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            
            # 转换数据类型
            price_cols = ['open', 'high', 'low', 'close']
            df[price_cols] = df[price_cols].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # 按时间排序
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"加载数据: {file_path}, 形状: {df.shape}")
            return df
            
        except Exception as e:
            print(f"加载数据失败 {file_path}: {e}")
            return pd.DataFrame()
    
    def calculate_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算对数收益率
        
        Args:
            df: 输入数据框
            
        Returns:
            DataFrame: 添加对数收益率的数据框
        """
        df = df.copy()
        
        # 对数收益率
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
        return df
    
    def calculate_hl_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算高低价范围
        
        Args:
            df: 输入数据框
            
        Returns:
            DataFrame: 添加高低价范围的数据框
        """
        df = df.copy()
        
        # 高低价范围 (相对于前一收盘价)
        df['hl_range'] = (df['high'] - df['low']) / df['close'].shift(1)
        
        return df
    
    def calculate_volume_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量Z-score
        
        Args:
            df: 输入数据框
            
        Returns:
            DataFrame: 添加成交量Z-score的数据框
        """
        df = df.copy()
        
        # 成交量归一化：先取对数，再计算Z-score
        df['log_volume'] = np.log1p(df['volume'])  # log(1+x)避免log(0)
        
        # 滚动Z-score
        rolling_mean = df['log_volume'].rolling(window=self.window_size, min_periods=1).mean()
        rolling_std = df['log_volume'].rolling(window=self.window_size, min_periods=1).std()
        
        df['vol_z'] = (df['log_volume'] - rolling_mean) / (rolling_std + 1e-8)
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            df: 输入数据框
            
        Returns:
            DataFrame: 添加技术指标的数据框
        """
        df = df.copy()
        
        # ATR (Average True Range)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=self.atr_period, min_periods=1).mean()
        df['atr_norm'] = df['atr'] / df['close'].shift(1)  # 归一化ATR
        
        # 对数收益率的滚动标准差
        df['log_ret_std'] = df['log_ret'].rolling(window=self.window_size, min_periods=1).std()
        
        # EMA
        df['ema'] = df['close'].ewm(span=self.ema_period).mean()
        df['ema_ratio'] = df['close'] / df['ema'] - 1  # EMA比率
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_norm'] = (df['rsi'] - 50) / 50  # 归一化到[-1,1]
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_delta'] = df['macd'] - df['macd_signal']
        df['macd_delta_norm'] = df['macd_delta'] / df['close'].shift(1)  # 归一化MACD
        
        # 清理临时列
        temp_cols = ['tr1', 'tr2', 'tr3', 'tr', 'ema', 'macd', 'macd_signal']
        df = df.drop(columns=[col for col in temp_cols if col in df.columns])
        
        return df
    
    def calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算时间特征
        
        Args:
            df: 输入数据框
            
        Returns:
            DataFrame: 添加时间特征的数据框
        """
        df = df.copy()
        
        # 提取时间特征
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_month'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        
        # 周期性编码 (sin/cos)
        # 小时 (24小时周期)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # 星期 (7天周期)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # 月份天数 (30天周期，近似)
        df['dom_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 30)
        df['dom_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 30)
        
        # 月份 (12个月周期)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 清理原始时间列
        time_cols = ['hour', 'day_of_week', 'day_of_month', 'month']
        df = df.drop(columns=time_cols)
        
        return df
    
    def apply_tanh_scaling(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        应用tanh缩放到指定特征
        
        Args:
            df: 输入数据框
            feature_cols: 需要缩放的特征列
            
        Returns:
            DataFrame: 缩放后的数据框
        """
        df = df.copy()
        
        for col in feature_cols:
            if col in df.columns:
                # 先进行波动缩放（除以滚动标准差）
                rolling_std = df[col].rolling(window=self.window_size, min_periods=1).std()
                scaled_col = df[col] / (rolling_std + 1e-8)
                
                # 再应用tanh压缩到[-1,1]
                df[f'{col}_scaled'] = np.tanh(scaled_col)
            else:
                print(f"警告: 列 '{col}' 不存在")
        
        return df
    
    def preprocess_single_file(self, file_path: str) -> pd.DataFrame:
        """
        预处理单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            DataFrame: 预处理后的数据
        """
        print(f"\n处理文件: {file_path}")
        
        # 加载数据
        df = self.load_kline_data(file_path)
        if df.empty:
            return pd.DataFrame()
        
        # 计算基础特征
        df = self.calculate_log_returns(df)
        df = self.calculate_hl_range(df)
        df = self.calculate_volume_zscore(df)
        
        # 计算技术指标
        df = self.calculate_technical_indicators(df)
        
        # 计算时间特征
        df = self.calculate_time_features(df)
        
        # 需要tanh缩放的特征
        scale_features = [
            'log_ret', 'hl_range', 'vol_z', 'atr_norm', 'log_ret_std',
            'ema_ratio', 'rsi_norm', 'macd_delta_norm'
        ]
        
        # 应用tanh缩放
        df = self.apply_tanh_scaling(df, scale_features)
        
        # 选择最终特征列
        feature_cols = [
            'datetime', 'timestamp',   #基础信息，移除close列
            'log_ret_scaled', 'hl_range_scaled', 'vol_z_scaled',  # 基础特征
            'atr_norm_scaled', 'log_ret_std_scaled', 'ema_ratio_scaled',  # 技术指标
            'rsi_norm_scaled', 'macd_delta_norm_scaled',  # 更多技术指标
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',  # 时间特征
            'dom_sin', 'dom_cos', 'month_sin', 'month_cos'
        ]
        
        # 确保所有列都存在
        existing_cols = [col for col in feature_cols if col in df.columns]
        df_final = df[existing_cols].copy()
        
        # 删除包含NaN的行
        initial_rows = len(df_final)
        df_final = df_final.dropna()
        final_rows = len(df_final)
        
        print(f"删除NaN行: {initial_rows} -> {final_rows} (删除 {initial_rows - final_rows} 行)")
        
        return df_final
    
    def preprocess_directory(self, input_dir: str, output_path: str, 
                           file_pattern: str = "*.csv") -> None:
        """
        预处理目录中的所有文件
        
        Args:
            input_dir: 输入目录
            output_path: 输出文件路径
            file_pattern: 文件匹配模式
        """
        print(f"\n开始预处理目录: {input_dir}")
        print(f"输出文件: {output_path}")
        
        # 查找所有CSV文件
        search_pattern = os.path.join(input_dir, "**", file_pattern)
        csv_files = glob.glob(search_pattern, recursive=True)
        
        if not csv_files:
            print(f"未找到匹配的文件: {search_pattern}")
            return
        
        print(f"找到 {len(csv_files)} 个文件")
        
        all_data = []
        
        for file_path in sorted(csv_files):
            try:
                df_processed = self.preprocess_single_file(file_path)
                if not df_processed.empty:
                    # 不再添加文件来源信息，避免在数据转换时出现问题
                    all_data.append(df_processed)
                    print(f"✓ 成功处理: {os.path.basename(file_path)} ({len(df_processed)} 行)")
                else:
                    print(f"✗ 跳过空文件: {os.path.basename(file_path)}")
                    
            except Exception as e:
                print(f"✗ 处理失败 {os.path.basename(file_path)}: {e}")
        
        if not all_data:
            print("没有成功处理的数据")
            return
        
        # 合并所有数据
        print(f"\n合并 {len(all_data)} 个数据文件...")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # 按时间排序
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存结果
        combined_df.to_csv(output_path, index=False)
        
        print(f"\n预处理完成!")
        print(f"总行数: {len(combined_df)}")
        print(f"特征数: {len(combined_df.columns) - 3}")
        print(f"时间范围: {combined_df['datetime'].min()} 到 {combined_df['datetime'].max()}")
        print(f"输出文件: {output_path}")
        
        # 显示特征统计
        print("\n特征统计:")
        feature_cols = [col for col in combined_df.columns 
                       if col.endswith('_scaled') or col.endswith('_sin') or col.endswith('_cos')]
        
        for col in feature_cols:
            stats = combined_df[col].describe()
            print(f"{col:20s}: min={stats['min']:8.4f}, max={stats['max']:8.4f}, mean={stats['mean']:8.4f}")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='K线数据预处理')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入数据目录')
    parser.add_argument('--output_path', type=str, required=True,
                       help='输出文件路径')
    parser.add_argument('--window_size', type=int, default=20,
                       help='滚动窗口大小 (默认: 20)')
    parser.add_argument('--atr_period', type=int, default=14,
                       help='ATR周期 (默认: 14)')
    parser.add_argument('--ema_period', type=int, default=12,
                       help='EMA周期 (默认: 12)')
    parser.add_argument('--rsi_period', type=int, default=14,
                       help='RSI周期 (默认: 14)')
    
    args = parser.parse_args()
    
    # 创建预处理器
    preprocessor = KlinePreprocessor(
        window_size=args.window_size,
        atr_period=args.atr_period,
        ema_period=args.ema_period,
        rsi_period=args.rsi_period
    )
    
    # 执行预处理
    preprocessor.preprocess_directory(
        input_dir=args.input_dir,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()