#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI量化交易预测系统 - 主程序
提供命令行界面来运行各种功能模块
"""

import sys
import os
from pathlib import Path
import argparse
from datetime import datetime, timedelta

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "backend"))

# 导入模块
try:
    from backend.data.binance_data_fetcher import BinanceDataFetcher, interactive_data_fetcher
    from backend.data.data_preprocessor import DataPreprocessor
    from backend.models.model_trainer import ModelTrainer
    from backend.utils.visualizer import TradingVisualizer
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保所有依赖已安装: pip install -r requirements.txt")
    sys.exit(1)

def print_banner():
    """
    打印程序横幅
    """
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                AI量化交易预测系统                              ║
    ║              基于Stable Baseline的智能交易平台                 ║
    ║                                                              ║
    ║  功能特性:                                                    ║
    ║  • 币安历史数据获取                                           ║
    ║  • 智能数据预处理                                             ║
    ║  • 多种机器学习模型                                           ║
    ║  • 强化学习交易策略                                           ║
    ║  • 可视化分析工具                                             ║
    ║  • 回测与风险管理                                             ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def fetch_data_command(args):
    """
    数据获取命令
    """
    print("\n=== 数据获取模块 ===")
    
    if args.interactive:
        # 交互式模式
        interactive_data_fetcher()
    else:
        # 命令行模式
        fetcher = BinanceDataFetcher()
        
        symbol = args.symbol.upper()
        interval = args.interval
        
        if args.days:
            # 获取最近N天数据
            df = fetcher.fetch_recent_data(symbol, interval, args.days)
        else:
            # 指定日期范围
            if not args.start_date or not args.end_date:
                print("错误: 请指定开始和结束日期，或使用 --days 参数")
                return
            df = fetcher.fetch_historical_data(symbol, interval, args.start_date, args.end_date)
        
        print(f"\n数据获取完成!")
        print(f"数据形状: {df.shape}")
        print(f"时间范围: {df['open_time'].min()} 到 {df['open_time'].max()}")
        print("\n前5行数据:")
        print(df.head())

def preprocess_data_command(args):
    """
    数据预处理命令
    """
    print("\n=== 数据预处理模块 ===")
    
    if not args.input_file:
        print("错误: 请指定输入数据文件路径")
        return
    
    if not Path(args.input_file).exists():
        print(f"错误: 文件不存在 {args.input_file}")
        return
    
    # 创建预处理器
    preprocessor = DataPreprocessor(
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon
    )
    
    # 处理数据
    try:
        data = preprocessor.process_data(
            args.input_file,
            target_type=args.target_type,
            scaler_type=args.scaler_type
        )
        
        # 保存处理后的数据
        if args.output_dir:
            symbol = Path(args.input_file).stem.split('_')[0]  # 从文件名提取交易对
            preprocessor.save_processed_data(data, args.output_dir, symbol)
        
        print("\n数据预处理完成!")
        
    except Exception as e:
        print(f"数据预处理失败: {e}")

def train_model_command(args):
    """
    模型训练命令
    """
    print("\n=== 模型训练模块 ===")
    print("注意: 完整的模型训练功能将在后续版本中实现")
    print("当前版本提供了训练框架，支持:")
    print("• 传统机器学习模型 (随机森林、梯度提升等)")
    print("• 深度学习模型 (LSTM、GRU、CNN-LSTM)")
    print("• 强化学习环境 (为Stable Baseline准备)")
    
    # 创建训练器
    trainer = ModelTrainer()
    print(f"\n模型保存目录: {trainer.model_save_dir}")
    
    # 显示可用模型
    traditional_models = trainer.create_traditional_models()
    print(f"\n可用传统模型: {list(traditional_models.keys())}")
    
    try:
        lstm_model = trainer.create_lstm_model((60, 10))
        if lstm_model:
            print("深度学习模型创建成功")
    except Exception as e:
        print(f"深度学习模型创建失败: {e}")

def visualize_command(args):
    """
    可视化命令
    """
    print("\n=== 数据可视化模块 ===")
    
    if not args.input_file:
        print("错误: 请指定输入数据文件路径")
        return
    
    if not Path(args.input_file).exists():
        print(f"错误: 文件不存在 {args.input_file}")
        return
    
    try:
        import pandas as pd
        
        # 读取数据
        df = pd.read_csv(args.input_file)
        
        # 确保时间列格式正确
        if 'open_time' in df.columns:
            df['open_time'] = pd.to_datetime(df['open_time'])
        
        # 创建可视化器
        visualizer = TradingVisualizer()
        
        # 绘制K线图
        if args.chart_type == 'matplotlib':
            visualizer.plot_candlestick_matplotlib(
                df, 
                title=f"{Path(args.input_file).stem} K线图",
                save_path=args.output_file
            )
        elif args.chart_type == 'plotly':
            visualizer.plot_candlestick_plotly(
                df,
                title=f"{Path(args.input_file).stem} 交互式K线图",
                save_path=args.output_file
            )
        
        print("\n可视化完成!")
        
    except Exception as e:
        print(f"可视化失败: {e}")

def list_data_command(args):
    """
    列出可用数据文件
    """
    print("\n=== 可用数据文件 ===")
    
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print("数据目录不存在，请先获取数据")
        return
    
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("未找到数据文件，请先使用 fetch-data 命令获取数据")
        return
    
    print(f"找到 {len(csv_files)} 个数据文件:")
    for i, file in enumerate(csv_files, 1):
        file_size = file.stat().st_size / (1024 * 1024)  # MB
        print(f"{i:2d}. {file.name} ({file_size:.2f} MB)")

def main():
    """
    主函数
    """
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="AI量化交易预测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 数据获取命令
    fetch_parser = subparsers.add_parser('fetch-data', help='获取历史数据')
    fetch_parser.add_argument('--interactive', '-i', action='store_true', help='交互式模式')
    fetch_parser.add_argument('--symbol', '-s', default='BTCUSDT', help='交易对符号 (默认: BTCUSDT)')
    fetch_parser.add_argument('--interval', default='1h', help='时间间隔 (默认: 1h)')
    fetch_parser.add_argument('--days', type=int, help='获取最近N天数据')
    fetch_parser.add_argument('--start-date', help='开始日期 (YYYY-MM-DD)')
    fetch_parser.add_argument('--end-date', help='结束日期 (YYYY-MM-DD)')
    
    # 数据预处理命令
    preprocess_parser = subparsers.add_parser('preprocess', help='数据预处理')
    preprocess_parser.add_argument('--input-file', '-i', required=True, help='输入数据文件')
    preprocess_parser.add_argument('--output-dir', '-o', default='data/processed', help='输出目录')
    preprocess_parser.add_argument('--sequence-length', type=int, default=60, help='序列长度')
    preprocess_parser.add_argument('--prediction-horizon', type=int, default=1, help='预测时间跨度')
    preprocess_parser.add_argument('--target-type', choices=['price_direction', 'price_return', 'price_level'], 
                                 default='price_direction', help='目标类型')
    preprocess_parser.add_argument('--scaler-type', choices=['minmax', 'standard'], 
                                 default='minmax', help='缩放类型')
    
    # 模型训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--data-dir', default='data/processed', help='处理后数据目录')
    train_parser.add_argument('--model-type', choices=['traditional', 'deep_learning', 'reinforcement'], 
                            default='traditional', help='模型类型')
    
    # 可视化命令
    viz_parser = subparsers.add_parser('visualize', help='数据可视化')
    viz_parser.add_argument('--input-file', '-i', required=True, help='输入数据文件')
    viz_parser.add_argument('--chart-type', choices=['matplotlib', 'plotly'], 
                          default='matplotlib', help='图表类型')
    viz_parser.add_argument('--output-file', '-o', help='输出文件路径')
    
    # 列出数据命令
    list_parser = subparsers.add_parser('list-data', help='列出可用数据文件')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 执行对应命令
    if args.command == 'fetch-data':
        fetch_data_command(args)
    elif args.command == 'preprocess':
        preprocess_data_command(args)
    elif args.command == 'train':
        train_model_command(args)
    elif args.command == 'visualize':
        visualize_command(args)
    elif args.command == 'list-data':
        list_data_command(args)

if __name__ == "__main__":
    main()