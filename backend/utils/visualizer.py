#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化模块
用于展示K线图、技术指标和AI预测结果
支持交互式图表和回测结果展示
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 绘图库
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib未安装，基础绘图功能将不可用")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly未安装，交互式图表功能将不可用")

# 设置中文字体
if MATPLOTLIB_AVAILABLE:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")

class TradingVisualizer:
    """
    交易可视化器
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        初始化可视化器
        
        Args:
            figsize: 图表大小
        """
        self.figsize = figsize
        self.colors = {
            'up': '#00ff88',      # 上涨绿色
            'down': '#ff4444',    # 下跌红色
            'volume': '#1f77b4',  # 成交量蓝色
            'ma5': '#ff7f0e',     # MA5橙色
            'ma20': '#2ca02c',    # MA20绿色
            'ma50': '#d62728',    # MA50红色
            'buy': '#00ff00',     # 买入信号
            'sell': '#ff0000',    # 卖出信号
            'prediction': '#9467bd'  # 预测紫色
        }
    
    def plot_candlestick_matplotlib(self, df: pd.DataFrame, 
                                  title: str = "K线图",
                                  volume: bool = True,
                                  ma_periods: List[int] = [5, 20, 50],
                                  save_path: Optional[str] = None) -> None:
        """
        使用Matplotlib绘制K线图
        
        Args:
            df: 包含OHLCV数据的DataFrame
            title: 图表标题
            volume: 是否显示成交量
            ma_periods: 移动平均线周期
            save_path: 保存路径
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib未安装，无法绘制图表")
            return
        
        # 确保数据包含必要列
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            print(f"数据缺少必要列: {required_cols}")
            return
        
        # 创建子图
        if volume and 'volume' in df.columns:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                         gridspec_kw={'height_ratios': [3, 1]},
                                         sharex=True)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=self.figsize)
            ax2 = None
        
        # 绘制K线
        for i in range(len(df)):
            open_price = df.iloc[i]['open']
            high_price = df.iloc[i]['high']
            low_price = df.iloc[i]['low']
            close_price = df.iloc[i]['close']
            
            # 确定颜色
            color = self.colors['up'] if close_price >= open_price else self.colors['down']
            
            # 绘制影线
            ax1.plot([i, i], [low_price, high_price], color='black', linewidth=1)
            
            # 绘制实体
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            rect = Rectangle((i-0.3, body_bottom), 0.6, body_height, 
                           facecolor=color, edgecolor='black', linewidth=0.5)
            ax1.add_patch(rect)
        
        # 绘制移动平均线
        for period in ma_periods:
            if len(df) >= period:
                ma = df['close'].rolling(window=period).mean()
                ax1.plot(range(len(df)), ma, 
                        label=f'MA{period}', 
                        color=self.colors.get(f'ma{period}', f'C{period//10}'),
                        linewidth=1.5)
        
        # 设置主图
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_ylabel('价格', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制成交量
        if ax2 is not None and 'volume' in df.columns:
            colors = [self.colors['up'] if df.iloc[i]['close'] >= df.iloc[i]['open'] 
                     else self.colors['down'] for i in range(len(df))]
            ax2.bar(range(len(df)), df['volume'], color=colors, alpha=0.7)
            ax2.set_ylabel('成交量', fontsize=12)
            ax2.grid(True, alpha=0.3)
        
        # 设置x轴
        if 'open_time' in df.columns:
            # 设置x轴标签
            step = max(1, len(df) // 10)
            x_ticks = range(0, len(df), step)
            x_labels = [df.iloc[i]['open_time'].strftime('%m-%d %H:%M') 
                       if pd.notna(df.iloc[i]['open_time']) else '' 
                       for i in x_ticks]
            
            if ax2 is not None:
                ax2.set_xticks(x_ticks)
                ax2.set_xticklabels(x_labels, rotation=45)
                ax2.set_xlabel('时间', fontsize=12)
            else:
                ax1.set_xticks(x_ticks)
                ax1.set_xticklabels(x_labels, rotation=45)
                ax1.set_xlabel('时间', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def plot_candlestick_plotly(self, df: pd.DataFrame,
                              title: str = "交互式K线图",
                              volume: bool = True,
                              ma_periods: List[int] = [5, 20, 50],
                              save_path: Optional[str] = None) -> None:
        """
        使用Plotly绘制交互式K线图
        
        Args:
            df: 包含OHLCV数据的DataFrame
            title: 图表标题
            volume: 是否显示成交量
            ma_periods: 移动平均线周期
            save_path: 保存路径
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly未安装，无法绘制交互式图表")
            return
        
        # 确保数据包含必要列
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            print(f"数据缺少必要列: {required_cols}")
            return
        
        # 创建子图
        if volume and 'volume' in df.columns:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=('价格', '成交量'),
                row_width=[0.2, 0.7]
            )
        else:
            fig = make_subplots(rows=1, cols=1)
        
        # 绘制K线
        candlestick = go.Candlestick(
            x=df.index if 'open_time' not in df.columns else df['open_time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K线',
            increasing_line_color=self.colors['up'],
            decreasing_line_color=self.colors['down']
        )
        
        fig.add_trace(candlestick, row=1, col=1)
        
        # 绘制移动平均线
        for period in ma_periods:
            if len(df) >= period:
                ma = df['close'].rolling(window=period).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df.index if 'open_time' not in df.columns else df['open_time'],
                        y=ma,
                        mode='lines',
                        name=f'MA{period}',
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
        
        # 绘制成交量
        if volume and 'volume' in df.columns:
            colors = ['green' if df.iloc[i]['close'] >= df.iloc[i]['open'] 
                     else 'red' for i in range(len(df))]
            
            fig.add_trace(
                go.Bar(
                    x=df.index if 'open_time' not in df.columns else df['open_time'],
                    y=df['volume'],
                    name='成交量',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # 更新布局
        fig.update_layout(
            title=title,
            xaxis_title='时间',
            yaxis_title='价格',
            template='plotly_white',
            height=800,
            showlegend=True
        )
        
        # 隐藏x轴范围选择器
        fig.update_layout(xaxis_rangeslider_visible=False)
        
        if save_path:
            fig.write_html(save_path)
            print(f"交互式图表已保存到: {save_path}")
        
        fig.show()
    
    def plot_predictions(self, df: pd.DataFrame, 
                        predictions: np.ndarray,
                        actual: np.ndarray,
                        title: str = "AI预测结果",
                        save_path: Optional[str] = None) -> None:
        """
        绘制AI预测结果对比图
        
        Args:
            df: 原始数据
            predictions: 预测结果
            actual: 实际结果
            title: 图表标题
            save_path: 保存路径
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib未安装，无法绘制图表")
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.figsize, 
                                          gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # 绘制价格走势
        ax1.plot(df['close'], label='收盘价', color='black', linewidth=1)
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_ylabel('价格', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制预测vs实际
        x_range = range(len(predictions))
        ax2.plot(x_range, actual, label='实际', color='blue', linewidth=2)
        ax2.plot(x_range, predictions, label='预测', color='red', linewidth=2, alpha=0.7)
        ax2.set_ylabel('信号', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 绘制预测准确性
        accuracy = (predictions == actual).astype(int)
        colors = ['green' if acc else 'red' for acc in accuracy]
        ax3.bar(x_range, accuracy, color=colors, alpha=0.7)
        ax3.set_ylabel('准确性', fontsize=12)
        ax3.set_xlabel('时间', fontsize=12)
        ax3.set_ylim(0, 1.2)
        ax3.grid(True, alpha=0.3)
        
        # 添加准确率文本
        overall_accuracy = np.mean(accuracy)
        ax3.text(0.02, 0.98, f'总体准确率: {overall_accuracy:.2%}', 
                transform=ax3.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"预测结果图已保存到: {save_path}")
        
        plt.show()
    
    def plot_backtest_results(self, backtest_results: Dict,
                            title: str = "回测结果",
                            save_path: Optional[str] = None) -> None:
        """
        绘制回测结果
        
        Args:
            backtest_results: 回测结果字典
            title: 图表标题
            save_path: 保存路径
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib未安装，无法绘制图表")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
        # 1. 资金曲线
        if 'equity_curve' in backtest_results:
            equity = backtest_results['equity_curve']
            ax1.plot(equity, color='blue', linewidth=2)
            ax1.set_title('资金曲线', fontsize=14)
            ax1.set_ylabel('资金', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # 添加最大回撤标记
            if 'max_drawdown' in backtest_results:
                max_dd = backtest_results['max_drawdown']
                ax1.text(0.02, 0.98, f'最大回撤: {max_dd:.2%}', 
                        transform=ax1.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral'))
        
        # 2. 收益分布
        if 'returns' in backtest_results:
            returns = backtest_results['returns']
            ax2.hist(returns, bins=50, alpha=0.7, color='green', edgecolor='black')
            ax2.set_title('收益分布', fontsize=14)
            ax2.set_xlabel('收益率', fontsize=12)
            ax2.set_ylabel('频次', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            ax2.axvline(mean_return, color='red', linestyle='--', label=f'均值: {mean_return:.3f}')
            ax2.legend()
        
        # 3. 交易信号
        if 'signals' in backtest_results:
            signals = backtest_results['signals']
            buy_signals = [i for i, s in enumerate(signals) if s == 1]
            sell_signals = [i for i, s in enumerate(signals) if s == -1]
            
            ax3.scatter(buy_signals, [1]*len(buy_signals), 
                       color='green', marker='^', s=50, label='买入', alpha=0.7)
            ax3.scatter(sell_signals, [-1]*len(sell_signals), 
                       color='red', marker='v', s=50, label='卖出', alpha=0.7)
            ax3.set_title('交易信号', fontsize=14)
            ax3.set_ylabel('信号', fontsize=12)
            ax3.set_ylim(-1.5, 1.5)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 关键指标
        ax4.axis('off')
        if 'metrics' in backtest_results:
            metrics = backtest_results['metrics']
            metrics_text = "\n".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                    for k, v in metrics.items()])
            ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"回测结果图已保存到: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str], 
                              importance_scores: np.ndarray,
                              title: str = "特征重要性",
                              top_n: int = 20,
                              save_path: Optional[str] = None) -> None:
        """
        绘制特征重要性图
        
        Args:
            feature_names: 特征名称列表
            importance_scores: 重要性分数
            title: 图表标题
            top_n: 显示前N个重要特征
            save_path: 保存路径
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib未安装，无法绘制图表")
            return
        
        # 排序并选择前N个
        indices = np.argsort(importance_scores)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_scores = importance_scores[indices]
        
        # 绘制水平条形图
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
        
        y_pos = np.arange(len(top_features))
        bars = ax.barh(y_pos, top_scores, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, top_scores)):
            ax.text(bar.get_width() + max(top_scores) * 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', ha='left', va='center', fontsize=10)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel('重要性分数', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征重要性图已保存到: {save_path}")
        
        plt.show()
    
    def create_dashboard(self, df: pd.DataFrame, 
                        predictions: Optional[np.ndarray] = None,
                        backtest_results: Optional[Dict] = None,
                        save_path: Optional[str] = None) -> None:
        """
        创建综合仪表板
        
        Args:
            df: 市场数据
            predictions: 预测结果
            backtest_results: 回测结果
            save_path: 保存路径
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly未安装，无法创建仪表板")
            return
        
        # 创建子图
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('K线图', '成交量', '技术指标', '预测结果', '收益曲线', '关键指标'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "table"}]],
            vertical_spacing=0.08
        )
        
        # 1. K线图
        candlestick = go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K线'
        )
        fig.add_trace(candlestick, row=1, col=1)
        
        # 2. 成交量
        if 'volume' in df.columns:
            fig.add_trace(
                go.Bar(x=df.index, y=df['volume'], name='成交量', opacity=0.7),
                row=1, col=2
            )
        
        # 3. 技术指标 (RSI示例)
        if 'rsi_14' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['rsi_14'], name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # 4. 预测结果
        if predictions is not None:
            fig.add_trace(
                go.Scatter(x=df.index[-len(predictions):], y=predictions, 
                          name='预测', line=dict(color='red')),
                row=2, col=2
            )
        
        # 5. 收益曲线
        if backtest_results and 'equity_curve' in backtest_results:
            equity = backtest_results['equity_curve']
            fig.add_trace(
                go.Scatter(x=list(range(len(equity))), y=equity, 
                          name='资金曲线', line=dict(color='blue')),
                row=3, col=1
            )
        
        # 6. 关键指标表格
        if backtest_results and 'metrics' in backtest_results:
            metrics = backtest_results['metrics']
            fig.add_trace(
                go.Table(
                    header=dict(values=['指标', '数值']),
                    cells=dict(values=[list(metrics.keys()), list(metrics.values())])
                ),
                row=3, col=2
            )
        
        # 更新布局
        fig.update_layout(
            title='AI量化交易仪表板',
            height=1200,
            showlegend=True,
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"仪表板已保存到: {save_path}")
        
        fig.show()

if __name__ == "__main__":
    # 示例用法
    print("可视化模块初始化完成")
    print(f"Matplotlib可用: {MATPLOTLIB_AVAILABLE}")
    print(f"Plotly可用: {PLOTLY_AVAILABLE}")
    
    # 创建可视化器
    visualizer = TradingVisualizer()
    
    # 生成示例数据
    dates = pd.date_range('2023-01-01', periods=100, freq='H')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'open_time': dates,
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # 计算high, low, close
    for i in range(len(sample_data)):
        open_price = sample_data.iloc[i]['open']
        change = np.random.randn() * 2
        close_price = open_price + change
        high_price = max(open_price, close_price) + abs(np.random.randn() * 0.5)
        low_price = min(open_price, close_price) - abs(np.random.randn() * 0.5)
        
        sample_data.iloc[i, sample_data.columns.get_loc('high')] = high_price
        sample_data.iloc[i, sample_data.columns.get_loc('low')] = low_price
        sample_data.iloc[i, sample_data.columns.get_loc('close')] = close_price
    
    print("示例数据生成完成，可以使用以下方法测试:")
    print("visualizer.plot_candlestick_matplotlib(sample_data)")
    if PLOTLY_AVAILABLE:
        print("visualizer.plot_candlestick_plotly(sample_data)")