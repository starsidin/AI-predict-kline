# AI量化交易预测系统

基于Stable Baseline的股票、虚拟货币AI量化交易预测系统，集成数据获取、模型训练、预测分析和可视化展示功能。

## 🚀 项目特性

- **📊 多源数据获取**: 支持币安(Binance)历史数据获取，涵盖多种加密货币和时间间隔
- **🔧 智能数据预处理**: 自动化数据清洗、特征工程和技术指标计算
- **🤖 多模型支持**: 集成传统机器学习、深度学习和强化学习模型
- **📈 可视化分析**: 交互式K线图、技术指标和AI预测结果展示
- **🔄 回测系统**: 完整的策略回测和风险评估框架
- **⚡ 实时预测**: 支持实时数据获取和模型预测

## 📁 项目结构

```
AI_predict/
├── backend/                    # 后端核心模块
│   ├── api/                   # API接口
│   ├── data/                  # 数据处理模块
│   │   ├── binance_data_fetcher.py  # 币安数据获取器
│   │   └── data_preprocessor.py     # 数据预处理器
│   ├── models/                # 模型训练模块
│   │   └── model_trainer.py   # 模型训练框架
│   └── utils/                 # 工具模块
│       └── visualizer.py      # 可视化工具
├── data/                      # 数据存储目录
│   ├── raw/                   # 原始数据
│   └── processed/             # 处理后数据
├── frontend/                  # 前端界面
│   └── src/
│       └── components/
├── models/                    # 训练好的模型
├── main.py                    # 主程序入口
├── requirements.txt           # 依赖包列表
└── README.md                  # 项目说明文档
```

## 🛠️ 安装指南

### 环境要求

- Python 3.8+
- pip 或 conda
- 8GB+ RAM (推荐)
- GPU支持 (可选，用于深度学习加速)

### 快速安装

1. **克隆项目**
```bash
git clone <repository-url>
cd AI_predict
```

2. **创建虚拟环境** (推荐)
```bash
# 使用conda
conda create -n ai_trading python=3.9
conda activate ai_trading

# 或使用venv
python -m venv ai_trading
# Windows
ai_trading\Scripts\activate
# Linux/Mac
source ai_trading/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

### 可选依赖安装

**TA-Lib技术指标库** (推荐)
```bash
# Windows
pip install TA-Lib

# Linux/Mac
# 先安装系统依赖
sudo apt-get install libta-lib-dev  # Ubuntu/Debian
brew install ta-lib                 # macOS
# 然后安装Python包
pip install TA-Lib
```

**GPU支持** (可选)
```bash
# NVIDIA GPU支持
pip install tensorflow-gpu
```

## 🎯 快速开始

### 1. 数据获取

#### 交互式模式 (推荐新手)
```bash
python main.py fetch-data --interactive
```

#### 命令行模式
```bash
# 获取BTC最近30天的1小时数据
python main.py fetch-data --symbol BTCUSDT --interval 1h --days 30

# 获取指定日期范围的数据
python main.py fetch-data --symbol ETHUSDT --interval 1d --start-date 2023-01-01 --end-date 2023-12-31
```

#### 支持的交易对
- 主流币: BTCUSDT, ETHUSDT, BNBUSDT
- 热门币: ADAUSDT, XRPUSDT, SOLUSDT, DOTUSDT
- 更多: DOGEUSDT, AVAXUSDT, MATICUSDT, LTCUSDT

#### 支持的时间间隔
- 分钟级: 1m, 3m, 5m, 15m, 30m
- 小时级: 1h, 2h, 4h, 6h, 8h, 12h
- 日级: 1d, 3d, 1w, 1M

### 2. 数据预处理

```bash
# 基础预处理
python main.py preprocess --input-file data/raw/BTCUSDT_1h_2023-01-01_to_2023-12-31.csv

# 自定义参数
python main.py preprocess \
    --input-file data/raw/BTCUSDT_1h_2023-01-01_to_2023-12-31.csv \
    --sequence-length 120 \
    --target-type price_direction \
    --scaler-type minmax
```

### 3. 数据可视化

```bash
# 使用Matplotlib绘制K线图
python main.py visualize --input-file data/raw/BTCUSDT_1h_2023-01-01_to_2023-12-31.csv

# 使用Plotly绘制交互式图表
python main.py visualize --input-file data/raw/BTCUSDT_1h_2023-01-01_to_2023-12-31.csv --chart-type plotly
```

### 4. 查看可用数据

```bash
python main.py list-data
```

## 📊 功能模块详解

### 数据获取模块 (binance_data_fetcher.py)

**主要功能:**
- 从币安官方数据源获取历史K线数据
- 支持多种时间间隔和交易对
- 自动数据验证和格式化
- 支持批量下载和断点续传

**核心类:**
```python
from backend.data.binance_data_fetcher import BinanceDataFetcher

# 创建数据获取器
fetcher = BinanceDataFetcher()

# 获取数据
df = fetcher.fetch_historical_data('BTCUSDT', '1h', '2023-01-01', '2023-12-31')
```

### 数据预处理模块 (data_preprocessor.py)

**主要功能:**
- 数据清洗和异常值处理
- 技术指标计算 (MA, RSI, MACD, 布林带等)
- 特征工程和序列化
- 目标变量生成

**技术指标包括:**
- 移动平均线: SMA, EMA (5, 10, 20, 50期)
- 动量指标: RSI, 随机指标, 威廉指标
- 趋势指标: MACD, CCI
- 波动率指标: 布林带, ATR
- 成交量指标: 成交量比率

**核心类:**
```python
from backend.data.data_preprocessor import DataPreprocessor

# 创建预处理器
preprocessor = DataPreprocessor(sequence_length=60)

# 处理数据
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.process_data(
    'data/raw/BTCUSDT_1h_2023-01-01_to_2023-12-31.csv'
)
```

### 模型训练框架 (model_trainer.py)

**支持的模型类型:**

1. **传统机器学习**
   - 随机森林 (Random Forest)
   - 梯度提升 (Gradient Boosting)
   - 逻辑回归 (Logistic Regression)
   - 支持向量机 (SVM)

2. **深度学习模型**
   - LSTM (长短期记忆网络)
   - GRU (门控循环单元)
   - CNN-LSTM (卷积-循环混合网络)

3. **强化学习环境**
   - 交易环境基类 (为Stable Baseline准备)
   - 动作空间: 买入/卖出/持有
   - 奖励函数: 基于收益率和风险

**核心类:**
```python
from backend.models.model_trainer import ModelTrainer

# 创建训练器
trainer = ModelTrainer()

# 训练传统模型
results = trainer.train_traditional_models(X_train, y_train, X_val, y_val)

# 创建深度学习模型
lstm_model = trainer.create_lstm_model((60, 50))  # 60时间步，50特征
```

### 可视化模块 (visualizer.py)

**支持的图表类型:**
- K线图 (Candlestick Chart)
- 成交量图
- 技术指标图
- AI预测结果对比
- 回测结果展示
- 特征重要性分析

**核心类:**
```python
from backend.utils.visualizer import TradingVisualizer

# 创建可视化器
visualizer = TradingVisualizer()

# 绘制K线图
visualizer.plot_candlestick_matplotlib(df)
visualizer.plot_candlestick_plotly(df)  # 交互式图表
```

## 🔧 高级配置

### 自定义数据源

可以扩展数据获取器以支持其他数据源:

```python
class CustomDataFetcher(BinanceDataFetcher):
    def __init__(self):
        super().__init__()
        self.base_url = "your_custom_api_url"
    
    def fetch_custom_data(self, symbol, start_date, end_date):
        # 实现自定义数据获取逻辑
        pass
```

### 自定义技术指标

```python
def custom_indicator(df):
    """自定义技术指标"""
    # 实现自定义指标计算
    return indicator_values

# 在预处理器中添加
preprocessor = DataPreprocessor()
df['custom_indicator'] = custom_indicator(df)
```

### 模型参数调优

```python
# 使用Optuna进行超参数优化
import optuna

def objective(trial):
    # 定义超参数搜索空间
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    
    # 训练模型并返回验证分数
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    # ... 训练和评估逻辑
    return validation_score

# 运行优化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

## 📈 使用示例

### 完整的交易策略开发流程

```python
# 1. 数据获取
fetcher = BinanceDataFetcher()
df = fetcher.fetch_historical_data('BTCUSDT', '1h', '2023-01-01', '2023-12-31')

# 2. 数据预处理
preprocessor = DataPreprocessor(sequence_length=60)
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.process_data(
    'data/raw/BTCUSDT_1h_2023-01-01_to_2023-12-31.csv',
    target_type='price_direction'
)

# 3. 模型训练
trainer = ModelTrainer()
results = trainer.train_traditional_models(X_train, y_train, X_val, y_val)

# 4. 模型评估
best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
print(f"最佳模型: {best_model[0]}, F1分数: {best_model[1]['f1_score']:.4f}")

# 5. 可视化结果
visualizer = TradingVisualizer()
visualizer.plot_candlestick_plotly(df)
```

## 🚨 注意事项

### 数据使用
- 币安数据仅供研究和学习使用
- 请遵守相关法律法规和平台使用条款
- 建议使用VPN以确保数据获取稳定性

### 模型风险
- 历史数据不代表未来表现
- 模型预测存在不确定性
- 请谨慎使用于实际交易
- 建议充分回测和风险评估

### 性能优化
- 大数据集处理可能需要较长时间
- 建议使用SSD存储以提高I/O性能
- GPU加速可显著提升深度学习训练速度

## 🔮 后续开发计划

### 短期目标 (1-2个月)
- [ ] 集成Stable Baseline3强化学习算法
- [ ] 实现GRU-TCN和Transformer模型
- [ ] 添加更多技术指标和特征工程
- [ ] 完善回测系统和风险管理

### 中期目标 (3-6个月)
- [ ] Web界面开发
- [ ] 实时数据流处理
- [ ] 多资产组合优化
- [ ] 情感分析和新闻数据集成

### 长期目标 (6个月+)
- [ ] 云端部署和API服务
- [ ] 移动端应用
- [ ] 社区功能和策略分享
- [ ] 机构级风险管理系统

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- 项目维护者: [Your Name]
- 邮箱: [your.email@example.com]
- 项目链接: [https://github.com/yourusername/AI_predict]

## 🙏 致谢

- [Binance](https://www.binance.com/) - 提供免费的历史数据API
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - 强化学习算法库
- [TA-Lib](https://ta-lib.org/) - 技术分析指标库
- [Plotly](https://plotly.com/) - 交互式可视化库

---

**免责声明**: 本项目仅供教育和研究目的使用。任何投资决策都应基于您自己的研究和风险承受能力。作者不对使用本软件造成的任何损失承担责任。