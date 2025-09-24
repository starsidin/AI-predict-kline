#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练框架
支持传统机器学习模型和深度学习模型
为后续集成Stable Baseline强化学习做准备
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import joblib
import json
from datetime import datetime

# 传统机器学习模型
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 深度学习模型
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        LSTM,
        GRU,
        Dense,
        Dropout,
        Input,
        Conv1D,
        MaxPooling1D,
        Flatten,
        GlobalAveragePooling1D,
        Concatenate,
        LayerNormalization,
        Add,
        Lambda,
        MultiHeadAttention,
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow未安装，深度学习功能将不可用")

# 强化学习环境基类（为Stable Baseline准备）
try:
    import gym
    from gym import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    print("Gym未安装，强化学习环境将不可用")

DEFAULT_FEATURE_COLUMNS = [
    "log_ret_scaled",
    "hl_range_scaled",
    "vol_z_scaled",
    "atr_norm_scaled",
    "log_ret_std_scaled",
    "ema_ratio_scaled",
    "rsi_norm_scaled",
    "macd_delta_norm_scaled",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "dom_sin",
    "dom_cos",
    "month_sin",
    "month_cos",
]

class TradingEnvironment:
    """
    交易环境基类（为强化学习准备）
    """
    
    def __init__(self, data: np.ndarray, initial_balance: float = 10000.0):
        """
        初始化交易环境
        
        Args:
            data: 市场数据
            initial_balance: 初始资金
        """
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # 0: 空仓, 1: 多头, -1: 空头
        self.entry_price = 0
        self.total_trades = 0
        self.winning_trades = 0
        
        # 动作空间: 0=持有, 1=买入, 2=卖出
        self.action_space_size = 3
        
        # 观察空间维度
        self.observation_space_size = data.shape[1] if len(data.shape) > 1 else 1
    
    def reset(self):
        """
        重置环境
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.total_trades = 0
        self.winning_trades = 0
        return self._get_observation()
    
    def _get_observation(self):
        """
        获取当前观察状态
        """
        if self.current_step >= len(self.data):
            return np.zeros(self.observation_space_size)
        
        obs = self.data[self.current_step]
        if len(obs.shape) == 0:
            obs = np.array([obs])
        
        # 添加账户信息
        account_info = np.array([self.balance / self.initial_balance, self.position])
        return np.concatenate([obs.flatten(), account_info])
    
    def step(self, action: int):
        """
        执行动作
        
        Args:
            action: 动作 (0=持有, 1=买入, 2=卖出)
            
        Returns:
            (observation, reward, done, info)
        """
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, {}
        
        current_price = self.data[self.current_step, 0] if len(self.data.shape) > 1 else self.data[self.current_step]
        
        reward = 0
        
        # 执行交易动作
        if action == 1 and self.position <= 0:  # 买入
            if self.position == -1:  # 平空头
                reward += (self.entry_price - current_price) / self.entry_price
            self.position = 1
            self.entry_price = current_price
            self.total_trades += 1
            
        elif action == 2 and self.position >= 0:  # 卖出
            if self.position == 1:  # 平多头
                reward += (current_price - self.entry_price) / self.entry_price
            self.position = -1
            self.entry_price = current_price
            self.total_trades += 1
        
        # 计算持仓收益
        if self.position != 0:
            next_price = self.data[self.current_step + 1, 0] if len(self.data.shape) > 1 else self.data[self.current_step + 1]
            unrealized_pnl = self.position * (next_price - current_price) / current_price
            reward += unrealized_pnl * 0.1  # 给予部分未实现收益
        
        if reward > 0:
            self.winning_trades += 1
        
        self.current_step += 1
        
        done = self.current_step >= len(self.data) - 1
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades)
        }
        
        return self._get_observation(), reward, done, info

class ModelTrainer:
    """
    模型训练器
    """
    
    def __init__(
        self,
        model_save_dir: str = "../../models",
        feature_columns: Optional[List[str]] = None,
    ):
        """
        初始化训练器
        
        Args:
            model_save_dir: 模型保存目录
            feature_columns: 预处理特征列顺序，默认为预处理流水线产出的列顺序
        """
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.training_history = {}
        self.feature_columns = (
            list(feature_columns)
            if feature_columns is not None
            else list(DEFAULT_FEATURE_COLUMNS)
        )

    def _prepare_features(self, data: Any):
        """将输入数据转换为numpy数组并按特征顺序排列"""
        if isinstance(data, (list, tuple)):
            return [self._prepare_features(item) for item in data]
        if isinstance(data, pd.DataFrame):
            available = [col for col in self.feature_columns if col in data.columns]
            if not available:
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    raise ValueError("DataFrame中没有可用的数值特征列")
                available = numeric_cols
            return data[available].to_numpy(dtype=np.float32)
        if isinstance(data, pd.Series):
            return data.to_numpy(dtype=np.float32)
        array = np.asarray(data)
        if array.dtype == np.float64:
            array = array.astype(np.float32)
        return array

    def _prepare_targets(self, data: Any):
        """确保标签数据为合适的numpy数组"""
        if isinstance(data, pd.DataFrame):
            array = data.iloc[:, 0].to_numpy()
        elif isinstance(data, pd.Series):
            array = data.to_numpy()
        else:
            array = np.asarray(data)
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim == 1:
            if array.dtype == bool:
                array = array.astype(np.int64)
            elif np.issubdtype(array.dtype, np.integer):
                array = array.astype(np.int64)
            elif np.issubdtype(array.dtype, np.floating):
                if np.all(np.isclose(array, np.round(array))):
                    array = np.round(array).astype(np.int64)
                else:
                    array = array.astype(np.float32)
        else:
            if np.issubdtype(array.dtype, np.floating):
                array = array.astype(np.float32)
        return array

    def create_traditional_models(self) -> Dict[str, Any]:
        """
        创建传统机器学习模型
        
        Returns:
            模型字典
        """
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        }
        return models
    
    def create_lstm_model(self, input_shape: Tuple[int, int], num_classes: int = 2) -> Optional[Model]:
        """
        创建LSTM模型
        
        Args:
            input_shape: 输入形状 (sequence_length, features)
            num_classes: 分类数量
            
        Returns:
            LSTM模型
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow未安装，无法创建LSTM模型")
            return None
        
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def create_tcn_model(
        self,
        input_shape: Tuple[int, int],
        num_classes: int = 2,
        loss_type: str = "crossentropy",
    ) -> Optional[Model]:
        """创建TCN/1D CNN模型

        该模型采用空洞卷积构建的时间卷积网络（Temporal Convolutional Network, TCN）。
        通过不同空洞率的卷积堆叠捕捉不同时间尺度的特征，并构建短、长两条分支后拼接。

        Args:
            input_shape: 输入张量形状 ``(时间步, 特征数)``
            num_classes: 分类类别数
            loss_type: ``"crossentropy"`` 使用交叉熵；``"focal"`` 使用focal loss

        Returns:
            构建好的 ``tf.keras.Model`` 模型实例
        """

        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow未安装，无法创建TCN模型")
            return None

        inputs = Input(shape=input_shape)

        def tcn_block(x: tf.Tensor, filters: int) -> tf.Tensor:
            """TCN核心模块，包含多层空洞卷积与残差连接"""
            for d in [1, 2, 4, 8, 16]:
                residual = x
                x = Conv1D(
                    filters,
                    kernel_size=3,
                    padding="causal",
                    dilation_rate=d,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                )(x)
                x = LayerNormalization()(x)
                x = Dropout(0.1)(x)
                if residual.shape[-1] != x.shape[-1]:
                    residual = Conv1D(filters, 1, padding="same")(residual)
                x = Add()([x, residual])
            return x

        short_branch = tcn_block(inputs, 32)
        short_pool = GlobalAveragePooling1D()(short_branch)

        long_branch = tcn_block(inputs, 48)
        long_pool = GlobalAveragePooling1D()(long_branch)

        merged = Concatenate()([short_pool, long_pool])
        x = Dense(128, activation="relu")(merged)
        x = Dropout(0.1)(x)
        x = Dense(64, activation="relu")(x)
        outputs = Dense(num_classes, activation="softmax")(x)

        model = Model(inputs, outputs)

        def focal_loss(y_true, y_pred, gamma: float = 2.0, alpha: float = 0.25):
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            ce = -y_true * tf.math.log(y_pred)
            weight = alpha * tf.pow(1 - y_pred, gamma)
            return tf.reduce_sum(weight * ce, axis=-1)

        loss_fn = focal_loss if loss_type == "focal" else "sparse_categorical_crossentropy"
        optimizer = Adam(learning_rate=1e-3, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
        return model

    def create_dual_gru_model(
        self,
        main_input_shape: Tuple[int, int],
        aux_input_shape: Tuple[int, int],
        num_classes: int = 2,
    ) -> Optional[Model]:
        """创建双输入GRU模型

        一个分支接收15分钟特征序列, 另一个分支接收运行时计算的
        均线特征序列。"""

        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow未安装，无法创建GRU模型")
            return None

        main_inputs = Input(shape=main_input_shape, name="main_input")
        short = GRU(96, return_sequences=True, dropout=0.2)(main_inputs)
        short = GRU(96, dropout=0.2)(short)

        aux_inputs = Input(shape=aux_input_shape, name="aux_input")
        aux_branch = GRU(64, return_sequences=True)(aux_inputs)
        aux_branch = GRU(64)(aux_branch)

        merged = Concatenate()([short, aux_branch])
        x = Dense(128, activation="relu")(merged)
        x = Dropout(0.2)(x)
        x = Dense(128, activation="relu")(x)
        outputs = Dense(num_classes, activation="softmax")(x)

        optimizer = Adam(learning_rate=1e-3, clipnorm=1.0)
        model = Model(inputs=[main_inputs, aux_inputs], outputs=outputs)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def create_tiny_transformer_model(
        self, input_shape: Tuple[int, int], num_classes: int = 2
    ) -> Optional[Model]:
        """创建双分支Tiny-Transformer模型"""

        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow未安装，无法创建Transformer模型")
            return None

        inputs = Input(shape=input_shape)

        x = Dense(64)(inputs)

        class PositionalEncoding(tf.keras.layers.Layer):
            def __init__(self, d_model: int):
                super().__init__()
                self.d_model = d_model

            def call(self, x):
                pos = tf.range(tf.shape(x)[1], dtype=tf.float32)[:, tf.newaxis]
                i = tf.range(self.d_model, dtype=tf.float32)[tf.newaxis, :]
                angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / self.d_model)
                angle_rads = pos * angle_rates
                sines = tf.sin(angle_rads[:, 0::2])
                cosines = tf.cos(angle_rads[:, 1::2])
                pos_encoding = tf.concat([sines, cosines], axis=-1)
                pos_encoding = pos_encoding[tf.newaxis, ...]
                return x + pos_encoding

        x = PositionalEncoding(64)(x)

        def transformer_block(x: tf.Tensor, ffn_dim: int) -> tf.Tensor:
            attn = MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.1)(x, x)
            attn = Dropout(0.1)(attn)
            x = LayerNormalization()(Add()([x, attn]))
            ffn = Dense(ffn_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
            ffn = Dropout(0.1)(ffn)
            ffn = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(1e-5))(ffn)
            x = LayerNormalization()(Add()([x, ffn]))
            return x

        short = transformer_block(x, 128)
        short = transformer_block(short, 128)
        short_pool = Lambda(lambda t: t[:, 0, :])(short)

        long = transformer_block(x, 256)
        long = transformer_block(long, 256)
        long_pool = Lambda(lambda t: t[:, 0, :])(long)

        merged = Concatenate()([short_pool, long_pool])
        x = Dense(128, activation="relu")(merged)
        x = Dropout(0.1)(x)
        x = Dense(128, activation="relu")(x)
        outputs = Dense(num_classes, activation="softmax")(x)

        optimizer = Adam(learning_rate=1e-3, clipnorm=0.5)
        model = Model(inputs, outputs)
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def create_model_by_name(
        self,
        name: str,
        input_shape: Tuple[int, int] | Tuple[Tuple[int, int], Tuple[int, int]],
        num_classes: int = 2,
    ) -> Optional[Model]:
        """根据名称创建模型，便于在训练时灵活选择"""

        if name == "tcn":
            return self.create_tcn_model(input_shape, num_classes)
        if name == "gru_dual":
            main_shape, aux_shape = input_shape  # type: ignore[misc]
            return self.create_dual_gru_model(main_shape, aux_shape, num_classes)
        if name == "tiny_transformer":
            return self.create_tiny_transformer_model(input_shape, num_classes)  # type: ignore[arg-type]
        if name == "lstm":
            return self.create_lstm_model(input_shape, num_classes)  # type: ignore[arg-type]
        if name == "gru":
            return self.create_gru_model(input_shape, num_classes)  # type: ignore[arg-type]
        if name == "cnn_lstm":
            return self.create_cnn_lstm_model(input_shape, num_classes)  # type: ignore[arg-type]
        raise ValueError(f"未知模型类型: {name}")
    
    def create_gru_model(self, input_shape: Tuple[int, int], num_classes: int = 2) -> Optional[Model]:
        """
        创建GRU模型
        
        Args:
            input_shape: 输入形状
            num_classes: 分类数量
            
        Returns:
            GRU模型
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow未安装，无法创建GRU模型")
            return None
        
        model = Sequential([
            GRU(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_cnn_lstm_model(self, input_shape: Tuple[int, int], num_classes: int = 2) -> Optional[Model]:
        """
        创建CNN-LSTM混合模型
        
        Args:
            input_shape: 输入形状
            num_classes: 分类数量
            
        Returns:
            CNN-LSTM模型
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow未安装，无法创建CNN-LSTM模型")
            return None
        
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_traditional_models(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Dict]:
        """
        训练传统机器学习模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            
        Returns:
            训练结果
        """
        X_train = self._prepare_features(X_train)
        X_val = self._prepare_features(X_val)
        y_train = self._prepare_targets(y_train)
        y_val = self._prepare_targets(y_val)

        if isinstance(X_train, (list, tuple)) or isinstance(X_val, (list, tuple)):
            raise ValueError("传统机器学习模型不支持多输入特征，请提供单输入特征数组。")

        # 将序列数据展平（如果是3D）
        if len(X_train.shape) == 3:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
        else:
            X_train_flat = X_train
            X_val_flat = X_val
        
        models = self.create_traditional_models()
        results = {}
        
        for name, model in models.items():
            print(f"训练 {name} 模型...")
            
            # 训练模型
            model.fit(X_train_flat, y_train)
            
            # 预测
            y_pred = model.predict(X_val_flat)
            
            # 计算指标
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, average='weighted')
            recall = recall_score(y_val, y_pred, average='weighted')
            f1 = f1_score(y_val, y_pred, average='weighted')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            print(f"{name} - 准确率: {accuracy:.4f}, F1分数: {f1:.4f}")
            
            # 保存模型
            model_path = self.model_save_dir / f"{name}_model.pkl"
            joblib.dump(model, model_path)
        
        self.models.update({name: result['model'] for name, result in results.items()})
        return results
    
    def train_deep_learning_model(self, model: Model, model_name: str,
                                X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                epochs: int = 100, batch_size: int = 32) -> Dict:
        """
        训练深度学习模型
        
        Args:
            model: 深度学习模型
            model_name: 模型名称
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批次大小
            
        Returns:
            训练结果
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow未安装，无法训练深度学习模型")
            return {}
        
        X_train = self._prepare_features(X_train)
        X_val = self._prepare_features(X_val)
        y_train = self._prepare_targets(y_train)
        y_val = self._prepare_targets(y_val)

        print(f"训练 {model_name} 模型...")
        
        # 回调函数
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5),
            ModelCheckpoint(
                filepath=str(self.model_save_dir / f"{model_name}_best.h5"),
                save_best_only=True
            )
        ]
        
        # 训练模型
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # 评估模型
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        
        # 预测
        y_pred_proba = model.predict(X_val)
        y_pred = np.argmax(y_pred_proba, axis=1) if y_pred_proba.shape[1] > 1 else (y_pred_proba > 0.5).astype(int).flatten()
        
        # 计算指标
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        results = {
            'model': model,
            'history': history.history,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f"{model_name} - 验证准确率: {val_accuracy:.4f}, F1分数: {f1:.4f}")
        
        # 保存模型
        model.save(self.model_save_dir / f"{model_name}_final.h5")
        
        self.models[model_name] = model
        self.training_history[model_name] = history.history

        return results

    def evaluate_model(
        self, model: Model, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """在测试集上评估深度学习模型"""

        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow未安装，无法评估模型")
            return {}

        X_test = self._prepare_features(X_test)
        y_test = self._prepare_targets(y_test)

        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        y_proba = model.predict(X_test)
        y_pred = (
            np.argmax(y_proba, axis=1)
            if y_proba.shape[1] > 1
            else (y_proba > 0.5).astype(int).flatten()
        )
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(
            f"测试集 - 损失: {loss:.4f}, 准确率: {accuracy:.4f}, F1: {f1:.4f}"
        )

        return {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
    
    def create_trading_environment(self, data: np.ndarray) -> TradingEnvironment:
        """
        创建交易环境（为强化学习准备）
        
        Args:
            data: 市场数据
            
        Returns:
            交易环境
        """
        return TradingEnvironment(data)
    
    def save_training_results(self, results: Dict, symbol: str, timestamp: str = None):
        """
        保存训练结果
        
        Args:
            results: 训练结果
            symbol: 交易对符号
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 准备保存的数据
        save_data = {}
        for name, result in results.items():
            save_data[name] = {
                'accuracy': float(result.get('accuracy', 0)),
                'precision': float(result.get('precision', 0)),
                'recall': float(result.get('recall', 0)),
                'f1_score': float(result.get('f1_score', 0)),
                'val_loss': float(result.get('val_loss', 0)) if 'val_loss' in result else None,
                'val_accuracy': float(result.get('val_accuracy', 0)) if 'val_accuracy' in result else None
            }
        
        # 保存结果
        results_file = self.model_save_dir / f"{symbol}_training_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"训练结果已保存到: {results_file}")
    
    def load_model(self, model_path: str, model_type: str = 'traditional'):
        """
        加载已训练的模型
        
        Args:
            model_path: 模型路径
            model_type: 模型类型 ('traditional' 或 'deep_learning')
            
        Returns:
            加载的模型
        """
        if model_type == 'traditional':
            return joblib.load(model_path)
        elif model_type == 'deep_learning' and TENSORFLOW_AVAILABLE:
            return tf.keras.models.load_model(model_path)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

if __name__ == "__main__":
    # 示例用法
    print("模型训练框架初始化完成")
    print(f"TensorFlow可用: {TENSORFLOW_AVAILABLE}")
    print(f"Gym可用: {GYM_AVAILABLE}")
    
    # 创建训练器
    trainer = ModelTrainer()
    
    # 示例：创建模型
    if TENSORFLOW_AVAILABLE:
        lstm_model = trainer.create_lstm_model((60, 10))  # 60个时间步，10个特征
        gru_model = trainer.create_gru_model((60, 10))
        # 演示新加入的模型创建方式
        tcn_model = trainer.create_model_by_name("tcn", (60, 10))
        tiny_transformer = trainer.create_model_by_name("tiny_transformer", (60, 10))
        print("深度学习模型创建完成")
    
    traditional_models = trainer.create_traditional_models()
    print(f"传统机器学习模型创建完成: {list(traditional_models.keys())}")