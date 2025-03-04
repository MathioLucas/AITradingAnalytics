
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import yfinance as yf
import datetime
import threading
import time
import queue
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from multiprocessing import Pool, cpu_count
import warnings
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import logging
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_platform.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AI-Trading-Platform")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class DataProcessor:
    """Handles data acquisition, preprocessing, and feature engineering"""
    
    def __init__(self, symbols, start_date, end_date, interval='1d', historical_window=365):
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.historical_window = historical_window
        self.data = {}
        self.scalers = {}
        self.logger = logging.getLogger(__name__)
        
    def fetch_data(self):
        """Fetch historical data for all symbols"""
        self.logger.info(f"Fetching historical data for {len(self.symbols)} symbols")
        
        # Calculate extended start date to allow for technical indicators calculation
        extended_start = (datetime.datetime.strptime(self.start_date, '%Y-%m-%d') - 
                          datetime.timedelta(days=self.historical_window)).strftime('%Y-%m-%d')
        
        with ThreadPoolExecutor(max_workers=min(10, len(self.symbols))) as executor:
            futures = {executor.submit(self._fetch_single_symbol, symbol, extended_start): symbol 
                      for symbol in self.symbols}
            
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    symbol = futures[future]
                    self.logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
        
        self.logger.info(f"Successfully fetched data for {len(self.data)} symbols")
        return self.data
    
    def _fetch_single_symbol(self, symbol, extended_start):
        """Fetch data for a single symbol"""
        try:
            df = yf.download(symbol, start=extended_start, end=self.end_date, interval=self.interval)
            if df.empty or len(df) < 30:
                self.logger.warning(f"Insufficient data for {symbol}, skipping")
                return
                
            df.index = pd.to_datetime(df.index)
            self.data[symbol] = df
            self.logger.info(f"Successfully fetched {len(df)} records for {symbol}")
        except Exception as e:
            self.logger.error(f"Error fetching {symbol}: {str(e)}")
            raise
    
    def engineer_features(self, add_technical_indicators=True, add_fundamental_data=False):
        """Add technical and fundamental analysis features to the datasets"""
        self.logger.info("Engineering features for all symbols")
        
        for symbol, df in self.data.items():
            # Make a copy to avoid SettingWithCopyWarning
            self.data[symbol] = self._engineer_features_single(df.copy(), symbol, 
                                                            add_technical_indicators,
                                                            add_fundamental_data)
        
        return self.data
    
    def _engineer_features_single(self, df, symbol, add_technical=True, add_fundamental=False):
        """Engineer features for a single symbol"""
        # Basic price and volume features
        df['Returns'] = df['Close'].pct_change()
        df['LogReturns'] = np.log(df['Close']/df['Close'].shift(1))
        df['Range'] = df['High'] - df['Low']
        df['VolumeDelta'] = df['Volume'].pct_change()
        
        if add_technical:
            # Moving averages
            for window in [5, 10, 20, 50, 100, 200]:
                df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
                df[f'MA_Diff_{window}'] = df['Close'] / df[f'MA_{window}'] - 1
            
            # Bollinger Bands
            for window in [20]:
                df[f'BB_Mid_{window}'] = df['Close'].rolling(window=window).mean()
                df[f'BB_Std_{window}'] = df['Close'].rolling(window=window).std()
                df[f'BB_Upper_{window}'] = df[f'BB_Mid_{window}'] + 2 * df[f'BB_Std_{window}']
                df[f'BB_Lower_{window}'] = df[f'BB_Mid_{window}'] - 2 * df[f'BB_Std_{window}']
                df[f'BB_Position_{window}'] = (df['Close'] - df[f'BB_Lower_{window}']) / (df[f'BB_Upper_{window}'] - df[f'BB_Lower_{window}'])
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            
            for window in [14, 28]:
                avg_gain = gain.rolling(window=window).mean()
                avg_loss = loss.rolling(window=window).mean()
                rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
                df[f'RSI_{window}'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # Volatility indicators
            for window in [21]:
                df[f'Volatility_{window}'] = df['LogReturns'].rolling(window=window).std() * np.sqrt(252)
            
            # Momentum
            for window in [14, 28]:
                df[f'Momentum_{window}'] = df['Close'] / df['Close'].shift(window) - 1
            
            # ADX (Average Directional Index)
            high_diff = df['High'] - df['High'].shift(1)
            low_diff = df['Low'].shift(1) - df['Low']
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            atr_period = 14
            tr = np.maximum(df['High'] - df['Low'], 
                  np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                             abs(df['Low'] - df['Close'].shift(1))))
            
            atr = pd.Series(tr).rolling(atr_period).mean()
            plus_di = 100 * pd.Series(plus_dm).rolling(atr_period).mean() / atr
            minus_di = 100 * pd.Series(minus_dm).rolling(atr_period).mean() / atr
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 0.001)
            df['ADX'] = pd.Series(dx).rolling(atr_period).mean()
        
        if add_fundamental:
            # If available, add fundamental data from yfinance
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                quarterly_financials = stock.quarterly_financials
                
                # Add some fundamental data if available
                df['MarketCap'] = info.get('marketCap', np.nan)
                df['PE_Ratio'] = info.get('trailingPE', np.nan)
                df['Dividend_Yield'] = info.get('dividendYield', np.nan)
                
                # Fill fundamental data across the dataframe
                for col in ['MarketCap', 'PE_Ratio', 'Dividend_Yield']:
                    df[col] = df[col].fillna(method='ffill')
            except Exception as e:
                self.logger.warning(f"Could not fetch fundamental data for {symbol}: {str(e)}")
        
        # Drop rows with NaN values from feature engineering
        df = df.dropna()
        return df
    
    def prepare_ml_data(self, symbol, target_col='Close', prediction_horizon=5, sequence_length=60, test_size=0.2):
        """Prepare data for machine learning - creates sequences for LSTM and scales features"""
        if symbol not in self.data:
            self.logger.error(f"Symbol {symbol} not found in processed data")
            return None, None, None, None
        
        df = self.data[symbol].copy()
        
        # Create future target (what we want to predict)
        df[f'Target_{prediction_horizon}d'] = df[target_col].shift(-prediction_horizon)
        
        # Drop rows with NaN in target
        df = df.dropna()
        
        # Get features and target
        features = df.drop([f'Target_{prediction_horizon}d', 'Date'] if 'Date' in df.columns else [f'Target_{prediction_horizon}d'], axis=1)
        target = df[f'Target_{prediction_horizon}d']
        
        # Normalize features
        scaler = MinMaxScaler()
        feature_names = features.columns
        scaled_features = scaler.fit_transform(features)
        self.scalers[symbol] = {'feature_scaler': scaler}
        
        # Create target scaler separately to be able to inverse transform predictions
        target_scaler = MinMaxScaler()
        scaled_target = target_scaler.fit_transform(target.values.reshape(-1, 1))
        self.scalers[symbol]['target_scaler'] = target_scaler
        
        # Create sequences for LSTM
        X, y = self._create_sequences(scaled_features, scaled_target, sequence_length)
        
        # Split into train and test sets
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, X_test, y_train, y_test, feature_names
    
    def _create_sequences(self, features, target, sequence_length):
        """Create sequences for LSTM model"""
        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features[i:i+sequence_length])
            y.append(target[i+sequence_length])
        return np.array(X), np.array(y)

class ModelBuilder:
    """Builds, trains and optimizes machine learning models"""
    
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.models = {}
        self.histories = {}
        self.logger = logging.getLogger(__name__)
        
    def build_lstm_model(self, input_shape, layers=[100, 50], dropout_rate=0.2):
        """Build an LSTM model with the given architecture"""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(layers[0], return_sequences=len(layers) > 1, 
                       input_shape=input_shape, 
                       activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # Additional LSTM layers
        for i in range(1, len(layers)):
            model.add(LSTM(layers[i], return_sequences=(i < len(layers) - 1), 
                           activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile the model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def train_model(self, symbol, prediction_horizon=5, sequence_length=60, epochs=50, batch_size=32):
        """Train a model for the given symbol"""
        self.logger.info(f"Training model for {symbol} with prediction horizon {prediction_horizon} days")
        
        # Prepare data
        X_train, X_test, y_train, y_test, feature_names = self.data_processor.prepare_ml_data(
            symbol, prediction_horizon=prediction_horizon, sequence_length=sequence_length
        )
        
        if X_train is None:
            self.logger.error(f"Failed to prepare data for {symbol}")
            return None
        
        # Build model
        model = self.build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Callbacks for early stopping and model checkpointing
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Create directory for model checkpoints if it doesn't exist
        os.makedirs('models', exist_ok=True)
        mc = ModelCheckpoint(f'models/{symbol}_lstm_h{prediction_horizon}_best.h5', 
                             monitor='val_loss', save_best_only=True)
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es, mc],
            verbose=1
        )
        
        # Store model and history
        self.models[symbol] = model
        self.histories[symbol] = history.history
        
        # Calculate metrics
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        # Inverse transform predictions back to original scale
        scaler = self.data_processor.scalers[symbol]['target_scaler']
        train_predictions = scaler.inverse_transform(train_predictions)
        test_predictions = scaler.inverse_transform(test_predictions)
        y_train_orig = scaler.inverse_transform(y_train)
        y_test_orig = scaler.inverse_transform(y_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train_orig, train_predictions)
        test_mse = mean_squared_error(y_test_orig, test_predictions)
        train_mae = mean_absolute_error(y_train_orig, train_predictions)
        test_mae = mean_absolute_error(y_test_orig, test_predictions)
        
        self.logger.info(f"Training completed for {symbol}:")
        self.logger.info(f"Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
        self.logger.info(f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
        
        return model, history
    
    def optimize_hyperparameters(self, symbol, n_trials=50, prediction_horizon=5, sequence_length=60):
        """Use Optuna to optimize hyperparameters for the model"""
        self.logger.info(f"Optimizing hyperparameters for {symbol} model")
        
        # Prepare data
        X_train, X_test, y_train, y_test, _ = self.data_processor.prepare_ml_data(
            symbol, prediction_horizon=prediction_horizon, sequence_length=sequence_length
        )
        
        if X_train is None:
            self.logger.error(f"Failed to prepare data for {symbol}")
            return None
        
        def objective(trial):
            # Define hyperparameters to optimize
            layers = [
                trial.suggest_int('lstm_units_1', 32, 256),
                trial.suggest_int('lstm_units_2', 16, 128)
            ]
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            
            # Build model
            model = Sequential()
            model.add(LSTM(layers[0], return_sequences=True, 
                           input_shape=(X_train.shape[1], X_train.shape[2]), 
                           activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            model.add(LSTM(layers[1], activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            model.add(Dense(1))
            
            # Compile the model
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            # Callbacks for early stopping
            es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            # Train the model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=50,  # We'll use early stopping
                batch_size=batch_size,
                callbacks=[es],
                verbose=0
            )
            
            # Return validation loss
            return history.history['val_loss'][-1]
        
        # Create Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        self.logger.info(f"Best hyperparameters for {symbol}: {best_params}")
        
        # Train model with best parameters
        layers = [best_params['lstm_units_1'], best_params['lstm_units_2']]
        dropout_rate = best_params['dropout_rate']
        learning_rate = best_params['learning_rate']
        batch_size = best_params['batch_size']
        
        # Build optimized model
        model = Sequential()
        model.add(LSTM(layers[0], return_sequences=True, 
                       input_shape=(X_train.shape[1], X_train.shape[2]), 
                       activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(LSTM(layers[1], activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(1))
        
        # Compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        # Callbacks
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        mc = ModelCheckpoint(f'models/{symbol}_lstm_optimized_h{prediction_horizon}.h5', 
                             monitor='val_loss', save_best_only=True)
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,  # We'll use early stopping
            batch_size=batch_size,
            callbacks=[es, mc],
            verbose=1
        )
        
        # Store optimized model and history
        self.models[f"{symbol}_optimized"] = model
        self.histories[f"{symbol}_optimized"] = history.history
        
        return model, history, best_params

class TradingStrategies:
    """Implements various trading strategies based on technical analysis and ML predictions"""
    
    def __init__(self, data_processor, model_builder):
        self.data_processor = data_processor
        self.model_builder = model_builder
        self.logger = logging.getLogger(__name__)
        
    def moving_average_crossover(self, symbol, short_window=20, long_window=50):
        """Implement a simple moving average crossover strategy"""
        if symbol not in self.data_processor.data:
            self.logger.error(f"Symbol {symbol} not found in processed data")
            return None
        
        df = self.data_processor.data[symbol].copy()
        
        # Calculate short and long moving averages
        df['ShortMA'] = df['Close'].rolling(window=short_window).mean()
        df['LongMA'] = df['Close'].rolling(window=long_window).mean()
        
        # Generate signals
        df['Signal'] = 0
        df.loc[df['ShortMA'] > df['LongMA'], 'Signal'] = 1  # Buy signal
        df.loc[df['ShortMA'] < df['LongMA'], 'Signal'] = -1  # Sell signal
        
        # Generate actual positions (1 = long, 0 = neutral, -1 = short)
        df['Position'] = df['Signal'].diff()
        
        return df
    
    def rsi_strategy(self, symbol, window=14, overbought=70, oversold=30):
        """Implement an RSI-based trading strategy"""
        if symbol not in self.data_processor.data:
            self.logger.error(f"Symbol {symbol} not found in processed data")
            return None
        
        df = self.data_processor.data[symbol].copy()
        
        # Calculate RSI if not already calculated
        if f'RSI_{window}' not in df.columns:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            rs = avg_gain / avg_loss.replace(0, 0.001)
            df[f'RSI_{window}'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        df['Signal'] = 0
        df.loc[df[f'RSI_{window}'] < oversold, 'Signal'] = 1  # Buy signal when RSI is oversold
        df.loc[df[f'RSI_{window}'] > overbought, 'Signal'] = -1  # Sell signal when RSI is overbought
        
        # Generate positions
        df['Position'] = df['Signal']
        
        return df
    
    def ml_based_strategy(self, symbol, prediction_horizon=5, threshold=0.01):
        """Implement a trading strategy based on ML predictions"""
        if symbol not in self.model_builder.models:
            self.logger.error(f"No trained model found for {symbol}")
            return None
        
        df = self.data_processor.data[symbol].copy()
        
        # Get the model and prepare the data
        model = self.model_builder.models[symbol]
        
        # We need to create sequences for prediction
        sequence_length = model.input_shape[1]
        
        # Scale the features
        feature_scaler = self.data_processor.scalers[symbol]['feature_scaler']
        
        # Drop target column if exists
        features = df.drop([f'Target_{prediction_horizon}d'] if f'Target_{prediction_horizon}d' in df.columns else [], axis=1)
        
        # Scale features
        scaled_features = feature_scaler.transform(features)
        
        # Create sequences
        X = []
        for i in range(len(scaled_features) - sequence_length + 1):
            X.append(scaled_features[i:i+sequence_length])
        X = np.array(X)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Inverse transform predictions
        target_scaler = self.data_processor.scalers[symbol]['target_scaler']
        predictions = target_scaler.inverse_transform(predictions)
        
        # Create a dataframe with predictions
        pred_df = pd.DataFrame(index=df.index[sequence_length-1:], data={'Prediction': predictions.flatten()})
        
        # Calculate predicted returns
        pred_df['PredictedReturn'] = pred_df['Prediction'] / df['Close'].iloc[sequence_length-1:].values - 1
        
        # Generate signals
        pred_df['Signal'] = 0
        pred_df.loc[pred_df['PredictedReturn'] > threshold, 'Signal'] = 1  # Buy signal
        pred_df.loc[pred_df['PredictedReturn'] < -threshold, 'Signal'] = -1  # Sell signal
        
        # Merge predictions with original data
        result = pd.concat([df.iloc[sequence_length-1:], pred_df], axis=1)
        
        return result
    
    def ensemble_strategy(self, symbol, strategies=['ma_crossover', 'rsi', 'ml'], weights=[0.3, 0.3, 0.4]):
        """Combine multiple strategies with specified weights"""
        if symbol not in self.data_processor.data:
            self.logger.error(f"Symbol {symbol} not found in processed data")
            return None
        
        results = {}
        
        # Get results from each strategy
        if 'ma_crossover' in strategies:
            results['ma_crossover'] = self.moving_average_crossover(symbol)
        
        if 'rsi' in strategies:
            results['rsi'] = self.rsi_strategy(symbol)
        
        if 'ml' in strategies and symbol in self.model_builder.models:
            results['ml'] = self.ml_based_strategy(symbol)
        
        # Ensure all strategies have results
        if len(results) < len(strategies):
            missing = set(strategies) - set(results.keys())
            self.logger.warning(f"Missing results for strategies: {missing}")
            # Adjust weights for missing strategies
            available_strategies = list(results.keys())
            available_weights = [weights[strategies.index(s)] for s in available_strategies]
            total_weight = sum(available_weights)
            weights = [w/total_weight for w in available_weights]
            strategies = available_strategies
        
        # Initialize ensemble result dataframe
        base_df = self.data_processor.data[symbol].copy()
        ensemble_result = base_df.copy()
        
        # Combine signals using weighted approach
        ensemble_result['EnsembleSignal'] = 0
        
        for strat, weight in zip(strategies, weights):
            if strat in results and 'Signal' in results[strat].columns:
                # Align indices and fill NaN with 0
                signal_series = results[strat]['Signal'].reindex(ensemble_result.index).fillna(0)
                ensemble_result['EnsembleSignal'] += signal_series * weight
        
        # Threshold for final decision
        ensemble_result['Signal'] = 0
        ensemble_result.loc[ensemble_result['EnsembleSignal'] > 0.2, 'Signal'] = 1  # Strong buy
        ensemble_result.loc[ensemble_result['EnsembleSignal'] < -0.2, 'Signal'] = -1  # Strong sell
        
        # Calculate positions and returns
        ensemble_result['Position'] = ensemble_result['Signal'].diff()
        
        return ensemble_result
    
    def backtest_strategy(self, symbol, strategy_df, initial_capital=10000, position_size=1.0):
        """Backtest a trading strategy and calculate performance metrics"""
        if 'Signal' not in strategy_df.columns:
            self.logger.error("Strategy dataframe must contain 'Signal' column")
            return None
        
        # Copy the strategy dataframe
        backtest = strategy_df.copy()
        
        # Calculate returns
        backtest['Returns'] = backtest['Close'].pct_change()
        
        # Calculate strategy returns (position at previous day * today's return)
        backtest['StrategyReturns'] = backtest['Signal'].shift(1) * backtest['Returns']
        
        # Calculate cumulative returns
        backtest['CumulativeReturns'] = (1 + backtest['Returns']).cumprod()
        backtest['CumulativeStrategyReturns'] = (1 + backtest['StrategyReturns']).cumprod()
        
        # Calculate portfolio value
        backtest['PortfolioValue'] = initial_capital * backtest['CumulativeStrategyReturns']
        
        # Calculate drawdowns
        backtest['PeakValue'] = backtest['PortfolioValue'].cummax()
        backtest['Drawdown'] = (backtest['PortfolioValue'] - backtest['PeakValue']) / backtest['PeakValue']
        
        # Calculate performance metrics
        total_return = backtest['CumulativeStrategyReturns'].iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(backtest)) - 1
        annualized_volatility = backtest['StrategyReturns'].std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        max_drawdown = backtest['Drawdown'].min()
        
        # Count trades
        trades = backtest['Signal'].diff().fillna(0)
        num_trades = (trades != 0).sum()
        
        # Calculate win rate
        winning_trades = ((backtest['StrategyReturns'] > 0) & (backtest['Signal'].shift(1) != 0)).sum()
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        
        metrics = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Number of Trades': num_trades,
            'Win Rate': win_rate
        }
        
        self.logger.info(f"Backtest results for {symbol}:")
        for key, value in metrics.items
