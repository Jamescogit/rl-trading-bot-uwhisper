import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import urllib.parse
import uuid
import time
from collections import deque
import hashlib

# Auto-refresh setup
try:
    from streamlit_autorefresh import st_autorefresh
    refresh_counter = st_autorefresh(interval=10_000, limit=None, key="auto_refresh")
except ImportError:
    refresh_counter = 0

# Page configuration
st.set_page_config(
    page_title="RL Trading Bot - uWhisper",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Authentication System
def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["app_password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h1>🤖 RL Trading Bot</h1>
            <h3>uWhisper.com - Private Access</h3>
            <p>Please enter your access password:</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.text_input(
            "Password", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h1>🤖 RL Trading Bot</h1>
            <h3>uWhisper.com - Private Access</h3>
            <p style="color: red;">❌ Password incorrect. Please try again:</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.text_input(
            "Password", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        return False
    else:
        # Password correct.
        return True

# Only run the app if password is correct
if check_password():
    
    # Configuration
    SYMBOLS = {
        "EURUSD": {"name": "EUR/USD", "type": "forex", "description": "Euro vs US Dollar"},
        "EURJPY": {"name": "EUR/JPY", "type": "forex", "description": "Euro vs Japanese Yen"},
        "USDJPY": {"name": "USD/JPY", "type": "forex", "description": "US Dollar vs Japanese Yen"},
        "XAUUSD": {"name": "XAU/USD", "type": "commodity", "description": "Gold vs US Dollar"},
        "NAS": {"name": "NASDAQ", "type": "index", "description": "NASDAQ 100 Index"}
    }

    # Custom CSS for Soft UI Dashboard styling
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            max-width: 100%;
        }
        
        .stApp {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(310deg, #f8f9fa 0%, #dee2e6 100%);
        }
        
        /* Header Card */
        .header-card {
            background: linear-gradient(310deg, #2152ff 0%, #21d4fd 100%);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 20px 27px 0 rgba(0,0,0,.05);
            border: none;
        }
        
        .header-title {
            color: white;
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .header-subtitle {
            color: rgba(255, 255, 255, 0.8);
            font-size: 1.1rem;
            margin: 0.5rem 0 0 0;
            font-weight: 400;
        }
        
        /* Metric Cards */
        .metric-card {
            background: white;
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 20px 27px 0 rgba(0,0,0,.05);
            border: none;
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 25px 35px 0 rgba(0,0,0,.1);
        }
        
        .metric-title {
            font-size: 0.875rem;
            font-weight: 600;
            color: #67748e;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #344767;
            margin-bottom: 0.25rem;
        }
        
        .metric-change {
            font-size: 0.875rem;
            font-weight: 600;
        }
        
        .metric-change.positive {
            color: #4caf50;
        }
        
        .metric-change.negative {
            color: #f44336;
        }
        
        /* Signal Card */
        .signal-card {
            background: white;
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 20px 27px 0 rgba(0,0,0,.05);
            border: none;
            position: relative;
            overflow: hidden;
        }
        
        .signal-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #2152ff 0%, #21d4fd 100%);
        }
        
        .signal-buy {
            border-left: 5px solid #4caf50;
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.05) 0%, rgba(76, 175, 80, 0.1) 100%);
        }
        
        .signal-sell {
            border-left: 5px solid #f44336;
            background: linear-gradient(135deg, rgba(244, 67, 54, 0.05) 0%, rgba(244, 67, 54, 0.1) 100%);
        }
        
        .signal-hold {
            border-left: 5px solid #ff9800;
            background: linear-gradient(135deg, rgba(255, 152, 0, 0.05) 0%, rgba(255, 152, 0, 0.1) 100%);
        }
        
        .signal-title {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .signal-buy .signal-title {
            color: #4caf50;
        }
        
        .signal-sell .signal-title {
            color: #f44336;
        }
        
        .signal-hold .signal-title {
            color: #ff9800;
        }
        
        /* Data Tables */
        .data-card {
            background: white;
            border-radius: 20px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 20px 27px 0 rgba(0,0,0,.05);
            border: none;
        }
        
        .data-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #344767;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        /* Chat Card */
        .chat-card {
            background: white;
            border-radius: 20px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 20px 27px 0 rgba(0,0,0,.05);
            border: none;
            border-left: 5px solid #2152ff;
        }
        
        .bot-response {
            background: linear-gradient(135deg, rgba(33, 82, 255, 0.05) 0%, rgba(33, 212, 253, 0.05) 100%);
            border-radius: 12px;
            padding: 1rem;
            margin-top: 0.5rem;
            border-left: 3px solid #2152ff;
        }
        
        /* Performance Card */
        .performance-card {
            background: white;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 20px 27px 0 rgba(0,0,0,.05);
            border: none;
            height: fit-content;
        }
        
        .performance-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #344767;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .performance-metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 0;
            border-bottom: 1px solid #f0f2f5;
        }
        
        .performance-metric:last-child {
            border-bottom: none;
        }
        
        .performance-label {
            font-size: 0.875rem;
            font-weight: 500;
            color: #67748e;
        }
        
        .performance-value {
            font-size: 0.875rem;
            font-weight: 700;
            color: #344767;
        }
        
        .performance-positive {
            color: #4caf50 !important;
        }
        
        .performance-negative {
            color: #f44336 !important;
        }
        
        /* Button Styles */
        .stButton > button {
            background: linear-gradient(310deg, #2152ff 0%, #21d4fd 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(33, 82, 255, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(33, 82, 255, 0.4);
        }
        
        /* Sidebar Styles */
        .css-1d391kg {
            background: white;
            border-radius: 20px;
            margin: 1rem;
            padding: 1rem;
            box-shadow: 0 20px 27px 0 rgba(0,0,0,.05);
        }
        
        .sidebar-section {
            background: linear-gradient(135deg, rgba(33, 82, 255, 0.05) 0%, rgba(33, 212, 253, 0.05) 100%);
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
            border: 1px solid rgba(33, 82, 255, 0.1);
        }
        
        /* Alert Styles */
        .alert-success {
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(76, 175, 80, 0.05) 100%);
            border: 1px solid rgba(76, 175, 80, 0.2);
            border-radius: 12px;
            padding: 1rem;
            color: #2e7d32;
            font-weight: 500;
        }
        
        .alert-warning {
            background: linear-gradient(135deg, rgba(255, 152, 0, 0.1) 0%, rgba(255, 152, 0, 0.05) 100%);
            border: 1px solid rgba(255, 152, 0, 0.2);
            border-radius: 12px;
            padding: 1rem;
            color: #f57c00;
            font-weight: 500;
        }
        
        /* Status Badges */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .status-success {
            background: rgba(76, 175, 80, 0.1);
            color: #2e7d32;
        }
        
        .status-warning {
            background: rgba(255, 152, 0, 0.1);
            color: #f57c00;
        }
        
        .status-info {
            background: rgba(33, 150, 243, 0.1);
            color: #1565c0;
        }
        
        /* Private access indicator */
        .private-badge {
            position: fixed;
            top: 10px;
            right: 10px;
            background: rgba(244, 67, 54, 0.9);
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 20px;
            font-size: 0.7rem;
            font-weight: 600;
            z-index: 1000;
        }
    </style>
    """, unsafe_allow_html=True)

    # Private access indicator
    st.markdown('<div class="private-badge">🔒 PRIVATE ACCESS</div>', unsafe_allow_html=True)

    # Initialize session state for logs and signals
    if "signal_log" not in st.session_state:
        st.session_state.signal_log = []
    if "last_signal" not in st.session_state:
        st.session_state.last_signal = "HOLD"

    # Technical Indicators Helper Functions
    def add_indicators(df):
        """Add technical indicators to the dataframe"""
        if df.empty or len(df) < 50:
            return df
            
        # EMAs
        df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema55"] = df["close"].ewm(span=55, adjust=False).mean()
        df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
        
        # RSI
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        df["rsi14"] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        return df.fillna(method="bfill").fillna(method="ffill")

    # Trading Environment
    class TradingEnv:
        def __init__(self, df, window_size=10):
            self.df = df.reset_index(drop=True)
            self.window_size = window_size
            self.reset()
            
        def reset(self):
            self.current_step = self.window_size
            self.position = 0
            self.entry_price = None
            self.total_profit = 0
            self.total_trades = 0
            self.wins = 0
            self.done = False
            return self._get_state()
            
        def _get_state(self):
            if self.current_step < self.window_size:
                return np.zeros(12)
                
            window = self.df.iloc[self.current_step-self.window_size:self.current_step]
            close_prices = window["close"].values
            high_prices = window["high"].values
            low_prices = window["low"].values
            volumes = window["volume"].values
            
            # Calculate features
            price_changes = np.diff(close_prices[-5:]) / close_prices[-5] if len(close_prices) >= 5 else [0]
            volatility = np.std(close_prices[-5:]) / np.mean(close_prices[-5:]) if len(close_prices) >= 5 else 0
            volume_ratio = volumes[-1] / np.mean(volumes) if np.mean(volumes) > 0 else 1
            
            sma5 = np.mean(close_prices[-5:])
            sma10 = np.mean(close_prices)
            price_position = (close_prices[-1] - sma5) / sma5 if sma5 > 0 else 0
            momentum = (close_prices[-1] - close_prices[-3]) / close_prices[-3] if len(close_prices) >= 3 else 0
            
            support = np.min(low_prices)
            resistance = np.max(high_prices)
            support_distance = (close_prices[-1] - support) / close_prices[-1] if close_prices[-1] > 0 else 0
            resistance_distance = (resistance - close_prices[-1]) / close_prices[-1] if close_prices[-1] > 0 else 0
            
            up_movement = 1 if close_prices[-1] > close_prices[-2] else 0
            
            # Technical indicators
            rsi = window["rsi14"].iloc[-1] if "rsi14" in window.columns else 50
            macd_hist = window["macd_hist"].iloc[-1] if "macd_hist" in window.columns else 0
            
            state = np.array([
                price_changes[-1] if len(price_changes) > 0 else 0,
                volatility, volume_ratio, price_position, momentum,
                support_distance, resistance_distance, up_movement,
                (sma5 - sma10) / sma10 if sma10 > 0 else 0,
                (high_prices[-1] - low_prices[-1]) / close_prices[-1] if close_prices[-1] > 0 else 0,
                rsi / 100.0, macd_hist
            ])
            
            return np.nan_to_num(state)
            
        def step(self, action):
            current_price = self.df.at[self.current_step, "close"]
            reward = 0
            
            # Execute action
            if action == 1 and self.position != 1:  # Buy
                self.position = 1
                self.entry_price = current_price
                reward -= 0.02  # Transaction cost
            elif action == 2 and self.position != -1:  # Sell
                self.position = -1
                self.entry_price = current_price
                reward -= 0.02  # Transaction cost
                
            # Calculate profit/loss
            if self.position != 0 and self.entry_price is not None:
                price_diff = (current_price - self.entry_price) * self.position
                reward += price_diff * 1000 - 0.001  # Scaled reward
                
                # Exit conditions
                if abs(price_diff) >= 0.005 or self.current_step % 30 == 0:
                    profit_percent = price_diff / self.entry_price * 100 - 0.02
                    self.total_profit += profit_percent
                    self.total_trades += 1
                    
                    if profit_percent > 0:
                        self.wins += 1
                        reward += 0.1
                    else:
                        reward -= 0.1
                        
                    self.position = 0
                    self.entry_price = None
                    
            self.current_step += 1
            if self.current_step >= len(self.df) - 1:
                self.done = True
                
            return self._get_state(), reward, self.done, False, {
                "profit": self.total_profit,
                "trades": self.total_trades,
                "win_rate": self.wins / self.total_trades if self.total_trades > 0 else 0
            }

    # Enhanced RL Scalping Bot
    class ScalpingRLBot:
        def __init__(self):
            self.state_size = 12
            self.action_size = 3  # Hold, Buy, Sell
            self.epsilon = 1.0
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            self.gamma = 0.95
            self.learning_rate = 0.001
            self._init_model()
            self.position = 0
            self.entry_price = None
            self.trades_history = []
            
        def _init_model(self):
            """Initialize neural network weights"""
            self.model = {
                'layer1': np.random.randn(self.state_size, 24) * 0.1,
                'bias1': np.zeros((1, 24)),
                'layer2': np.random.randn(24, 24) * 0.1,
                'bias2': np.zeros((1, 24)),
                'layer3': np.random.randn(24, self.action_size) * 0.1,
                'bias3': np.zeros((1, self.action_size))
            }
            
        def predict(self, state):
            """Forward pass through neural network"""
            z1 = state.reshape(1, -1) @ self.model['layer1'] + self.model['bias1']
            a1 = np.tanh(z1)
            z2 = a1 @ self.model['layer2'] + self.model['bias2']
            a2 = np.tanh(z2)
            z3 = a2 @ self.model['layer3'] + self.model['bias3']
            return z3.flatten(), (z1, a1, z2, a2)
            
        def get_signals(self, df):
            """Generate trading signals"""
            # 30% exploration for better learning
            if np.random.rand() < 0.3:
                action = "BUY" if np.random.rand() < 0.5 else "SELL"
                confidence = np.random.uniform(0.4, 0.7)
                return action, confidence
                
            # RL-based decision
            env = TradingEnv(df)
            state = env._get_state()
            q_values, _ = self.predict(state)
            
            if np.random.rand() < self.epsilon:
                action_idx = np.random.choice(self.action_size)
                confidence = 0.3
            else:
                action_idx = np.argmax(q_values)
                confidence = min(0.9, abs(q_values[action_idx]) / (np.sum(np.abs(q_values)) + 1e-8))
                
            action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
            action = action_map[action_idx]
            
            return action, confidence
            
        def simulate_trade(self, df, action, confidence):
            """Simulate trade execution"""
            current_price = df["close"].iloc[-1]
            current_time = datetime.now()
            
            if self.position == 0 and action in ("BUY", "SELL"):
                self.position = 1 if action == "BUY" else -1
                self.entry_price = current_price
                self.trades_history.append({
                    "entry_time": current_time,
                    "position": action,
                    "entry_price": current_price,
                    "confidence": confidence
                })
            elif self.position != 0:
                # Exit trade
                last_trade = self.trades_history[-1]
                profit_pips = (current_price - self.entry_price) * self.position
                profit_percent = profit_pips / self.entry_price * 100
                
                last_trade.update({
                    "exit_time": current_time,
                    "exit_price": current_price,
                    "profit_pips": profit_pips,
                    "profit_percent": profit_percent,
                    "exit_reason": "auto"
                })
                
                self.position = 0
                self.entry_price = None
                
        def train_one_episode(self, df):
            """Train the RL model on one episode"""
            env = TradingEnv(df)
            state = env.reset()
            total_loss = 0.0
            
            while not env.done:
                q_values, cache = self.predict(state)
                z1, a1, z2, a2 = cache
                
                # Choose action
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(self.action_size)
                else:
                    action = np.argmax(q_values)
                    
                next_state, reward, done, _, info = env.step(action)
                
                # Q-learning update
                q_next, _ = self.predict(next_state)
                target = reward + self.gamma * np.max(q_next) * (0 if done else 1)
                error = q_values[action] - target
                total_loss += error ** 2
                
                # Backpropagation
                grad_z3 = np.zeros(self.action_size)
                grad_z3[action] = error
                
                # Update layer 3
                self.model['layer3'][:, action] -= self.learning_rate * a2.flatten() * grad_z3[action]
                self.model['bias3'][0, action] -= self.learning_rate * grad_z3[action]
                
                # Update layer 2
                delta2 = (grad_z3 @ self.model['layer3'].T) * (1 - a2**2)
                self.model['layer2'] -= self.learning_rate * np.outer(a1.flatten(), delta2.flatten())
                self.model['bias2'] -= self.learning_rate * delta2.flatten()
                
                # Update layer 1
                delta1 = (delta2 @ self.model['layer2'].T) * (1 - a1**2)
                self.model['layer1'] -= self.learning_rate * np.outer(state, delta1.flatten())
                self.model['bias1'] -= self.learning_rate * delta1.flatten()
                
                state = next_state
                
            # Update epsilon
            if info["win_rate"] > 0.5:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            else:
                self.epsilon = min(0.5, self.epsilon * 1.01)
                
            return info, total_loss
            
        def get_performance_stats(self):
            """Get bot performance statistics"""
            completed_trades = [t for t in self.trades_history if "profit_percent" in t]
            
            if not completed_trades:
                return {
                    "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                    "avg_confidence": 0, "total_profit": 0, "avg_profit": 0
                }
                
            df_trades = pd.DataFrame(completed_trades)
            wins = (df_trades["profit_percent"] > 0).sum()
            total = len(df_trades)
            
            return {
                "total_trades": total,
                "wins": wins,
                "losses": total - wins,
                "win_rate": wins / total,
                "avg_confidence": df_trades["confidence"].mean(),
                "total_profit": df_trades["profit_percent"].sum(),
                "avg_profit": df_trades["profit_percent"].mean()
            }

    # Initialize RL Bot
    @st.cache_resource
    def get_rl_bot():
        return ScalpingRLBot()

    # Data fetching functions
    def get_api_endpoint(symbol):
        symbol_info = SYMBOLS.get(symbol, {})
        asset_type = symbol_info.get("type", "forex")
        return {
            "forex": "https://quote.tradeswitcher.com/quote-b-api",
            "commodity": "https://quote.tradeswitcher.com/quote-b-api", 
            "index": "https://quote.tradeswitcher.com/quote-b-api"
        }[asset_type]

    def fetch_real_kline_data(symbol, kline_type=1, num_candles=100):
        try:
            api_base = get_api_endpoint(symbol)
            trace_id = f"python_kline_{uuid.uuid4().hex[:8]}"
            
            query_data = {
                "trace": trace_id,
                "data": {
                    "code": symbol,
                    "kline_type": kline_type,
                    "kline_timestamp_end": 0,
                    "query_kline_num": num_candles,
                    "adjust_type": 0
                }
            }
            
            query_string = urllib.parse.quote(json.dumps(query_data))
            api_url = f"{api_base}/kline?token={st.secrets['ALLTICK_API_KEY']}&query={query_string}"
            
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'AllTick-Python-Client/1.0'
            }
            
            response = requests.get(api_url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("ret") == 200 and "data" in data:
                    kline_list = data["data"].get("kline_list", [])
                    
                    if kline_list:
                        df_data = []
                        for kline in kline_list:
                            df_data.append({
                                'time': pd.to_datetime(int(kline['timestamp']), unit='s'),
                                'open': float(kline['open_price']),
                                'high': float(kline['high_price']),
                                'low': float(kline['low_price']),
                                'close': float(kline['close_price']),
                                'volume': float(kline.get('volume', 0))
                            })
                        
                        df = pd.DataFrame(df_data)
                        df = df.sort_values('time').reset_index(drop=True)
                        return add_indicators(df)
                    else:
                        return None
                else:
                    return None
            else:
                return None
                
        except Exception as e:
            return None

    def format_price(price, symbol):
        if symbol in ["EURUSD", "EURJPY", "USDJPY"]:
            return f"{price:.5f}"
        elif symbol == "XAUUSD":
            return f"${price:.2f}"
        elif symbol == "NAS":
            return f"{price:.1f}"
        else:
            return f"{price:.4f}"

    def create_mock_data(symbol, num_points=100):
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=num_points/60)
        timestamps = pd.date_range(start=start_time, end=end_time, periods=num_points)
        
        symbol_config = {
            "EURUSD": {"base": 1.0850, "volatility": 0.005},
            "EURJPY": {"base": 158.50, "volatility": 0.008},
            "USDJPY": {"base": 146.20, "volatility": 0.007},
            "XAUUSD": {"base": 2045.50, "volatility": 0.015},
            "NAS": {"base": 16800.0, "volatility": 0.012}
        }
        
        config = symbol_config.get(symbol, {"base": 1.0000, "volatility": 0.01})
        base_price = config["base"]
        volatility = config["volatility"]
        
        returns = np.random.normal(0, volatility, num_points)
        prices = [base_price]
        
        for ret in returns:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 0.01))
        
        closes = prices[1:]
        opens = prices[:-1]
        
        highs = []
        lows = []
        volumes = []
        
        for i in range(num_points):
            high = max(opens[i], closes[i]) * np.random.uniform(1.0, 1.005)
            low = min(opens[i], closes[i]) * np.random.uniform(0.995, 1.0)
            
            if symbol in ["EURUSD", "EURJPY", "USDJPY"]:
                volume = np.random.randint(50000, 500000)
            elif symbol == "XAUUSD":
                volume = np.random.randint(10000, 100000)
            else:
                volume = np.random.randint(1000000, 10000000)
            
            highs.append(high)
            lows.append(low)
            volumes.append(volume)
        
        df = pd.DataFrame({
            'time': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        return add_indicators(df)

    @st.cache_data(ttl=10)  # Cache for 10 seconds
    def get_market_data(symbol, timeframe, num_candles, use_real_data, refresh_count):
        timeframe_mapping = {
            "1 Min": 1, "5 Min": 2, "15 Min": 3,
            "30 Min": 4, "1 Hr": 5, "Daily": 8
        }
        
        if use_real_data and "ALLTICK_API_KEY" in st.secrets:
            kline_type = timeframe_mapping.get(timeframe, 3)
            df = fetch_real_kline_data(symbol, kline_type, num_candles)
            if df is None:
                df = create_mock_data(symbol, num_candles)
            return df
        else:
            return create_mock_data(symbol, num_candles)

    # Initialize RL Bot
    rl_bot = get_rl_bot()

    # Header
    st.markdown("""
    <div class="header-card">
        <h1 class="header-title">🤖 RL Scalping Trading Bot</h1>
        <p class="header-subtitle">uWhisper.com • Advanced Reinforcement Learning • Real-time Market Data • Auto-refresh 10s</p>
    </div>
    """, unsafe_allow_html=True)

    # Check API key with refresh indicator
    refresh_time = datetime.now().strftime("%H:%M:%S")
    if "ALLTICK_API_KEY" in st.secrets:
        st.markdown(f'<div class="alert-success">✅ API Key found! Real data mode available. Last refresh: {refresh_time}</div>', unsafe_allow_html=True)
        api_available = True
    else:
        st.markdown(f'<div class="alert-warning">⚠️ AllTick API key not found! Using demo mode only. Last refresh: {refresh_time}</div>', unsafe_allow_html=True)
        api_available = False

    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Trading Controls")
        
        # Symbol selection
        symbol_options = list(SYMBOLS.keys())
        symbol_labels = [f"{SYMBOLS[s]['name']} ({SYMBOLS[s]['description']})" for s in symbol_options]

        selected_index = st.selectbox(
            "📈 Select Trading Instrument",
            range(len(symbol_options)),
            format_func=lambda x: symbol_labels[x],
            index=3  # Default to XAUUSD
        )
        symbol = symbol_options[selected_index]
        symbol_info = SYMBOLS[symbol]

        st.markdown(f"""
        <div class="sidebar-section">
            <strong>📊 Instrument Details:</strong><br>
            <strong>Symbol:</strong> {symbol}<br>
            <strong>Name:</strong> {symbol_info['name']}<br>
            <strong>Type:</strong> {symbol_info['type'].title()}
        </div>
        """, unsafe_allow_html=True)

        # Timeframe
        timeframe_options = ["1 Min", "5 Min", "15 Min", "30 Min", "1 Hr", "Daily"]
        timeframe = st.selectbox("⏰ Timeframe", timeframe_options, index=0)

        # Number of candles
        num_candles = st.slider("📊 Number of Candles", 50, 500, 100, 50)

        # Data source toggle
        use_real_data = st.checkbox(
            "🔄 Use Real AllTick Data",
            value=api_available,
            disabled=not api_available,
            help="Fetch real market data from AllTick API"
        )

        if use_real_data and api_available:
            st.markdown('<span class="status-badge status-success">🟢 REAL DATA MODE</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-info">🔵 DEMO DATA MODE</span>', unsafe_allow_html=True)

    # Get fresh data
    with st.spinner(f"🔄 Fetching latest market data for {symbol}..."):
        df = get_market_data(symbol, timeframe, num_candles, use_real_data, refresh_counter)

    # Market Stats Section with real-time indicator
    current_timestamp = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"## 📊 Market Overview <small style='color: #67748e; font-size: 0.8rem;'>(Updated: {current_timestamp})</small>", unsafe_allow_html=True)

    if df is not None and not df.empty:
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
        price_change = ((current_price / prev_price) - 1) * 100 if prev_price != 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            change_class = "positive" if price_change >= 0 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Current Price</div>
                <div class="metric-value">{format_price(current_price, symbol)}</div>
                <div class="metric-change {change_class}">
                    {"↗" if price_change >= 0 else "↘"} {price_change:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Session High</div>
                <div class="metric-value">{format_price(df['high'].max(), symbol)}</div>
                <div class="metric-change">Peak Value</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Session Low</div>
                <div class="metric-value">{format_price(df['low'].min(), symbol)}</div>
                <div class="metric-change">Minimum Value</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Volume</div>
                <div class="metric-value">{df['volume'].iloc[-1]:,.0f}</div>
                <div class="metric-change">Latest Period</div>
            </div>
            """, unsafe_allow_html=True)

        # Generate and simulate signal
        action, confidence = rl_bot.get_signals(df)
        rl_bot.simulate_trade(df, action, confidence)

        # Log new signals
        if action in ("BUY", "SELL") and action != st.session_state.last_signal:
            st.session_state.signal_log.append({
                "Time": current_timestamp,
                "Signal": action,
                "Price": format_price(current_price, symbol),
                "Confidence": f"{confidence:.1%}"
            })
            st.session_state.last_signal = action

        # RL Scalping Bot Section
        st.markdown("## 🤖 AI Scalping Bot Analysis")

        # Display RL signals
        col_signal, col_performance = st.columns([2.5, 1.5])
        
        with col_signal:
            signal_class = f"signal-{action.lower()}"
            signal_icon = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "🟡"
            
            st.markdown(f"""
            <div class="signal-card {signal_class}">
                <div class="signal-title">
                    {signal_icon} RL SCALPING SIGNAL: {action}
                </div>
                <div style="margin-bottom: 1rem;">
                    <strong>Confidence:</strong> {confidence:.1%} | 
                    <strong>Strategy:</strong> Reinforcement Learning Model
                </div>
                <div style="color: #67748e; line-height: 1.6;">
                    <strong>Analysis:</strong> Advanced RL neural network with 30% exploration rate for optimal learning
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Technical indicators display
            if "rsi14" in df.columns and "macd_hist" in df.columns:
                rsi = df["rsi14"].iloc[-1]
                macd_hist = df["macd_hist"].iloc[-1]
                ema9 = df["ema9"].iloc[-1] if "ema9" in df.columns else current_price
                ema55 = df["ema55"].iloc[-1] if "ema55" in df.columns else current_price
                
                st.markdown(f"""
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-top: 1rem;">
                    <div style="padding: 0.75rem; background: rgba(255, 255, 255, 0.7); border-radius: 8px;">
                        <div style="font-size: 0.75rem; color: #67748e; font-weight: 600;">RSI (14)</div>
                        <div style="font-size: 1rem; font-weight: 700; color: {'#f44336' if rsi > 70 else '#4caf50' if rsi < 30 else '#344767'}">{rsi:.1f}</div>
                    </div>
                    <div style="padding: 0.75rem; background: rgba(255, 255, 255, 0.7); border-radius: 8px;">
                        <div style="font-size: 0.75rem; color: #67748e; font-weight: 600;">MACD Hist</div>
                        <div style="font-size: 1rem; font-weight: 700; color: {'#4caf50' if macd_hist > 0 else '#f44336'}">{macd_hist:.4f}</div>
                    </div>
                    <div style="padding: 0.75rem; background: rgba(255, 255, 255, 0.7); border-radius: 8px;">
                        <div style="font-size: 0.75rem; color: #67748e; font-weight: 600;">EMA Trend</div>
                        <div style="font-size: 1rem; font-weight: 700; color: {'#4caf50' if ema9 > ema55 else '#f44336'}">{"Bullish" if ema9 > ema55 else "Bearish"}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col_performance:
            # Get performance stats
            performance = rl_bot.get_performance_stats()
            
            st.markdown(f"""
            <div class="performance-card">
                <div class="performance-title">🏆 Bot Performance</div>
            """, unsafe_allow_html=True)
            
            if performance["total_trades"] > 0:
                win_rate_class = "performance-positive" if performance["win_rate"] > 0.5 else "performance-negative"
                profit_class = "performance-positive" if performance["total_profit"] > 0 else "performance-negative"
                
                st.markdown(f"""
                <div class="performance-metric">
                    <span class="performance-label">Total Trades</span>
                    <span class="performance-value">{performance["total_trades"]}</span>
                </div>
                <div class="performance-metric">
                    <span class="performance-label">Win Rate</span>
                    <span class="performance-value {win_rate_class}">{performance['win_rate']:.1%}</span>
                </div>
                <div class="performance-metric">
                    <span class="performance-label">Total Profit</span>
                    <span class="performance-value {profit_class}">{performance['total_profit']:.2f}%</span>
                </div>
                <div class="performance-metric">
                    <span class="performance-label">Avg Confidence</span>
                    <span class="performance-value">{performance['avg_confidence']:.1%}</span>
                </div>
                <div class="performance-metric">
                    <span class="performance-label">Exploration Rate</span>
                    <span class="performance-value">{rl_bot.epsilon:.1%}</span>
                </div>
                """, unsafe_allow_html=True)
                
                if performance["win_rate"] > 0.6:
                    st.markdown('<div style="margin-top: 1rem;"><span class="status-badge status-success">🟢 Strong Performance</span></div>', unsafe_allow_html=True)
                elif performance["win_rate"] > 0.4:
                    st.markdown('<div style="margin-top: 1rem;"><span class="status-badge status-warning">🟡 Learning Phase</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="margin-top: 1rem;"><span class="status-badge status-warning">🔴 Needs Training</span></div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="performance-metric">
                    <span class="performance-label">Status</span>
                    <span class="performance-value">🤖 Bot is learning...</span>
                </div>
                <div style="margin-top: 1rem; color: #67748e; font-size: 0.875rem;">
                    The bot is analyzing market patterns and will start generating performance metrics after completing trades.
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

        # Signal Log and Trade History
        col_log, col_trades = st.columns(2)
        
        with col_log:
            st.markdown("""
            <div class="data-card">
                <div class="data-title">📝 Signal Log</div>
            """, unsafe_allow_html=True)
            
            if st.session_state.signal_log:
                signal_df = pd.DataFrame(st.session_state.signal_log)
                st.dataframe(signal_df.tail(10), use_container_width=True)
            else:
                st.write("No signals generated yet.")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_trades:
            st.markdown("""
            <div class="data-card">
                <div class="data-title">📋 Executed Trades</div>
            """, unsafe_allow_html=True)
            
            if rl_bot.trades_history:
                trades_df = pd.DataFrame(rl_bot.trades_history)
                if "entry_time" in trades_df.columns:
                    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"]).dt.strftime("%H:%M:%S")
                if "exit_time" in trades_df.columns:
                    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"]).dt.strftime("%H:%M:%S")
                
                display_cols = ["entry_time", "position", "entry_price", "confidence"]
                if "exit_time" in trades_df.columns:
                    display_cols.extend(["exit_time", "profit_percent"])
                
                st.dataframe(trades_df[display_cols].tail(10), use_container_width=True)
            else:
                st.write("No trades executed yet.")
            
            st.markdown("</div>", unsafe_allow_html=True)

        # Training Controls Section
        st.markdown("## 🧠 RL Bot Training & Management")

        col_train1, col_train2, col_train3 = st.columns(3)

        with col_train1:
            if st.button("🚀 Train One Episode", type="primary"):
                with st.spinner("Training RL bot on current market data..."):
                    info, loss = rl_bot.train_one_episode(df)
                    st.success(
                        f"✅ Episode completed: Trades={info['trades']} | "
                        f"Win Rate={info['win_rate']:.1%} | "
                        f"Profit={info['profit']:.2f}% | Loss={loss:.2f}"
                    )

        with col_train2:
            if st.button("🎯 Reset Bot Performance"):
                rl_bot.trades_history = []
                rl_bot.position = 0
                rl_bot.entry_price = None
                rl_bot.epsilon = 1.0
                st.session_state.signal_log = []
                st.session_state.last_signal = "HOLD"
                st.success("✅ Bot performance and logs reset!")

        with col_train3:
            if st.button("📊 Bot Statistics"):
                st.info(f"""
                **Learning Stats:**
                - Exploration Rate: {rl_bot.epsilon:.1%}
                - Position: {"LONG" if rl_bot.position == 1 else "SHORT" if rl_bot.position == -1 else "NEUTRAL"}
                - Learning Rate: {rl_bot.learning_rate}
                - Gamma (Discount): {rl_bot.gamma}
                """)

        # Chat with Bot Section
        st.markdown("""
        <div class="chat-card">
            <div class="data-title">💬 Chat with Bot</div>
        """, unsafe_allow_html=True)
        
        user_question = st.text_input("Ask me about my sentiment, status, strategy, or market patterns:")
        
        if user_question:
            def get_bot_response(message):
                msg_lower = message.lower()
                performance = rl_bot.get_performance_stats()
                
                if "status" in msg_lower or "performance" in msg_lower:
                    return (f"I've executed {performance['total_trades']} trades with a {performance['win_rate']:.1%} win rate. "
                           f"Total profit: {performance['total_profit']:.2f}%")
                
                elif "sentiment" in msg_lower:
                    if performance["win_rate"] > 0.6:
                        return "🟢 I'm crushing it! High win rate and strong performance."
                    elif performance["win_rate"] > 0.4:
                        return "🟡 I'm in learning mode. Cautious optimism as I adapt to market patterns."
                    else:
                        return "🔴 I need more training. Market conditions are challenging."
                
                elif "strategy" in msg_lower or "epsilon" in msg_lower:
                    return (f"My strategy uses reinforcement learning with ε-greedy exploration at {rl_bot.epsilon:.2%}. "
                           f"Learning rate: {rl_bot.learning_rate}, Discount factor: {rl_bot.gamma}")
                
                elif "pattern" in msg_lower or "indicator" in msg_lower:
                    if "rsi14" in df.columns and "macd_hist" in df.columns:
                        rsi = df["rsi14"].iloc[-1]
                        macd_hist = df["macd_hist"].iloc[-1]
                        ema9 = df["ema9"].iloc[-1] if "ema9" in df.columns else current_price
                        ema55 = df["ema55"].iloc[-1] if "ema55" in df.columns else current_price
                        
                        patterns = []
                        if ema9 > ema55:
                            patterns.append("EMA bullish trend")
                        else:
                            patterns.append("EMA bearish trend")
                            
                        if rsi > 70:
                            patterns.append("RSI overbought")
                        elif rsi < 30:
                            patterns.append("RSI oversold")
                        else:
                            patterns.append("RSI neutral")
                            
                        if macd_hist > 0:
                            patterns.append("MACD positive momentum")
                        else:
                            patterns.append("MACD negative momentum")
                            
                        return f"Current market patterns: {', '.join(patterns)}."
                    else:
                        return "Technical indicators are being calculated. Please wait for more data."
                
                else:
                    return "I can discuss my performance, sentiment, trading strategy, or current market patterns. What would you like to know?"
            
            response = get_bot_response(user_question)
            st.markdown(f"""
            <div class="bot-response">
                <strong>🤖 Bot:</strong> {response}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Footer Status with live indicator
    st.markdown("---")
    data_source = "REAL AllTick Data" if use_real_data and api_available else "Demo Data"
    rl_status = f"RL: {action} ({confidence:.0%})"
    live_indicator = "🟢 LIVE" if use_real_data and api_available else "🔵 DEMO"

    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: white; border-radius: 12px; margin-top: 2rem;">
        <span class="status-badge status-info">📍 {symbol_info['name']}</span>
        <span class="status-badge status-success">{rl_status}</span>
        <span class="status-badge status-info">{data_source}</span>
        <span class="status-badge status-success">{live_indicator}</span>
        <div style="margin-top: 0.5rem; font-size: 0.75rem; color: #67748e;">
            Auto-refresh: Every 10 seconds | Last update: {current_timestamp} | uWhisper.com Private Trading Bot
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Force refresh button for manual updates
    if st.button("🔄 Force Refresh Now", help="Manually refresh market data"):
        st.cache_data.clear()
        st.rerun()
