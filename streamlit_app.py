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
import os
import pickle
import base64
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Auto-refresh setup
try:
    from streamlit_autorefresh import st_autorefresh
    refresh_counter = st_autorefresh(interval=10_000, limit=None, key="auto_refresh")
except ImportError:
    refresh_counter = 0

# Expert Configuration
STATE_FILE = "expert_bot_state.pkl"
CACHE_FOLDER = "cache"
os.makedirs(CACHE_FOLDER, exist_ok=True)

# EODHD API Configuration
EODHD_API_KEY = "686e628db5f664.24674585"  # Store in secrets for production
EODHD_SYMBOLS = {
    "EURUSD": "EURUSD.FOREX",
    "EURJPY": "EURJPY.FOREX", 
    "USDJPY": "USDJPY.FOREX",
    "XAUUSD": "XAUUSD.FOREX",
    "NAS": "NDX.INDX"
}

# Trading symbols configuration
SYMBOLS = {
    "EURUSD": {"name": "EUR/USD", "type": "forex", "description": "Euro vs US Dollar", "pip_value": 10000},
    "EURJPY": {"name": "EUR/JPY", "type": "forex", "description": "Euro vs Japanese Yen", "pip_value": 100},
    "USDJPY": {"name": "USD/JPY", "type": "forex", "description": "US Dollar vs Japanese Yen", "pip_value": 100},
    "XAUUSD": {"name": "XAU/USD", "type": "commodity", "description": "Gold vs US Dollar", "pip_value": 1},
    "NAS": {"name": "NASDAQ", "type": "index", "description": "NASDAQ 100 Index", "pip_value": 1}
}

# Enhanced backup functions
def enhanced_save_bot_state_v2(bot_state):
    """Enhanced save with multiple backup layers"""
    success_count = 0
    
    # 1. Save locally
    try:
        with open(STATE_FILE, 'wb') as f:
            pickle.dump(bot_state, f)
        success_count += 1
    except Exception as e:
        st.sidebar.warning(f"Local save failed: {e}")
    
    # 2. Save to session state
    try:
        if 'backup_bot_states' not in st.session_state:
            st.session_state.backup_bot_states = []
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        backup_entry = {
            'timestamp': timestamp,
            'bot_state': bot_state
        }
        
        st.session_state.backup_bot_states.append(backup_entry)
        
        if len(st.session_state.backup_bot_states) > 20:
            st.session_state.backup_bot_states = st.session_state.backup_bot_states[-20:]
        
        success_count += 1
        
    except Exception as e:
        st.sidebar.warning(f"Session backup failed: {e}")
    
    # 3. Save as downloadable JSON
    try:
        backup_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'bot_state': bot_state,
            'type': 'expert_rl_trading_bot_backup'
        }
        
        st.session_state.downloadable_backup = json.dumps(backup_data, default=str, indent=2)
        success_count += 1
        
    except Exception as e:
        st.sidebar.warning(f"Downloadable backup failed: {e}")
    
    return success_count > 0

def enhanced_load_bot_state_v2():
    """Enhanced load with multiple fallback options"""
    # Try local first
    try:
        with open(STATE_FILE, 'rb') as f:
            bot_state = pickle.load(f)
        return bot_state
    except FileNotFoundError:
        pass
    except Exception as e:
        st.sidebar.warning(f"Local load error: {e}")
    
    # Try session state backup
    try:
        if 'backup_bot_states' in st.session_state and st.session_state.backup_bot_states:
            latest_backup = st.session_state.backup_bot_states[-1]
            bot_state = latest_backup['bot_state']
            
            try:
                with open(STATE_FILE, 'wb') as f:
                    pickle.dump(bot_state, f)
            except:
                pass
            
            return bot_state
    except Exception as e:
        st.sidebar.warning(f"Session load error: {e}")
    
    return {}

# EODHD API Functions
def load_cached_data(symbol):
    """Load cached EODHD data"""
    filepath = os.path.join(CACHE_FOLDER, f"{symbol}.json")
    if os.path.exists(filepath):
        cache_time = os.path.getmtime(filepath)
        if time.time() - cache_time < 3600:  # 1 hour cache
            with open(filepath, "r") as f:
                return json.load(f)
    return None

def save_cached_data(symbol, data):
    """Save EODHD data to cache"""
    filepath = os.path.join(CACHE_FOLDER, f"{symbol}.json")
    with open(filepath, "w") as f:
        json.dump(data, f)

def get_eodhd_data(symbol, limit=100):
    """Get EODHD market data"""
    mapped_symbol = EODHD_SYMBOLS.get(symbol)
    if not mapped_symbol:
        return None
    
    # Try cache first
    cached = load_cached_data(symbol)
    if cached:
        df = pd.DataFrame(cached)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values("date")
    
    # Fetch from API
    url = f"https://eodhd.com/api/eod/{mapped_symbol}?api_token={EODHD_API_KEY}&fmt=json&order=d&limit={limit}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        save_cached_data(symbol, data)
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values("date")
    except:
        return pd.DataFrame(cached) if cached else pd.DataFrame()

def get_economic_events():
    """Get today's high-impact economic events"""
    event_cache = os.path.join(CACHE_FOLDER, "events.json")
    if os.path.exists(event_cache):
        cache_time = os.path.getmtime(event_cache)
        if time.time() - cache_time < 3600:  # 1 hour cache
            with open(event_cache, "r") as f:
                return json.load(f)
    
    url = f"https://eodhd.com/api/economic-events/?api_token={EODHD_API_KEY}&limit=50"
    try:
        resp = requests.get(url)
        data = resp.json()
        today = datetime.utcnow().strftime('%Y-%m-%d')
        today_events = [e for e in data if e.get('date', '').startswith(today) and 'High' in e.get('importance', '')]
        
        with open(event_cache, "w") as f:
            json.dump(today_events, f)
        return today_events
    except:
        return []

def get_news_sentiment():
    """Get market sentiment from news"""
    news_cache = os.path.join(CACHE_FOLDER, "news.json")
    if os.path.exists(news_cache):
        cache_time = os.path.getmtime(news_cache)
        if time.time() - cache_time < 1800:  # 30 min cache
            with open(news_cache, "r") as f:
                return json.load(f).get('sentiment', 0)
    
    try:
        url = f"https://eodhd.com/api/news?api_token={EODHD_API_KEY}&limit=10"
        resp = requests.get(url)
        data = resp.json()
        sentiment_score = 0
        
        for item in data:
            title = item.get("title", "").lower()
            if any(w in title for w in ["rise", "gain", "bullish", "up", "surge", "boost"]):
                sentiment_score += 1
            elif any(w in title for w in ["drop", "fall", "bearish", "down", "crash", "decline"]):
                sentiment_score -= 1
        
        # Normalize sentiment to -1 to 1
        score = max(-1, min(1, sentiment_score / 5))
        
        with open(news_cache, "w") as f:
            json.dump({'sentiment': score}, f)
        return score
    except:
        return 0

# Page configuration
st.set_page_config(
    page_title="Expert RL Trading Bot - uWhisper",
    page_icon="🚀",
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
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h1>🚀 Expert RL Trading Bot</h1>
            <h3>uWhisper.com - Elite Trading System</h3>
            <p>Please enter your access password:</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h1>🚀 Expert RL Trading Bot</h1>
            <h3>uWhisper.com - Elite Trading System</h3>
            <p style="color: red;">❌ Password incorrect. Please try again:</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    else:
        return True

# Only run if password is correct
if check_password():
    
    # Enhanced CSS for Expert UI
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            max-width: 100%;
        }
        
        .stApp {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        /* Expert Header */
        .expert-header {
            background: linear-gradient(135deg, #ff6b6b 0%, #4ecdc4 100%);
            border-radius: 25px;
            padding: 2.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 25px 50px rgba(0,0,0,0.15);
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .expert-title {
            color: white;
            font-size: 3rem;
            font-weight: 800;
            margin: 0;
            text-shadow: 0 4px 8px rgba(0,0,0,0.2);
            background: linear-gradient(45deg, #fff, #f0f0f0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .expert-subtitle {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.3rem;
            margin: 0.5rem 0 0 0;
            font-weight: 500;
        }
        
        /* Report Card */
        .report-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.1);
            color: white;
        }
        
        .report-title {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            text-align: center;
            color: white;
        }
        
        .report-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 1rem;
        }
        
        .report-item {
            text-align: center;
            padding: 1rem;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .report-value {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .report-label {
            font-size: 0.9rem;
            opacity: 0.9;
        }
        
        /* Enhanced Cards */
        .expert-card {
            background: white;
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            border: 1px solid rgba(0,0,0,0.05);
            transition: transform 0.3s ease;
        }
        
        .expert-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 25px 50px rgba(0,0,0,0.15);
        }
        
        .card-title {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            color: #2c3e50;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        /* Signal Cards */
        .signal-expert-buy {
            background: linear-gradient(135deg, #00f260 0%, #0575e6 100%);
            color: white;
            border: none;
        }
        
        .signal-expert-sell {
            background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
            color: white;
            border: none;
        }
        
        .signal-expert-hold {
            background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
            color: #2c3e50;
            border: none;
        }
        
        /* Performance Chart */
        .chart-container {
            background: white;
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        
        /* Trade Table */
        .trade-table {
            background: white;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        
        .profit-cell {
            background: #d4edda;
            color: #155724;
            font-weight: 600;
            padding: 0.5rem;
            border-radius: 8px;
        }
        
        .loss-cell {
            background: #f8d7da;
            color: #721c24;
            font-weight: 600;
            padding: 0.5rem;
            border-radius: 8px;
        }
        
        /* Status Indicators */
        .status-expert {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-size: 0.9rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .status-live {
            background: linear-gradient(135deg, #00f260 0%, #0575e6 100%);
            color: white;
        }
        
        .status-demo {
            background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
            color: #2c3e50;
        }
        
        /* Elite Badge */
        .elite-badge {
            position: fixed;
            top: 15px;
            right: 15px;
            background: linear-gradient(135deg, #ff6b6b 0%, #4ecdc4 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-size: 0.8rem;
            font-weight: 700;
            z-index: 1000;
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
    </style>
    """, unsafe_allow_html=True)

    # Elite badge
    st.markdown('<div class="elite-badge">🚀 EXPERT SYSTEM</div>', unsafe_allow_html=True)

    # Technical Indicators
    def add_indicators(df):
        """Add comprehensive technical indicators"""
        if df.empty or len(df) < 50:
            return df
        
        # Use appropriate price columns
        if 'close' in df.columns:
            close_col = 'close'
        elif 'adjusted_close' in df.columns:
            close_col = 'adjusted_close'
        else:
            return df
        
        # EMAs
        df["ema9"] = df[close_col].ewm(span=9, adjust=False).mean()
        df["ema21"] = df[close_col].ewm(span=21, adjust=False).mean()
        df["ema55"] = df[close_col].ewm(span=55, adjust=False).mean()
        
        # RSI
        delta = df[close_col].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        df["rsi14"] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df[close_col].ewm(span=12, adjust=False).mean()
        ema26 = df[close_col].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # Bollinger Bands
        df["bb_middle"] = df[close_col].rolling(20).mean()
        bb_std = df[close_col].rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
        
        # Stochastic
        low_14 = df["low"].rolling(14).min() if "low" in df.columns else df[close_col].rolling(14).min()
        high_14 = df["high"].rolling(14).max() if "high" in df.columns else df[close_col].rolling(14).max()
        df["stoch_k"] = 100 * (df[close_col] - low_14) / (high_14 - low_14)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()
        
        return df.fillna(method="bfill").fillna(method="ffill")

    # Enhanced Trading Environment
    class ExpertTradingEnv:
        def __init__(self, df, window_size=20):
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
                return np.zeros(15)  # Increased state size
            
            window = self.df.iloc[self.current_step-self.window_size:self.current_step]
            
            # Use appropriate price column
            if 'close' in window.columns:
                close_prices = window["close"].values
            elif 'adjusted_close' in window.columns:
                close_prices = window["adjusted_close"].values
            else:
                return np.zeros(15)
            
            # Enhanced features
            features = []
            
            # Price features
            features.append(np.mean(np.diff(close_prices[-10:]) / close_prices[-10:-1]) if len(close_prices) >= 10 else 0)
            features.append(np.std(close_prices[-10:]) / np.mean(close_prices[-10:]) if len(close_prices) >= 10 else 0)
            features.append((close_prices[-1] - close_prices[-5]) / close_prices[-5] if len(close_prices) >= 5 else 0)
            
            # Technical indicators
            if "rsi14" in window.columns:
                features.append(window["rsi14"].iloc[-1] / 100)
            else:
                features.append(0.5)
            
            if "macd_hist" in window.columns:
                features.append(np.tanh(window["macd_hist"].iloc[-1]))
            else:
                features.append(0)
            
            if "ema9" in window.columns and "ema21" in window.columns:
                features.append((window["ema9"].iloc[-1] - window["ema21"].iloc[-1]) / window["ema21"].iloc[-1])
            else:
                features.append(0)
            
            # Bollinger Bands
            if "bb_upper" in window.columns and "bb_lower" in window.columns:
                bb_position = (close_prices[-1] - window["bb_lower"].iloc[-1]) / (window["bb_upper"].iloc[-1] - window["bb_lower"].iloc[-1])
                features.append(bb_position)
            else:
                features.append(0.5)
            
            # Stochastic
            if "stoch_k" in window.columns:
                features.append(window["stoch_k"].iloc[-1] / 100)
            else:
                features.append(0.5)
            
            # Volume features (if available)
            if "volume" in window.columns:
                vol_ratio = window["volume"].iloc[-1] / window["volume"].mean()
                features.append(np.tanh(vol_ratio))
            else:
                features.append(0)
            
            # Time features
            features.append(np.sin(2 * np.pi * datetime.now().hour / 24))  # Hour of day
            features.append(np.cos(2 * np.pi * datetime.now().hour / 24))
            features.append(np.sin(2 * np.pi * datetime.now().weekday() / 7))  # Day of week
            
            # Market regime
            sma_short = np.mean(close_prices[-5:])
            sma_long = np.mean(close_prices[-20:])
            features.append((sma_short - sma_long) / sma_long)
            
            # Volatility regime
            volatility = np.std(close_prices[-20:]) / np.mean(close_prices[-20:])
            features.append(volatility)
            
            # Momentum
            momentum = (close_prices[-1] - close_prices[-10]) / close_prices[-10] if len(close_prices) >= 10 else 0
            features.append(momentum)
            
            return np.array(features[:15])  # Ensure exactly 15 features
        
        def step(self, action):
            if 'close' in self.df.columns:
                current_price = self.df.at[self.current_step, "close"]
            elif 'adjusted_close' in self.df.columns:
                current_price = self.df.at[self.current_step, "adjusted_close"]
            else:
                current_price = 100  # Default price
            
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
                if abs(price_diff) >= 0.01 or self.current_step % 50 == 0:
                    profit_percent = price_diff / self.entry_price * 100
                    self.total_profit += profit_percent
                    self.total_trades += 1
                    
                    if profit_percent > 0:
                        self.wins += 1
                        reward += 0.2
                    else:
                        reward -= 0.2
                    
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

    # Expert RL Bot
    class ExpertRLBot:
        def __init__(self):
            self.state_size = 15  # Increased state size
            self.action_size = 3  # Hold, Buy, Sell
            self.epsilon = 0.95  # Higher initial exploration
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            self.gamma = 0.99  # Higher discount factor
            self.learning_rate = 0.001
            self._init_model()
            self.position = 0
            self.entry_price = None
            self.trades_history = []
            self.training_episodes = 0
            self.last_training_date = None
            
        def _init_model(self):
            """Initialize enhanced neural network"""
            self.model = {
                'l1': np.random.randn(self.state_size, 64) * 0.1,
                'b1': np.zeros((1, 64)),
                'l2': np.random.randn(64, 32) * 0.1,
                'b2': np.zeros((1, 32)),
                'l3': np.random.randn(32, 16) * 0.1,
                'b3': np.zeros((1, 16)),
                'l4': np.random.randn(16, self.action_size) * 0.1,
                'b4': np.zeros((1, self.action_size))
            }
        
        def load_state(self):
            """Load bot state with enhanced backup"""
            saved = enhanced_load_bot_state_v2()
            if saved:
                self.model = saved.get('model', self.model)
                self.epsilon = saved.get('epsilon', self.epsilon)
                self.trades_history = saved.get('trades_history', [])
                self.training_episodes = saved.get('training_episodes', 0)
                self.last_training_date = saved.get('last_training_date', None)
                return True
            return False
        
        def save_state(self):
            """Save bot state with enhanced backup"""
            bot_state = {
                'model': self.model,
                'epsilon': self.epsilon,
                'trades_history': self.trades_history,
                'training_episodes': self.training_episodes,
                'last_training_date': self.last_training_date
            }
            return enhanced_save_bot_state_v2(bot_state)
        
        def predict(self, state):
            """Enhanced forward pass through neural network"""
            z1 = state.reshape(1, -1) @ self.model['l1'] + self.model['b1']
            a1 = np.tanh(z1)
            z2 = a1 @ self.model['l2'] + self.model['b2']
            a2 = np.tanh(z2)
            z3 = a2 @ self.model['l3'] + self.model['b3']
            a3 = np.tanh(z3)
            z4 = a3 @ self.model['l4'] + self.model['b4']
            return z4.flatten(), (z1, a1, z2, a2, z3, a3)
        
        def get_signals(self, df, news_sentiment=0, economic_events=[]):
            """Generate expert trading signals with market context"""
            # 20% exploration for learning
            if np.random.rand() < 0.2:
                action = np.random.choice(["BUY", "SELL", "HOLD"])
                confidence = np.random.uniform(0.3, 0.6)
                return action, confidence
            
            # RL-based decision with market context
            env = ExpertTradingEnv(df)
            state = env._get_state()
            
            # Add market context to state
            state = np.append(state, [news_sentiment, len(economic_events) / 10])
            if len(state) > 15:
                state = state[:15]
            
            q_values, _ = self.predict(state)
            
            if np.random.rand() < self.epsilon:
                action_idx = np.random.choice(self.action_size)
                confidence = 0.4
            else:
                action_idx = np.argmax(q_values)
                confidence = min(0.95, abs(q_values[action_idx]) / (np.sum(np.abs(q_values)) + 1e-8))
                
                # Boost confidence based on market context
                if news_sentiment > 0.3 and action_idx == 1:  # Bullish news + Buy
                    confidence = min(0.95, confidence * 1.2)
                elif news_sentiment < -0.3 and action_idx == 2:  # Bearish news + Sell
                    confidence = min(0.95, confidence * 1.2)
                
                # Reduce confidence during high-impact events
                if len(economic_events) > 2:
                    confidence *= 0.8
            
            action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
            action = action_map[action_idx]
            
            return action, confidence
        
        def calculate_pips(self, entry_price, exit_price, symbol):
            """Calculate pips won/lost"""
            if symbol in SYMBOLS:
                pip_value = SYMBOLS[symbol]["pip_value"]
                return (exit_price - entry_price) * pip_value
            return exit_price - entry_price
        
        def simulate_trade(self, df, action, confidence, symbol="EURUSD"):
            """Simulate trade execution with enhanced tracking"""
            if 'close' in df.columns:
                current_price = df["close"].iloc[-1]
            elif 'adjusted_close' in df.columns:
                current_price = df["adjusted_close"].iloc[-1]
            else:
                current_price = 100
            
            current_time = datetime.now()
            
            if self.position == 0 and action in ("BUY", "SELL"):
                self.position = 1 if action == "BUY" else -1
                self.entry_price = current_price
                
                trade_entry = {
                    "entry_time": current_time,
                    "position": action,
                    "entry_price": current_price,
                    "confidence": confidence,
                    "symbol": symbol,
                    "news_sentiment": st.session_state.get('news_sentiment', 0),
                    "economic_events": len(st.session_state.get('economic_events', []))
                }
                
                self.trades_history.append(trade_entry)
                self.save_state()
                
            elif self.position != 0:
                # Exit trade
                last_trade = self.trades_history[-1]
                
                # Calculate profits and pips
                profit_pips = self.calculate_pips(self.entry_price, current_price, symbol) * self.position
                profit_percent = (current_price - self.entry_price) * self.position / self.entry_price * 100
                
                # Calculate trade duration
                trade_duration = (current_time - last_trade["entry_time"]).total_seconds() / 60  # minutes
                
                last_trade.update({
                    "exit_time": current_time,
                    "exit_price": current_price,
                    "profit_pips": profit_pips,
                    "profit_percent": profit_percent,
                    "trade_duration": trade_duration,
                    "exit_reason": "auto"
                })
                
                self.position = 0
                self.entry_price = None
                self.save_state()
        
        def train_one_episode(self, df):
            """Train with enhanced learning"""
            env = ExpertTradingEnv(df)
            state = env.reset()
            total_reward = 0
            
            while not env.done:
                # Add market context
                state_with_context = np.append(state, [st.session_state.get('news_sentiment', 0), 0])
                if len(state_with_context) > 15:
                    state_with_context = state_with_context[:15]
                
                q_values, cache = self.predict(state_with_context)
                z1, a1, z2, a2, z3, a3 = cache
                
                # Choose action
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(self.action_size)
                else:
                    action = np.argmax(q_values)
                
                next_state, reward, done, _, info = env.step(action)
                total_reward += reward
                
                # Q-learning update
                next_state_with_context = np.append(next_state, [st.session_state.get('news_sentiment', 0), 0])
                if len(next_state_with_context) > 15:
                    next_state_with_context = next_state_with_context[:15]
                
                q_next, _ = self.predict(next_state_with_context)
                target = reward + self.gamma * np.max(q_next) * (0 if done else 1)
                error = q_values[action] - target
                
                # Enhanced backpropagation
                grad_z4 = np.zeros(self.action_size)
                grad_z4[action] = error * self.learning_rate
                
                # Update layer 4
                self.model['l4'][:, action] -= a3.flatten() * grad_z4[action]
                self.model['b4'][0, action] -= grad_z4[action]
                
                # Update layer 3
                delta3 = (grad_z4 @ self.model['l4'].T) * (1 - a3**2)
                self.model['l3'] -= self.learning_rate * np.outer(a2.flatten(), delta3.flatten())
                self.model['b3'] -= self.learning_rate * delta3.flatten()
                
                # Update layer 2
                delta2 = (delta3 @ self.model['l3'].T) * (1 - a2**2)
                self.model['l2'] -= self.learning_rate * np.outer(a1.flatten(), delta2.flatten())
                self.model['b2'] -= self.learning_rate * delta2.flatten()
                
                # Update layer 1
                delta1 = (delta2 @ self.model['l2'].T) * (1 - a1**2)
                self.model['l1'] -= self.learning_rate * np.outer(state_with_context, delta1.flatten())
                self.model['b1'] -= self.learning_rate * delta1.flatten()
                
                state = next_state
            
            # Update epsilon and training stats
            if info.get("win_rate", 0) > 0.6:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            else:
                self.epsilon = min(0.8, self.epsilon * 1.005)
            
            self.training_episodes += 1
            self.last_training_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.save_state()
            
            return info, total_reward
        
        def get_performance_stats(self):
            """Get comprehensive performance statistics"""
            completed_trades = [t for t in self.trades_history if "profit_percent" in t]
            
            if not completed_trades:
                return {
                    "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                    "avg_confidence": 0, "total_profit": 0, "avg_profit": 0,
                    "avg_win": 0, "avg_loss": 0, "avg_duration": 0,
                    "best_trade": 0, "worst_trade": 0, "total_pips": 0
                }
            
            df_trades = pd.DataFrame(completed_trades)
            wins = df_trades[df_trades["profit_percent"] > 0]
            losses = df_trades[df_trades["profit_percent"] <= 0]
            
            return {
                "total_trades": len(df_trades),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": len(wins) / len(df_trades),
                "avg_confidence": df_trades["confidence"].mean(),
                "total_profit": df_trades["profit_percent"].sum(),
                "avg_profit": df_trades["profit_percent"].mean(),
                "avg_win": wins["profit_percent"].mean() if len(wins) > 0 else 0,
                "avg_loss": losses["profit_percent"].mean() if len(losses) > 0 else 0,
                "avg_duration": df_trades["trade_duration"].mean() if "trade_duration" in df_trades.columns else 0,
                "best_trade": df_trades["profit_percent"].max(),
                "worst_trade": df_trades["profit_percent"].min(),
                "total_pips": df_trades["profit_pips"].sum() if "profit_pips" in df_trades.columns else 0
            }

    # Helper functions
    def create_performance_chart(trades_history):
        """Create cumulative performance chart"""
        if not trades_history:
            return None
        
        completed_trades = [t for t in trades_history if "profit_percent" in t]
        if not completed_trades:
            return None
        
        df = pd.DataFrame(completed_trades)
        df['cumulative_profit'] = df['profit_percent'].cumsum()
        df['trade_number'] = range(1, len(df) + 1)
        
        fig = go.Figure()
        
        # Add cumulative profit line
        fig.add_trace(go.Scatter(
            x=df['trade_number'],
            y=df['cumulative_profit'],
            mode='lines+markers',
            name='Cumulative Profit %',
            line=dict(color='#00f260', width=3),
            marker=dict(size=6)
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
        
        fig.update_layout(
            title="📈 Cumulative Performance",
            xaxis_title="Trade Number",
            yaxis_title="Cumulative Profit (%)",
            template="plotly_white",
            height=400,
            showlegend=True
        )
        
        return fig

    def show_enhanced_backup_status():
        """Show enhanced backup status"""
        with st.sidebar:
            st.markdown("### 💾 Expert Backup System")
            
            local_exists = os.path.exists(STATE_FILE)
            session_backups = len(st.session_state.get('backup_bot_states', []))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Local", "✅" if local_exists else "❌")
            with col2:
                st.metric("Cloud", session_backups)
            
            if st.button("💾 Manual Backup"):
                if 'rl_bot' in st.session_state:
                    bot_state = {
                        'model': st.session_state.rl_bot.model,
                        'epsilon': st.session_state.rl_bot.epsilon,
                        'trades_history': st.session_state.rl_bot.trades_history,
                        'training_episodes': st.session_state.rl_bot.training_episodes
                    }
                    if enhanced_save_bot_state_v2(bot_state):
                        st.success("✅ Backup created!")
            
            # Download backup
            if st.session_state.get('downloadable_backup'):
                backup_data = st.session_state.downloadable_backup
                b64 = base64.b64encode(backup_data.encode()).decode()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"expert_bot_backup_{timestamp}.json"
                
                st.download_button(
                    label="📥 Download Backup",
                    data=backup_data,
                    file_name=filename,
                    mime="application/json"
                )

    # Initialize Expert RL Bot
    @st.cache_resource
    def get_expert_rl_bot():
        bot = ExpertRLBot()
        if bot.load_state():
            st.success("🚀 Expert bot loaded with enhanced memory!")
        else:
            st.info("🆕 Initializing expert trading system...")
        return bot

    # Initialize session state
    if "signal_log" not in st.session_state:
        st.session_state.signal_log = []
    if "last_signal" not in st.session_state:
        st.session_state.last_signal = "HOLD"

    # Initialize expert bot
    rl_bot = get_expert_rl_bot()
    st.session_state.rl_bot = rl_bot

    # Get market data and context
    news_sentiment = get_news_sentiment()
    economic_events = get_economic_events()
    st.session_state.news_sentiment = news_sentiment
    st.session_state.economic_events = economic_events

    # Header
    st.markdown("""
    <div class="expert-header">
        <h1 class="expert-title">🚀 Expert RL Trading Bot</h1>
        <p class="expert-subtitle">Advanced AI • Market Intelligence • News Sentiment • Economic Events • Elite Performance</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Expert Controls")
        
        # Symbol selection
        symbol_options = list(SYMBOLS.keys())
        symbol_labels = [f"{SYMBOLS[s]['name']} ({SYMBOLS[s]['description']})" for s in symbol_options]
        
        selected_index = st.selectbox(
            "📈 Trading Instrument",
            range(len(symbol_options)),
            format_func=lambda x: symbol_labels[x],
            index=0  # Default to EURUSD
        )
        symbol = symbol_options[selected_index]
        symbol_info = SYMBOLS[symbol]
        
        # Data source
        use_eodhd = st.checkbox("🔄 Use EODHD Data", value=True, help="Professional market data")
        
        if use_eodhd:
            st.markdown('<div class="status-expert status-live">🟢 PROFESSIONAL DATA</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-expert status-demo">🟡 DEMO DATA</div>', unsafe_allow_html=True)
        
        # Market context
        st.markdown("### 🌍 Market Context")
        
        # News sentiment
        sentiment_color = "green" if news_sentiment > 0 else "red" if news_sentiment < 0 else "gray"
        sentiment_text = "Bullish" if news_sentiment > 0 else "Bearish" if news_sentiment < 0 else "Neutral"
        st.markdown(f"**News Sentiment:** <span style='color:{sentiment_color}'>{sentiment_text} ({news_sentiment:.2f})</span>", unsafe_allow_html=True)
        
        # Economic events
        st.markdown(f"**Economic Events:** {len(economic_events)} high-impact today")
        
        if economic_events:
            st.markdown("**Today's Events:**")
            for event in economic_events[:3]:  # Show top 3
                st.markdown(f"• {event.get('event', 'Unknown')}")
        
        # Enhanced backup system
        show_enhanced_backup_status()

    # Get market data
    with st.spinner(f"🔄 Loading {symbol} data..."):
        if use_eodhd:
            df = get_eodhd_data(symbol)
        else:
            # Fallback to demo data
            df = pd.DataFrame({
                'date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
                'close': np.random.randn(100).cumsum() + 100,
                'high': np.random.randn(100).cumsum() + 105,
                'low': np.random.randn(100).cumsum() + 95,
                'volume': np.random.randint(1000, 10000, 100)
            })
        
        if df is not None and not df.empty:
            df = add_indicators(df)

    if df is not None and not df.empty:
        # Current price
        if 'close' in df.columns:
            current_price = df['close'].iloc[-1]
        elif 'adjusted_close' in df.columns:
            current_price = df['adjusted_close'].iloc[-1]
        else:
            current_price = 100
        
        # Generate signals
        action, confidence = rl_bot.get_signals(df, news_sentiment, economic_events)
        rl_bot.simulate_trade(df, action, confidence, symbol)
        
        # Get performance stats
        performance = rl_bot.get_performance_stats()
        
        # Report Card
        st.markdown(f"""
        <div class="report-card">
            <div class="report-title">📊 Bot Report Card</div>
            <div class="report-grid">
                <div class="report-item">
                    <div class="report-value">{performance['total_trades']}</div>
                    <div class="report-label">Total Trades</div>
                </div>
                <div class="report-item">
                    <div class="report-value">{performance['wins']}</div>
                    <div class="report-label">🟢 Wins</div>
                </div>
                <div class="report-item">
                    <div class="report-value">{performance['losses']}</div>
                    <div class="report-label">🔴 Losses</div>
                </div>
                <div class="report-item">
                    <div class="report-value">{performance['win_rate']:.1%}</div>
                    <div class="report-label">Win Rate</div>
                </div>
                <div class="report-item">
                    <div class="report-value">{performance['avg_win']:.2f}%</div>
                    <div class="report-label">📈 Avg Win</div>
                </div>
                <div class="report-item">
                    <div class="report-value">{performance['avg_loss']:.2f}%</div>
                    <div class="report-label">📉 Avg Loss</div>
                </div>
                <div class="report-item">
                    <div class="report-value">{performance['avg_confidence']:.1%}</div>
                    <div class="report-label">🧠 Confidence</div>
                </div>
                <div class="report-item">
                    <div class="report-value">{performance['avg_duration']:.1f}m</div>
                    <div class="report-label">⏳ Avg Duration</div>
                </div>
                <div class="report-item">
                    <div class="report-value">{rl_bot.training_episodes}</div>
                    <div class="report-label">🔁 Episodes</div>
                </div>
                <div class="report-item">
                    <div class="report-value">{performance['total_pips']:.1f}</div>
                    <div class="report-label">📊 Total Pips</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Signal and Performance sections
        col_signal, col_chart = st.columns([1, 1])
        
        with col_signal:
            signal_class = f"signal-expert-{action.lower()}"
            signal_icon = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "🟡"
            
            st.markdown(f"""
            <div class="expert-card {signal_class}">
                <div class="card-title">
                    {signal_icon} EXPERT SIGNAL: {action}
                </div>
                <div style="margin-bottom: 1rem;">
                    <strong>Confidence:</strong> {confidence:.1%}<br>
                    <strong>Price:</strong> {current_price:.5f}<br>
                    <strong>News Sentiment:</strong> {sentiment_text}<br>
                    <strong>Economic Events:</strong> {len(economic_events)}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_chart:
            # Performance chart
            chart = create_performance_chart(rl_bot.trades_history)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.info("📈 Performance chart will appear after trades")
        
        # Trade Analysis
        st.markdown("## 📋 Trade Analysis")
        
        if rl_bot.trades_history:
            completed_trades = [t for t in rl_bot.trades_history if "profit_percent" in t]
            if completed_trades:
                df_trades = pd.DataFrame(completed_trades)
                
                # Format the dataframe
                df_display = df_trades.copy()
                df_display["entry_time"] = pd.to_datetime(df_display["entry_time"]).dt.strftime("%H:%M:%S")
                if "exit_time" in df_display.columns:
                    df_display["exit_time"] = pd.to_datetime(df_display["exit_time"]).dt.strftime("%H:%M:%S")
                
                # Color-code profits/losses
                def color_profit(val):
                    if val > 0:
                        return 'background-color: #d4edda; color: #155724'
                    elif val < 0:
                        return 'background-color: #f8d7da; color: #721c24'
                    else:
                        return ''
                
                # Select columns to display
                display_cols = ["entry_time", "position", "entry_price", "confidence"]
                if "exit_time" in df_display.columns:
                    display_cols.extend(["exit_time", "exit_price", "profit_percent", "profit_pips", "trade_duration"])
                
                available_cols = [col for col in display_cols if col in df_display.columns]
                
                # Style the dataframe
                styled_df = df_display[available_cols].tail(10).style.applymap(
                    color_profit, 
                    subset=['profit_percent'] if 'profit_percent' in available_cols else []
                )
                
                st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No trades executed yet. The expert bot is analyzing market conditions.")
        
        # Training Controls
        st.markdown("## 🧠 Expert Training")
        
        col_train1, col_train2, col_train3 = st.columns(3)
        
        with col_train1:
            if st.button("🚀 Train Episode", type="primary"):
                with st.spinner("Training expert bot..."):
                    info, total_reward = rl_bot.train_one_episode(df)
                    st.success(f"✅ Training complete! Reward: {total_reward:.2f}")
        
        with col_train2:
            if st.button("📊 Performance Report"):
                st.balloons()
                st.markdown(f"""
                **🎯 Expert Bot Performance Report**
                
                **Training Stats:**
                - Episodes: {rl_bot.training_episodes}
                - Last Training: {rl_bot.last_training_date or 'Never'}
                - Exploration Rate: {rl_bot.epsilon:.1%}
                
                **Trading Stats:**
                - Total Trades: {performance['total_trades']}
                - Win Rate: {performance['win_rate']:.1%}
                - Total Profit: {performance['total_profit']:.2f}%
                - Best Trade: {performance['best_trade']:.2f}%
                - Worst Trade: {performance['worst_trade']:.2f}%
                """)
        
        with col_train3:
            if st.button("🔄 Reset System"):
                if os.path.exists(STATE_FILE):
                    os.remove(STATE_FILE)
                
                # Reset bot
                rl_bot.trades_history = []
                rl_bot.position = 0
                rl_bot.entry_price = None
                rl_bot.epsilon = 0.95
                rl_bot.training_episodes = 0
                rl_bot._init_model()
                
                # Clear session state
                for key in ['backup_bot_states', 'downloadable_backup', 'signal_log']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.success("✅ Expert system reset complete!")
    
    # Footer
    st.markdown("---")
    current_time = datetime.now().strftime("%H:%M:%S")
    
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 20px; margin-top: 2rem; color: white;">
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <span class="status-expert status-live">🚀 EXPERT SYSTEM</span>
            <span class="status-expert status-live">📊 {symbol_info['name']}</span>
            <span class="status-expert status-live">🧠 {performance['total_trades']} Trades</span>
            <span class="status-expert status-live">💰 {performance['total_profit']:.1f}% Profit</span>
            <span class="status-expert status-live">🎯 {performance['win_rate']:.1%} Win Rate</span>
        </div>
        <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.9;">
            Expert RL Trading Bot • Enhanced Backup System • Market Intelligence • Last Update: {current_time}
        </div>
    </div>
    """, unsafe_allow_html=True)
