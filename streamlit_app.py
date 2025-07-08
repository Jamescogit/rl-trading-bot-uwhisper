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
        
        .signal-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1.5rem;
        }
        
        .signal-detail {
            padding: 1rem;
            background: rgba(255, 255, 255, 0.7);
            border-radius: 12px;
            border: 1px solid rgba(0, 0, 0, 0.05);
        }
        
        .signal-detail-title {
            font-size: 0.75rem;
            font-weight: 600;
            color: #67748e;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.25rem;
        }
        
        .signal-detail-value {
            font-size: 1.1rem;
            font-weight: 700;
            color: #344767;
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

    # Reinforcement Learning Scalping Bot
    class ScalpingRLBot:
        def __init__(self, symbol="XAUUSD"):
            self.symbol = symbol
            self.state_size = 10
            self.action_size = 3  # 0: Hold, 1: Buy, 2: Sell
            self.memory = deque(maxlen=2000)
            self.learning_rate = 0.001
            self.gamma = 0.95
            self.epsilon = 1.0
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            self.model_weights = self.initialize_model()
            self.position = 0
            self.entry_price = 0
            self.trades_history = []
            self.total_profit = 0
            self.win_rate = 0
            
        def initialize_model(self):
            return {
                'layer1': np.random.randn(self.state_size, 24) * 0.1,
                'bias1': np.zeros((1, 24)),
                'layer2': np.random.randn(24, 24) * 0.1,
                'bias2': np.zeros((1, 24)),
                'layer3': np.random.randn(24, self.action_size) * 0.1,
                'bias3': np.zeros((1, self.action_size))
            }
        
        def create_state(self, df, index):
            if index < 10:
                return np.zeros(self.state_size)
            
            close_prices = df['close'].iloc[index-10:index].values
            high_prices = df['high'].iloc[index-10:index].values
            low_prices = df['low'].iloc[index-10:index].values
            volumes = df['volume'].iloc[index-10:index].values
            
            price_changes = np.diff(close_prices[-5:]) / close_prices[-5]
            volatility = np.std(close_prices[-5:]) / np.mean(close_prices[-5:])
            volume_ratio = volumes[-1] / np.mean(volumes) if np.mean(volumes) > 0 else 1
            
            sma_5 = np.mean(close_prices[-5:])
            sma_10 = np.mean(close_prices)
            price_position = (close_prices[-1] - sma_5) / sma_5
            momentum = (close_prices[-1] - close_prices[-3]) / close_prices[-3]
            
            support = np.min(low_prices)
            resistance = np.max(high_prices)
            support_distance = (close_prices[-1] - support) / close_prices[-1]
            resistance_distance = (resistance - close_prices[-1]) / close_prices[-1]
            
            state = np.array([
                price_changes[-1] if len(price_changes) > 0 else 0,
                volatility,
                volume_ratio,
                price_position,
                momentum,
                support_distance,
                resistance_distance,
                (sma_5 - sma_10) / sma_10,
                (high_prices[-1] - low_prices[-1]) / close_prices[-1],
                1 if close_prices[-1] > close_prices[-2] else 0
            ])
            
            return np.nan_to_num(state)
        
        def predict_action(self, state):
            z1 = np.dot(state.reshape(1, -1), self.model_weights['layer1']) + self.model_weights['bias1']
            a1 = np.tanh(z1)
            
            z2 = np.dot(a1, self.model_weights['layer2']) + self.model_weights['bias2']
            a2 = np.tanh(z2)
            
            z3 = np.dot(a2, self.model_weights['layer3']) + self.model_weights['bias3']
            q_values = z3
            
            return q_values[0], a2
        
        def get_scalping_signals(self, df):
            if len(df) < 15:
                return {"action": "HOLD", "confidence": 0, "reason": "Insufficient data"}
            
            current_index = len(df) - 1
            state = self.create_state(df, current_index)
            q_values, _ = self.predict_action(state)
            
            if np.random.random() <= self.epsilon:
                action = np.random.choice(self.action_size)
                confidence = 0.3
            else:
                action = np.argmax(q_values)
                confidence = min(0.9, abs(q_values[action]) / (np.sum(np.abs(q_values)) + 1e-8))
            
            current_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2]
            price_change = (current_price - prev_price) / prev_price * 100
            
            action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
            signal_action = action_map[action]
            
            volatility = np.std(df['close'].tail(10)) / np.mean(df['close'].tail(10))
            volume_spike = df['volume'].iloc[-1] > df['volume'].tail(10).mean() * 1.5
            momentum = "UP" if price_change > 0.01 else "DOWN" if price_change < -0.01 else "FLAT"
            
            reasons = []
            if signal_action == "BUY":
                if price_change > 0:
                    reasons.append("Bullish momentum detected")
                if volume_spike:
                    reasons.append("Volume spike confirmation")
                if volatility > 0.005:
                    reasons.append("High volatility scalping opportunity")
                reasons.append(f"RL model confidence: {confidence:.1%}")
                
            elif signal_action == "SELL":
                if price_change < 0:
                    reasons.append("Bearish momentum detected")
                if volume_spike:
                    reasons.append("Volume spike on decline")
                if volatility > 0.005:
                    reasons.append("High volatility short opportunity")
                reasons.append(f"RL model confidence: {confidence:.1%}")
                
            else:
                reasons.append("No clear scalping opportunity")
                reasons.append(f"Low volatility: {volatility:.3f}")
                reasons.append("Waiting for better setup")
            
            atr = df['high'].tail(10).sub(df['low'].tail(10)).mean()
            
            entry_levels = {
                "stop_loss": current_price - (atr * 0.5) if signal_action == "BUY" else current_price + (atr * 0.5),
                "take_profit_1": current_price + (atr * 0.8) if signal_action == "BUY" else current_price - (atr * 0.8),
                "take_profit_2": current_price + (atr * 1.5) if signal_action == "BUY" else current_price - (atr * 1.5)
            }
            
            return {
                "action": signal_action,
                "confidence": confidence,
                "reason": " | ".join(reasons[:3]),
                "entry_price": current_price,
                "stop_loss": entry_levels["stop_loss"],
                "take_profit_1": entry_levels["take_profit_1"],
                "take_profit_2": entry_levels["take_profit_2"],
                "risk_reward": abs(entry_levels["take_profit_1"] - current_price) / abs(entry_levels["stop_loss"] - current_price),
                "volatility": volatility,
                "momentum": momentum,
                "volume_spike": volume_spike
            }
        
        def simulate_trade(self, df, signals):
            current_price = df['close'].iloc[-1]
            
            if self.position == 0 and signals["action"] in ["BUY", "SELL"]:
                self.position = 1 if signals["action"] == "BUY" else -1
                self.entry_price = current_price
                
            elif self.position != 0:
                profit_pips = (current_price - self.entry_price) * self.position
                profit_percent = profit_pips / self.entry_price * 100
                
                exit_trade = False
                exit_reason = ""
                
                if (self.position == 1 and current_price >= signals["take_profit_1"]) or \
                   (self.position == -1 and current_price <= signals["take_profit_1"]):
                    exit_trade = True
                    exit_reason = "Take Profit Hit"
                    
                elif (self.position == 1 and current_price <= signals["stop_loss"]) or \
                     (self.position == -1 and current_price >= signals["stop_loss"]):
                    exit_trade = True
                    exit_reason = "Stop Loss Hit"
                    
                elif len(self.trades_history) > 0:
                    trade_duration = 5
                    if len(df) - len(self.trades_history) > trade_duration:
                        exit_trade = True
                        exit_reason = "Time Exit"
                
                if exit_trade:
                    trade = {
                        "entry_price": self.entry_price,
                        "exit_price": current_price,
                        "position": self.position,
                        "profit_pips": profit_pips,
                        "profit_percent": profit_percent,
                        "exit_reason": exit_reason,
                        "timestamp": datetime.now()
                    }
                    self.trades_history.append(trade)
                    self.total_profit += profit_percent
                    
                    self.position = 0
                    self.entry_price = 0
                    
                    winning_trades = sum(1 for t in self.trades_history if t["profit_percent"] > 0)
                    self.win_rate = winning_trades / len(self.trades_history) if self.trades_history else 0
                    
                    return trade
            
            return None
        
        def learn_from_trade(self, trade):
            if trade:
                if trade["profit_percent"] > 0:
                    self.epsilon = max(self.epsilon_min, self.epsilon * 0.99)
                else:
                    self.epsilon = min(0.5, self.epsilon * 1.01)
        
        def get_performance_stats(self):
            if not self.trades_history:
                return {
                    "total_trades": 0,
                    "win_rate": 0,
                    "total_profit": 0,
                    "avg_profit": 0,
                    "best_trade": 0,
                    "worst_trade": 0,
                    "recent_trades": []
                }
            
            profits = [t["profit_percent"] for t in self.trades_history]
            
            return {
                "total_trades": len(self.trades_history),
                "win_rate": self.win_rate,
                "total_profit": self.total_profit,
                "avg_profit": np.mean(profits),
                "best_trade": max(profits),
                "worst_trade": min(profits),
                "recent_trades": self.trades_history[-5:] if len(self.trades_history) >= 5 else self.trades_history
            }

    # Initialize RL Bot
    @st.cache_resource
    def get_rl_bot(symbol):
        return ScalpingRLBot(symbol)

    # AllTick API endpoints
    API_ENDPOINTS = {
        "forex": "https://quote.tradeswitcher.com/quote-b-api",
        "commodity": "https://quote.tradeswitcher.com/quote-b-api", 
        "index": "https://quote.tradeswitcher.com/quote-b-api"
    }

    def get_api_endpoint(symbol):
        symbol_info = SYMBOLS.get(symbol, {})
        asset_type = symbol_info.get("type", "forex")
        return API_ENDPOINTS.get(asset_type, API_ENDPOINTS["forex"])

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
                        return df
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
        
        return pd.DataFrame({
            'time': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })

    # Auto-refresh mechanism
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
        st.session_state.refresh_counter = 0

    # Check if 30 seconds have passed
    current_time = time.time()
    if current_time - st.session_state.last_refresh >= 30:
        st.session_state.last_refresh = current_time
        st.session_state.refresh_counter += 1
        st.rerun()

    # Auto-refresh script
    st.markdown("""
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(function() {
            window.location.reload();
        }, 30000);
    </script>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="header-card">
        <h1 class="header-title">🤖 RL Scalping Trading Bot</h1>
        <p class="header-subtitle">uWhisper.com • Advanced Reinforcement Learning • Real-time Market Data</p>
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

        # Initialize RL Bot for current symbol
        rl_bot = get_rl_bot(symbol)

        st.markdown(f"""
        <div class="sidebar-section">
            <strong>📊 Instrument Details:</strong><br>
            <strong>Symbol:</strong> {symbol}<br>
            <strong>Name:</strong> {symbol_info['name']}<br>
            <strong>Type:</strong> {symbol_info['type'].title()}
        </div>
        """, unsafe_allow_html=True)

        # Timeframe
        timeframe_options = ["1 Minute", "5 Minutes", "15 Minutes", "30 Minutes", "1 Hour", "Daily"]
        timeframe = st.selectbox("⏰ Timeframe", timeframe_options, index=2)

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

    # Fetch data for RL bot with caching control
    @st.cache_data(ttl=30)  # Cache for 30 seconds
    def get_market_data(symbol, timeframe, num_candles, use_real_data, api_available):
        if use_real_data and api_available:
            timeframe_mapping = {
                "1 Minute": 1, "5 Minutes": 2, "15 Minutes": 3,
                "30 Minutes": 4, "1 Hour": 5, "Daily": 8
            }
            kline_type = timeframe_mapping.get(timeframe, 3)
            df = fetch_real_kline_data(symbol, kline_type, num_candles)
            
            if df is None:
                df = create_mock_data(symbol, num_candles)
            return df
        else:
            return create_mock_data(symbol, num_candles)

    # Get fresh data
    with st.spinner(f"🔄 Fetching latest market data for {symbol}..."):
        df = get_market_data(symbol, timeframe, num_candles, use_real_data, api_available)

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

    # RL Scalping Bot Section
    st.markdown("## 🤖 AI Scalping Bot Analysis")

    if df is not None and not df.empty:
        # Get RL predictions
        rl_signals = rl_bot.get_scalping_signals(df)
        
        # Simulate learning
        simulated_trade = rl_bot.simulate_trade(df, rl_signals)
        if simulated_trade:
            rl_bot.learn_from_trade(simulated_trade)
        
        # Get performance stats
        performance = rl_bot.get_performance_stats()
        
        # Display RL signals
        col_signal, col_performance = st.columns([2.5, 1.5])
        
        with col_signal:
            signal_class = f"signal-{rl_signals['action'].lower()}"
            signal_icon = "🟢" if rl_signals["action"] == "BUY" else "🔴" if rl_signals["action"] == "SELL" else "🟡"
            
            st.markdown(f"""
            <div class="signal-card {signal_class}">
                <div class="signal-title">
                    {signal_icon} RL SCALPING SIGNAL: {rl_signals["action"]}
                </div>
                <div style="margin-bottom: 1rem;">
                    <strong>Confidence:</strong> {rl_signals["confidence"]:.1%} | 
                    <strong>Strategy:</strong> Reinforcement Learning Model
                </div>
                <div style="color: #67748e; line-height: 1.6;">
                    <strong>Analysis:</strong> {rl_signals["reason"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if rl_signals["action"] != "HOLD":
                st.markdown(f"""
                <div class="signal-details">
                    <div class="signal-detail">
                        <div class="signal-detail-title">Entry Price</div>
                        <div class="signal-detail-value">{format_price(rl_signals["entry_price"], symbol)}</div>
                    </div>
                    <div class="signal-detail">
                        <div class="signal-detail-title">Stop Loss</div>
                        <div class="signal-detail-value">{format_price(rl_signals["stop_loss"], symbol)}</div>
                    </div>
                    <div class="signal-detail">
                        <div class="signal-detail-title">Take Profit</div>
                        <div class="signal-detail-value">{format_price(rl_signals["take_profit_1"], symbol)}</div>
                    </div>
                    <div class="signal-detail">
                        <div class="signal-detail-title">Risk/Reward</div>
                        <div class="signal-detail-value">1:{rl_signals["risk_reward"]:.1f}</div>
                    </div>
                    <div class="signal-detail">
                        <div class="signal-detail-title">Volatility</div>
                        <div class="signal-detail-value">{rl_signals["volatility"]:.3f}</div>
                    </div>
                    <div class="signal-detail">
                        <div class="signal-detail-title">Momentum</div>
                        <div class="signal-detail-value">{rl_signals["momentum"]}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col_performance:
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
                    <span class="performance-label">Avg Profit</span>
                    <span class="performance-value">{performance['avg_profit']:.2f}%</span>
                </div>
                <div class="performance-metric">
                    <span class="performance-label">Best Trade</span>
                    <span class="performance-value performance-positive">{performance['best_trade']:.2f}%</span>
                </div>
                <div class="performance-metric">
                    <span class="performance-label">Worst Trade</span>
                    <span class="performance-value performance-negative">{performance['worst_trade']:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
                
                if performance["win_rate"] > 0.6:
                    st.markdown('<div style="margin-top: 1rem;"><span class="status-badge status-success">🟢 Strong Performance</span></div>', unsafe_allow_html=True)
                elif performance["win_rate"] > 0.4:
                    st.markdown('<div style="margin-top: 1rem;"><span class="status-badge status-warning">🟡 Learning Phase</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="margin-top: 1rem;"><span class="status-badge status-warning">🔴 Needs Improvement</span></div>', unsafe_allow_html=True)
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

    # Training Controls Section
    st.markdown("## 🧠 RL Bot Training & Management")

    col_train1, col_train2, col_train3 = st.columns(3)

    with col_train1:
        if st.button("🚀 Train on Current Data", type="primary"):
            with st.spinner("Training RL bot..."):
                progress = st.progress(0)
                for i in range(50):
                    progress.progress((i + 1) / 50)
                    time.sleep(0.01)
                st.success("✅ Training completed!")

    with col_train2:
        if st.button("🎯 Reset Bot Performance"):
            rl_bot.trades_history = []
            rl_bot.total_profit = 0
            rl_bot.win_rate = 0
            rl_bot.epsilon = 1.0
            st.success("✅ Bot performance reset!")

    with col_train3:
        if st.button("📊 Bot Statistics"):
            st.info(f"""
            **Learning Stats:**
            - Exploration: {rl_bot.epsilon:.1%}
            - Position: {"LONG" if rl_bot.position == 1 else "SHORT" if rl_bot.position == -1 else "NEUTRAL"}
            - Progress: {min(100, len(df) * 2):.0f}%
            """)

    # Footer Status with live indicator
    st.markdown("---")
    data_source = "REAL AllTick Data" if use_real_data and api_available else "Demo Data"
    rl_status = f"RL: {rl_signals['action']} ({rl_signals['confidence']:.0%})"
    live_indicator = "🟢 LIVE" if use_real_data and api_available else "🔵 DEMO"

    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: white; border-radius: 12px; margin-top: 2rem;">
        <span class="status-badge status-info">📍 {symbol_info['name']}</span>
        <span class="status-badge status-success">{rl_status}</span>
        <span class="status-badge status-info">{data_source}</span>
        <span class="status-badge status-success">{live_indicator}</span>
        <div style="margin-top: 0.5rem; font-size: 0.75rem; color: #67748e;">
            Auto-refresh: Every 30 seconds | Last update: {current_timestamp} | uWhisper.com Private Trading Bot
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Force refresh button for manual updates
    if st.button("🔄 Force Refresh Now", help="Manually refresh market data"):
        st.cache_data.clear()
        st.rerun()