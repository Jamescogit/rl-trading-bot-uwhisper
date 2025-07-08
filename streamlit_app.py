# trading_bot.py

# ---------------- Auto‐refresh every 10s ----------------
import subprocess, sys

# 1) Ensure streamlit‐autorefresh is installed
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit-autorefresh"])
    from streamlit_autorefresh import st_autorefresh

import streamlit as st
# Trigger rerun every 10 seconds
refresh_counter = st_autorefresh(interval=10_000, limit=None, key="auto_refresh")

# 2) Meta‐refresh fallback
import streamlit.components.v1 as components
components.html("<meta http-equiv='refresh' content='10'>", height=0)

# ---------------- Imports ----------------
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests, json, urllib.parse, uuid

# ---------------- Config ----------------
SYMBOLS = {
    "EURUSD": {"name": "EUR/USD", "type": "forex"},
    "EURJPY": {"name": "EUR/JPY", "type": "forex"},
    "USDJPY": {"name": "USD/JPY", "type": "forex"},
    "XAUUSD": {"name": "XAU/USD", "type": "commodity"},
    "NAS":    {"name": "NASDAQ",  "type": "index"}
}

# ---------------- CSS ----------------
st.markdown("""
<style>
  /* Paste your existing CSS here */
</style>
""", unsafe_allow_html=True)

# ---------------- Stateful logs ----------------
if "signal_log" not in st.session_state:
    st.session_state.signal_log = []
if "last_signal" not in st.session_state:
    st.session_state.last_signal = "HOLD"

# ---------------- Indicator helpers ----------------
def add_indicators(df):
    df["ema9"]   = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema55"]  = df["close"].ewm(span=55, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200,adjust=False).mean()
    d = df["close"].diff()
    g, l = d.clip(lower=0), -d.clip(upper=0)
    ag, al = g.rolling(14).mean(), l.rolling(14).mean()
    df["rsi14"] = 100 - 100/(1 + ag/(al + 1e-8))
    e12 = df["close"].ewm(span=12,adjust=False).mean()
    e26 = df["close"].ewm(span=26,adjust=False).mean()
    df["macd"]      = e12 - e26
    df["macd_sig"]  = df["macd"].ewm(span=9,adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]
    return df.fillna(method="bfill").fillna(method="ffill")

# ---------------- Data fetch / mock ----------------
def get_api_endpoint(sym):
    return {
        "forex":     "https://quote.tradeswitcher.com/quote-b-api",
        "commodity": "https://quote.tradeswitcher.com/quote-b-api",
        "index":     "https://quote.tradeswitcher.com/quote-b-api"
    }[SYMBOLS[sym]["type"]]

def fetch_real(sym, kt, nc):
    try:
        trace_id = f"rl_{uuid.uuid4().hex[:8]}"
        payload = {"trace": trace_id, "data": {
            "code": sym,
            "kline_type": kt,
            "kline_timestamp_end": 0,
            "query_kline_num": nc,
            "adjust_type": 0
        }}
        url = (
            f"{get_api_endpoint(sym)}/kline?"
            f"token={st.secrets['ALLTICK_API_KEY']}&query="
            f"{urllib.parse.quote(json.dumps(payload))}"
        )
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            kl = r.json().get("data", {}).get("kline_list", [])
            rows = [{
                "time":   pd.to_datetime(int(k["timestamp"]), unit="s"),
                "open":   float(k["open_price"]),
                "high":   float(k["high_price"]),
                "low":    float(k["low_price"]),
                "close":  float(k["close_price"]),
                "volume": float(k.get("volume", 0))
            } for k in kl]
            df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
            return add_indicators(df)
    except:
        pass
    return None

def create_mock(sym, pts=100):
    now = datetime.now()
    times = pd.date_range(now - timedelta(hours=pts/60), now, periods=pts)
    cfg = {
        "EURUSD": (1.0850, 0.005),
        "EURJPY": (158.5, 0.008),
        "USDJPY": (146.2, 0.007),
        "XAUUSD": (2045.5, 0.015),
        "NAS":    (16800, 0.012)
    }
    base, vol = cfg.get(sym, (1.0, 0.01))
    rets = np.random.normal(0, vol, pts)
    px = [base]
    for r in rets:
        px.append(px[-1] * (1 + r))
    op, cl = px[:-1], px[1:]
    hs = [max(o, c) * 1.002 for o, c in zip(op, cl)]
    ls = [min(o, c) * 0.998 for o, c in zip(op, cl)]
    vs = np.random.randint(1000, 10000, pts)
    df = pd.DataFrame({
        "time":   times,
        "open":   op,
        "high":   hs,
        "low":    ls,
        "close":  cl,
        "volume": vs
    })
    return add_indicators(df)

@st.cache_data(ttl=10)
def load_data(sym, tf, nc, use_real, rc):
    mapping = {"1 Min":1, "5 Min":2, "15 Min":3, "30 Min":4, "1 Hr":5, "Daily":8}
    df = fetch_real(sym, mapping[tf], nc) if use_real else None
    return df if df is not None else create_mock(sym, nc)

# ---------------- Blueprint strategy ----------------
def blueprint_signal(df):
    e9, e55, e200 = df["ema9"].iat[-1], df["ema55"].iat[-1], df["ema200"].iat[-1]
    rsi  = df["rsi14"].iat[-1]
    mh   = df["macd_hist"].iat[-1]
    avg  = df["macd_hist"].rolling(20).mean().iat[-1]
    if e9>e55>e200 and rsi>51 and mh<avg*0.8: return "BUY"
    if e9<e55<e200 and rsi<49 and mh>avg*0.8: return "SELL"
    return "HOLD"

# ---------------- TradingEnv ----------------
class TradingEnv:
    def __init__(self, df, ws=10):
        self.df = df.reset_index(drop=True)
        self.ws = ws
        self.reset()
    def reset(self):
        self.i = self.ws
        self.pos = 0
        self.ep  = None
        self.tp = 0
        self.tr = 0
        self.win= 0
        self.done = False
        return self._state()
    def _state(self):
        s = self.df.iloc[self.i-self.ws:self.i]
        c, h, l, v = s["close"].values, s["high"].values, s["low"].values, s["volume"].values
        pc   = np.diff(c[-5:]) / c[-5] if len(c)>=5 else []
        vol  = np.std(c[-5:]) / np.mean(c[-5:]) if len(c)>=5 else 0
        vr   = v[-1]/v.mean() if v.mean()>0 else 1
        sma5, sma10 = c[-5:].mean(), c.mean()
        posp = (c[-1]-sma5)/sma5 if sma5 else 0
        mom  = (c[-1]-c[-3])/c[-3] if len(c)>=3 else 0
        sup, res = l.min(), h.max()
        sd   = (c[-1]-sup)/c[-1] if c[-1] else 0
        rd   = (res-c[-1])/c[-1] if c[-1] else 0
        up   = 1 if c[-1]>c[-2] else 0
        return np.nan_to_num([
            pc[-1] if len(pc)>0 else 0,
            vol, vr, posp, mom, sd, rd, up,
            (sma5-sma10)/sma10 if sma10 else 0,
            (h[-1]-l[-1])/c[-1] if c[-1] else 0,
            s["rsi14"].iat[-1], s["macd_hist"].iat[-1]
        ])
    def step(self, a):
        price = self.df.at[self.i, "close"]
        r = 0
        if a==1 and self.pos!=1:
            self.pos, self.ep = 1, price
            r -= 0.02
        elif a==2 and self.pos!=-1:
            self.pos, self.ep = -1, price
            r -= 0.02
        if self.pos!=0 and self.ep is not None:
            d = (price-self.ep)*self.pos
            r += d*1000 - 0.001
            if abs(d)>=0.005 or self.i%30==0:
                pct = d/self.ep*100 - 0.02
                self.tp += pct
                self.tr += 1
                if pct>0:
                    self.win += 1
                    r += 0.1
                else:
                    r -= 0.1
                self.pos, self.ep = 0, None
        self.i += 1
        if self.i >= len(self.df)-1:
            self.done = True
        return self._state(), r, self.done, False, {
            "profit":   self.tp,
            "trades":   self.tr,
            "win_rate": self.win/self.tr if self.tr else 0
        }

# ---------------- RL Scalping Bot ----------------
class ScalpingRLBot:
    def __init__(self):
        self.S = 12
        self.A = 3
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
        self.model = {
            'l1': np.random.randn(self.S,24)*0.1, 'b1':np.zeros((1,24)),
            'l2': np.random.randn(24,24)*0.1,     'b2':np.zeros((1,24)),
            'l3': np.random.randn(24,self.A)*0.1, 'b3':np.zeros((1,self.A))
        }

    def predict(self, state):
        z1 = state.reshape(1,-1) @ self.model['l1'] + self.model['b1']
        a1 = np.tanh(z1)
        z2 = a1 @ self.model['l2'] + self.model['b2']
        a2 = np.tanh(z2)
        z3 = a2 @ self.model['l3'] + self.model['b3']
        return z3.flatten(), (z1, a1, z2, a2)

    def get_signals(self, df):
        # 30% forced exploration
        if np.random.rand() < 0.3:
            act = "BUY" if np.random.rand()<0.5 else "SELL"
            conf = np.random.uniform(0.4, 0.7)
            return act, conf

        # RL policy
        env = TradingEnv(df)
        state = env._state()
        q_vals, _ = self.predict(state)
        if np.random.rand() < self.epsilon:
            idx, conf = np.random.choice(self.A), 0.3
        else:
            idx = q_vals.argmax()
            conf = min(0.9, abs(q_vals[idx])/(abs(q_vals).sum()+1e-8))
        action = {0:"HOLD",1:"BUY",2:"SELL"}[idx]
        return action, conf

    def simulate_trade(self, df, action, conf):
        price = df["close"].iat[-1]
        now   = datetime.now()
        if self.position==0 and action in ("BUY","SELL"):
            self.position    = 1 if action=="BUY" else -1
            self.entry_price = price
            self.trades_history.append({
                "entry_time":   now,
                "position":     action,
                "entry_price":  price,
                "confidence":   conf
            })
        elif self.position!=0:
            last = self.trades_history[-1]
            pips = (price - self.entry_price)*self.position
            pct  = pips/self.entry_price*100
            last.update({
                "exit_time":      now,
                "exit_price":     price,
                "profit_pips":    pips,
                "profit_percent": pct,
                "exit_reason":    "auto"
            })
            self.position    = 0
            self.entry_price = None

    def train_one_episode(self, df):
        env = TradingEnv(df)
        state = env.reset()
        total_loss = 0.0
        while not env.done:
            q_vals, cache = self.predict(state)
            z1, a1, z2, a2 = cache
            # choose action
            if np.random.rand() < self.epsilon:
                action = np.random.choice(self.A)
            else:
                action = q_vals.argmax()
            next_state, reward, done, _, info = env.step(action)
            q_next, _ = self.predict(next_state)
            target = reward + self.gamma * np.max(q_next) * (0 if done else 1)
            error = q_vals[action] - target
            total_loss += error**2
            # backprop
            # W3, b3
            grad_z3 = np.zeros(self.A)
            grad_z3[action] = error
            self.model['l3'][:,action] -= self.learning_rate * a2.flatten() * grad_z3[action]
            self.model['b3'][0,action]  -= self.learning_rate * grad_z3[action]
            # back to layer2
            delta2 = (grad_z3 @ self.model['l3'].T) * (1 - a2**2)
            self.model['l2'] -= self.learning_rate * np.outer(a1.flatten(), delta2.flatten())
            self.model['b2'] -= self.learning_rate * delta2.flatten()
            # back to layer1
            delta1 = (delta2 @ self.model['l2'].T) * (1 - a1**2)
            self.model['l1'] -= self.learning_rate * np.outer(state, delta1.flatten())
            self.model['b1'] -= self.learning_rate * delta1.flatten()

            state = next_state

        # epsilon decay
        if info["win_rate"] > 0.5:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        else:
            self.epsilon = min(0.5, self.epsilon * 1.01)
        return info, total_loss

    def performance(self):
        comp = [t for t in self.trades_history if "profit_percent" in t]
        if not comp:
            return {
                "total_trades":0,"wins":0,"losses":0,"win_rate":0,
                "avg_confidence":0,"total_profit":0,"avg_profit":0
            }
        dfc = pd.DataFrame(comp)
        wins  = (dfc["profit_percent"]>0).sum()
        total = len(dfc)
        return {
            "total_trades": total,
            "wins": wins,
            "losses": total-wins,
            "win_rate": wins/total,
            "avg_confidence": dfc["confidence"].mean(),
            "total_profit": dfc["profit_percent"].sum(),
            "avg_profit": dfc["profit_percent"].mean()
        }

@st.cache_resource
def get_bot():
    return ScalpingRLBot()

# ---------------- Main App ----------------
st.set_page_config(page_title="RL Scalping Bot", layout="wide")
bot = get_bot()

# --- Sidebar Controls ---
with st.sidebar:
    st.header("🎛️ Controls")
    symbol    = st.selectbox("Instrument", list(SYMBOLS.keys()),
                             format_func=lambda s: SYMBOLS[s]["name"], index=3)
    timeframe = st.selectbox("Timeframe", ["1 Min","5 Min","15 Min","30 Min","1 Hr","Daily"], index=0)
    nc        = st.slider("Candles", 50, 500, 100)
    use_real  = st.checkbox("Use Real Data", value="ALLTICK_API_KEY" in st.secrets)

df = load_data(symbol, timeframe, nc, use_real, refresh_counter)

# --- Header ---
st.markdown(f"<h1>🤖 RL Scalping Bot — {SYMBOLS[symbol]['name']}</h1>", unsafe_allow_html=True)
st.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')}")

# --- Market Overview ---
st.subheader("📊 Market Overview")
if not df.empty:
    curr, prev = df["close"].iat[-1], df["close"].iat[-2]
    pct = ((curr/prev)-1)*100 if prev else 0
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Price", f"{curr:.4f}", f"{pct:.2f}%")
    c2.metric("High",  f"{df['high'].max():.4f}")
    c3.metric("Low",   f"{df['low'].min():.4f}")
    c4.metric("Volume", f"{int(df['volume'].iat[-1])}")

# --- Generate & simulate signal ---
action, conf = bot.get_signals(df)
bot.simulate_trade(df, action, conf)

# --- Signal display & log ---
st.subheader("🚦 Signal")
st.write(f"**{action}** — Confidence: {conf:.1%}")

if action in ("BUY","SELL") and action != st.session_state.last_signal:
    st.session_state.signal_log.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "signal": action,
        "price": f"{df['close'].iat[-1]:.4f}",
        "confidence": f"{conf:.1%}"
    })
    st.session_state.last_signal = action

st.subheader("📝 Signal Log")
if st.session_state.signal_log:
    st.dataframe(pd.DataFrame(st.session_state.signal_log))
else:
    st.write("No signals yet.")

# --- Executed trades ---
st.subheader("📋 Executed Trades")
trades_df = pd.DataFrame(bot.trades_history)
if not trades_df.empty:
    if "exit_time" in trades_df.columns:
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"]).dt.strftime("%H:%M:%S")
    if "entry_time" in trades_df.columns:
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"]).dt.strftime("%H:%M:%S")
    st.dataframe(trades_df)
else:
    st.write("No trades executed yet.")

# --- Performance stats & sentiment ---
stats = bot.performance()
st.subheader("📈 Performance")
st.write(f"- Trades: {stats['total_trades']}")
st.write(f"- Wins: {stats['wins']}, Losses: {stats['losses']} (Win%: {stats['win_rate']:.1%})")
st.write(f"- Total Profit: {stats['total_profit']:.2f}% (Avg: {stats['avg_profit']:.2f}%)")
st.write(f"- Avg Confidence: {stats['avg_confidence']:.1%}")

if stats["win_rate"] > 0.6:
    sentiment = "🟢 Bot is crushing it!"
elif stats["win_rate"] > 0.4:
    sentiment = "🟡 Bot is learning—caution advised."
else:
    sentiment = "🔴 Bot needs more training."
st.markdown(f"**Sentiment:** {sentiment}")

# --- Training controls (now with real weight updates!) ---
st.subheader("🧠 Training Controls")
col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 Train One Episode"):
        info, loss = bot.train_one_episode(df)
        st.success(
            f"Episode done: Trades={info['trades']} Win%={info['win_rate']:.1%} "
            f"Profit={info['profit']:.2f}% | Loss={loss:.2f}"
        )
with col2:
    if st.button("🔄 Reset Bot"):
        st.session_state.clear()
        st.experimental_rerun()

# --- Chat with Bot ---
st.subheader("💬 Chat with Bot")
user_q = st.text_input("Ask me about my sentiment, status, strategy, or patterns:")
if user_q:
    def get_response(msg):
        ml = msg.lower()
        if "status" in ml or "performance" in ml:
            return (f"I've taken {stats['total_trades']} trades, win rate {stats['win_rate']:.1%}, "
                    f"total profit {stats['total_profit']:.2f}%")
        if "sentiment" in ml:
            return sentiment
        if "strategy" in ml or "epsilon" in ml:
            return f"My ε is {bot.epsilon:.2f}, learning rate {bot.learning_rate}, γ={bot.gamma}."
        if "pattern" in ml or "indicator" in ml:
            last = df.iloc[-1]
            pts = []
            if last['ema9']>last['ema55']>last['ema200']:
                pts.append("EMA uptrend")
            elif last['ema9']<last['ema55']<last['ema200']:
                pts.append("EMA downtrend")
            if last['rsi14']>70:
                pts.append("RSI overbought")
            elif last['rsi14']<30:
                pts.append("RSI oversold")
            if last['macd_hist']>0:
                pts.append("MACD+")
            else:
                pts.append("MACD–")
            return "I see: " + ", ".join(pts) + "."
        return "I can discuss performance, sentiment, strategy, or patterns!"
    st.markdown(f"**Bot:** {get_response(user_q)}")

# --- Footer ---
st.markdown("---")
mode = "REAL" if use_real else "DEMO"
st.caption(f"Mode: {mode} | Auto-refresh every 10s")
