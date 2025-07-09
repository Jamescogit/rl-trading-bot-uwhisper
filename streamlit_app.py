# 🚀 Enhanced Backup System for Your RL Trading Bot
# Add this code directly to your existing streamlit_app.py

import json
import base64
from datetime import datetime
import pickle
import os

# Enhanced backup functions to ADD to your existing bot
def enhanced_save_bot_state_v2(bot_state):
    """Enhanced save with multiple backup layers"""
    success_count = 0
    
    # 1. Save locally (your existing method)
    try:
        with open('bot_state.pkl', 'wb') as f:
            pickle.dump(bot_state, f)
        success_count += 1
    except Exception as e:
        st.sidebar.warning(f"Local save failed: {e}")
    
    # 2. Save to session state (persistent during session)
    try:
        if 'backup_bot_states' not in st.session_state:
            st.session_state.backup_bot_states = []
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        backup_entry = {
            'timestamp': timestamp,
            'bot_state': bot_state
        }
        
        st.session_state.backup_bot_states.append(backup_entry)
        
        # Keep only last 10 backups to manage memory
        if len(st.session_state.backup_bot_states) > 10:
            st.session_state.backup_bot_states = st.session_state.backup_bot_states[-10:]
        
        success_count += 1
        st.sidebar.success(f"☁️ Backup saved! ({len(st.session_state.backup_bot_states)} total)")
        
    except Exception as e:
        st.sidebar.warning(f"Session backup failed: {e}")
    
    # 3. Save as downloadable JSON (manual backup)
    try:
        if 'downloadable_backup' not in st.session_state:
            st.session_state.downloadable_backup = None
        
        # Create downloadable backup
        backup_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'bot_state': bot_state,
            'type': 'rl_trading_bot_backup'
        }
        
        st.session_state.downloadable_backup = json.dumps(backup_data, default=str, indent=2)
        success_count += 1
        
    except Exception as e:
        st.sidebar.warning(f"Downloadable backup failed: {e}")
    
    return success_count > 0

def enhanced_load_bot_state_v2():
    """Enhanced load with multiple fallback options"""
    
    # 1. Try loading from local file first
    try:
        with open('bot_state.pkl', 'rb') as f:
            bot_state = pickle.load(f)
        st.sidebar.info("📱 Loaded from local storage")
        return bot_state
    except FileNotFoundError:
        pass  # Try next option
    except Exception as e:
        st.sidebar.warning(f"Local load error: {e}")
    
    # 2. Try loading from session state backup
    try:
        if 'backup_bot_states' in st.session_state and st.session_state.backup_bot_states:
            latest_backup = st.session_state.backup_bot_states[-1]
            bot_state = latest_backup['bot_state']
            
            # Save locally for next time
            try:
                with open('bot_state.pkl', 'wb') as f:
                    pickle.dump(bot_state, f)
            except:
                pass
            
            st.sidebar.success(f"☁️ Loaded from session backup! ({latest_backup['timestamp']})")
            return bot_state
    except Exception as e:
        st.sidebar.warning(f"Session load error: {e}")
    
    # 3. Return empty state if nothing works
    st.sidebar.info("🆕 Starting fresh - no previous state found")
    return {}

def show_enhanced_backup_status():
    """Show enhanced backup status and controls"""
    with st.sidebar:
        st.markdown("### 💾 Enhanced Backup System")
        
        # Show backup stats
        local_exists = os.path.exists('bot_state.pkl')
        session_backups = len(st.session_state.get('backup_bot_states', []))
        
        if local_exists:
            st.success("✅ Local backup available")
        else:
            st.info("📱 No local backup")
        
        if session_backups > 0:
            st.success(f"☁️ {session_backups} session backups")
            
            # Show timestamps of backups
            if st.checkbox("Show backup history"):
                for i, backup in enumerate(st.session_state.backup_bot_states[-5:]):  # Show last 5
                    st.text(f"{i+1}. {backup['timestamp']}")
        else:
            st.info("☁️ No session backups")
        
        # Manual backup button
        if st.button("💾 Create Manual Backup"):
            if hasattr(st.session_state, 'rl_bot') and st.session_state.rl_bot:
                bot_state = {
                    'model': st.session_state.rl_bot.model,
                    'epsilon': st.session_state.rl_bot.epsilon,
                    'trades_history': st.session_state.rl_bot.trades_history
                }
                if enhanced_save_bot_state_v2(bot_state):
                    st.success("✅ Manual backup created!")
            else:
                st.warning("No bot state to backup")
        
        # Download backup button
        if st.session_state.get('downloadable_backup'):
            st.markdown("### 📥 Download Backup")
            
            # Create download link
            backup_data = st.session_state.downloadable_backup
            b64 = base64.b64encode(backup_data.encode()).decode()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rl_bot_backup_{timestamp}.json"
            
            st.markdown(f'''
            <a href="data:application/json;base64,{b64}" download="{filename}">
                <button style="background:#4CAF50;color:white;border:none;padding:10px 20px;border-radius:5px;cursor:pointer;width:100%;">
                    📥 Download Backup File
                </button>
            </a>
            ''', unsafe_allow_html=True)
        
        # Upload backup section
        st.markdown("### 📤 Restore from Backup")
        uploaded_file = st.file_uploader("Upload backup file", type=['json'], key="backup_upload")
        
        if uploaded_file is not None:
            try:
                backup_data = json.loads(uploaded_file.getvalue().decode())
                
                if backup_data.get('type') == 'rl_trading_bot_backup':
                    if st.button("🔄 Restore from Backup"):
                        # Restore the bot state
                        bot_state = backup_data['bot_state']
                        
                        # Save locally
                        with open('bot_state.pkl', 'wb') as f:
                            pickle.dump(bot_state, f)
                        
                        st.success("✅ Backup restored! Please refresh the page.")
                        st.info("The bot will load the restored state on next refresh.")
                else:
                    st.error("Invalid backup file format")
            except Exception as e:
                st.error(f"Error reading backup file: {e}")

def auto_backup_trades_v2(trades_history, frequency=25):
    """Auto-backup trades every N completed trades"""
    if not hasattr(st.session_state, 'last_auto_backup_count'):
        st.session_state.last_auto_backup_count = 0
    
    completed_trades = len([t for t in trades_history if "profit_percent" in t])
    
    if completed_trades > 0 and completed_trades - st.session_state.last_auto_backup_count >= frequency:
        try:
            # Create trades backup
            trades_backup = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'trades_count': completed_trades,
                'trades_history': trades_history,
                'type': 'trades_backup'
            }
            
            # Save to session state
            if 'trades_backups' not in st.session_state:
                st.session_state.trades_backups = []
            
            st.session_state.trades_backups.append(trades_backup)
            
            # Keep only last 5 trade backups
            if len(st.session_state.trades_backups) > 5:
                st.session_state.trades_backups = st.session_state.trades_backups[-5:]
            
            st.session_state.last_auto_backup_count = completed_trades
            
            st.sidebar.info(f"📈 Auto-backup: {completed_trades} trades saved!")
            
        except Exception as e:
            st.sidebar.warning(f"Auto-backup failed: {e}")

# INTEGRATION INSTRUCTIONS FOR YOUR EXISTING BOT:

"""
🔧 HOW TO ADD THIS TO YOUR EXISTING BOT:

1. ADD these imports to the top of your streamlit_app.py:
   (You probably already have most of these)

2. REPLACE your ScalpingRLBot.save_state() method:
   
   OLD CODE:
   def save_state(self):
       try:
           with open(STATE_FILE, "wb") as f:
               pickle.dump({
                   'model': self.model,
                   'epsilon': self.epsilon,
                   'trades_history': self.trades_history
               }, f)
           return True
       except Exception as e:
           st.warning(f"Could not save state: {e}")
           return False
   
   NEW CODE:
   def save_state(self):
       bot_state = {
           'model': self.model,
           'epsilon': self.epsilon,
           'trades_history': self.trades_history
       }
       return enhanced_save_bot_state_v2(bot_state)

3. REPLACE your ScalpingRLBot.load_state() method:
   
   OLD CODE:
   def load_state(self):
       if os.path.exists(STATE_FILE):
           try:
               with open(STATE_FILE, "rb") as f:
                   saved = pickle.load(f)
               self.model = saved.get('model', self.model)
               self.epsilon = saved.get('epsilon', self.epsilon)
               self.trades_history = saved.get('trades_history', [])
               return True
           except Exception as e:
               st.warning(f"Could not load previous state: {e}")
               return False
       return False
   
   NEW CODE:
   def load_state(self):
       saved = enhanced_load_bot_state_v2()
       if saved:
           self.model = saved.get('model', self.model)
           self.epsilon = saved.get('epsilon', self.epsilon)
           self.trades_history = saved.get('trades_history', [])
           return True
       return False

4. ADD to your sidebar (around line 800+ in your code):
   
   ADD THIS LINE:
   show_enhanced_backup_status()

5. ADD to your main loop (after trade execution, around line 1100):
   
   ADD THIS LINE:
   auto_backup_trades_v2(rl_bot.trades_history)

That's it! Your bot will now have:
✅ Multi-layer backup system
✅ Session persistence 
✅ Download/upload capability
✅ Auto-backup every 25 trades
✅ Manual backup controls
✅ Backup history tracking
"""
