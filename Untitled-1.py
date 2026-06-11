import streamlit as st
import ccxt
import requests
import pandas as pd
import yfinance as yf
import numpy as np
import time
import concurrent.futures
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg') # Fix for Streamlit/Matplotlib GUI errors
import matplotlib.pyplot as plt

# Set page config at the top level to avoid errors and define layout
st.set_page_config(page_title="Quant Scalper 1h", layout="wide")

@st.cache_data
def fetch_fear_and_greed_history():
    """
    Fetches historical Fear and Greed Index data (Proxy for Market News Sentiment).
    """
    url = "https://api.alternative.me/fng/?limit=0"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()['data']
        df = pd.DataFrame(data)
        df['value'] = pd.to_numeric(df['value'])
        # Convert timestamp to datetime
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('date').sort_index()
        return df[['value']]
    except Exception as e:
        return None

@st.cache_data(ttl=300)
def fetch_and_analyze(symbol='BTC/USDT', timeframe='1h', start_date=None, end_date=None, silent=False):
    """
    Fetches Crypto Data (via ccxt/Binance) and calculates Multi-Strategy Factors.
    """
    try:
        # Handle Symbol Formatting (e.g., BTC-USD -> BTC/USDT for Binance)
        if symbol.endswith('-USD'):
            symbol = symbol.replace('-USD', '/USDT')
        elif '-' in symbol:
            symbol = symbol.replace('-', '/')
        
        print(f"Fetching data for {symbol} via Binance...")
        
        df = pd.DataFrame()
        
        # Check if symbol is a Stock/ETF to fetch via yfinance
        if symbol in ['SPY', 'QQQ', 'DIA', '^VIX', 'DX-Y.NYB', 'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AVGO', 'LLY', 'JPM']:
            # Fetch Stock Data via yfinance
            print(f"Fetching data for {symbol} via yfinance...")
            yf_interval = timeframe  # '1h', '1d' match yfinance usually
            
            if start_date:
                df = yf.download(symbol, start=start_date, interval=yf_interval, progress=False, prepost=True)
            else:
                # Live mode - last 60 days for 1h to get enough candles for indicators
                period = '60d' if timeframe == '1h' else '2y'
                df = yf.download(symbol, period=period, interval=yf_interval, progress=False, prepost=True)
            
            # Flatten MultiIndex columns (yfinance v0.2+)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            # Normalize columns
            df.columns = [c.lower() for c in df.columns]
            df.index.name = 'date'
            
            # Strip timezone to avoid merge_asof crashes with Fear & Greed data
            if df.index.tz is not None:
                df.index = df.index.tz_convert('UTC').tz_localize(None)
            
        else:
            # Fetch Crypto Data via Binance (CCXT)
            exchange = ccxt.binance({'enableRateLimit': True})
            limit = 1000 # Binance API limit per request
            all_ohlcv = []

            if start_date:
                # Backtesting mode: Fetch all data since start_date in a loop
                since = exchange.parse8601(start_date + 'T00:00:00Z')
                while True:
                    print(f"Fetching historical data chunk for {symbol} since {datetime.fromtimestamp(since/1000)}...")
                    ohlcv_chunk = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                    if not ohlcv_chunk:
                        break # No more data
                    
                    all_ohlcv.extend(ohlcv_chunk)
                    
                    if len(ohlcv_chunk) < limit:
                        break # Reached the end of available history for the period
                    
                    # Set 'since' to the timestamp of the last candle + 1ms for the next chunk
                    since = ohlcv_chunk[-1][0] + 1 
            else:
                # Live analysis mode: Fetch latest N candles
                print(f"Fetching latest {limit} candles for live analysis of {symbol}...")
                all_ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if all_ohlcv:
                df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('date')

        if df.empty:
             if not silent:
                 st.warning(f"No data returned for {symbol}. Check if the ticker is correct (e.g., BTC/USDT) and internet connection.")
             return None
            
        # Remove potential duplicates from overlapping fetches and sort
        if 'timestamp' in df.columns:
            df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
        else:
            df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True) # Ensure data is chronological
        cols = ['open', 'high', 'low', 'close', 'volume']
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

        # --- STRATEGY 1: Trend Following (Moving Averages) ---
        df['sma_200'] = df['close'].rolling(window=200).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # --- STRATEGY 2: Mean Reversion (Bollinger Bands & RSI) ---
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mid'] + (2 * df['bb_std'])
        df['bb_lower'] = df['bb_mid'] - (2 * df['bb_std'])

        # --- STRATEGY 3: Statistical Arbitrage (Simplified) ---
        # Using Volatility (Standard Deviation) as a proxy for mean-reverting regimes
        df['volatility'] = df['close'].rolling(window=20).std()
        df['z_score'] = (df['close'] - df['bb_mid']) / df['bb_std']

        # --- STRATEGY 4: Factor Investing (Momentum) ---
        # Rate of Change (30 days for Crypto)
        df['momentum'] = df['close'].pct_change(periods=30)
        
        # --- STRATEGY 5: Sentiment Analysis (High Impact News Proxy) ---
        # We merge Fear & Greed index. If news is bad (Fear), we look for bottoms.
        fg_df = fetch_fear_and_greed_history()
        if fg_df is not None:
            # Merge using merge_asof to align daily sentiment with candle times
            # We align backward to use the most recent known sentiment value
            df = pd.merge_asof(df, fg_df, left_index=True, right_index=True, direction='backward')
            df.rename(columns={'value': 'sentiment'}, inplace=True)
        else:
            df['sentiment'] = 50 # Default Neutral if API fails

        # --- STRATEGY 6: Institutional Order Flow (Chaikin Money Flow) ---
        # CMF measures buying/selling pressure over a period
        ad = np.where(df['high'] == df['low'], 0, (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']))
        df['vol_ad'] = ad * df['volume']
        df['cmf'] = df['vol_ad'].rolling(window=20).sum() / df['volume'].rolling(window=20).sum()

        # --- STRATEGY 7: Order Blocks (Support/Resistance Zones) ---
        # We identify the lowest low (Demand/Bullish Block) and highest high (Supply/Bearish Block)
        # of the last 50 candles to find where institutions placed orders.
        df['ob_bull'] = df['low'].rolling(window=50).min().shift(1) # Recent swing low
        df['ob_bear'] = df['high'].rolling(window=50).max().shift(1) # Recent swing high
        
        # --- STRATEGY 8: Volume Spikes (Squeeze Detection) ---
        df['vol_ma'] = df['volume'].rolling(window=20).mean()
        
        # --- STRATEGY 9: ADX (Trend Strength & Regime Filter) ---
        # ADX identifies if the market is trending or chopping
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
        df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        
        # Wilder's Smoothing (Window 14)
        df['tr_s'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
        df['plus_dm_s'] = df['plus_dm'].ewm(alpha=1/14, adjust=False).mean()
        df['minus_dm_s'] = df['minus_dm'].ewm(alpha=1/14, adjust=False).mean()
        df['plus_di'] = 100 * (df['plus_dm_s'] / df['tr_s'])
        df['minus_di'] = 100 * (df['minus_dm_s'] / df['tr_s'])
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].ewm(alpha=1/14, adjust=False).mean()

        # --- STRATEGY 10: Fair Value Gaps (FVG) ---
        # Identifying imbalances (gaps) left by aggressive orders
        # Bullish FVG: Low of candle i > High of candle i-2
        df['is_fvg_bull'] = (df['low'] > df['high'].shift(2)) & (df['close'] > df['open'])
        df['is_fvg_bear'] = (df['high'] < df['low'].shift(2)) & (df['close'] < df['open'])
        
        # Track the most recent FVG zones (Forward Fill)
        df['last_bull_fvg_top'] = df['low'].where(df['is_fvg_bull']).ffill()
        df['last_bull_fvg_bottom'] = df['high'].shift(2).where(df['is_fvg_bull']).ffill()
        df['last_bear_fvg_bottom'] = df['high'].where(df['is_fvg_bear']).ffill()
        df['last_bear_fvg_top'] = df['low'].shift(2).where(df['is_fvg_bear']).ffill()
        
        # Fill NaN values created by rolling windows
        df['cmf'] = df['cmf'].fillna(0)
        df['ob_bull'] = df['ob_bull'].fillna(df['low'])
        df['ob_bear'] = df['ob_bear'].fillna(df['high'])
        df['adx'] = df['adx'].fillna(0)
        df['last_bull_fvg_top'] = df['last_bull_fvg_top'].fillna(0)
        df['last_bull_fvg_bottom'] = df['last_bull_fvg_bottom'].fillna(0)
        df['last_bear_fvg_top'] = df['last_bear_fvg_top'].fillna(10000000)
        df['last_bear_fvg_bottom'] = df['last_bear_fvg_bottom'].fillna(10000000)

        # --- STRATEGY 11: Rising Momentum & Volume (User Request) ---
        # 12-Hour Volume (Assuming 1h candles, 12h = 12 periods)
        # We check if the total volume traded in the last 12 hours is increasing
        df['vol_12h'] = df['volume'].rolling(window=12).sum()
        df['vol_12h_prev'] = df['vol_12h'].shift(1)
        df['momentum_prev'] = df['momentum'].shift(1)
        
        return df

    except Exception as e:
        if not silent:
            st.error(f"Data Fetch Error: {e}")
        print(f"Error fetching data for {symbol}: {e}")
        return None

def generate_signal(df, symbol, start_hour=0, end_hour=24):
    """
    Generates a Composite Score based on the 5 quantitative strategies.
    """
    if df is None or df.empty:
        return
    
    # Get the latest data point
    current = df.iloc[-1]
    
    # Logic Variables
    price = current['close']
    ema_50 = current['ema_50']
    rsi = current['rsi']
    z_score = current['z_score']
    momentum = current['momentum']
    sentiment = current.get('sentiment', 50)
    cmf = current['cmf']
    ob_bull = current['ob_bull']
    ob_bear = current['ob_bear']
    volume = current['volume']
    vol_ma = current['vol_ma']
    adx = current['adx']
    vol_12h = current.get('vol_12h', 0)
    vol_12h_prev = current.get('vol_12h_prev', 0)
    momentum_prev = current.get('momentum_prev', 0)
    
    # FVG Zones
    price = current['close']
    in_bull_fvg = (price <= current['last_bull_fvg_top']) and (price >= current['last_bull_fvg_bottom'])
    in_bear_fvg = (price >= current['last_bear_fvg_bottom']) and (price <= current['last_bear_fvg_top'])
    
    # --- TIME FILTER ---
    # Check if current candle is within active trading hours
    current_hour = current.name.hour
    is_active_time = False
    if start_hour <= end_hour:
        is_active_time = (start_hour <= current_hour < end_hour)
    else: # Spans midnight (e.g., 22:00 to 06:00)
        is_active_time = (current_hour >= start_hour or current_hour < end_hour)


    # --- COMPOSITE SCORING SYSTEM ---
    score = 0
    reasons = []
    
    # 1. Trend Following (+1 if Bullish)
    if price > ema_50:
        score += 1
        reasons.append("Trend Bullish")
    else:
        score -= 1
        reasons.append("Trend Bearish")
        
    # 2. Mean Reversion (+1 if Oversold, -1 if Overbought)
    if rsi < 30:
        score += 1
        reasons.append("RSI Oversold")
    elif rsi > 70:
        score -= 1
        reasons.append("RSI Overbought")
        
    # 3. Statistical Arbitrage / Mean Rev (Z-Score extreme)
    if z_score < -2:
        score += 1
        reasons.append("Price < 2 StdDev (Cheap)")
    elif z_score > 2:
        score -= 1
        reasons.append("Price > 2 StdDev (Expensive)")
        
    # 4. Factor Investing (Momentum)
    if momentum > 0:
        score += 0.5
        reasons.append("Pos Momentum")
    else:
        score -= 0.5
        reasons.append("Neg Momentum")
    
    # 5. Sentiment / News Analysis
    # Buy when others are fearful (Bad News Overreaction)
    if sentiment < 20:
        score += 1
        reasons.append(f"Extreme Fear ({sentiment})")
    # Sell when others are greedy (Good News Hype)
    elif sentiment > 80:
        score -= 1
        reasons.append(f"Extreme Greed ({sentiment})")
        
    # 6. Order Flow (CMF) - Tracking Smart Money
    if cmf > 0.1:
        score += 1
        reasons.append(f"Inst. Accumulation (CMF {cmf:.2f})")
    elif cmf < -0.1:
        score -= 1
        reasons.append(f"Inst. Distribution (CMF {cmf:.2f})")
        
    # 7. Order Blocks (Re-testing Liquidity Zones)
    # If price is within 1% of the Bullish Order Block (Support)
    if 0 <= (price - ob_bull) / price <= 0.01:
        score += 1.5 # High weight for support bounces
        reasons.append("Testing Bullish Order Block")
    elif 0 <= (ob_bear - price) / price <= 0.01:
        score -= 1.5 # High weight for resistance rejection
        reasons.append("Testing Bearish Order Block")
        
    # 8. Short Squeeze / Blow-off Top Detector (Fade the "Fake" Spike)
    # Logic: Price > 3 StdDevs (Extreme) + RSI Hot + Massive Volume Spike = Liquidation Wick
    if z_score > 3 and rsi > 75 and volume > (vol_ma * 3):
        score -= 2.0 # Strong Sell signal (expecting rapid reversion)
        reasons.append("Short Squeeze / Fake Pump Detected")
        
    # 9. Regime Filter (ADX)
    if adx > 25:
        score *= 1.1 # Amplify score if trend is strong
        reasons.append(f"Strong Trend (ADX {adx:.1f})")
    elif adx < 20:
        score *= 0.5 # Reduce score confidence in chop
        reasons.append(f"Weak Trend/Chop (ADX {adx:.1f})")
        
    # 10. Fair Value Gaps (Sniper Entries)
    if in_bull_fvg:
        score += 2.0
        reasons.append("In Bullish FVG Zone (Buy Zone)")
    elif in_bear_fvg:
        score -= 2.0
        reasons.append("In Bearish FVG Zone (Sell Zone)")
        
    # 11. Rising Momentum & Volume (Dominant Factor)
    # "Mainly based on rising momentum and rising 12-hour volume"
    volume_rising = vol_12h > vol_12h_prev
    
    if volume_rising:
        # Bullish: Positive Momentum that is getting stronger
        if momentum > 0 and momentum > momentum_prev:
            score += 2.5
            reasons.append("Rising Mom & 12H Vol (Strong Buy)")
        # Bearish: Negative Momentum that is getting stronger (falling price speeding up)
        elif momentum < 0 and momentum < momentum_prev:
            score -= 2.5
            reasons.append("Rising Bearish Mom & 12H Vol (Strong Sell)")

    # --- Prediction Logic ---
    signal = "NEUTRAL"
    
    # Market Neutral Approach:
    # We only take positions if multiple factors align (High Confidence)
    if not is_active_time:
        signal = "NEUTRAL (Outside Hours)"
        reasons.insert(0, f"Inactive Time (UTC {current_hour}:00)")
    elif score >= 1.5: # Lowered slightly to show more activity
        signal = "BUY / LONG"
    elif score <= -1.5:
        signal = "SELL / SHORT"
    else:
        signal = "NEUTRAL / HEDGE"

    # Output formatting
    return {
        "symbol": symbol,
        "price": price,
        "score": score,
        "factors": reasons,
        "signal": signal,
        "date": current.name.date()
    }

def plot_backtest(df, trade_history, symbol):
    """
    Plots the price, indicators, and trades from the backtest.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Price and EMAs
    ax.plot(df.index, df['close'], label=f'{symbol} (Price)', color='black', alpha=0.5)
    ax.plot(df.index, df['ema_50'], label='Trend (50 EMA)', color='blue', alpha=0.7)
    ax.plot(df.index, df['bb_upper'], label='BB Upper', color='gray', linestyle='--', alpha=0.3)
    ax.plot(df.index, df['bb_lower'], label='BB Lower', color='gray', linestyle='--', alpha=0.3)
    ax.plot(df.index, df['ob_bull'], label='Bullish Order Block', color='green', linestyle=':', alpha=0.6)
    ax.plot(df.index, df['ob_bear'], label='Bearish Order Block', color='red', linestyle=':', alpha=0.6)
    
    # Plot FVG Zones (Fair Value Gaps)
    # Replace initialization values with NaN so they don't mess up the chart scale
    bull_top = df['last_bull_fvg_top'].replace(0, np.nan)
    bull_bot = df['last_bull_fvg_bottom'].replace(0, np.nan)
    bear_top = df['last_bear_fvg_top'].replace(10000000, np.nan)
    bear_bot = df['last_bear_fvg_bottom'].replace(10000000, np.nan)
    
    ax.fill_between(df.index, bull_top, bull_bot, color='green', alpha=0.15, label='Bullish FVG')
    ax.fill_between(df.index, bear_top, bear_bot, color='red', alpha=0.15, label='Bearish FVG')

    # Plot Trades
    for trade in trade_history:
        entry_date = trade['entry_date']
        exit_date = trade['exit_date']
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        
        color = 'green' if trade['type'] == 'LONG' else 'red'
        marker = '^' if trade['type'] == 'LONG' else 'v'
            
        # Mark Entry and Exit
        ax.scatter(entry_date, entry_price, color=color, marker=marker, s=100, zorder=5)
        ax.scatter(exit_date, exit_price, color='black', marker='x', s=50, zorder=5)
        ax.plot([entry_date, exit_date], [entry_price, exit_price], color=color, linestyle='--', alpha=0.5)

    ax.set_title(f'Backtest Analysis: {symbol}')
    ax.legend()
    plt.close(fig) # Prevent memory leaks
    return fig

def send_telegram_alert(token, chat_id, symbol, signal, price, score, reasons):
    """
    Sends a formatted trade alert to Telegram.
    """
    if not token or not chat_id:
        return

    emoji = "🟢" if "BUY" in signal else "🔴"
    msg = (
        f"{emoji} *TRADE ALERT: {symbol}*\n"
        f"*Signal:* {signal}\n"
        f"*Price:* ${price:,.2f}\n"
        f"*Score:* {score}\n\n"
        f"*Drivers:*\n• " + "\n• ".join(reasons)
    )
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}, timeout=5)
    except Exception as e:
        print(f"Telegram Error: {e}")

def backtest_strategy(df, symbol, start_hour=0, end_hour=24, rsi_lower=30, rsi_upper=70):
    """
    Simple backtest to verify strategy performance on historical data.
    """
    if df is None or df.empty:
        return

    initial_balance = 10000
    balance = initial_balance
    position = None # None, 'LONG', 'SHORT'
    entry_price = 0
    entry_date = None
    trades = []
    trade_history = []
    peak_balance = initial_balance
    max_drawdown = 0
    
    status_text = f"Running Backtest for {symbol} ({len(df)} days)..."
    
    # Iterate through history (skip first 50 for EMA warmup)
    for i in range(50, len(df) - 1):
        current = df.iloc[i]
        # We execute trades at the OPEN of the next candle based on CURRENT signals
        next_open = df.iloc[i+1]['open']
        
        # Logic Variables
        price = current['close']
        ema_50 = current['ema_50']
        rsi = current['rsi']
        z_score = current['z_score']
        momentum = current['momentum']
        sentiment = current.get('sentiment', 50)
        cmf = current['cmf']
        ob_bull = current['ob_bull']
        ob_bear = current['ob_bear']
        volume = current['volume']
        vol_ma = current['vol_ma']
        adx = current['adx']
        vol_12h = current.get('vol_12h', 0)
        vol_12h_prev = current.get('vol_12h_prev', 0)
        momentum_prev = current.get('momentum_prev', 0)
        
        # FVG
        in_bull_fvg = (price <= current['last_bull_fvg_top']) and (price >= current['last_bull_fvg_bottom'])
        in_bear_fvg = (price >= current['last_bear_fvg_bottom']) and (price <= current['last_bear_fvg_top'])
        
        # Time Filter Logic
        current_hour = current.name.hour
        is_active_time = False
        if start_hour <= end_hour:
            is_active_time = (start_hour <= current_hour < end_hour)
        else:
            is_active_time = (current_hour >= start_hour or current_hour < end_hour)

        # Calculate Composite Score
        score = 0
        if price > ema_50: score += 1
        else: score -= 1
        if rsi < rsi_lower: score += 1
        elif rsi > rsi_upper: score -= 1
        if z_score < -2: score += 1
        elif z_score > 2: score -= 1
        if momentum > 0: score += 0.5
        else: score -= 0.5
        
        # Sentiment Logic
        if sentiment < 20: score += 1
        elif sentiment > 80: score -= 1
        
        # Order Flow
        if cmf > 0.1: score += 1
        elif cmf < -0.1: score -= 1
        
        # Order Blocks (Bounce trading)
        if 0 <= (price - ob_bull) / price <= 0.01: score += 1.5
        elif 0 <= (ob_bear - price) / price <= 0.01: score -= 1.5
        
        # Short Squeeze Detector
        if z_score > 3 and rsi > 75 and volume > (vol_ma * 3): score -= 2.0
        
        # ADX Filter
        if adx > 25: score *= 1.1
        elif adx < 20: score *= 0.5
        
        # FVG
        if in_bull_fvg: score += 2.0
        elif in_bear_fvg: score -= 2.0
        
        # 11. Rising Momentum & Volume (Dominant Factor)
        volume_rising = vol_12h > vol_12h_prev
        if volume_rising:
            if momentum > 0 and momentum > momentum_prev:
                score += 2.5
            elif momentum < 0 and momentum < momentum_prev:
                score -= 2.5
        
        # Exit Conditions
        if position == 'LONG' and score < 1:
            pnl = (next_open - entry_price) / entry_price
            balance *= (1 + pnl)
            trades.append(pnl)
            trade_history.append({
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': df.index[i+1],
                'exit_price': next_open,
                'type': 'LONG'
            })
            position = None
        elif position == 'SHORT' and score > -1:
            pnl = (entry_price - next_open) / entry_price
            balance *= (1 + pnl)
            trades.append(pnl)
            trade_history.append({
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': df.index[i+1],
                'exit_price': next_open,
                'type': 'SHORT'
            })
            position = None
            
        # Track Drawdown
        if balance > peak_balance:
            peak_balance = balance
        drawdown = (peak_balance - balance) / peak_balance
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            
        # Entry Conditions
        if position is None and is_active_time:
            # Strong Buy Signal (Multiple factors align)
            if score >= 1.5:
                position = 'LONG'
                entry_price = next_open
                entry_date = df.index[i+1]
            # Strong Sell Signal (Trend Down + Overbought + High Vol)
            elif score <= -1.5:
                position = 'SHORT'
                entry_price = next_open
                entry_date = df.index[i+1]
                
    # Final PnL
    roi = ((balance - initial_balance) / initial_balance) * 100
    win_rate = (len([t for t in trades if t > 0]) / len(trades) * 100) if trades else 0
    
    gross_profit = sum([t for t in trades if t > 0])
    gross_loss = abs(sum([t for t in trades if t < 0]))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
    
    return {
        "final_balance": balance,
        "roi": roi,
        "total_trades": len(trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "history": trade_history
    }

def scan_and_rank_crypto():
    """
    Dynamically fetches the Top 50 Crypto assets (by 24h volume) and ranks them by Momentum.
    """
    exchange = ccxt.binance({'enableRateLimit': True})
    
    try:
        # Fetch all live tickers from Binance
        all_tickers = exchange.fetch_tickers()
        # Filter for active USDT pairs and extract quote volume
        usdt_pairs = [
            data for symbol, data in all_tickers.items() 
            if symbol.endswith('/USDT') and data.get('quoteVolume') is not None and data.get('active', True)
        ]
        # Sort by 24h quote volume to get the most liquid/top assets
        usdt_pairs.sort(key=lambda x: x['quoteVolume'], reverse=True)
        # Grab the top 50 symbols, plus our standard stocks
        tickers = [x['symbol'] for x in usdt_pairs[:50]] + ['SPY', 'QQQ', 'DIA', '^VIX', 'DX-Y.NYB', 'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AVGO', 'LLY', 'JPM']
    except Exception as e:
        st.error(f"Could not fetch dynamic tickers: {e}")
        tickers = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'SPY', 'QQQ', 'DIA', '^VIX', 'DX-Y.NYB', 'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AVGO', 'LLY', 'JPM'] # Fallback
        
    # --- Fetch Futures Metrics (Funding, Volume, OI) ---
    funding_rates = {}
    futures_24h_vol = {}
    futures_oi = {}
    swap_sym_map = {}
    exchange_swap = ccxt.binance({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})
    try:
        funding_rates_data = exchange_swap.fetch_funding_rates()
        swap_tickers_data = exchange_swap.fetch_tickers()
        
        try:
            oi_data = exchange_swap.fetch_open_interests()
        except Exception:
            oi_data = {}
            
        for swap_sym, data in funding_rates_data.items():
            # Convert 'BTC/USDT:USDT' to 'BTC/USDT' to match spot tickers
            spot_sym = swap_sym.split(':')[0]
            fr = data.get('fundingRate')
            funding_rates[spot_sym] = fr if fr is not None else 0.0
            swap_sym_map[spot_sym] = swap_sym
            
        for swap_sym, data in swap_tickers_data.items():
            spot_sym = swap_sym.split(':')[0]
            vol = data.get('quoteVolume')
            futures_24h_vol[spot_sym] = vol if vol is not None else 0.0
            
        for swap_sym, data in oi_data.items():
            spot_sym = swap_sym.split(':')[0]
            futures_oi[spot_sym] = data
    except Exception as e:
        st.warning(f"Could not fetch futures metrics: {e}")
        
    stats = []
    progress_bar = st.progress(0)
    total_tickers = len(tickers)
    
    def process_ticker(sym):
        df = fetch_and_analyze(sym, timeframe='1h', silent=True)
        if df is not None and not df.empty:
            mom = df['momentum'].iloc[-1]
            price = df['close'].iloc[-1]
            fr = funding_rates.get(sym, 0.0)
            vol_24h = futures_24h_vol.get(sym, 0.0)
            
            vol_12h = 0.0
            if sym in swap_sym_map:
                try:
                    ohlcv_1h = exchange_swap.fetch_ohlcv(swap_sym_map[sym], timeframe='1h', limit=12)
                    if ohlcv_1h:
                        vol_12h = sum(c[5] * c[4] for c in ohlcv_1h)
                except Exception:
                    pass
                    
            oi_info = futures_oi.get(sym, {})
            notional_oi = oi_info.get('openInterestValue')
            if notional_oi is None:
                base_oi = oi_info.get('openInterest', 0.0)
                notional_oi = base_oi * price
                
            trend = df['close'].tail(12).tolist()
            vol_profile = df['volume'].tail(12).tolist()
            return {
                'symbol': sym, 
                'momentum': mom, 
                'price': price, 
                'funding_rate': fr, 
                'notional_oi': notional_oi,
                'futures_12h_vol': vol_12h,
                'futures_24h_vol': vol_24h,
                '12h_trend': trend, 
                '1h_volume': vol_profile
            }
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_ticker, sym) for sym in tickers]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            if result:
                stats.append(result)
            # Update progress bar
            progress_bar.progress((i + 1) / total_tickers)
            
    # Sort descending
    stats.sort(key=lambda x: x['momentum'], reverse=True)
    
    return pd.DataFrame(stats)

def optimize_parameters(df, symbol, active_hours):
    """
    Runs a Grid Search to find the best RSI parameters.
    """
    rsi_lowers = [20, 25, 30, 35]
    rsi_uppers = [65, 70, 75, 80]
    results = []
    
    total_iterations = len(rsi_lowers) * len(rsi_uppers)
    progress_bar = st.progress(0)
    iteration = 0
    
    for lower in rsi_lowers:
        for upper in rsi_uppers:
            # Run backtest with specific params
            stats = backtest_strategy(df, symbol, start_hour=active_hours[0], end_hour=active_hours[1], rsi_lower=lower, rsi_upper=upper)
            stats['rsi_lower'] = lower
            stats['rsi_upper'] = upper
            results.append(stats)
            iteration += 1
            progress_bar.progress(iteration / total_iterations)
            
    return pd.DataFrame(results)

def color_metrics(val):
    if isinstance(val, (int, float)):
        color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
        return f'color: {color}'
    return ''

def format_large_number(x):
    try:
        if pd.isna(x):
            return "$0"
        x = float(x)
        is_negative = x < 0
        x = abs(x)
        
        if x >= 1e9:
            val = f"${x/1e9:.2f}B"
        elif x >= 1e6:
            val = f"${x/1e6:.2f}M"
        elif x >= 1e3:
            val = f"${x/1e3:.2f}K"
        else:
            val = f"${x:.2f}"
        return f"-{val}" if is_negative else val
    except (ValueError, TypeError):
        return "$0"

def main():
    st.title("Quantitative Scalping Dashboard (1h) 📈")
    
    # Sidebar
    if st.sidebar.button("🔄 Force Data Refresh"):
        st.cache_data.clear()
        st.sidebar.success("Cache cleared! Data will be fetched fresh on next action.")
        
    st.sidebar.header("Configuration")
    # Dropdown list for asset selection
    asset_options = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'DOGE/USDT', 'ADA/USDT', 
        'TRX/USDT', 'AVAX/USDT', 'SHIB/USDT', 'DOT/USDT', 'LINK/USDT', 'BCH/USDT', 'NEAR/USDT', 
        'LTC/USDT', 'MATIC/USDT', 'UNI/USDT', 'APT/USDT', 'ICP/USDT', 'FIL/USDT',
        'TON/USDT', 'XLM/USDT', 'ETC/USDT', 'XMR/USDT', 'OKB/USDT', 'ATOM/USDT', 'HBAR/USDT', 
        'VET/USDT', 'CRO/USDT', 'AR/USDT', 'MNT/USDT', 'OP/USDT', 'INJ/USDT', 'RNDR/USDT', 
        'GRT/USDT', 'IMX/USDT', 'STX/USDT', 'THETA/USDT', 'EGLD/USDT', 'FTM/USDT', 'ALGO/USDT', 
        'TIA/USDT', 'AAVE/USDT', 'FLOW/USDT', 'QNT/USDT', 'SNX/USDT',
        'SPY', 'QQQ', 'DIA', '^VIX', 'DX-Y.NYB', 'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AVGO', 'LLY', 'JPM'
    ]
    symbol = st.sidebar.selectbox("Ticker Symbol", options=asset_options)
    # Default backtest to 30 days for 1h timeframe to avoid huge data loads
    backtest_start = st.sidebar.date_input("Backtest Start", value=datetime.now() - timedelta(days=30))
    
    st.sidebar.header("Strategy Settings")
    
    # Dynamically adjust hours based on asset type
    us_stocks_indices = ['SPY', 'QQQ', 'DIA', '^VIX', 'DX-Y.NYB', 'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AVGO', 'LLY', 'JPM']
    is_stock = symbol in us_stocks_indices
    
    default_hours = (8, 21) if is_stock else (0, 23)
    label = "Active Hours (UTC) - Stocks" if is_stock else "Active Hours (UTC) - Crypto (24/7)"
    active_hours = st.sidebar.slider(label, 0, 23, default_hours)
    
    st.sidebar.header("Notifications")
    tg_token = st.sidebar.text_input("Telegram Bot Token", type="password")
    tg_chat_id = st.sidebar.text_input("Telegram Chat ID")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Overview", "⚡ Top Crypto Ranking", "🛠️ Backtest Engine", "🏛️ US Indices"])

    with tab1:
        st.subheader(f"Live Analysis: {symbol}")
        if st.button("Analyze Current Market"):
            df = fetch_and_analyze(symbol)
            if df is not None:
                sig = generate_signal(df, symbol, start_hour=active_hours[0], end_hour=active_hours[1])
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"${sig['price']:.2f}")
                col2.metric("Composite Score", f"{sig['score']}")
                col3.metric("Signal", sig['signal'])
                st.write(f"**Factors Driving Signal:** {', '.join(sig['factors'])}")
                
                # Send Alert if Strong Signal detected during analysis
                if ("BUY" in sig['signal'] or "SELL" in sig['signal']) and tg_token and tg_chat_id:
                    send_telegram_alert(tg_token, tg_chat_id, symbol, sig['signal'], sig['price'], sig['score'], sig['factors'])
                    st.success(f"Telegram Alert sent to {tg_chat_id}!")
            else:
                st.error(f"Could not load data for {symbol}. Please check the ticker.")

    with tab2:
        st.subheader("Top Crypto Momentum Ranking")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            scan_btn = st.button("Scan Top Crypto")
        with col2:
            auto_scan = st.checkbox("🔄 Auto-Scan (Refresh every 5 mins)")
            
        if scan_btn or auto_scan:
            with st.spinner("Scanning crypto markets..."):
                df_rank = scan_and_rank_crypto()
                
                styler = df_rank.style.format({
                    "momentum": "{:.2%}", 
                    "price": "${:.2f}", 
                    "funding_rate": "{:.4%}",
                    "notional_oi": format_large_number,
                    "futures_12h_vol": format_large_number,
                    "futures_24h_vol": format_large_number
                })
                
                if hasattr(styler, 'map'):
                    styler = styler.map(color_metrics, subset=['momentum', 'funding_rate'])
                else:
                    styler = styler.applymap(color_metrics, subset=['momentum', 'funding_rate'])
                    
                styler = styler.background_gradient(subset=['futures_12h_vol', 'futures_24h_vol'], cmap='Blues')

                st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                st.dataframe(
                    styler,
                    column_config={
                        "notional_oi": st.column_config.Column(
                            "Open Interest", 
                            help="Notional Open Interest (USDT)"
                        ),
                        "futures_12h_vol": st.column_config.Column(
                            "12H Futures Vol", 
                            help="Rolling 12-Hour Futures Trading Volume (USDT)"
                        ),
                        "futures_24h_vol": st.column_config.Column(
                            "24H Futures Vol", 
                            help="24-Hour Futures Trading Volume (USDT)"
                        ),
                        "12h_trend": st.column_config.LineChartColumn(
                            "12H Trend", help="Price movement over the last 12 hours"
                        ),
                        "1h_volume": st.column_config.BarChartColumn(
                            "1h Volume Profile", help="1-hour volume bars over the last 12 hours"
                        )
                    }
                )
                
            if auto_scan:
                time.sleep(301) # Wait slightly over 5m to clear the 300s cache TTL
                if hasattr(st, 'rerun'):
                    st.rerun()
                else:
                    st.experimental_rerun()
                        
        # --- 12-HOUR MICRO-MOMENTUM BREAKDOWN ---
        st.divider()
        st.subheader("🔍 12-Hour Micro-Momentum Breakdown")
        st.write("Breaks down the past 12 hours into 1-hour returns to spot building bullish or bearish pressure.")
        
        micro_options = st.session_state.get('scanned_tickers', asset_options)
        micro_idx = micro_options.index(symbol) if symbol in micro_options else 0
        micro_sym = st.selectbox("Select Asset to Analyze", options=micro_options, index=micro_idx, key='micro_sym')
        
        micro_df = fetch_and_analyze(micro_sym, timeframe='1h', silent=True)
        if micro_df is not None and len(micro_df) >= 13:
            # Calculate 1h returns before slicing so the first candle has a valid return
            micro_df['1h_return'] = micro_df['close'].pct_change() * 100
            last_12h = micro_df.tail(12).copy()
            
            # Format index for better chart display (e.g., just the time '14:35')
            last_12h.index = last_12h.index.strftime('%H:%M')
            
            # Plotting bar chart of 1h returns
            st.bar_chart(last_12h['1h_return'])
            
            # Summary metrics
            bullish_candles = (last_12h['1h_return'] > 0).sum()
            bearish_candles = (last_12h['1h_return'] < 0).sum()
            net_12h_return = ((micro_df['close'].iloc[-1] - micro_df['close'].iloc[-13]) / micro_df['close'].iloc[-13]) * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Net 12H Price Change", f"{net_12h_return:.2f}%")
            c2.metric("Bullish 1h Candles", int(bullish_candles))
            c3.metric("Bearish 1h Candles", int(bearish_candles))
            
            # Momentum Verdict
            if net_12h_return > 0 and bullish_candles > bearish_candles:
                st.success(f"**Verdict:** Momentum is steadily building **BULLISH** 📈")
            elif net_12h_return < 0 and bearish_candles > bullish_candles:
                st.error(f"**Verdict:** Momentum is steadily building **BEARISH** 📉")
            else:
                st.warning(f"**Verdict:** Momentum is currently **MIXED / CONSOLIDATING** ⚖️")

    with tab3:
        st.subheader(f"Strategy Backtest: {symbol}")
        if st.button("Run Backtest"):
            df = fetch_and_analyze(symbol, start_date=backtest_start.strftime('%Y-%m-%d'))
            if df is not None:
                with st.spinner("Simulating strategy..."):
                    stats = backtest_strategy(df, symbol, start_hour=active_hours[0], end_hour=active_hours[1])
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Final Balance", f"${stats['final_balance']:.2f}")
                    m2.metric("ROI", f"{stats['roi']:.2f}%")
                    m3.metric("Profit Factor", f"{stats['profit_factor']:.2f}")
                    m4.metric("Win Rate", f"{stats['win_rate']:.2f}%")
                    
                    fig = plot_backtest(df, stats['history'], symbol)
                    st.pyplot(fig)
            else:
                st.error(f"Could not load backtest data for {symbol}.")

        st.divider()
        st.subheader("🤖 AI Parameter Optimizer")
        st.write("Automatically find the best RSI thresholds for this asset and timeframe.")
        
        if st.button("Run Optimization Loop"):
            df = fetch_and_analyze(symbol, start_date=backtest_start.strftime('%Y-%m-%d'))
            if df is not None:
                with st.spinner("Testing 16 parameter combinations..."):
                    results_df = optimize_parameters(df, symbol, active_hours)
                    
                    # Find best result by ROI
                    best = results_df.sort_values(by='roi', ascending=False).iloc[0]
                    
                    st.success(f"💎 Best Parameters Found: RSI < {best['rsi_lower']} and RSI > {best['rsi_upper']}")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Optimized ROI", f"{best['roi']:.2f}%")
                    col2.metric("Win Rate", f"{best['win_rate']:.2f}%")
                    col3.metric("Total Trades", best['total_trades'])
                    
                    top_results = results_df[['rsi_lower', 'rsi_upper', 'roi', 'win_rate', 'profit_factor']].sort_values(by='roi', ascending=False).head(5)
                    styler = top_results.style.format({
                        "roi": "{:.2f}%",
                        "win_rate": "{:.2f}%",
                        "profit_factor": "{:.2f}"
                    })
                    
                    if hasattr(styler, 'map'):
                        styler = styler.map(color_metrics, subset=['roi'])
                    else:
                        styler = styler.applymap(color_metrics, subset=['roi'])
                        
                    st.dataframe(styler)

    with tab4:
        st.subheader("🏛️ Top US Indices & VIX Overview")
        st.write("Tracking S&P 500 (SPY), Nasdaq 100 (QQQ), Dow Jones (DIA), Volatility Index (^VIX), and US Dollar Index (DX-Y.NYB).")
        
        indices = ['SPY', 'QQQ', 'DIA', '^VIX', 'DX-Y.NYB']
        index_stats = []
        
        if st.button("Refresh Indices Data"):
            with st.spinner("Fetching US Indices..."):
                for sym in indices:
                    df_idx = fetch_and_analyze(sym, timeframe='1h', silent=True)
                    if df_idx is not None and not df_idx.empty:
                        current = df_idx.iloc[-1]
                        # Estimate daily volume (last 7 1-hour bars = 7 trading hours)
                        est_vol = df_idx['volume'].tail(7).sum()
                        
                        index_stats.append({
                            "Symbol": sym,
                            "Price": current['close'],
                            "Momentum": current['momentum'],
                            "RSI": current['rsi'],
                            "Trend": "Bullish 🟢" if current['close'] > current['ema_50'] else "Bearish 🔴",
                            "Est. Daily Volume": est_vol,
                            "Z-Score": current['z_score']
                        })
                        
                if index_stats:
                    df_ind = pd.DataFrame(index_stats)
                    styler = df_ind.style.format({
                        "Price": "${:.2f}",
                        "Momentum": "{:.2%}",
                        "RSI": "{:.2f}",
                        "Est. Daily Volume": format_large_number,
                        "Z-Score": "{:.2f}"
                    })
                    
                    if hasattr(styler, 'map'):
                        styler = styler.map(color_metrics, subset=['Momentum', 'Z-Score'])
                    else:
                        styler = styler.applymap(color_metrics, subset=['Momentum', 'Z-Score'])
                        
                    st.dataframe(styler, use_container_width=True)
        else:
            st.info("Click 'Refresh Indices Data' to load the latest metrics for US Markets.")
            
        st.divider()
        st.subheader("🏢 Top 10 US Stocks Overview")
        st.write("Tracking the top 10 US companies by market cap.")
        
        top_stocks = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AVGO', 'LLY', 'JPM']
        stock_stats = []
        
        if st.button("Refresh Stocks Data"):
            with st.spinner("Fetching US Stocks..."):
                bullish_trends = 0
                positive_momentum = 0
                valid_stocks = 0
                
                for sym in top_stocks:
                    df_stock = fetch_and_analyze(sym, timeframe='1h', silent=True)
                    if df_stock is not None and not df_stock.empty:
                        current = df_stock.iloc[-1]
                        est_vol = df_stock['volume'].tail(7).sum()
                        
                        is_bullish = current['close'] > current['ema_50']
                        if is_bullish: bullish_trends += 1
                        if current['momentum'] > 0: positive_momentum += 1
                        valid_stocks += 1
                        
                        stock_stats.append({
                            "Symbol": sym,
                            "Price": current['close'],
                            "Momentum": current['momentum'],
                            "RSI": current['rsi'],
                            "Trend": "Bullish 🟢" if is_bullish else "Bearish 🔴",
                            "Est. Daily Volume": est_vol,
                            "Z-Score": current['z_score']
                        })
                        
                if stock_stats:
                    # --- Market Health Score UI ---
                    health_score = (bullish_trends / valid_stocks) * 100
                    health_status = "🟢 STRONG BULL" if health_score >= 70 else "🔴 BEARISH" if health_score <= 30 else "🟡 NEUTRAL"
                    
                    hc1, hc2, hc3 = st.columns(3)
                    hc1.metric("Stocks in Bullish Trend", f"{bullish_trends} / {valid_stocks}")
                    hc2.metric("Positive Momentum", f"{positive_momentum} / {valid_stocks}")
                    hc3.metric("Overall Health", health_status)
                    
                    st.write("") # Spacer
                    
                    df_stocks = pd.DataFrame(stock_stats)
                    styler_stocks = df_stocks.style.format({
                        "Price": "${:.2f}",
                        "Momentum": "{:.2%}",
                        "RSI": "{:.2f}",
                        "Est. Daily Volume": format_large_number,
                        "Z-Score": "{:.2f}"
                    })
                    
                    if hasattr(styler_stocks, 'map'):
                        styler_stocks = styler_stocks.map(color_metrics, subset=['Momentum', 'Z-Score'])
                    else:
                        styler_stocks = styler_stocks.applymap(color_metrics, subset=['Momentum', 'Z-Score'])
                        
                    st.dataframe(styler_stocks, use_container_width=True)
        else:
            st.info("Click 'Refresh Stocks Data' to load the latest metrics for Top US Stocks.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
