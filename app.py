import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Configure Streamlit page architecture to dark wide layout
st.set_page_config(page_title="GQPE MSFT Single-Asset Execution Desk", layout="wide", initial_sidebar_state="collapsed")

# 1. MSFT LIVE DATA PIPELINE INGESTION NODE
@st.cache_data(ttl=1800)
def get_historical_market_data(ticker="MSFT"):
    stock = yf.Ticker(ticker)
    df = stock.history(period="3mo", interval="1d")
    
    if df is None or df.empty:
        return pd.DataFrame()
        
    # تنظيف وتحويل الفهرس إلى تواريخ صافية بدون أوقات أو مناطق زمنية
    df.index = pd.to_datetime(df.index).normalize()
    
    # Structural Filters
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    high_low = df['High'] - df['Low']
    high_cp = np.abs(df['High'] - df['Close'].shift())
    low_cp = np.abs(df['Low'] - df['Close'].shift())
    df['ATR_20'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1).rolling(20).mean()
    
    cleaned_df = df.dropna()
    return cleaned_df.tail(40)

# 2. BALANCED INTRADAY PROBABILITY ENGINE (BIDIRECTIONAL ENABLED)
def compute_gqpe_probability(row, prev_row):
    price_vs_ema = (row['Close'] - row['EMA_20']) / (row['EMA_20'] + 1e-9)
    momentum_factor = (row['Close'] - prev_row['Close']) / (prev_row['Close'] + 1e-9)
    
    z = (price_vs_ema * 10.0) + (momentum_factor * 15.0)
    return 1.0 / (1.0 + np.exp(-z))

# 3. MSFT SINGLE-ASSET SIMULATOR
def run_msft_simulation(df, kelly_fraction=0.50):
    if df is None or len(df) < 2:
        raise ValueError("Insufficient historical data retrieved for MSFT.")

    capital = 1000.00
    log = []
    dates = df.index
    
    for i in range(1, len(dates)):
        current_date = dates[i]
        prev_date = dates[i-1]
        
        current_row = df.loc[current_date]
        prev_row = df.loc[prev_date]
        
        p_y = compute_gqpe_probability(current_row, prev_row)
        
        open_to_close_ret = (current_row['Close'] - current_row['Open']) / (current_row['Open'] + 1e-9)
        friction_decay = 0.02
        
        allocated_capital = capital * kelly_fraction
        cash_buffer = capital * (1.0 - kelly_fraction)
        
        if p_y >= 0.50:
            action = "Buy Call (MSFT)"
            option_return = (open_to_close_ret * 20.0) - friction_decay 
        else:
            action = "Buy Put (MSFT)"
            option_return = (-open_to_close_ret * 20.0) - friction_decay
            
        allocated_capital *= (1.0 + option_return)
        capital = cash_buffer + allocated_capital
        
        log.append({
            "Date": current_date.strftime('%Y-%m-%d'),
            "Selected Asset": "MSFT",
            "Probability": round(p_y, 4),
            "Action": action,
            "Trade Return (%)": round(option_return * 100, 2),
            "Equity ($)": round(capital, 2)
        })
        
    if not log:
        raise ValueError("Simulation log is empty.")
        
    return pd.DataFrame(log)

# Initialize pipeline execution tracking
try:
    df_market = get_historical_market_data("MSFT")
    results_df = run_msft_simulation(df_market)
    latest_state = results_df.iloc[-1]
    
    # 4. STREAMLIT VISUAL METRIC DASHBOARD PANEL
    st.markdown("<h1 style='text-align: center; color: white;'>📊 GQPE MSFT Single-Asset Execution Desk</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #9ca3af;'>Dynamic Intraday Execution & Compound Control Matrix for Microsoft</p>", unsafe_allow_html=True)
    st.divider()
    
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Net Portfolio Equity", f"${latest_state['Equity ($)']:,}")
    kpi2.metric("Initial Capital", "$1,000.00")
    kpi3.metric("Strategy Net ROI", f"+{(latest_state['Equity ($)'] - 1000) / 10:.2f}%", "MSFT Alpha Velocity")
    
    st.divider()
    
    if latest_state['Probability'] >= 0.50:
        st.success(f"🚀 LATEST SIGNAL: **{latest_state['Selected Asset']}** | Action: **{latest_state['Action']}** (Prob: {latest_state['Probability']})")
    else:
        st.error(f"🔴 LATEST SIGNAL: **{latest_state['Selected Asset']}** | Action: **{latest_state['Action']}** (Prob: {latest_state['Probability']})")
        
    m1, m2, m3 = st.columns(3)
    m1.write(f"**Active Asset:** Microsoft Corporation (MSFT)")
    m2.write(f"**Allocated Premium Target:** ${round(latest_state['Equity ($)'] * 0.50, 2)} USD")
    m3.write(f"**Session Strategy:** Bidirectional Daily Directional")
    
    st.divider()
    
    st.subheader("📈 MSFT Strategy Capital Growth Curve")
    st.line_chart(data=results_df, x="Date", y="Equity ($)", use_container_width=True)
    
    st.divider()
    
    st.subheader("📋 MSFT Execution Backtest Ledger")
    st.dataframe(results_df.iloc[::-1], use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"[-] Execution Pipeline Failure: {str(e)}")