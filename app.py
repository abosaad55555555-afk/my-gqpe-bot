import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Configure Streamlit page architecture to dark wide layout
st.set_page_config(page_title="GQPE Multi-Stock Rotational Execution Desk", layout="wide", initial_sidebar_state="collapsed")

# 1. MULTI-STOCK LIVE DATA PIPELINE INGESTION NODE
@st.cache_data(ttl=1800)
def get_historical_market_data(tickers=["MSFT", "NVDA", "AAPL"]):
    combined_data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        df = stock.history(period="3mo", interval="1d") # توسيع الفترة لضمان توفر بيانات كافية
        
        if df is None or df.empty:
            continue
            
        # إزالة المنطقة الزمنية من الفهرس لتجنب مشاكل التطابق
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
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
        if not cleaned_df.empty:
            combined_data[ticker] = cleaned_df.tail(30)
            
    return combined_data

# 2. BALANCED INTRADAY PROBABILITY ENGINE (BIDIRECTIONAL ENABLED)
def compute_gqpe_probability(row, prev_row):
    price_vs_ema = (row['Close'] - row['EMA_20']) / (row['EMA_20'] + 1e-9)
    momentum_factor = (row['Close'] - prev_row['Close']) / (prev_row['Close'] + 1e-9)
    
    z = (price_vs_ema * 10.0) + (momentum_factor * 15.0)
    return 1.0 / (1.0 + np.exp(-z))

# 3. ROBUST ROTATIONAL MULTI-STOCK SIMULATOR
def run_rotational_simulation(market_data_dict, kelly_fraction=0.50):
    tickers = list(market_data_dict.keys())
    if not tickers:
        raise ValueError("No market data retrieved for any ticker.")
        
    all_dates = set()
    for t in tickers:
        all_dates.update(market_data_dict[t].index)
    sorted_dates = sorted(list(all_dates))
    
    if len(sorted_dates) < 2:
        raise ValueError("Insufficient historical dates found.")

    capital = 1000.00
    log = []
    
    for i in range(1, len(sorted_dates)):
        current_date = sorted_dates[i]
        prev_date = sorted_dates[i-1]
        
        best_ticker = None
        best_p_y = -1
        best_row = None
        
        for ticker in tickers:
            df = market_data_dict[ticker]
            if current_date in df.index and prev_date in df.index:
                current_row = df.loc[current_date]
                prev_row = df.loc[prev_date]
                p_y = compute_gqpe_probability(current_row, prev_row)
                
                if abs(p_y - 0.5) > abs(best_p_y - 0.5):
                    best_p_y = p_y
                    best_ticker = ticker
                    best_row = current_row
                    
        if best_ticker is None or best_row is None:
            continue
                
        open_to_close_ret = (best_row['Close'] - best_row['Open']) / (best_row['Open'] + 1e-9)
        friction_decay = 0.02
        
        allocated_capital = capital * kelly_fraction
        cash_buffer = capital * (1.0 - kelly_fraction)
        
        if best_p_y >= 0.50:
            action = f"Buy Call ({best_ticker})"
            option_return = (open_to_close_ret * 20.0) - friction_decay 
        else:
            action = f"Buy Put ({best_ticker})"
            option_return = (-open_to_close_ret * 20.0) - friction_decay
            
        allocated_capital *= (1.0 + option_return)
        capital = cash_buffer + allocated_capital
        
        log.append({
            "Date": current_date.strftime('%Y-%m-%d'),
            "Selected Asset": best_ticker,
            "Probability": round(best_p_y, 4),
            "Action": action,
            "Trade Return (%)": round(option_return * 100, 2),
            "Equity ($)": round(capital, 2)
        })
        
    if not log:
        raise ValueError("Simulation log is empty after union processing.")
        
    return pd.DataFrame(log)

# Initialize pipeline execution tracking
try:
    tickers_list = ["MSFT", "NVDA", "AAPL"]
    market_dict = get_historical_market_data(tickers_list)
    results_df = run_rotational_simulation(market_dict)
    latest_state = results_df.iloc[-1]
    
    # 4. STREAMLIT VISUAL METRIC DASHBOARD PANEL
    st.markdown("<h1 style='text-align: center; color: white;'>📊 GQPE 3-Stock Rotational Execution Desk</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #9ca3af;'>Dynamic Multi-Asset Selection & Compound Control Matrix</p>", unsafe_allow_html=True)
    st.divider()
    
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Net Portfolio Equity", f"${latest_state['Equity ($)']:,}")
    kpi2.metric("Initial Capital", "$1,000.00")
    kpi3.metric("Rotational Net ROI", f"+{(latest_state['Equity ($)'] - 1000) / 10:.2f}%", "Multi-Asset Alpha Velocity")
    
    st.divider()
    
    if latest_state['Probability'] >= 0.50:
        st.success(f"🚀 TOP ROTATIONAL PICK: **{latest_state['Selected Asset']}** | Action: **{latest_state['Action']}** (Prob: {latest_state['Probability']})")
    else:
        st.error(f"🔴 TOP ROTATIONAL PICK: **{latest_state['Selected Asset']}** | Action: **{latest_state['Action']}** (Prob: {latest_state['Probability']})")
        
    m1, m2, m3 = st.columns(3)
    m1.write(f"**Active Asset Pool:** MSFT, NVDA, AAPL")
    m2.write(f"**Allocated Premium Target:** ${round(latest_state['Equity ($)'] * 0.50, 2)} USD")
    m3.write(f"**Session Strategy:** Bidirectional Daily Asset Rotation")
    
    st.divider()
    
    st.subheader("📈 Multi-Asset Rotational Capital Growth Curve")
    st.line_chart(data=results_df, x="Date", y="Equity ($)", use_container_width=True)
    
    st.divider()
    
    st.subheader("📋 3-Stock Rotation Strategy Backtest Ledger")
    st.dataframe(results_df.iloc[::-1], use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"[-] Execution Pipeline Failure: {str(e)}")