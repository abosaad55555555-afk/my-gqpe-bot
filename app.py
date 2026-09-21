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
        df = stock.history(period="2mo", interval="1d") # Fetch past 60 days
        
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
        
        combined_data[ticker] = df.dropna().tail(21)
    return combined_data

# 2. INTRADAY PROBABILITY ENGINE
def compute_gqpe_probability(row, prev_row):
    alpha_rt = 0.08 if row['Close'] > row['EMA_20'] else -0.06
    phi_ot = np.tanh(1.2) * 0.12 
    gap_pct = (row['Open'] - prev_row['Close']) / (prev_row['Close'] + 1e-9)
    atr_bounds = np.abs(row['Open'] - prev_row['Close']) / (row['ATR_20'] + 1e-9)
    
    if atr_bounds > 2.5: return 0.5 # Avert outlier gap traps
    
    lambda_pre = gap_pct * 3.5 
    rsi_penalty = 1.0 / (1.0 + np.exp((row['RSI_14'] - 70) / 4.0))
    t_t = rsi_penalty * np.exp(-np.abs(row['Open'] - row['EMA_20']) / (row['Close'] * 0.05))
    
    z = alpha_rt + phi_ot + lambda_pre + (t_t * 0.1)
    return 1.0 / (1.0 + np.exp(-z))

# 3. ROTATIONAL MULTI-STOCK SIMULATOR
def run_rotational_simulation(market_data_dict, kelly_fraction=0.50):
    tickers = list(market_data_dict.keys())
    # Align dates across all tickers
    common_dates = market_data_dict[tickers[0]].index
    for t in tickers[1:]:
        common_dates = common_dates.intersection(market_data_dict[t].index)
    
    capital = 1000.00
    log = []
    
    for i in range(1, len(common_dates)):
        current_date = common_dates[i]
        prev_date = common_dates[i-1]
        
        best_ticker = None
        best_p_y = -1
        best_row = None
        best_prev_row = None
        
        # Scan all 3 stocks to find the highest probability asset for the day
        for ticker in tickers:
            df = market_data_dict[ticker]
            current_row = df.loc[current_date]
            prev_row = df.loc[prev_date]
            p_y = compute_gqpe_probability(current_row, prev_row)
            
            if p_y > best_p_y:
                best_p_y = p_y
                best_ticker = ticker
                best_row = current_row
                best_prev_row = prev_row
                
        # Execute trade on the top-ranked stock of the day
        open_to_close_ret = (best_row['Close'] - best_row['Open']) / best_row['Open']
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
    
    # Render key KPI tracking metrics
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Net Portfolio Equity", f"${latest_state['Equity ($)']:,}")
    kpi2.metric("Initial Capital", "$1,000.00")
    kpi3.metric("Rotational Net ROI", f"+{(latest_state['Equity ($)'] - 1000) / 10:.2f}%", "Multi-Asset Alpha Velocity")
    
    st.divider()
    
    # Render Active Live Trigger Ticket
    st.success(f"🚀 TOP ROTATIONAL PICK: **{latest_state['Selected Asset']}** | Action: **{latest_state['Action']}** (Prob: {latest_state['Probability']})")
        
    # Micro asset allocation matrix stats
    m1, m2, m3 = st.columns(3)
    m1.write(f"**Active Asset Pool:** MSFT, NVDA, AAPL")
    m2.write(f"**Allocated Premium Target:** ${round(latest_state['Equity ($)'] * 0.50, 2)} USD")
    m3.write(f"**Session Strategy:** Dynamic Daily Asset Rotation")
    
    st.divider()
    
    # Render Parabolic Line Curve
    st.subheader("📈 Multi-Asset Rotational Capital Growth Curve")
    st.line_chart(data=results_df, x="Date", y="Equity ($)", use_container_width=True)
    
    st.divider()
    
    # Output comprehensive validation matrix tables ledger
    st.subheader("📋 3-Stock Rotation Strategy Backtest Ledger")
    st.dataframe(results_df.iloc[::-1], use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"[-] Execution Pipeline Failure: {str(e)}")