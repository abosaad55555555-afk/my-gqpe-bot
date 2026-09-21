import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# Configure Streamlit page architecture to dark wide layout
st.set_page_config(page_title="GQPE Institutional Multi-Asset Execution Desk", layout="wide", initial_sidebar_state="expanded")

# Sidebar Control Panel for Dynamic Ticker Input
st.sidebar.markdown("## ⚙️ Execution Parameters")
user_ticker = st.sidebar.text_input("Enter Stock Ticker", value="MSFT").upper().strip()
kelly_fraction = st.sidebar.slider("Kelly Risk Fraction", min_value=0.05, max_value=0.50, value=0.15, step=0.05)

# 1. QUANTITATIVE BLACK-SCHOLES PRICING ENGINE
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
    return price

# 2. ROBUST LIVE DATA INGESTION & FEATURE ENGINEERING
@st.cache_data(ttl=1800)
def get_historical_market_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo", interval="1d")
    
    if df is None or df.empty:
        return pd.DataFrame()
        
    df.index = pd.to_datetime(df.index).normalize()
    
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    
    cleaned_df = df.dropna()
    return cleaned_df.tail(60)

# 3. STATISTICAL PROBABILITY ENGINE
def compute_gqpe_probability(row, prev_row):
    price_vs_ema = (row['Close'] - row['EMA_20']) / (row['EMA_20'] + 1e-9)
    momentum_factor = (row['Close'] - prev_row['Close']) / (prev_row['Close'] + 1e-9)
    
    z = (price_vs_ema * 8.0) + (momentum_factor * 12.0)
    return 1.0 / (1.0 + np.exp(-z))

# 4. OPTIMIZED INSTITUTIONAL BACKTESTING SIMULATOR
def run_institutional_simulation(df, ticker, kelly_fraction=0.15, risk_free_rate=0.045):
    if df is None or len(df) < 2:
        raise ValueError(f"Insufficient historical data retrieved for ticker: {ticker}.")

    capital = 10000.00
    log = []
    dates = df.index
    
    for i in range(1, len(dates)):
        current_date = dates[i]
        prev_date = dates[i-1]
        
        current_row = df.loc[current_date]
        prev_row = df.loc[prev_date]
        
        p_y = compute_gqpe_probability(current_row, prev_row)
        
        # فلتر الأمان للحد من التداول العشوائي
        if 0.46 <= p_y <= 0.54:
            log.append({
                "Date": current_date.strftime('%Y-%m-%d'),
                "Asset": ticker,
                "Probability": round(p_y, 4),
                "Action": "Hold / Cash (Filter Active)",
                "Volatility": round(max(current_row['Volatility_20'], 0.10) * 100, 2),
                "Trade Return (%)": 0.0,
                "Portfolio Equity ($)": round(capital, 2)
            })
            continue

        open_price = current_row['Open']
        close_price = current_row['Close']
        volatility = max(current_row['Volatility_20'], 0.10)
        
        strike_price = open_price 
        time_to_expiry = 30.0 / 252.0  
        
        if p_y > 0.54:
            action = f"Buy Call ({ticker})"
            opt_price_open = black_scholes_price(open_price, strike_price, time_to_expiry, risk_free_rate, volatility, "call")
            opt_price_close = black_scholes_price(close_price, strike_price, time_to_expiry - (1.0/252.0), risk_free_rate, volatility, "call")
        else:
            action = f"Buy Put ({ticker})"
            opt_price_open = black_scholes_price(open_price, strike_price, time_to_expiry, risk_free_rate, volatility, "put")
            opt_price_close = black_scholes_price(close_price, strike_price, time_to_expiry - (1.0/252.0), risk_free_rate, volatility, "put")
            
        if opt_price_open <= 0:
            continue
            
        option_return = (opt_price_close - opt_price_open) / opt_price_open
        friction_cost = 0.005  
        net_option_return = option_return - friction_cost
        
        allocated_capital = capital * kelly_fraction
        cash_buffer = capital * (1.0 - kelly_fraction)
        
        allocated_capital *= (1.0 + net_option_return)
        capital = cash_buffer + allocated_capital
        
        log.append({
            "Date": current_date.strftime('%Y-%m-%d'),
            "Asset": ticker,
            "Probability": round(p_y, 4),
            "Action": action,
            "Volatility": round(volatility * 100, 2),
            "Trade Return (%)": round(net_option_return * 100, 2),
            "Portfolio Equity ($)": round(capital, 2)
        })
        
    if not log:
        raise ValueError("Simulation log is empty.")
        
    return pd.DataFrame(log)

# 5. EXECUTION PIPELINE INTEGRATION
try:
    if not user_ticker:
        st.warning("Please enter a valid stock ticker in the sidebar.")
        st.stop()
        
    df_market = get_historical_market_data(user_ticker)
    if df_market.empty:
        st.error(f"[-] No market data found for symbol: {user_ticker}. Please verify the ticker symbol.")
        st.stop()
        
    results_df = run_institutional_simulation(df_market, user_ticker, kelly_fraction)
    latest_state = results_df.iloc[-1]
    net_roi = ((latest_state['Portfolio Equity ($)'] - 10000.0) / 10000.0) * 100
    
    # STREAMLIT VISUAL DASHBOARD PANEL
    st.markdown(f"<h1 style='text-align: center; color: white;'>🏛️ GQPE Institutional Execution Desk</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #9ca3af;'>Black-Scholes Filtered Options Engine — Asset: <b>{user_ticker}</b></p>", unsafe_allow_html=True)
    st.divider()
    
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Net Portfolio Equity", f"${latest_state['Portfolio Equity ($)']:,}")
    kpi2.metric("Initial Baseline Capital", "$10,000.00")
    kpi3.metric("Strategy Alpha ROI", f"{net_roi:+.2f}%", f"Filtered Delta ({user_ticker})")
    
    st.divider()
    
    if "Call" in latest_state['Action']:
        st.success(f"🚀 LIVE SIGNAL: **{latest_state['Asset']}** | Execution: **{latest_state['Action']}** | Model Probability: **{latest_state['Probability']}**")
    elif "Put" in latest_state['Action']:
        st.error(f"🔴 LIVE SIGNAL: **{latest_state['Asset']}** | Execution: **{latest_state['Action']}** | Model Probability: **{latest_state['Probability']}**")
    else:
        st.info(f"⏳ LIVE SIGNAL: **{latest_state['Asset']}** | Execution: **{latest_state['Action']}** (Market in Neutral Zone)")
        
    m1, m2, m3 = st.columns(3)
    m1.write(f"**Valuation Model:** Black-Scholes (30-Day Expiry)")
    m2.write(f"**Dynamic Annualized Volatility:** {latest_state['Volatility']}%")
    m3.write(f"**Execution Risk Profile:** Kelly Fraction ({int(kelly_fraction*100)}%)")
    
    st.divider()
    
    st.subheader(f"📈 Institutional Equity Growth Curve ({user_ticker})")
    st.line_chart(data=results_df, x="Date", y="Portfolio Equity ($)", use_container_width=True)
    
    st.divider()
    
    st.subheader(f"📋 Execution & Pricing Audit Ledger ({user_ticker})")
    st.dataframe(results_df.iloc[::-1], use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"[-] Execution Pipeline Failure: {str(e)}")