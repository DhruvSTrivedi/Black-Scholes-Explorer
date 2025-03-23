import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(
    page_title="Black-Scholes Pro Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: 'Courier New', monospace;
}
h1, h2, h3, h4 {
    color: #58a6ff;
    font-weight: bold;
}
.stSidebar {
    background-color: #161b22;
    border-right: 2px solid #30363d;
}
.metric-box {
    background-color: #161b22;
    padding: 15px;
    border: 1px solid #30363d;
    border-radius: 8px;
    text-align: center;
}
.option-card {
    background: linear-gradient(135deg, #0d1117, #1f6feb);
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 20px;
    color: #ffffff;
    text-align: center;
}
.stNumberInput input, .stSlider > div > div > div > div {
    background-color: #21262d;
    color: #c9d1d9 !important;
    border-radius: 4px;
    border: 1px solid #30363d;
}
</style>
""", unsafe_allow_html=True)

# ⚙️ Black-Scholes + Delta
def black_scholes(S, K, T, r, sigma, option_type="call"):
    try:
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        elif option_type == "put":
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1
        return price, delta
    except:
        return 0.0, 0.0

# 🎲 Monte Carlo Simulation
def monte_carlo(S, K, T, r, sigma, option_type="call", simulations=10000):
    np.random.seed(42)  # For reproducibility
    dt = T
    Z = np.random.standard_normal(simulations)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return price

# 📊 Heatmap Generator
def generate_heatmap(S, K, T, r, sigma, spot_range, vol_range, option_type="call"):
    prices = np.zeros((len(vol_range), len(spot_range)))
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            prices[i, j], _ = black_scholes(spot, K, T, r, vol, option_type)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="none")
    sns.heatmap(
        prices,
        xticklabels=np.round(spot_range, 1)[::2],
        yticklabels=np.round(vol_range, 2)[::2],
        annot=True,
        fmt=".2f",
        cmap="viridis",
        ax=ax,
        cbar_kws={'label': f"{option_type.upper()} Price"},
        annot_kws={"size": 8, "color": "#e0e0e0"}
    )
    ax.set_title(f"{option_type.capitalize()} Price Heatmap", color="#58a6ff", fontsize=14)
    ax.set_xlabel("Spot Price", color="#c9d1d9")
    ax.set_ylabel("Volatility", color="#c9d1d9")
    ax.tick_params(colors="#c9d1d9")
    fig.patch.set_alpha(0)
    cbar = ax.collections[0].colorbar
    cbar.set_label(f"{option_type.upper()} Price", color="#c9d1d9")
    plt.setp(cbar.ax.get_yticklabels(), color="#c9d1d9")
    return fig

# 📡 Fetch Real-Time Data
def fetch_yahoo_data(ticker):
    stock = yf.Ticker(ticker)
    try:
        hist = stock.history(period="1d")
        S = hist["Close"].iloc[-1]
        # Fetch implied volatility from options (approximation using last 30 days)
        options = stock.option_chain(stock.options[0]) if stock.options else None
        sigma = stock.history(period="30d")["Close"].pct_change().std() * np.sqrt(252) if not options else 0.2
        return S, sigma
    except:
        return None, None

# 🧠 Sidebar
with st.sidebar:
    st.markdown("<h3>Enhanced by Grok</h3>", unsafe_allow_html=True)
    st.markdown("[Powered by xAI](#)", unsafe_allow_html=True)
    st.markdown("---")
    
    ticker = st.text_input("Ticker Symbol (e.g., AAPL)", "AAPL")
    if st.button("Fetch Real-Time Data"):
        S_live, sigma_live = fetch_yahoo_data(ticker)
        if S_live:
            st.session_state["S"] = S_live
            st.session_state["sigma"] = sigma_live
        else:
            st.error("Failed to fetch data. Using defaults.")

    st.subheader("📈 Parameters")
    S = st.number_input("Asset Price", 1.0, 1000.0, st.session_state.get("S", 100.0), 1.0)
    K = st.number_input("Strike Price", 1.0, 1000.0, 100.0, 1.0)
    T = st.number_input("Time (Years)", 0.01, 10.0, 1.0, 0.01)
    sigma = st.number_input("Volatility (σ)", 0.01, 1.0, st.session_state.get("sigma", 0.2), 0.01)
    r = st.number_input("Risk-Free Rate", 0.0, 1.0, 0.05, 0.01)

    st.subheader("🎯 Heatmap Settings")
    spot_min = st.number_input("Min Spot", value=S * 0.8, step=1.0)
    spot_max = st.number_input("Max Spot", value=S * 1.2, step=1.0)
    vol_min = st.slider("Min Volatility", 0.01, 1.0, sigma * 0.5, 0.01)
    vol_max = st.slider("Max Volatility", 0.01, 1.0, sigma * 1.5, 0.01)
    grid_size = st.slider("Heatmap Grid Size", 5, 20, 10, 1)

    st.subheader("🎲 Monte Carlo")
    sims = st.number_input("Simulations", 1000, 100000, 10000, 1000)

# 🧪 Main Interface with Tabs
st.title("📈 Black-Scholes Pro Terminal")
st.markdown("Real-time pricing, Greeks, and Monte Carlo simulations.")

tabs = st.tabs(["Black-Scholes", "Monte Carlo"])

# Tab 1: Black-Scholes
with tabs[0]:
    st.subheader("🔢 Current Inputs")
    cols = st.columns(5)
    metrics = [("Asset Price", S), ("Strike", K), ("Time", T), ("Volatility", sigma), ("Rate", r)]
    for col, (label, value) in zip(cols, metrics):
        col.markdown(f"<div class='metric-box'><h4>{label}</h4><p>{value:.2f}</p></div>", unsafe_allow_html=True)

    call_price, call_delta = black_scholes(S, K, T, r, sigma, "call")
    put_price, put_delta = black_scholes(S, K, T, r, sigma, "put")
    st.subheader("📊 Option Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='option-card'><h3>CALL</h3><h2>${call_price:.2f}</h2><p>Delta: {call_delta:.3f}</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='option-card'><h3>PUT</h3><h2>${put_price:.2f}</h2><p>Delta: {put_delta:.3f}</p></div>", unsafe_allow_html=True)

    st.subheader("📉 Price Heatmaps")
    spot_range = np.linspace(spot_min, spot_max, grid_size)
    vol_range = np.linspace(vol_min, vol_max, grid_size)
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(generate_heatmap(S, K, T, r, sigma, spot_range, vol_range, "call"))
    with col2:
        st.pyplot(generate_heatmap(S, K, T, r, sigma, spot_range, vol_range, "put"))

# Tab 2: Monte Carlo
with tabs[1]:
    st.subheader("🎲 Monte Carlo Simulation")
    st.markdown("Compare Black-Scholes with Monte Carlo pricing.")
    call_mc = monte_carlo(S, K, T, r, sigma, "call", sims)
    put_mc = monte_carlo(S, K, T, r, sigma, "put", sims)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='option-card'><h3>CALL</h3><h2>${call_mc:.2f}</h2><p>Sims: {sims}</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='option-card'><h3>PUT</h3><h2>${put_mc:.2f}</h2><p>Sims: {sims}</p></div>", unsafe_allow_html=True)

    st.markdown(f"**Black-Scholes Comparison:** Call: ${call_price:.2f} | Put: ${put_price:.2f}")