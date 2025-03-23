import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="Black-Scholes Pro Terminal",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
theme = """
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
"""
st.markdown(theme, unsafe_allow_html=True)

# Black-Scholes with Greeks
def black_scholes_with_greeks(S, K, T, r, sigma, option_type="call"):
    try:
        if T <= 0 or sigma <= 0:
            return {"price": 0, "delta": 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                     r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                     r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        return {"price": price, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
    except:
        return {"price": 0, "delta": 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}

# Monte Carlo with Steps
def monte_carlo_steps(S, K, T, r, sigma, option_type="call", simulations=10000, steps=100):
    dt = T / steps
    ST = S * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt +
                              sigma * np.sqrt(dt) * np.random.randn(simulations, steps), axis=1))[:, -1]
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    return np.exp(-r * T) * np.mean(payoffs)

# Heatmap
def generate_heatmap(S, K, T, r, sigma, spot_range, vol_range, option_type="call"):
    prices = np.zeros((len(vol_range), len(spot_range)))
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            prices[i, j] = black_scholes_with_greeks(spot, K, T, r, vol, option_type)['price']
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

# Real-time data
@st.cache_data
def fetch_yahoo_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        price = hist["Close"].iloc[-1]
        sigma = stock.history(period="30d")["Close"].pct_change().std() * np.sqrt(252)
        return price, sigma
    except:
        return None, None

# Sidebar UI
with st.sidebar:
    st.markdown("<h3>Sharjeel Jafri</h3>", unsafe_allow_html=True)
    st.markdown("---")
    ticker = st.text_input("Ticker Symbol", "AAPL")
    if st.button("Fetch Real-Time Data"):
        S_live, sigma_live = fetch_yahoo_data(ticker)
        if S_live:
            st.session_state["S"] = S_live
            st.session_state["sigma"] = sigma_live
        else:
            st.error("Failed to fetch data. Using defaults.")

    S = st.number_input("Asset Price", 1.0, 1000.0, st.session_state.get("S", 100.0), 1.0)
    K = st.number_input("Strike Price", 1.0, 1000.0, 100.0, 1.0)
    T = st.number_input("Time (Years)", 0.01, 10.0, 1.0, 0.01)
    sigma = st.number_input("Volatility (σ)", 0.01, 1.0, st.session_state.get("sigma", 0.2), 0.01)
    r = st.number_input("Risk-Free Rate", 0.0, 1.0, 0.05, 0.01)

    st.subheader("Heatmap Settings")
    spot_min = st.number_input("Min Spot", value=S * 0.8, step=1.0)
    spot_max = st.number_input("Max Spot", value=S * 1.2, step=1.0)
    vol_min = st.slider("Min Volatility", 0.01, 1.0, sigma * 0.5, 0.01)
    vol_max = st.slider("Max Volatility", 0.01, 1.0, sigma * 1.5, 0.01)
    grid_size = st.slider("Heatmap Grid Size", 5, 20, 10, 1)

    st.subheader("Monte Carlo")
    sims = st.number_input("Simulations", 1000, 100000, 10000, 1000)
    steps = st.slider("Time Steps", 10, 500, 100, 10)

# Main Interface
st.title("Black-Scholes Pro Terminal")
st.markdown("Real-time pricing, Greeks, Monte Carlo, and heatmaps.")
tabs = st.tabs(["Black-Scholes", "Monte Carlo", "Candlestick Chart"])

with tabs[0]:
    st.subheader("Inputs")
    for col, (label, value) in zip(st.columns(5), [("Asset Price", S), ("Strike", K), ("Time", T), ("Volatility", sigma), ("Rate", r)]):
        col.markdown(f"<div class='metric-box'><h4>{label}</h4><p>{value:.2f}</p></div>", unsafe_allow_html=True)

    call_data = black_scholes_with_greeks(S, K, T, r, sigma, "call")
    put_data = black_scholes_with_greeks(S, K, T, r, sigma, "put")

    st.subheader("Option Insights")
    col1, col2 = st.columns(2)
    for data, label, col in zip([call_data, put_data], ["CALL", "PUT"], [col1, col2]):
        col.markdown(f"""
        <div class='option-card'>
            <h3>{label}</h3>
            <h2>${data['price']:.2f}</h2>
            <p>Delta: {data['delta']:.3f}</p>
            <p>Gamma: {data['gamma']:.4f}</p>
            <p>Vega: {data['vega']:.4f}</p>
            <p>Theta: {data['theta']:.4f}</p>
            <p>Rho: {data['rho']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

    if spot_min < spot_max and vol_min < vol_max:
        st.subheader("Heatmaps")
        spot_range = np.linspace(spot_min, spot_max, grid_size)
        vol_range = np.linspace(vol_min, vol_max, grid_size)
        col1, col2 = st.columns(2)
        col1.pyplot(generate_heatmap(S, K, T, r, sigma, spot_range, vol_range, "call"))
        col2.pyplot(generate_heatmap(S, K, T, r, sigma, spot_range, vol_range, "put"))
    else:
        st.error("Ensure Spot Min < Max and Volatility Min < Max.")

with tabs[1]:
    st.subheader("Monte Carlo Simulation")
    call_mc = monte_carlo_steps(S, K, T, r, sigma, "call", sims, steps)
    put_mc = monte_carlo_steps(S, K, T, r, sigma, "put", sims, steps)
    col1, col2 = st.columns(2)
    col1.markdown(f"<div class='option-card'><h3>CALL</h3><h2>${call_mc:.2f}</h2><p>Sims: {sims}</p></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='option-card'><h3>PUT</h3><h2>${put_mc:.2f}</h2><p>Sims: {sims}</p></div>", unsafe_allow_html=True)

with tabs[2]:
    st.subheader("Live Candlestick Chart")
    stock = yf.Ticker(ticker)
    df = stock.history(period="7d", interval="1h")
    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"])
    ])
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font=dict(color="#c9d1d9"),
        title=f"{ticker} - Last 7 Days Candlestick"
    )
    st.plotly_chart(fig, use_container_width=True)
