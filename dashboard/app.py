import plotly.graph_objects as go
import streamlit as st

from dashboard.data_access import (
    compute_all_methods,
    compute_attribution,
    load_backtest_results,
    load_returns,
)

st.set_page_config(page_title="Market Risk Engine", page_icon="📉", layout="wide")

st.title("📉 Market Risk Engine")
st.caption(
    "A market risk engine comparing 4 VaR/ES methodologies, validated with "
    "out-of-sample backtesting (Kupiec, Christoffersen, Basel), served live. "
    "Adjust portfolio weights in the sidebar to see risk estimates and "
    "component attribution update in real time."
)

# ---------------------------------------------------------------------
# Sidebar — portfolio controls
# ---------------------------------------------------------------------
returns = load_returns()
tickers = list(returns.columns)

st.sidebar.title("Portfolio Controls")
st.sidebar.markdown("Adjust weights — they're normalized to sum to 1 automatically.")
st.sidebar.divider()

raw_weights = {}
for t in tickers:
    raw_weights[t] = st.sidebar.slider(t, 0.0, 1.0, 1.0 / len(tickers), 0.01)

total = sum(raw_weights.values())
if total == 0:
    st.error("At least one weight must be positive.")
    st.stop()
weights = {t: w / total for t, w in raw_weights.items()}

st.sidebar.caption("Normalized: " + ", ".join(f"{t} {w:.0%}" for t, w in weights.items()))

st.sidebar.divider()
st.sidebar.markdown("**Risk settings**")
confidence = st.sidebar.select_slider("Confidence level", options=[0.95, 0.99], value=0.99)
alpha = round(1 - confidence, 10)

# ---------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------
results = compute_all_methods(returns, weights, alpha=alpha)
headline = results.loc[results["Method"] == "Filtered Historical Sim"].iloc[0]

col1, col2, col3 = st.columns(3)
col1.metric(f"Portfolio VaR (FHS, {confidence:.0%})", f"{headline['VaR']:.2%}")
col2.metric(f"Portfolio ES (FHS, {confidence:.0%})", f"{headline['ES']:.2%}")
col3.metric("Assets in universe", len(tickers))

st.divider()

# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tab_live, tab_backtest, tab_attribution = st.tabs(
    ["📊 Live VaR/ES", "✅ Backtest Validation", "🔍 Component Attribution"]
)

with tab_live:
    st.subheader(f"VaR / ES comparison at {confidence:.0%} confidence")
    st.dataframe(
        results.style.format({"VaR": "{:.2%}", "ES": "{:.2%}", "ES/VaR": "{:.2f}"}),
        use_container_width=True,
    )

    fig = go.Figure()
    fig.add_bar(name="VaR", x=results["Method"], y=results["VaR"], marker_color="#d62728")
    fig.add_bar(name="ES", x=results["Method"], y=results["ES"], marker_color="#7f0000")
    fig.update_layout(
        barmode="group",
        yaxis_tickformat=".2%",
        yaxis_title="Loss (% of portfolio)",
        legend_title=None,
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_backtest:
    st.subheader("Backtest validation")
    st.caption("Equal-weight portfolio, fixed — not affected by the sidebar sliders above.")

    bt = load_backtest_results()
    method_choice = st.selectbox("Method", sorted(bt["method"].unique()))
    bt_method = bt[bt["method"] == method_choice]

    quantile_rows = bt_method[bt_method["confidence_level"] != "basel_summary"].copy()
    basel_row = bt_method[bt_method["confidence_level"] == "basel_summary"]

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.markdown("**Kupiec / Christoffersen**")
        display_cols = [
            "confidence_level",
            "breach_rate",
            "kupiec_p_value",
            "kupiec_reject",
            "christoffersen_p_ind",
            "christoffersen_reject_ind",
        ]
        bool_cols = ["kupiec_reject", "christoffersen_reject_ind"]
        display_df = quantile_rows[display_cols].copy()
        display_df[bool_cols] = display_df[bool_cols].astype(bool)
        st.dataframe(
            display_df.style.format(
                {
                    "breach_rate": "{:.2%}",
                    "kupiec_p_value": "{:.4f}",
                    "christoffersen_p_ind": "{:.4f}",
                }
            ),
            use_container_width=True,
        )

    with bcol2:
        st.markdown("**Basel traffic-light summary**")
        if not basel_row.empty:
            r = basel_row.iloc[0]
            pct_green = r["breach_rate"]  # reused column — see engineering-notes.md
            is_red_now = bool(r["combined_reject"])
            zone_label = "🔴 Red" if is_red_now else "🟢 / 🟡 Not currently Red"
            st.metric("Current zone", zone_label)
            st.metric("% of rolling windows in Green", f"{pct_green:.1%}")
            st.metric("Breaches in current window", int(r["n_breaches"]))
        else:
            st.info("No Basel summary logged for this method.")

with tab_attribution:
    st.subheader("Component VaR — who's actually driving portfolio risk?")
    st.caption(
        "Live: reflects the weights set in the sidebar. Parametric/EWMA only — "
        "component decomposition requires an explicit covariance matrix."
    )

    attribution = compute_attribution(returns, weights, confidence=confidence)

    fig2 = go.Figure()
    fig2.add_bar(
        name="% of capital",
        x=attribution["Ticker"],
        y=attribution["Weight"],
        marker_color="#1f77b4",
    )
    fig2.add_bar(
        name="% of portfolio risk",
        x=attribution["Ticker"],
        y=attribution["% of Portfolio VaR"],
        marker_color="#d62728",
    )
    fig2.update_layout(barmode="group", yaxis_tickformat=".0%", yaxis_title="Share")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.caption(
    "Built from first principles: EWMA/GARCH volatility, 4 VaR methodologies, "
    "regulatory backtesting (Kupiec, Christoffersen, Basel), Euler-decomposed "
    "component VaR. [Source](https://github.com/Aksh-19/market-risk-engine)"
)
