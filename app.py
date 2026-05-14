"""
FreshMart Demand Planner
========================
Streamlit app for the Corporación Favorita forecasting project.
Champion model: Prophet (or Random Forest — swap MODEL_TYPE below).

Run:
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle, json, os
from datetime import timedelta

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = "data/"
MODELS_DIR = "models/"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FreshMart Demand Planner",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    daily = pd.read_csv(DATA_DIR + "timeseries_clean.csv", parse_dates=["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily

@st.cache_resource
def load_models():
    models = {}
    meta_path = MODELS_DIR + "champion_metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            models["metadata"] = json.load(f)
    model_path = MODELS_DIR + "champion_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            models["champion"] = pickle.load(f)
    results_path = MODELS_DIR + "all_models_results.csv"
    if os.path.exists(results_path):
        models["results"] = pd.read_csv(results_path)
    feat_path = MODELS_DIR + "feature_columns.json"
    if os.path.exists(feat_path):
        with open(feat_path) as f:
            models["features"] = json.load(f)
    return models

def prophet_forecast(model, daily, cutoff_date, horizon_days):
    """Generate Prophet forecast from cutoff_date for horizon_days."""
    future_dates = pd.date_range(
        start=cutoff_date + timedelta(days=1), periods=horizon_days, freq="D"
    )
    # Oil price: forward-fill last known value
    last_oil = daily.loc[daily.date <= cutoff_date, "dcoilwtico"].iloc[-1]
    future_df = pd.DataFrame({
        "ds" : future_dates,
        "oil": last_oil
    })
    forecast = model.predict(future_df)
    forecast["yhat"] = forecast["yhat"].clip(lower=0)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

def rf_forecast(model, features, daily, cutoff_date, horizon_days):
    """Recursive forecast for Random Forest using lag/rolling features."""
    history = daily[daily.date <= cutoff_date]["unit_sales"].tolist()
    oil_series = daily.set_index("date")["dcoilwtico"]
    hol_series = daily.set_index("date")["is_national_holiday"]

    future_dates = pd.date_range(
        start=cutoff_date + timedelta(days=1), periods=horizon_days, freq="D"
    )
    predictions = []
    for fd in future_dates:
        row = {}
        n = len(history)
        for lag in [1, 7, 14, 30]:
            row[f"lag_{lag}"] = history[-lag] if n >= lag else 0
        for w in [7, 14, 28]:
            sl = history[-w:] if n >= w else history
            row[f"roll_mean_{w}"] = np.mean(sl)
            row[f"roll_std_{w}"]  = np.std(sl) if len(sl) > 1 else 0
        row["day_of_week"]        = fd.dayofweek
        row["month"]              = fd.month
        row["week"]               = fd.isocalendar()[1]
        row["is_weekend"]         = int(fd.dayofweek >= 5)
        row["dcoilwtico"]         = oil_series.get(fd, oil_series.iloc[-1])
        row["oil_lag_1"]          = oil_series.get(fd - timedelta(days=1), oil_series.iloc[-1])
        row["oil_lag_7"]          = oil_series.get(fd - timedelta(days=7), oil_series.iloc[-1])
        row["is_national_holiday"]= int(hol_series.get(fd, 0))
        row["store_open"]         = 0 if row["is_national_holiday"] else 1

        X = pd.DataFrame([row])[features]
        pred = max(model.predict(X)[0], 0)
        predictions.append(pred)
        history.append(pred)

    return pd.DataFrame({"ds": future_dates, "yhat": predictions})

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
daily  = load_data()
models = load_models()

champion_name = models.get("metadata", {}).get("champion_model", "Champion Model")
model_type    = "prophet" if "Prophet" in champion_name else "rf"

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shopping-cart.png", width=64)
    st.title("FreshMart\nDemand Planner")
    st.caption("Powered by " + champion_name)
    st.divider()

    st.subheader("⚙️ Forecast Settings")

    max_date = daily["date"].max().date()
    min_date = daily["date"].min().date() + timedelta(days=60)

    cutoff_date = st.date_input(
        "📅 Last known sales date",
        value=pd.Timestamp("2013-12-31").date(),
        min_value=min_date,
        max_value=max_date,
        help="The app will forecast sales for the days after this date.",
    )

    horizon_days = st.slider(
        "📆 Forecast horizon (days)", min_value=7, max_value=30, value=14, step=7
    )

    history_days = st.slider(
        "📊 Historical days to show", min_value=14, max_value=90, value=30, step=7
    )

    show_actuals = st.toggle("Show actual sales on chart", value=True)
    show_ci      = st.toggle("Show confidence interval",   value=True,
                              disabled=(model_type != "prophet"))

    run_btn = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_forecast, tab_compare, tab_eda, tab_about = st.tabs([
    "📈 Forecast", "📊 Model Comparison", "🔍 EDA Explorer", "ℹ️ About"
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — FORECAST
# ═══════════════════════════════════════════════════════════════
with tab_forecast:
    st.header("📈 Sales Forecast")
    st.caption(
        f"Forecast from **{cutoff_date + timedelta(days=1)}** "
        f"for **{horizon_days} days** · Champion: **{champion_name}**"
    )

    if not run_btn:
        st.info("Configure your settings in the sidebar and click **Generate Forecast** to begin.")
    else:
        cutoff_ts = pd.Timestamp(cutoff_date)
        champion  = models.get("champion")

        if champion is None:
            st.error("Champion model not found. Run notebooks 1–4 first to generate models/.")
            st.stop()

        with st.spinner("Running forecast..."):
            if model_type == "prophet":
                fc = prophet_forecast(champion, daily, cutoff_ts, horizon_days)
            else:
                fc = rf_forecast(champion, models["features"], daily, cutoff_ts, horizon_days)

        # ── KPIs ──────────────────────────────────────────────
        total_demand   = fc["yhat"].sum()
        avg_daily      = fc["yhat"].mean()
        peak_idx       = fc["yhat"].idxmax()
        peak_day       = fc.loc[peak_idx, "ds"].strftime("%a %b %d")
        peak_sales     = fc.loc[peak_idx, "yhat"]

        # Demand level
        hist_avg = daily[daily.date <= cutoff_ts]["unit_sales"].mean()
        level    = "🔴 High" if avg_daily > hist_avg * 1.1 else (
                   "🟡 Medium" if avg_daily > hist_avg * 0.9 else "🟢 Low")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Forecast Demand", f"{total_demand:,.0f} units")
        col2.metric("Avg Daily Forecast",    f"{avg_daily:,.0f} units")
        col3.metric("Peak Day",              peak_day, f"{peak_sales:,.0f} units")
        col4.metric("Demand Level",          level)

        # ── Chart ─────────────────────────────────────────────
        hist_slice = daily[
            (daily.date > cutoff_ts - timedelta(days=history_days)) &
            (daily.date <= cutoff_ts)
        ]

        fig = go.Figure()

        # Historical actuals
        fig.add_trace(go.Scatter(
            x=hist_slice.date, y=hist_slice.unit_sales,
            mode="lines", name="Historical Sales",
            line=dict(color="#2196F3", width=1.5)
        ))

        # Confidence interval (Prophet only)
        if show_ci and model_type == "prophet" and "yhat_lower" in fc.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([fc.ds, fc.ds[::-1]]),
                y=pd.concat([fc.yhat_upper, fc.yhat_lower[::-1]]),
                fill="toself", fillcolor="rgba(244,67,54,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="80% CI", showlegend=True
            ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=fc.ds, y=fc.yhat,
            mode="lines+markers", name="Forecast",
            line=dict(color="#F44336", width=2.5),
            marker=dict(size=5)
        ))

        # Actual test data (if available and toggle on)
        if show_actuals:
            actuals = daily[
                (daily.date > cutoff_ts) &
                (daily.date <= fc.ds.max())
            ]
            if not actuals.empty:
                fig.add_trace(go.Scatter(
                    x=actuals.date, y=actuals.unit_sales,
                    mode="lines", name="Actual Sales",
                    line=dict(color="#4CAF50", width=1.5, dash="dot")
                ))

        fig.add_vline(x=cutoff_ts.timestamp() * 1000, line_dash="dot", line_color="grey")
        fig.add_annotation(x=cutoff_ts, y=1, yref="paper", text="Cutoff",
                        showarrow=False, xanchor="left", font=dict(color="grey"))

        fig.update_layout(
            title=f"Sales Forecast — Next {horizon_days} Days",
            xaxis_title="Date", yaxis_title="Unit Sales",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified", height=420,
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Forecast Table ─────────────────────────────────────
        st.subheader("📋 Forecast Table")
        display_fc = fc[["ds", "yhat"]].copy()
        display_fc.columns = ["Date", "Forecasted Sales"]
        display_fc["Day"]  = display_fc["Date"].dt.day_name()
        display_fc["Forecasted Sales"] = display_fc["Forecasted Sales"].round(0).astype(int)
        display_fc = display_fc[["Date", "Day", "Forecasted Sales"]]

        st.dataframe(display_fc, use_container_width=True, hide_index=True)

        csv = display_fc.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Forecast CSV", csv, "forecast.csv", "text/csv")

        # ── Planning Interpretation ────────────────────────────
        with st.expander("📝 Planning Interpretation"):
            weekends = display_fc[display_fc["Day"].isin(["Saturday", "Sunday"])]
            st.markdown(f"""
**Business insights for the next {horizon_days} days:**

- 📦 **Total order target:** Plan for approximately **{total_demand:,.0f} units** over the period
- 📅 **Peak day:** Highest demand expected on **{peak_day}** ({peak_sales:,.0f} units)
- 📊 **Weekend uplift:** {len(weekends)} weekend days in forecast window — stock accordingly
- ⚠️ **Demand level vs. historical avg:** {level}

*Use these figures as input for your order quantities, staffing schedule, and shelf planning.*
""")

# ═══════════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════
with tab_compare:
    st.header("📊 Model Comparison")
    st.caption("All models evaluated on the same 90-day test period (Jan–Mar 2014)")

    if "results" not in models:
        st.warning("Run notebooks 1–4 to generate model results. Results file not found.")
    else:
        results_df = models["results"].sort_values("RMSE")
        champion_row = results_df.iloc[0]

        st.success(f"🏆 **Champion: {champion_row['Model']}**  ·  "
                   f"RMSE={champion_row['RMSE']}  MAE={champion_row['MAE']}  "
                   f"R²={champion_row['R²']}")

        col1, col2 = st.columns(2)

        # RMSE bar
        fig_rmse = px.bar(
            results_df, x="RMSE", y="Model", orientation="h",
            color="RMSE", color_continuous_scale="RdYlGn_r",
            title="RMSE (lower is better)",
            text=results_df["RMSE"].apply(lambda x: f"{x:.1f}")
        )
        fig_rmse.update_layout(showlegend=False, height=350, yaxis={"categoryorder": "total ascending"})
        col1.plotly_chart(fig_rmse, use_container_width=True)

        # R² bar
        fig_r2 = px.bar(
            results_df, x="R²", y="Model", orientation="h",
            color="R²", color_continuous_scale="RdYlGn",
            title="R² (higher is better)",
            text=results_df["R²"].apply(lambda x: f"{x:.3f}")
        )
        fig_r2.update_layout(showlegend=False, height=350, yaxis={"categoryorder": "total descending"})
        col2.plotly_chart(fig_r2, use_container_width=True)

        st.subheader("Full Results Table")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        with st.expander("💡 Why did we choose this champion?"):
            st.markdown(f"""
**{champion_row['Model']}** was selected as the champion model because:

1. **Lowest RMSE** on the held-out 90-day test period — the most important metric for demand planning
2. **Handles weekly seasonality** natively — critical for this dataset
3. **Holiday awareness** — automatically accounts for store-closed days
4. **Interpretable components** — trend, seasonality, and holiday effects are separately visible

**Note on ML models:** With only ~330 effective training rows (after lag feature warmup),
XGBoost and Random Forest face a *data starvation* problem. Statistical models that
encode seasonality structurally — rather than learning it from limited data — outperform
them here. This is a classic finding in time-series forecasting with short histories.
""")

# ═══════════════════════════════════════════════════════════════
# TAB 3 — EDA EXPLORER
# ═══════════════════════════════════════════════════════════════
with tab_eda:
    st.header("🔍 EDA Data Explorer")
    st.caption("Key findings from Notebook 1 — Exploratory Data Analysis")

    # Full series
    st.subheader("Full Sales Time Series")
    fig_ts = px.line(daily, x="date", y="unit_sales",
                     title="Daily Unit Sales (Jan 2013 – Mar 2014)",
                     labels={"unit_sales": "Unit Sales", "date": "Date"})
    fig_ts.add_vline(x=pd.Timestamp("2014-01-01").timestamp() * 1000,
                    line_dash="dash", line_color="tomato")
    fig_ts.add_annotation(x=pd.Timestamp("2014-01-01"), y=1, yref="paper",
                        text="Train/Test Split", showarrow=False,
                        xanchor="left", font=dict(color="tomato"))
    fig_ts.update_layout(height=350)
    st.plotly_chart(fig_ts, use_container_width=True)

    col1, col2 = st.columns(2)

    # Weekly seasonality
    with col1:
        st.subheader("Weekly Seasonality")
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow_avg = (daily[daily.store_open == 1]
                   .groupby("day_of_week")["unit_sales"].mean()
                   .reindex(range(7)).values)
        fig_dow = px.bar(x=day_labels, y=dow_avg,
                         labels={"x": "Day", "y": "Avg Unit Sales"},
                         title="Avg Sales by Day of Week",
                         color=dow_avg,
                         color_continuous_scale="Blues")
        fig_dow.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_dow, use_container_width=True)

        wk_mean = np.mean(dow_avg[:5])
        st.info(f"📊 Saturday +{(dow_avg[5]/wk_mean-1)*100:.0f}% · Sunday +{(dow_avg[6]/wk_mean-1)*100:.0f}% vs weekday avg")

    # Oil price overlay
    with col2:
        st.subheader("Oil Price Correlation")
        fig_oil = go.Figure()
        fig_oil.add_trace(go.Scatter(x=daily.date, y=daily.unit_sales,
                                     name="Unit Sales", yaxis="y1",
                                     line=dict(color="#2196F3", width=1)))
        fig_oil.add_trace(go.Scatter(x=daily.date, y=daily.dcoilwtico,
                                     name="Oil Price", yaxis="y2",
                                     line=dict(color="#FF9800", width=1.5)))
        fig_oil.update_layout(
            title="Unit Sales vs. Oil Price",
            yaxis=dict(title=dict(text="Unit Sales", font=dict(color="#2196F3"))),
            yaxis2=dict(title=dict(text="Oil Price (USD)", font=dict(color="#FF9800")),
                        overlaying="y", side="right"),
            hovermode="x unified", height=300, legend=dict(x=0, y=1.1, orientation="h")
        )
        st.plotly_chart(fig_oil, use_container_width=True)
        corr = daily["unit_sales"].corr(daily["dcoilwtico"])
        st.info(f"📊 Pearson correlation: {corr:.3f}")

    # Key findings summary
    st.subheader("Key EDA Findings")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Weekly Seasonality", "s = 7", "Dominant pattern")
    col_b.metric("ADF Stationarity", "p < 0.05", "Stationary ✅")
    col_c.metric("Missing Dates Imputed", "Dec 25 + Jan 1", "Set to 0, flagged")

    st.dataframe(daily[["date", "unit_sales", "dcoilwtico",
                         "is_national_holiday", "store_open",
                         "day_of_week", "is_weekend"]].head(20),
                 use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ═══════════════════════════════════════════════════════════════
with tab_about:
    st.header("ℹ️ About FreshMart Demand Planner")
    st.markdown("""
## Business Problem

Small grocery store managers struggle to know **how much stock to order** for the coming week.
Ordering too little means missed sales and frustrated customers (stockout).
Ordering too much means wasted perishable goods and tied-up cash (overstock).

**FreshMart Demand Planner** uses a data-driven forecasting model to answer:
> *"How many units should I expect to sell in the next 7–30 days?"*

---

## How It Works

1. **Historical data** (Jan 2013 – Mar 2014) was used to identify patterns
2. **EDA** revealed strong weekly seasonality and moderate oil-price correlation
3. **Five models** were trained and evaluated on a held-out 90-day test period
4. The **champion model** was selected on lowest RMSE and deployed here

---

## Dataset

Derived from the Corporación Favorita Grocery Sales dataset (Ecuador).
Daily unit sales for a single store/product combination.

---

## Project Structure

| Notebook | Purpose |
|---|---|
| `01_eda_and_cleaning.ipynb` | Data cleaning & exploratory analysis |
| `02_feature_engineering.ipynb` | 19 features: lags, rolling stats, calendar, oil, holidays |
| `03_statistical_models.ipynb` | SARIMA, Holt-Winters, Prophet |
| `04_ml_models.ipynb` | Random Forest, XGBoost, HyperOpt, MLflow, champion selection |
| `app.py` | This Streamlit app |
""")
