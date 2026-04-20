"""
House Price Prediction — Pakistan Real Estate
Streamlit App · XGBoost · Feature Engineering
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pakistan House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700&family=DM+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

  .metric-card {
    background: #F8FAFC;
    border: 0.5px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
  }
  .metric-label {
    font-size: 11px;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-family: 'DM Mono', monospace;
  }
  .metric-value {
    font-size: 26px;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.5px;
    margin-top: 4px;
  }
  .metric-sub {
    font-size: 11px;
    color: #94A3B8;
    font-family: 'DM Mono', monospace;
    margin-top: 2px;
  }
  .prediction-box {
    background: linear-gradient(135deg, #1E3A5F 0%, #2D5A8E 100%);
    border-radius: 14px;
    padding: 28px 32px;
    text-align: center;
    margin: 20px 0;
  }
  .prediction-label {
    font-size: 12px;
    color: #93C5FD;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'DM Mono', monospace;
  }
  .prediction-value {
    font-size: 42px;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -1px;
    margin: 8px 0;
  }
  .prediction-range {
    font-size: 13px;
    color: #BAC8FF;
    font-family: 'DM Mono', monospace;
  }
  .insight-pill {
    display: inline-block;
    background: #EFF6FF;
    color: #1D4ED8;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    font-family: 'DM Mono', monospace;
    margin: 3px 2px;
  }
  .section-title {
    font-size: 13px;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-family: 'DM Mono', monospace;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 0.5px solid #E2E8F0;
  }
  div[data-testid="stSidebar"] {
    background: #F8FAFC;
    border-right: 0.5px solid #E2E8F0;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT & ENTRAÎNEMENT (caché)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_and_prepare():
    df = pd.read_csv("Cleaned_data_for_model.csv")
    df = df[df["purpose"] == "For Sale"].copy()
    df = df[df["price"] > 0].copy()
    p99 = df["price"].quantile(0.99)
    df = df[df["price"] <= p99].copy()
    type_counts = df["property_type"].value_counts()
    valid_types = type_counts[type_counts >= 200].index
    df = df[df["property_type"].isin(valid_types)].copy()
    df["log_price"] = np.log1p(df["price"])
    return df, p99


@st.cache_data
def build_encoders(df):
    """Target encoding : retourne les tables de mapping."""
    global_mean = df["log_price"].mean()

    def _encode_table(col):
        stats = df.groupby(col)["log_price"].agg(["mean", "count"])
        smoothing = 1 / (1 + np.exp(-(stats["count"] - 10) / 10))
        stats["encoded"] = smoothing * stats["mean"] + (1 - smoothing) * global_mean
        return stats["encoded"].to_dict(), global_mean

    city_map, city_default     = _encode_table("city")
    location_map, loc_default  = _encode_table("location")

    prop_types = sorted([t for t in df["property_type"].unique()])

    return city_map, city_default, location_map, loc_default, prop_types, global_mean


@st.cache_resource
def train_model(df_hash):
    df, _   = load_and_prepare()
    city_map, city_def, loc_map, loc_def, prop_types, _ = build_encoders(df)

    df["city_encoded"]     = df["city"].map(city_map).fillna(city_def)
    df["location_encoded"] = df["location"].map(loc_map).fillna(loc_def)
    df["bath_bed_ratio"]   = df["baths"] / (df["bedrooms"] + 1)
    df = pd.get_dummies(df, columns=["property_type"], drop_first=True)

    feat_cols = (
        ["city_encoded", "location_encoded", "bedrooms", "baths", "Area_in_Marla", "bath_bed_ratio"]
        + [c for c in df.columns if c.startswith("property_type_")]
    )

    X = df[feat_cols].astype(float)
    y = df["log_price"]
    y_real = df["price"]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    xgb_params = dict(
        n_estimators=800, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=0
    )

    cv_results = []
    models     = []

    for tr, val in kf.split(X):
        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X.iloc[tr], y.iloc[tr],
              eval_set=[(X.iloc[val], y.iloc[val])], verbose=False)
        p = np.expm1(m.predict(X.iloc[val]))
        cv_results.append({
            "rmse": np.sqrt(mean_squared_error(y_real.iloc[val], p)),
            "mae":  mean_absolute_error(y_real.iloc[val], p),
            "r2":   r2_score(y_real.iloc[val], p),
        })
        models.append(m)

    best_idx   = int(np.argmax([r["r2"] for r in cv_results]))
    best_model = models[best_idx]

    return best_model, feat_cols, cv_results, df


# ── Chargement ────────────────────────────────────────────────────────────────
with st.spinner("Entraînement du modèle XGBoost..."):
    df_raw, p99 = load_and_prepare()
    city_map, city_def, loc_map, loc_def, prop_types, global_mean = build_encoders(df_raw)
    model, feat_cols, cv_results, df_trained = train_model(hash(str(df_raw.shape)))

cities    = sorted(df_raw["city"].unique().tolist())
locations = sorted(df_raw["location"].unique().tolist())

mean_r2   = np.mean([r["r2"]   for r in cv_results])
mean_rmse = np.mean([r["rmse"] for r in cv_results])
mean_mae  = np.mean([r["mae"]  for r in cv_results])


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Paramètres du bien
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏠 Caractéristiques du bien")
    st.markdown("---")

    city = st.selectbox("Ville", cities, index=cities.index("Lahore") if "Lahore" in cities else 0)

    city_locations = sorted(df_raw[df_raw["city"] == city]["location"].unique().tolist())
    location = st.selectbox("Quartier", city_locations)

    property_type = st.selectbox(
        "Type de bien",
        [t for t in prop_types if t not in ["Farm House", "Room"]],
        index=0
    )

    st.markdown("---")
    bedrooms  = st.slider("Chambres",  min_value=1, max_value=10, value=3)
    baths     = st.slider("Salles de bain", min_value=1, max_value=10, value=2)
    area      = st.number_input("Surface (Marla)", min_value=1.0, max_value=200.0,
                                value=10.0, step=0.5)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#94A3B8;font-family:DM Mono,monospace'>"
        f"Modèle : XGBoost · R² = {mean_r2:.4f}<br>"
        f"Entraîné sur {len(df_raw):,} biens · For Sale"
        "</div>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# PRÉDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict_price(city, location, property_type, bedrooms, baths, area):
    city_enc = city_map.get(city, city_def)
    loc_enc  = loc_map.get(location, city_def)
    bath_bed = baths / (bedrooms + 1)

    # One-hot property_type
    prop_dummies = {c: 0 for c in feat_cols if c.startswith("property_type_")}
    key = f"property_type_{property_type}"
    if key in prop_dummies:
        prop_dummies[key] = 1

    row = {
        "city_encoded":     city_enc,
        "location_encoded": loc_enc,
        "bedrooms":         bedrooms,
        "baths":            baths,
        "Area_in_Marla":    area,
        "bath_bed_ratio":   bath_bed,
        **prop_dummies,
    }

    X_pred = pd.DataFrame([row])[feat_cols].astype(float)
    log_pred = model.predict(X_pred)[0]
    price    = np.expm1(log_pred)

    # Intervalle de confiance approximatif (±RMSE)
    lower = max(0, price - mean_rmse)
    upper = price + mean_rmse

    return price, lower, upper


price, lower, upper = predict_price(city, location, property_type, bedrooms, baths, area)


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("# 🏠 Pakistan House Price Predictor")
st.markdown(
    f"Modèle **XGBoost** · R² = **{mean_r2:.4f}** · "
    f"Entraîné sur **{len(df_raw):,}** biens · For Sale uniquement"
)
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Prédiction", "📊 Performance du modèle", "🔍 EDA"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 : PRÉDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col_pred, col_info = st.columns([1, 1], gap="large")

    with col_pred:
        st.markdown(f"""
        <div class="prediction-box">
          <div class="prediction-label">Prix estimé</div>
          <div class="prediction-value">{price/1e6:.1f}M PKR</div>
          <div class="prediction-range">
            Fourchette · {lower/1e6:.1f}M – {upper/1e6:.1f}M PKR
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Contexte marché
        city_median = df_raw[df_raw["city"] == city]["price"].median()
        delta_pct   = ((price - city_median) / city_median) * 100
        direction   = "au-dessus" if delta_pct > 0 else "en-dessous"

        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Contexte marché — {city}</div>
          <div class="metric-value">{city_median/1e6:.1f}M PKR</div>
          <div class="metric-sub">
            Médiane ville · Ce bien est {abs(delta_pct):.0f}% {direction} du marché
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        st.markdown('<div class="section-title">Récapitulatif du bien</div>', unsafe_allow_html=True)

        details = {
            "Ville":          city,
            "Quartier":       location,
            "Type":           property_type,
            "Chambres":       str(bedrooms),
            "Salles de bain": str(baths),
            "Surface":        f"{area} Marla",
            "Prix / Marla":   f"{price/area/1e6:.2f}M PKR",
        }

        for k, v in details.items():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:6px 0;border-bottom:0.5px solid #F1F5F9;font-size:13px'>"
                f"<span style='color:#64748B'>{k}</span>"
                f"<span style='font-weight:600'>{v}</span></div>",
                unsafe_allow_html=True
            )

        # Similarité avec le marché
        st.markdown("")
        st.markdown('<div class="section-title">Biens similaires dans ce quartier</div>',
                    unsafe_allow_html=True)

        similar = df_raw[
            (df_raw["location"] == location) &
            (df_raw["bedrooms"].between(bedrooms - 1, bedrooms + 1))
        ]["price"]

        if len(similar) > 3:
            st.markdown(
                f"<span class='insight-pill'>Min : {similar.min()/1e6:.1f}M</span>"
                f"<span class='insight-pill'>Médiane : {similar.median()/1e6:.1f}M</span>"
                f"<span class='insight-pill'>Max : {similar.max()/1e6:.1f}M</span>"
                f"<span class='insight-pill'>{len(similar)} biens</span>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<span class='insight-pill'>Données insuffisantes pour ce quartier</span>",
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 : PERFORMANCE MODÈLE
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">R² Score</div>
          <div class="metric-value">{mean_r2:.4f}</div>
          <div class="metric-sub">Moyenne 5-Fold CV</div>
        </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">RMSE</div>
          <div class="metric-value">{mean_rmse/1e6:.2f}M</div>
          <div class="metric-sub">PKR · Moyenne 5-Fold</div>
        </div>""", unsafe_allow_html=True)

    with col_c:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">MAE</div>
          <div class="metric-value">{mean_mae/1e6:.2f}M</div>
          <div class="metric-sub">PKR · Moyenne 5-Fold</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown('<div class="section-title">R² par fold</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3))
        folds = [f"Fold {i+1}" for i in range(5)]
        r2s   = [r["r2"] for r in cv_results]
        colors = ["#3B82F6" if v == max(r2s) else "#CBD5E1" for v in r2s]
        ax.bar(folds, r2s, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_ylim(0.84, 0.88)
        ax.axhline(mean_r2, color="#EF4444", linestyle="--", linewidth=1,
                   label=f"Moyenne : {mean_r2:.4f}")
        ax.legend(fontsize=9)
        ax.set_ylabel("R²")
        fig.patch.set_facecolor("white")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_chart2:
        st.markdown('<div class="section-title">Feature Importance (XGBoost)</div>',
                    unsafe_allow_html=True)
        importance = pd.Series(
            model.feature_importances_, index=feat_cols
        ).sort_values(ascending=True).tail(8)

        fig, ax = plt.subplots(figsize=(5, 3))
        colors = ["#3B82F6" if v > importance.median() else "#CBD5E1"
                  for v in importance.values]
        ax.barh(importance.index, importance.values,
                color=colors, edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Importance (gain)")
        fig.patch.set_facecolor("white")
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 : EDA
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.markdown('<div class="section-title">Prix médian par ville</div>',
                    unsafe_allow_html=True)
        city_med = df_raw[df_raw["purpose"] == "For Sale"].groupby("city")["price"].median().sort_values()
        CITY_COLORS = {
            "Karachi": "#3B82F6", "Lahore": "#8B5CF6",
            "Islamabad": "#10B981", "Rawalpindi": "#F59E0B", "Faisalabad": "#EF4444"
        }
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(city_med.index, city_med.values / 1e6,
                color=[CITY_COLORS.get(c, "#94A3B8") for c in city_med.index],
                edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Prix médian (M PKR)")
        fig.patch.set_facecolor("white")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_e2:
        st.markdown('<div class="section-title">Distribution log(prix)</div>',
                    unsafe_allow_html=True)
        sale = df_raw[df_raw["purpose"] == "For Sale"]
        log_p = np.log1p(sale["price"])
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(log_p, bins=50, color="#8B5CF6", alpha=0.85,
                edgecolor="white", linewidth=0.3)
        ax.axvline(log_p.median(), color="#EF4444", linestyle="--",
                   linewidth=1.5, label=f"Médiane : {log_p.median():.2f}")
        ax.set_xlabel("log(Prix)")
        ax.legend(fontsize=9)
        fig.patch.set_facecolor("white")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("")

    col_e3, col_e4 = st.columns(2)

    with col_e3:
        st.markdown('<div class="section-title">Prix médian par type de bien</div>',
                    unsafe_allow_html=True)
        type_med = (
            df_raw[df_raw["purpose"] == "For Sale"]
            .groupby("property_type")["price"].median()
            .sort_values(ascending=True)
        )
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(type_med.index, type_med.values / 1e6,
                color="#10B981", alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Prix médian (M PKR)")
        fig.patch.set_facecolor("white")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_e4:
        st.markdown('<div class="section-title">Prix médian par nb de chambres</div>',
                    unsafe_allow_html=True)
        bed_med = (
            df_raw[
                (df_raw["purpose"] == "For Sale") &
                (df_raw["property_type"] == "House") &
                (df_raw["bedrooms"].between(1, 7))
            ]
            .groupby("bedrooms")["price"].median()
        )
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(bed_med.index, bed_med.values / 1e6,
               color="#F59E0B", edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Chambres")
        ax.set_ylabel("Prix médian (M PKR)")
        fig.patch.set_facecolor("white")
        st.pyplot(fig, use_container_width=True)
        plt.close()
