import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
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
    background: #F8FAFC; border: 0.5px solid #E2E8F0;
    border-radius: 10px; padding: 16px 20px; margin-bottom: 10px;
  }
  .metric-label {
    font-size: 11px; color: #64748B; text-transform: uppercase;
    letter-spacing: 0.8px; font-family: 'DM Mono', monospace;
  }
  .metric-value { font-size: 26px; font-weight: 700; color: #0F172A; letter-spacing: -0.5px; margin-top: 4px; }
  .metric-sub   { font-size: 11px; color: #94A3B8; font-family: 'DM Mono', monospace; margin-top: 2px; }
  .prediction-box {
    background: linear-gradient(135deg, #1E3A5F 0%, #2D5A8E 100%);
    border-radius: 14px; padding: 28px 32px; text-align: center; margin: 20px 0;
  }
  .prediction-label { font-size: 12px; color: #93C5FD; text-transform: uppercase; letter-spacing: 1px; font-family: 'DM Mono', monospace; }
  .prediction-value { font-size: 42px; font-weight: 700; color: #FFFFFF; letter-spacing: -1px; margin: 8px 0; }
  .prediction-range { font-size: 13px; color: #BAC8FF; font-family: 'DM Mono', monospace; }
  .insight-pill {
    display: inline-block; background: #EFF6FF; color: #1D4ED8;
    border-radius: 20px; padding: 4px 12px; font-size: 11px;
    font-family: 'DM Mono', monospace; margin: 3px 2px;
  }
  .section-title {
    font-size: 13px; font-weight: 600; color: #64748B; text-transform: uppercase;
    letter-spacing: 0.8px; font-family: 'DM Mono', monospace;
    margin-bottom: 12px; padding-bottom: 6px; border-bottom: 0.5px solid #E2E8F0;
  }
  div[data-testid="stSidebar"] { background: #F8FAFC; border-right: 0.5px solid #E2E8F0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT MODÈLE & METADATA (instantané)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, meta


@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_data_for_model.csv")
    return df[df["purpose"] == "For Sale"].copy()


model, meta  = load_model()
df_raw       = load_data()

feat_cols    = meta["feat_cols"]
city_map     = meta["city_map"]
location_map = meta["location_map"]
global_mean  = meta["global_mean"]
cv_results   = meta["cv_results"]
n_train      = meta["n_train"]

mean_r2   = np.mean([r["r2"]   for r in cv_results])
mean_rmse = np.mean([r["rmse"] for r in cv_results])
mean_mae  = np.mean([r["mae"]  for r in cv_results])
cities    = sorted(df_raw["city"].unique().tolist())


# ══════════════════════════════════════════════════════════════════════════════
# PRÉDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict_price(city, location, property_type, bedrooms, baths, area):
    city_enc = city_map.get(city, global_mean)
    loc_enc  = location_map.get(location, global_mean)
    bath_bed = baths / (bedrooms + 1)

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

    X_pred   = pd.DataFrame([row])[feat_cols].astype(float)
    log_pred = model.predict(X_pred)[0]
    price    = np.expm1(log_pred)
    return price, max(0, price - mean_rmse), price + mean_rmse


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏠 Caractéristiques du bien")
    st.markdown("---")

    city = st.selectbox("Ville", cities,
                        index=cities.index("Lahore") if "Lahore" in cities else 0)

    city_locs     = sorted(df_raw[df_raw["city"] == city]["location"].unique().tolist())
    location      = st.selectbox("Quartier", city_locs)
    valid_types   = [t for t in sorted(df_raw["property_type"].unique())
                     if t not in ["Farm House", "Room"]]
    property_type = st.selectbox("Type de bien", valid_types)

    st.markdown("---")
    bedrooms = st.slider("Chambres",       min_value=1, max_value=10, value=3)
    baths    = st.slider("Salles de bain", min_value=1, max_value=10, value=2)
    area     = st.number_input("Surface (Marla)", min_value=1.0,
                                max_value=200.0, value=10.0, step=0.5)

    st.markdown("---")
    st.markdown(
        f"<div style='font-size:11px;color:#94A3B8;font-family:DM Mono,monospace'>"
        f"Modèle : XGBoost · R² = {mean_r2:.4f}<br>"
        f"Entraîné sur {n_train:,} biens · For Sale"
        f"</div>", unsafe_allow_html=True
    )

price, lower, upper = predict_price(city, location, property_type, bedrooms, baths, area)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("# 🏠 Pakistan House Price Predictor")
st.markdown(
    f"Modèle **XGBoost** · R² = **{mean_r2:.4f}** · "
    f"Entraîné sur **{n_train:,}** biens · For Sale uniquement"
)
st.markdown("---")

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
          <div class="prediction-range">Fourchette · {lower/1e6:.1f}M – {upper/1e6:.1f}M PKR</div>
        </div>""", unsafe_allow_html=True)

        city_median = df_raw[df_raw["city"] == city]["price"].median()
        delta_pct   = ((price - city_median) / city_median) * 100
        direction   = "au-dessus" if delta_pct > 0 else "en-dessous"

        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Contexte marché — {city}</div>
          <div class="metric-value">{city_median/1e6:.1f}M PKR</div>
          <div class="metric-sub">Médiane ville · Ce bien est {abs(delta_pct):.0f}% {direction} du marché</div>
        </div>""", unsafe_allow_html=True)

    with col_info:
        st.markdown('<div class="section-title">Récapitulatif du bien</div>',
                    unsafe_allow_html=True)

        for k, v in {
            "Ville": city, "Quartier": location, "Type": property_type,
            "Chambres": str(bedrooms), "Salles de bain": str(baths),
            "Surface": f"{area} Marla", "Prix / Marla": f"{price/area/1e6:.2f}M PKR",
        }.items():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:6px 0;border-bottom:0.5px solid #F1F5F9;font-size:13px'>"
                f"<span style='color:#64748B'>{k}</span>"
                f"<span style='font-weight:600'>{v}</span></div>",
                unsafe_allow_html=True
            )

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
# TAB 2 : PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    col_a, col_b, col_c = st.columns(3)
    for col, label, val, sub in [
        (col_a, "R² Score", f"{mean_r2:.4f}",        "Moyenne 5-Fold CV"),
        (col_b, "RMSE",     f"{mean_rmse/1e6:.2f}M", "PKR · Moyenne 5-Fold"),
        (col_c, "MAE",      f"{mean_mae/1e6:.2f}M",  "PKR · Moyenne 5-Fold"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{val}</div>
              <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown('<div class="section-title">R² par fold</div>', unsafe_allow_html=True)
        r2s    = [r["r2"] for r in cv_results]
        colors = ["#3B82F6" if v == max(r2s) else "#CBD5E1" for v in r2s]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar([f"Fold {i+1}" for i in range(5)], r2s,
               color=colors, edgecolor="white", linewidth=0.5)
        ax.set_ylim(0.84, 0.88)
        ax.axhline(mean_r2, color="#EF4444", linestyle="--",
                   linewidth=1, label=f"Moyenne : {mean_r2:.4f}")
        ax.legend(fontsize=9)
        ax.set_ylabel("R²")
        fig.patch.set_facecolor("white")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_c2:
        st.markdown('<div class="section-title">Feature Importance</div>',
                    unsafe_allow_html=True)
        importance = pd.Series(
            model.feature_importances_, index=feat_cols
        ).sort_values(ascending=True).tail(8)
        colors = ["#3B82F6" if v > importance.median() else "#CBD5E1"
                  for v in importance.values]
        fig, ax = plt.subplots(figsize=(5, 3))
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
    CITY_COLORS = {
        "Karachi": "#3B82F6", "Lahore": "#8B5CF6", "Islamabad": "#10B981",
        "Rawalpindi": "#F59E0B", "Faisalabad": "#EF4444"
    }

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.markdown('<div class="section-title">Prix médian par ville</div>',
                    unsafe_allow_html=True)
        city_med = df_raw.groupby("city")["price"].median().sort_values()
        fig, ax  = plt.subplots(figsize=(5, 3))
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
        log_p = np.log1p(df_raw["price"])
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

    col_e3, col_e4 = st.columns(2)

    with col_e3:
        st.markdown('<div class="section-title">Prix médian par type de bien</div>',
                    unsafe_allow_html=True)
        type_med = df_raw.groupby("property_type")["price"].median().sort_values()
        fig, ax  = plt.subplots(figsize=(5, 3))
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
            df_raw[df_raw["bedrooms"].between(1, 7)]
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
