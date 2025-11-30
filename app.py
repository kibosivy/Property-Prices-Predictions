# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path

# ---------- CONFIG ----------
MODEL_FILE = "ridge_model.pkl"
FEATURES_FILE = "feature_list.pkl"

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏠 House Price Predictor")
st.markdown("Fill in the property details below to estimate its **Sale Price**.")

# ---------- HELPERS ----------
def load_feature_list(path=FEATURES_FILE):
    fl = joblib.load(path)
    fl_clean = [re.sub(r'[\x00-\x1f\x7f-\x9f]+', '', str(c)).strip() for c in fl]
    return fl_clean

def group_features(feature_list):
    known_quality = {
        "ExterQual","ExterCond","BsmtQual","BsmtCond","HeatingQC",
        "KitchenQual","GarageQual","GarageCond","FireplaceQu",
        "OverallQual","OverallCond","BsmtExposure","BsmtFinType1","BsmtFinType2",
        "Functional","PavedDrive","Fence","PoolQC","GarageFinish"
    }
    numeric_patterns = [
        "sf","area","year","yr","age","bath","bedroom",
        "grliv","lot","cars","garage","total","porch","remod","qual","score"
    ]

    numeric, quality, onehot, others = [], [], {}, []
    for feat in feature_list:
        if feat.lower() == "saleprice":
            continue
        if feat in known_quality:
            quality.append(feat); continue
        if "_" in feat:
            prefix, suffix = feat.split("_",1)
            onehot.setdefault(prefix, []).append(feat); continue
        if any(p in feat.lower() for p in numeric_patterns):
            numeric.append(feat); continue
        others.append(feat)
    return sorted(set(numeric)), sorted(set(quality)), {k: sorted(v) for k,v in onehot.items()}, sorted(set(others))

def build_input_df(feature_list, numeric_inputs, quality_inputs, onehot_inputs, other_inputs):
    row = {f: 0.0 for f in feature_list}
    for d in (numeric_inputs, other_inputs):
        for k,v in d.items():
            if k in row: row[k] = float(v)
    for k,v in quality_inputs.items():
        if k in row: row[k] = float(v)
    for prefix, selection in onehot_inputs.items():
        if selection == "(none)": continue
        full = f"{prefix}_{selection}"
        if full not in row:
            cand = full.replace(" ","")
            if cand in row:
                full = cand
        if full in row:
            row[full] = 1.0
    return pd.DataFrame([row])

# ---------- LOAD MODEL & FEATURES ----------
if not Path(MODEL_FILE).exists() or not Path(FEATURES_FILE).exists():
    st.error("Missing model or feature list. Save `ridge_model.pkl` and `feature_list.pkl` to this folder and reload.")
    st.stop()

try:
    model = joblib.load(MODEL_FILE)
except Exception as e:
    st.error(f"Could not load model `{MODEL_FILE}`: {e}")
    st.stop()

try:
    feature_list = load_feature_list(FEATURES_FILE)
except Exception as e:
    st.error(f"Could not load feature list: {e}")
    st.stop()

numeric_features, quality_features, onehot_groups, other_features = group_features(feature_list)

# ---------- SIDEBAR (simplified) ----------
with st.sidebar:
    st.header("⚙️ Model Info")
    st.markdown("**Model:** Ridge Regression")
    st.markdown(f"**Features used:** {len(feature_list)}")
    st.markdown("---")
    st.header("📦 Downloads")
    try:
        with open(MODEL_FILE, "rb") as f:
            st.download_button("Download model (.pkl)", f, file_name=MODEL_FILE, key="dl_model")
        with open(FEATURES_FILE, "rb") as f:
            st.download_button("Download features (.pkl)", f, file_name=FEATURES_FILE, key="dl_feats")
    except Exception:
        st.info("Model/feature files available in project folder.")
    st.markdown("---")
    st.caption("Built by Ivy Kibos")

# ---------- GROUP PICKERS ----------
def pick_group(feats, keywords):
    return sorted([f for f in feats if any(k.lower() in f.lower() for k in keywords)])

garage_feats    = pick_group(numeric_features + other_features, ["garage"])
basement_feats  = pick_group(numeric_features + other_features, ["bsmt"])
porch_feats     = pick_group(numeric_features + other_features, ["porch","deck","screen"])
pool_feats      = pick_group(numeric_features + other_features, ["pool"])
fireplace_feats = pick_group(numeric_features + other_features, ["fireplace"])

grouped = set(garage_feats + basement_feats + porch_feats + pool_feats + fireplace_feats)
numeric_display = [f for f in numeric_features if f not in grouped]

# ---------- INPUT UI ----------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📏 Core Numeric Features")
    numeric_inputs = {}
    core_priority = ["OverallQual","OverallCond","GrLivArea","TotalSF","Total_Bathrooms",
                     "LotArea","LotFrontage","YearBuilt","YearRemodAdd"]
    for feat in core_priority:
        if feat in numeric_display:
            numeric_inputs[feat] = st.number_input(feat, value=0.0, key=f"num_{feat}")

    if numeric_display:
        with st.expander("More numeric features"):
            for feat in numeric_display:
                if feat in numeric_inputs: 
                    continue
                numeric_inputs[feat] = st.number_input(feat, value=0.0, key=f"num_more_{feat}")

    # Basement (no emoji)
    st.subheader("Basement Features")
    basement_inputs = {}
    if basement_feats:
        for feat in basement_feats:
            basement_inputs[feat] = st.number_input(feat, value=0.0, key=f"bsmt_{feat}")
    else:
        st.write("No basement numeric features detected.")

    st.subheader("🪵 Porch / Deck Features")
    porch_inputs = {}
    if porch_feats:
        for feat in porch_feats:
            porch_inputs[feat] = st.number_input(feat, value=0.0, key=f"porch_{feat}")
    else:
        porch_inputs = {}

with col_right:
    st.subheader("🚗 Garage Features")
    garage_inputs = {}
    if garage_feats:
        for feat in garage_feats:
            if "yr" in feat.lower() or "year" in feat.lower():
                garage_inputs[feat] = st.number_input(feat, value=2005, step=1, key=f"garage_{feat}")
            else:
                garage_inputs[feat] = st.number_input(feat, value=0.0, key=f"garage_{feat}")
    else:
        garage_inputs = {}

    st.subheader("🔥 Fireplace & 🏊 Pool Features")
    fp_inputs = {}
    if fireplace_feats:
        for feat in fireplace_feats:
            fp_inputs[feat] = st.number_input(feat, value=0.0, key=f"fp_{feat}")
    pool_inputs = {}
    if pool_feats:
        for feat in pool_feats:
            pool_inputs[feat] = st.number_input(feat, value=0.0, key=f"pool_{feat}")

# ---------- QUALITY FEATURES ----------
st.subheader("⭐ Quality Features")
quality_inputs = {}
QUAL_ORDER = ["Ex","Gd","TA","Fa","Po","None"]
QUAL_MAP = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"None":0}
for feat in quality_features:
    if feat == "OverallQual":
        quality_inputs[feat] = st.number_input("OverallQual (1–10)", min_value=1, max_value=10, value=6, key="OverallQual_input")
    else:
        choice = st.selectbox(feat, QUAL_ORDER, index=2, key=f"qual_{feat}")
        quality_inputs[feat] = QUAL_MAP.get(choice, 0)

# ---------- CATEGORICAL (ONE-HOT) ----------
st.subheader("📚 Categorical Options")
onehot_inputs = {}
for prefix in sorted(onehot_groups.keys()):
    suffixes = [c.split("_",1)[1] for c in onehot_groups[prefix]]
    suffixes = [s if s != "" else "(blank)" for s in suffixes]
    options = ["(none)"] + suffixes
    onehot_inputs[prefix] = st.selectbox(prefix, options, key=f"oh_{prefix}")

# ---------- OTHER FEATURES ----------
st.subheader("🔢 Other Numeric Features")
other_inputs = {}
for feat in other_features:
    # skip features already added as numeric/quality/onehot
    if feat in numeric_inputs or feat in quality_inputs or any(feat in lst for lst in onehot_groups.values()):
        continue
    other_inputs[feat] = st.number_input(feat, value=0.0, key=f"other_{feat}")

# ---------- PREDICTION ----------
st.markdown("---")
if st.button("Predict"):
    input_df = build_input_df(
        feature_list,
        numeric_inputs,
        quality_inputs,
        onehot_inputs,
        {**garage_inputs, **basement_inputs, **porch_inputs, **fp_inputs, **pool_inputs, **other_inputs}
    )
    input_df = input_df.reindex(columns=feature_list, fill_value=0.0)
    try:
        pred = model.predict(input_df)[0]
        # assume model predicts in raw price (no log) — show directly
        pred_price = float(pred)
        st.success(f"💰 Estimated Sale Price: **${pred_price:,.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("| RMSE: 18723 | R²: 0.91 |")
st.caption("© 2025 – Built by Ivy Kibos")