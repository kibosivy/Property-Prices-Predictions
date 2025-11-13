import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path

MODEL_FILE = "best_xgb_model.pkl"
FEATURES_FILE = "feature_list.pkl"

def load_feature_list(path=FEATURES_FILE):
    fl = joblib.load(path)
    fl_clean = [re.sub(r'[\x00-\x1f\x7f-\x9f]+', '', str(c)).strip() for c in fl]
    return fl_clean

def group_onehot_features(feature_list, numeric_patterns):
    numeric = []
    ordinal = []
    onehot = {}

    known_ordinals = {
        "ExterQual","ExterCond","BsmtQual","BsmtCond","HeatingQC",
        "KitchenQual","GarageQual","GarageCond","FireplaceQu",
        "OverallQual","OverallCond"
    }

    for feat in feature_list:
        if feat in known_ordinals:
            ordinal.append(feat)
            continue
        if any(p.lower() in feat.lower() for p in numeric_patterns):
            numeric.append(feat)
            continue
        if "_" in feat:
            prefix, suffix = feat.split("_", 1)
            onehot.setdefault(prefix, []).append(feat)
        else:
            numeric.append(feat)

    numeric = [f for f in numeric if f not in ordinal]
    return numeric, ordinal, onehot


QUAL_MAPPING = {"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1, "NF":0, "Zero":0}
QUAL_ORDER = ["Zero","Po","Fa","TA","Gd","Ex"]

# APP
st.set_page_config(page_title="House Price Prediction App", layout="wide")
st.title("🏠 House Price Prediction")
st.markdown("Enter property details and click **Predict** to estimate SalePrice.")

if not Path(MODEL_FILE).exists() or not Path(FEATURES_FILE).exists():
    st.error(f"Missing `{MODEL_FILE}` or `{FEATURES_FILE}` in folder.")
    st.stop()

model = joblib.load(MODEL_FILE)
feature_list = load_feature_list(FEATURES_FILE)

numeric_patterns = [
    "sf","area","year","yr","age","bath","bedroom","grliv","lot","cars",
    "garagecars","total","porch","mssubclass","remod","qual","score"
]

numeric_features, ordinal_features, onehot_groups = group_onehot_features(feature_list, numeric_patterns)

preferred_order = ["MSSubClass","LotFrontage","LotArea","OverallQual","YearBuilt","YearRemodAdd",
                   "TotalSF","GrLivArea","Total_Bathrooms","GarageCars","GarageArea","TotalBsmtSF"]

numeric_features = sorted(numeric_features, key=lambda x: (0 if x in preferred_order else 1, preferred_order.index(x) if x in preferred_order else x))

left, right = st.columns([1,1])

# Numeric inputs
with left:
    st.subheader("Numeric Features")
    numeric_inputs = {}
    for feat in numeric_features:
        default = 0.0
        if "bath" in feat.lower(): default = 2.0
        if feat == "OverallQual": default = 6.0
        if feat == "GarageCars": default = 2.0
        if feat == "TotalSF": default = 1500.0

        try:
            numeric_inputs[feat] = st.number_input(label=feat, value=float(default), step=1.0, format="%.4f")
        except:
            numeric_inputs[feat] = st.number_input(label=feat, value=0.0)

# Ordinal quality inputs
with left:
    if ordinal_features:
        st.subheader("Ordinal Quality Features")
        ordinal_inputs = {}
        for feat in ordinal_features:
            if feat == "OverallQual":
                ordinal_inputs[feat] = st.number_input("OverallQual (1–10)", min_value=1, max_value=10, value=6)
                continue

            sel = st.selectbox(feat, options=QUAL_ORDER, index=3)
            ordinal_inputs[feat] = QUAL_MAPPING[sel]

# One-hot grouped categoricals
with right:
    st.subheader("Categorical Features")
    onehot_inputs = {}

    for prefix in sorted(onehot_groups.keys()):
        suffixes = [f.split("_",1)[1] for f in onehot_groups[prefix]]
        suffixes = [s if s != "" else "(blank)" for s in suffixes]

        choice = st.selectbox(prefix, options=["(none)"] + suffixes)
        onehot_inputs[prefix] = choice

# Build input row
def build_input_df(feature_list, numeric_inputs, ordinal_inputs, onehot_groups, onehot_inputs):
    row = {f:0.0 for f in feature_list}

    for k,v in numeric_inputs.items():
        if k in row: row[k] = float(v)

    for k,v in ordinal_inputs.items():
        if k in row: row[k] = float(v)

    for prefix, selection in onehot_inputs.items():
        if selection == "(none)": continue
        full = f"{prefix}_{selection}"

        if full not in row:
            cand = full.replace(" ","")
            if cand in row: full = cand

        if full in row:
            row[full] = 1.0

    return pd.DataFrame([row])

# Prediction
st.markdown("---")
if st.button("Predict Sale Price"):
    input_df = build_input_df(feature_list, numeric_inputs, ordinal_inputs if 'ordinal_inputs' in locals() else {}, onehot_groups, onehot_inputs)
    input_df = input_df.reindex(columns=feature_list, fill_value=0.0)

    pred_log = model.predict(input_df)[0]
    pred_price = np.expm1(pred_log)

    st.success(f"Estimated Sale Price: **${pred_price:,.2f}**")

