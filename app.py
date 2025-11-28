import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path

MODEL_FILE = "ridge_model.pkl"
FEATURES_FILE = "feature_list.pkl"

st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title("House Price Predictor")

def load_feature_list(path=FEATURES_FILE):
    fl = joblib.load(path)
    fl_clean = [re.sub(r'[\x00-\x1f\x7f-\x9f]+', '', str(c)).strip() for c in fl]
    return fl_clean

def group_features(feature_list):
    """Return numeric, ordinal, onehot_groups, other_features"""
    known_ordinals = {
        "ExterQual","ExterCond","BsmtQual","BsmtCond","HeatingQC",
        "KitchenQual","GarageQual","GarageCond","FireplaceQu",
        "OverallQual","OverallCond","BsmtExposure","BsmtFinType1","BsmtFinType2",
        "Functional","PavedDrive","Fence","PoolQC","GarageFinish"
    }
    numeric_patterns = ["sf","area","year","yr","age","bath","bedroom","grliv","lot","cars","garage","total","porch","mssubclass","remod","qual","score"]

    numeric, ordinal, onehot, others = [], [], {}, []
    for feat in feature_list:
        if feat.lower() == "saleprice":
            continue
        if feat in known_ordinals:
            ordinal.append(feat); continue
        if "_" in feat:
            prefix, suffix = feat.split("_",1)
            onehot.setdefault(prefix, []).append(feat); continue
        if any(p in feat.lower() for p in numeric_patterns):
            numeric.append(feat); continue
        others.append(feat)
    return sorted(set(numeric)), sorted(set(ordinal)), {k: sorted(v) for k,v in onehot.items()}, sorted(set(others))

def build_input_df(feature_list, numeric_inputs, ordinal_inputs, onehot_inputs, other_inputs):
    row = {f: 0.0 for f in feature_list}
    for d in (numeric_inputs, ordinal_inputs, other_inputs):
        for k,v in d.items():
            if k in row:
                row[k] = float(v)
    for prefix, selection in onehot_inputs.items():
        if selection == "(none)":
            continue
        full = f"{prefix}_{selection}"
        if full not in row:
            cand = full.replace(" ","")
            if cand in row: full = cand
        if full in row:
            row[full] = 1.0
    return pd.DataFrame([row])

if not Path(MODEL_FILE).exists() or not Path(FEATURES_FILE).exists():
    st.error(f"Missing `{MODEL_FILE}` or `{FEATURES_FILE}`. Save them to this folder and reload.")
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

numeric_features, ordinal_features, onehot_groups, other_features = group_features(feature_list)

def pick_group(feats, keywords):
    return sorted([f for f in feats if any(k.lower() in f.lower() for k in keywords)])

garage_feats    = pick_group(numeric_features + other_features, ["garage"])
basement_feats  = pick_group(numeric_features + other_features, ["bsmt","basement"])
porch_feats     = pick_group(numeric_features + other_features, ["porch","deck","openporch","screenporch","3ssn"])
pool_feats      = pick_group(numeric_features + other_features, ["pool"])
fireplace_feats = pick_group(numeric_features + other_features, ["fireplace"])

grouped = set(garage_feats + basement_feats + porch_feats + pool_feats + fireplace_feats)
numeric_display = [f for f in numeric_features if f not in grouped]

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Core numeric features")
    numeric_inputs = {}
    core_priority = ["OverallQual","OverallCond","GrLivArea","TotalSF","Total_Bathrooms","LotArea","LotFrontage","YearBuilt","YearRemodAdd"]
    for feat in core_priority:
        if feat in numeric_display:
            key = f"num_{feat}"
            default = 1500.0 if "SF" in feat or "Total" in feat or "GrLivArea" in feat else 0.0
            numeric_inputs[feat] = st.number_input(label=feat, value=float(default), key=key)
    if numeric_display:
        with st.expander("More numeric features"):
            for feat in numeric_display:
                if feat in numeric_inputs: continue
                key = f"num_{feat}"
                numeric_inputs[feat] = st.number_input(label=feat, value=0.0, key=key)

    st.subheader("Basement")
    basement_inputs = {}
    if basement_feats:
        for feat in basement_feats:
            key = f"bsmt_{feat}"
            basement_inputs[feat] = st.number_input(label=feat, value=0.0, key=key)
    else:
        st.write("No basement numeric features detected.")

    st.subheader("Porch / Deck")
    porch_inputs = {}
    if porch_feats:
        for feat in porch_feats:
            key = f"porch_{feat}"
            porch_inputs[feat] = st.number_input(label=feat, value=0.0, key=key)
    else:
        st.write("No porch features detected.")

with col_right:
    st.subheader("Garage")
    garage_inputs = {}
    if garage_feats:
        for feat in garage_feats:
            key = f"garage_{feat}"
            if "yr" in feat.lower() or "year" in feat.lower():
                garage_inputs[feat] = st.number_input(label=feat, value=2005, step=1, key=key)
            else:
                garage_inputs[feat] = st.number_input(label=feat, value=0.0, key=key)
    else:
        st.write("No garage numeric features detected.")

    st.subheader("Fireplace & Pool")
    fp_inputs = {}
    if fireplace_feats:
        for feat in fireplace_feats:
            key = f"fp_{feat}"
            fp_inputs[feat] = st.number_input(label=feat, value=0.0, key=key)
    if pool_feats:
        pool_inputs = {}
        for feat in pool_feats:
            key = f"pool_{feat}"
            pool_inputs[feat] = st.number_input(label=feat, value=0.0, key=key)
    else:
        pool_inputs = {}

st.subheader("Ordinal / quality features")
ordinal_inputs = {}
if ordinal_features:
    QUAL_ORDER = ["Ex","Gd","TA","Fa","Po","None"]
    QUAL_MAP = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"None":0}
    
    for feat in ordinal_features:
        key = f"ord_{feat}"
        if feat == "OverallQual":
            ordinal_inputs[feat] = st.number_input("OverallQual (1–10)", min_value=1, max_value=10, value=6, key=key)
        else:
            sel = st.selectbox(label=f"{feat} (encoded)", options=QUAL_ORDER, index=2, key=key)
            ordinal_inputs[feat] = QUAL_MAP.get(sel, 0)
else:
    st.write("No ordinal features detected.")

st.subheader("Categorical groups")
onehot_inputs = {}
for prefix in sorted(onehot_groups.keys()):
    suffixes = [col.split("_",1)[1] for col in onehot_groups[prefix]]
    suffixes = [s if s != "" else "(blank)" for s in suffixes]
    options = ["(none)"] + suffixes
    key = f"oh_{prefix}"
    choice = st.selectbox(f"{prefix} (select category)", options=options, key=key)
    onehot_inputs[prefix] = choice

st.subheader("Other features")
other_inputs = {}
for feat in other_features:
    if feat in numeric_inputs or feat in ordinal_inputs or any(feat in lst for lst in onehot_groups.values()):
        continue
    key = f"other_{feat}"
    if any(p in feat.lower() for p in ["yr","year","mo","sold","id"]):
        other_inputs[feat] = st.number_input(label=feat, value=0, step=1, key=key)
    else:
        other_inputs[feat] = st.number_input(label=feat, value=0.0, key=key)

st.markdown("---")
if st.button("Predict"):
    input_df = build_input_df(feature_list,
                              numeric_inputs,
                              ordinal_inputs,
                              onehot_inputs,
                              {**garage_inputs, **basement_inputs, **porch_inputs, **fp_inputs, **pool_inputs, **other_inputs})
    input_df = input_df.reindex(columns=feature_list, fill_value=0.0)

    try:
        pred_log = model.predict(input_df)[0]
        pred_price = np.expm1(pred_log)
        st.success(f"Estimated Sale Price: **${pred_price:,.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
