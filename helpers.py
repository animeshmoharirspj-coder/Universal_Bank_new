"""
helpers.py — shared data loading, model training, and colour palette
"""
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, average_precision_score, precision_recall_curve
from imblearn.over_sampling import SMOTE

# ── Colour palette ────────────────────────────────────────────────────────────
GOLD   = "#C9A84C"
GOLD2  = "#E8C97A"
NAVY   = "#0B1629"
NAVY2  = "#112240"
NAVY3  = "#1A3357"
TEAL   = "#0EA5E9"
GREEN  = "#22C55E"
RED    = "#EF4444"
GREY   = "#64748B"
TEXT   = "#E2DFD8"
SUBTEXT= "#94A3B8"

CHART_THEME = dict(
    paper_bgcolor=NAVY2,
    plot_bgcolor="#0D1B2E",
    font=dict(family="Georgia, serif", color=SUBTEXT, size=12),
    xaxis=dict(gridcolor="#1E3455", linecolor="#1E3455", tickcolor=SUBTEXT),
    yaxis=dict(gridcolor="#1E3455", linecolor="#1E3455", tickcolor=SUBTEXT),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=SUBTEXT)),
    title=dict(font=dict(color=GOLD, size=15, family="Georgia, serif")),
)

ACCEPT_COLORS = {0: RED, 1: GREEN}

FEATURES = ["Age","Experience","Income","Family","CCAvg",
            "Education","Mortgage","Securities_Account","CD_Account","Online","CreditCard"]
TARGET   = "Personal_Loan"

EDU_MAP  = {1:"Undergrad", 2:"Graduate", 3:"Advanced/Prof"}
FAM_MAP  = {1:"Single (1)", 2:"Couple (2)", 3:"Family (3)", 4:"Family (4)"}

# ── Data loader ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("UniversalBank.csv")
    df.drop(columns=["ID","ZIP Code"], errors="ignore", inplace=True)
    df.columns = df.columns.str.strip().str.replace(" ","_")
    df["Education_Label"] = df["Education"].map(EDU_MAP)
    df["Family_Label"]    = df["Family"].map(FAM_MAP)
    df["Income_Group"] = pd.cut(df["Income"], bins=[0,40,80,120,225],
                                labels=["Low (<40K)","Mid (40–80K)","High (80–120K)","Very High (>120K)"])
    df["CCAvg_Group"]  = pd.cut(df["CCAvg"], bins=[-0.01,1,3,6,10.1],
                                labels=["<1K","1–3K","3–6K",">6K"])
    df["Age_Group"]    = pd.cut(df["Age"], bins=[22,35,45,55,67],
                                labels=["23–35","36–45","46–55","56–67"])
    df["Mortgage_Group"] = pd.cut(df["Mortgage"], bins=[-1,0,200,400,640],
                                  labels=["None","Low (<200K)","Mid (200–400K)","High (>400K)"])
    return df

# ── Model trainer ─────────────────────────────────────────────────────────────
@st.cache_resource
def train_model(df):
    X = df[FEATURES]; y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
    rf = RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=2,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_res, y_res)
    y_prob = rf.predict_proba(X_te)[:,1]
    y_pred = rf.predict(X_te)
    metrics = dict(
        auc   = roc_auc_score(y_te, y_prob),
        ap    = average_precision_score(y_te, y_prob),
        report= classification_report(y_te, y_pred, output_dict=True),
        cm    = confusion_matrix(y_te, y_pred),
        fpr   = roc_curve(y_te, y_prob)[0],
        tpr   = roc_curve(y_te, y_prob)[1],
        prec  = precision_recall_curve(y_te, y_prob)[0],
        rec   = precision_recall_curve(y_te, y_prob)[1],
        fi    = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True),
    )
    return rf, metrics

