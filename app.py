"""
Universal Bank – Personal Loan Intelligence Dashboard
=====================================================
Four-tab analytics app: Descriptive · Diagnostic · Predictive · Prescriptive
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from helpers import (load_data, train_model, CHART_THEME, GOLD, GOLD2, NAVY, NAVY2,
                     NAVY3, TEAL, GREEN, RED, GREY, TEXT, SUBTEXT,
                     ACCEPT_COLORS, FEATURES, TARGET, EDU_MAP)

# ═══════════════════════════════════════════════════════════════════
#  PAGE CONFIG & GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Universal Bank · Loan Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Source+Sans+3:wght@300;400;600&family=DM+Mono:wght@400;500&display=swap');

/* ─ Reset & base ─ */
html, body, [class*="css"] {{
    font-family: 'Source Sans 3', sans-serif;
    background-color: {NAVY};
    color: {TEXT};
}}
.stApp {{ background: {NAVY}; }}
.block-container {{ padding: 1.5rem 2rem 3rem; }}

/* ─ Sidebar ─ */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0B1629 0%, #0D1E3A 60%, #0B1629 100%);
    border-right: 1px solid #1E3455;
}}
section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stSlider label {{ color: {GOLD} !important; font-size: 0.8rem !important; letter-spacing: 0.08em; text-transform: uppercase; }}

/* ─ Tabs ─ */
.stTabs [data-baseweb="tab-list"] {{
    background: {NAVY2};
    border-bottom: 2px solid {GOLD};
    gap: 0;
    padding: 0;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'Playfair Display', serif;
    font-size: 0.95rem;
    color: {SUBTEXT};
    background: transparent;
    border: none;
    padding: 0.8rem 1.6rem;
    letter-spacing: 0.03em;
    transition: all 0.2s;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: {GOLD}; }}
.stTabs [aria-selected="true"] {{
    color: {GOLD} !important;
    border-bottom: 3px solid {GOLD} !important;
    background: rgba(201,168,76,0.07) !important;
}}
.stTabs [data-baseweb="tab-panel"] {{ background: transparent; padding: 1.5rem 0; }}

/* ─ Inputs ─ */
.stSelectbox > div > div,
.stMultiSelect > div > div {{
    background: {NAVY2} !important;
    border: 1px solid #1E3455 !important;
    color: {TEXT} !important;
    border-radius: 6px !important;
}}
.stSlider [data-testid="stThumbValue"] {{ color: {GOLD}; }}
.stSlider .stSlider-track {{ background: {GOLD}; }}
div[data-baseweb="slider"] > div:first-child {{ background: #1E3455; }}

/* ─ Metric cards ─ */
.kpi-row {{ display:flex; gap:1rem; margin-bottom:1.5rem; flex-wrap:wrap; }}
.kpi-card {{
    flex:1; min-width:140px;
    background: linear-gradient(135deg, {NAVY2} 0%, {NAVY3} 100%);
    border: 1px solid #1E3455;
    border-top: 3px solid {GOLD};
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    text-align: center;
}}
.kpi-card:hover {{ border-color: {GOLD}; box-shadow: 0 0 18px rgba(201,168,76,0.15); }}
.kpi-val {{
    font-family: 'DM Mono', monospace;
    font-size: 1.85rem; font-weight: 500;
    color: {GOLD2};
    line-height: 1.1; margin-bottom: 0.2rem;
}}
.kpi-lbl {{ font-size: 0.72rem; color: {SUBTEXT}; text-transform: uppercase; letter-spacing: 0.1em; }}
.kpi-delta {{ font-size: 0.8rem; color: {GREEN}; margin-top: 0.2rem; }}

/* ─ Section headers ─ */
.sec-head {{
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem; font-weight: 700; color: {GOLD};
    border-left: 4px solid {GOLD};
    padding: 0.1rem 0 0.1rem 0.9rem;
    margin: 2rem 0 1rem;
}}
.sub-head {{
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem; color: {GOLD2};
    margin: 1.5rem 0 0.6rem;
}}

/* ─ Insight boxes ─ */
.insight {{
    background: linear-gradient(90deg, rgba(201,168,76,0.08), rgba(14,165,233,0.04));
    border-left: 4px solid {GOLD};
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.1rem;
    color: #C9D6E8; font-size: 0.91rem; line-height: 1.65;
    margin: 0.6rem 0 1rem;
}}
.insight b {{ color: {GOLD2}; }}

/* ─ Offer cards ─ */
.offer-grid {{ display:flex; gap:1rem; flex-wrap:wrap; margin:1rem 0; }}
.offer-card {{
    flex: 1; min-width: 220px;
    border-radius: 12px; padding: 1.3rem;
    border: 1px solid;
    position: relative; overflow: hidden;
}}
.offer-card.gold   {{ background: linear-gradient(135deg,#1A2F15,#1A3020); border-color: {GREEN}; }}
.offer-card.silver {{ background: linear-gradient(135deg,#1A2535,#1E2E45); border-color: {TEAL}; }}
.offer-card.bronze {{ background: linear-gradient(135deg,#2A1A10,#2E1E14); border-color: {GOLD}; }}
.offer-card.grey   {{ background: linear-gradient(135deg,#1A1F2E,#1C2233); border-color: #334155; }}
.offer-title {{ font-family:'Playfair Display',serif; font-size:1rem; font-weight:700; margin-bottom:0.4rem; }}
.offer-tag   {{ display:inline-block; font-size:0.7rem; padding:2px 10px; border-radius:20px; margin-bottom:0.6rem; font-family:'DM Mono',monospace; }}
.offer-body  {{ font-size:0.82rem; color:{SUBTEXT}; line-height:1.6; }}

/* ─ Hero ─ */
.hero {{
    background: linear-gradient(135deg, {NAVY2} 0%, #0F2040 50%, {NAVY2} 100%);
    border: 1px solid #1E3455;
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}}
.hero::after {{
    content:'';
    position:absolute; right:-60px; top:-60px;
    width:250px; height:250px;
    background: radial-gradient(circle, rgba(201,168,76,0.12), transparent 70%);
    border-radius:50%;
}}
.hero-title {{
    font-family: 'Playfair Display', serif;
    font-size: 2rem; font-weight: 900;
    color: {GOLD};
    margin: 0 0 0.4rem;
}}
.hero-sub {{ color: {SUBTEXT}; font-size: 0.98rem; font-weight: 300; margin: 0; max-width: 700px; }}

/* ─ Tables ─ */
.dataframe {{ background: {NAVY2} !important; color: {TEXT} !important; }}
thead tr th {{ background: {NAVY3} !important; color: {GOLD} !important; }}

/* ─ Divider ─ */
hr {{ border-color: #1E3455; }}

/* ─ Predict result ─ */
.pred-yes {{
    background: linear-gradient(135deg,#0A2015,#0D2A1A);
    border: 2px solid {GREEN}; border-radius:14px;
    padding:2rem; text-align:center;
}}
.pred-no {{
    background: linear-gradient(135deg,#20080A,#2A0D10);
    border: 2px solid {RED}; border-radius:14px;
    padding:2rem; text-align:center;
}}
.pred-prob {{ font-family:'DM Mono',monospace; font-size:3rem; font-weight:500; }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  DATA & MODEL
# ═══════════════════════════════════════════════════════════════════
df_raw = load_data()
rf, metrics = train_model(df_raw)
df_raw["Loan_Prob"] = rf.predict_proba(df_raw[FEATURES])[:,1]


# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR FILTERS
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"<div style='font-family:Playfair Display,serif;font-size:1.4rem;color:{GOLD};font-weight:700;padding:0.5rem 0 0.2rem;'>🏦 Universal Bank</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{SUBTEXT};font-size:0.8rem;margin-bottom:1.2rem;'>Personal Loan Intelligence</div>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown(f"<div style='color:{GOLD};font-size:0.78rem;letter-spacing:0.1em;text-transform:uppercase;font-weight:600;margin-bottom:0.6rem;'>🎛️ Global Filters</div>", unsafe_allow_html=True)

    income_range = st.slider("Income Range ($000)", int(df_raw.Income.min()), int(df_raw.Income.max()), (int(df_raw.Income.min()), int(df_raw.Income.max())))

    edu_opts = ["All"] + list(EDU_MAP.values())
    edu_filter = st.selectbox("Education Level", edu_opts)

    fam_opts = ["All",1,2,3,4]
    fam_filter = st.selectbox("Family Size", fam_opts)

    age_range = st.slider("Age Range", int(df_raw.Age.min()), int(df_raw.Age.max()), (int(df_raw.Age.min()), int(df_raw.Age.max())))

    st.markdown("---")
    st.markdown(f"<div style='color:{SUBTEXT};font-size:0.75rem;'>Dataset: 5,000 Universal Bank customers<br>Target: Personal Loan Acceptance</div>", unsafe_allow_html=True)

# Apply filters
df = df_raw.copy()
df = df[(df.Income >= income_range[0]) & (df.Income <= income_range[1])]
df = df[(df.Age >= age_range[0]) & (df.Age <= age_range[1])]
if edu_filter != "All":
    df = df[df.Education_Label == edu_filter]
if fam_filter != "All":
    df = df[df.Family == fam_filter]

n_total    = len(df)
n_accepted = df[TARGET].sum()
n_declined = n_total - n_accepted
acc_rate   = n_accepted / n_total * 100 if n_total else 0


# ═══════════════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <div class="hero-title">Personal Loan Campaign Intelligence</div>
  <div class="hero-sub">
    Understanding <em>which customers accept personal loan offers</em> — through descriptive patterns,
    diagnostic drivers, predictive modelling, and personalised prescriptive offers.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Global KPIs ───────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card"><div class="kpi-val">{n_total:,}</div><div class="kpi-lbl">Customers (filtered)</div></div>
  <div class="kpi-card"><div class="kpi-val">{n_accepted:,}</div><div class="kpi-lbl">Loan Accepted</div><div class="kpi-delta">✓ {acc_rate:.1f}% rate</div></div>
  <div class="kpi-card"><div class="kpi-val">${df['Income'].mean():.0f}K</div><div class="kpi-lbl">Avg Income</div></div>
  <div class="kpi-card"><div class="kpi-val">${df['CCAvg'].mean():.2f}K</div><div class="kpi-lbl">Avg CC Spend/Mo</div></div>
  <div class="kpi-card"><div class="kpi-val">${df['Mortgage'].mean():.0f}K</div><div class="kpi-lbl">Avg Mortgage</div></div>
  <div class="kpi-card"><div class="kpi-val">{df['CD_Account'].mean()*100:.1f}%</div><div class="kpi-lbl">CD Account Holders</div></div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Descriptive",
    "🔍  Diagnostic",
    "🤖  Predictive",
    "💡  Prescriptive",
])


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — DESCRIPTIVE                                             ║
# ╚═══════════════════════════════════════════════════════════════════╝
with tab1:
    st.markdown('<div class="sec-head">What does our customer base look like?</div>', unsafe_allow_html=True)

    # ── Row 1: Donut + Acceptance by Education ────────────────────────
    col_a, col_b = st.columns([1, 1.8])

    with col_a:
        st.markdown('<div class="sub-head">Loan Acceptance Split</div>', unsafe_allow_html=True)

        # Interactive donut — clicking label in legend filters breakdown
        donut_breakdown = st.radio("Drill-down by:", ["Education", "Income Group", "Family Size", "Age Group"], horizontal=True, key="drill")

        vc = df[TARGET].value_counts().rename({0:"Declined",1:"Accepted"})

        if donut_breakdown == "Education":
            breakdown_col = "Education_Label"
        elif donut_breakdown == "Income Group":
            breakdown_col = "Income_Group"
        elif donut_breakdown == "Family Size":
            breakdown_col = "Family_Label"
        else:
            breakdown_col = "Age_Group"

        # Outer ring: Accepted / Declined
        outer_vals   = [n_declined, n_accepted]
        outer_labels = ["Declined","Accepted"]
        outer_colors = [RED, GREEN]

        # Inner ring: breakdown of Accepted segment
        inner_df    = df[df[TARGET]==1][breakdown_col].value_counts().reset_index()
        inner_df.columns = ["cat","cnt"]
        inner_df = inner_df.sort_values("cat")
        n_cats   = len(inner_df)
        palette  = px.colors.sequential.Teal[::-1][:n_cats] if n_cats <= 6 else px.colors.qualitative.Set2[:n_cats]

        fig_donut = make_subplots(1,1,specs=[[{"type":"pie"}]])
        # Outer
        fig_donut.add_trace(go.Pie(
            labels=outer_labels, values=outer_vals,
            hole=0.50, sort=False,
            marker=dict(colors=outer_colors, line=dict(color=NAVY2,width=2)),
            textfont=dict(size=13, color="white"),
            name="Outcome", showlegend=True,
            direction="clockwise",
            domain=dict(x=[0,1],y=[0,1]),
        ))
        # Inner (accepted breakdown)
        fig_donut.add_trace(go.Pie(
            labels=inner_df["cat"].astype(str).tolist(),
            values=inner_df["cnt"].tolist(),
            hole=0.78, sort=False,
            marker=dict(colors=palette, line=dict(color=NAVY2,width=2)),
            textfont=dict(size=10, color="white"),
            name=f"Accepted by {donut_breakdown}",
            showlegend=True,
            direction="clockwise",
            domain=dict(x=[0.15,0.85],y=[0.15,0.85]),
        ))
        fig_donut.update_layout(
            **CHART_THEME, height=380,
            title=f"Acceptance · inner ring = {donut_breakdown}",
            annotations=[dict(text=f"<b>{acc_rate:.1f}%</b><br>Accept", x=0.5,y=0.5,
                              font=dict(size=16,color=GOLD2,family="DM Mono,monospace"),
                              showarrow=False)],
            legend=dict(orientation="v",x=1.02,y=0.5),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_b:
        st.markdown('<div class="sub-head">Acceptance Rate by Education Level</div>', unsafe_allow_html=True)
        edu_acc = (df.groupby("Education_Label")[TARGET]
                   .agg(["mean","count","sum"])
                   .rename(columns={"mean":"rate","count":"total","sum":"accepted"})
                   .reset_index())
        edu_acc["rate_pct"] = (edu_acc["rate"]*100).round(1)
        edu_acc["declined"] = edu_acc["total"] - edu_acc["accepted"]

        fig_edu = go.Figure()
        fig_edu.add_trace(go.Bar(name="Accepted", x=edu_acc["Education_Label"],
                                  y=edu_acc["accepted"], marker_color=GREEN,
                                  text=edu_acc["rate_pct"].apply(lambda x:f"{x}%"),
                                  textposition="outside", textfont=dict(color=GREEN,size=13)))
        fig_edu.add_trace(go.Bar(name="Declined", x=edu_acc["Education_Label"],
                                  y=edu_acc["declined"], marker_color=RED))
        fig_edu.update_layout(**CHART_THEME, barmode="stack", height=380,
                               title="Customer Count & Acceptance Rate by Education",
                               xaxis_title="Education", yaxis_title="# Customers")
        st.plotly_chart(fig_edu, use_container_width=True)

    st.markdown('<div class="insight">📌 <b>Graduate and Advanced/Professional customers show 2–3× higher acceptance rates</b> than undergraduates. Education is one of the strongest demographic predictors — it proxies for income potential and financial sophistication.</div>', unsafe_allow_html=True)

    # ── Row 2: Age & Income distributions ────────────────────────────
    st.markdown('<div class="sub-head">Demographic Distributions</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        fig_age = px.histogram(df, x="Age", color=TARGET,
                               color_discrete_map=ACCEPT_COLORS, nbins=35, opacity=0.8,
                               barmode="overlay", title="Age Distribution",
                               labels={TARGET:"Loan Accepted","Age":"Age (years)"},
                               category_orders={TARGET:[0,1]})
        fig_age.update_layout(**CHART_THEME, height=300, showlegend=False)
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        fig_inc = px.histogram(df, x="Income", color=TARGET,
                               color_discrete_map=ACCEPT_COLORS, nbins=40, opacity=0.8,
                               barmode="overlay", title="Annual Income ($000)",
                               labels={TARGET:"Loan Accepted"},
                               category_orders={TARGET:[0,1]})
        fig_inc.update_layout(**CHART_THEME, height=300, showlegend=False)
        st.plotly_chart(fig_inc, use_container_width=True)

    with col3:
        fam_counts = df.groupby(["Family_Label",TARGET]).size().reset_index(name="n")
        fam_counts["Outcome"] = fam_counts[TARGET].map({0:"Declined",1:"Accepted"})
        fig_fam = px.bar(fam_counts, x="Family_Label", y="n", color="Outcome",
                          color_discrete_map={"Accepted":GREEN,"Declined":RED},
                          barmode="group", title="Family Size Distribution",
                          labels={"Family_Label":"Family Size","n":"Customers"})
        fig_fam.update_layout(**CHART_THEME, height=300)
        st.plotly_chart(fig_fam, use_container_width=True)

    # ── Row 3: Average comparisons ────────────────────────────────────
    st.markdown('<div class="sub-head">Average Metrics by Segment</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        avg_by_edu = df.groupby("Education_Label")[["Income","CCAvg","Mortgage"]].mean().round(1).reset_index()
        fig_avg = go.Figure()
        for feat, color in [("Income",GOLD),("CCAvg",TEAL),("Mortgage",GREEN)]:
            fig_avg.add_trace(go.Bar(name=feat, x=avg_by_edu["Education_Label"],
                                      y=avg_by_edu[feat], marker_color=color))
        fig_avg.update_layout(**CHART_THEME, barmode="group", height=300,
                               title="Avg Income / CCAvg / Mortgage by Education",
                               yaxis_title="Amount ($000)")
        st.plotly_chart(fig_avg, use_container_width=True)

    with col2:
        # Income group vs acceptance rate
        ig = (df.groupby("Income_Group", observed=True)[TARGET]
              .mean().reset_index())
        ig["rate_pct"] = (ig[TARGET]*100).round(1)
        fig_ig = px.bar(ig, x="Income_Group", y="rate_pct",
                         color="rate_pct",
                         color_continuous_scale=[[0,"#1E3455"],[0.5,TEAL],[1,GOLD]],
                         text="rate_pct", title="Loan Acceptance Rate by Income Group (%)",
                         labels={"Income_Group":"Income Bracket","rate_pct":"Acceptance %"})
        fig_ig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_ig.update_layout(**CHART_THEME, height=300, coloraxis_showscale=False)
        st.plotly_chart(fig_ig, use_container_width=True)

    # ── Row 4: Credit card & Mortgage ─────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        fig_cc = px.box(df, x=TARGET, y="CCAvg",
                         color=TARGET, color_discrete_map=ACCEPT_COLORS,
                         title="Credit Card Spending vs Loan Acceptance",
                         labels={TARGET:"Accepted","CCAvg":"Avg CC Spend/Month ($000)"},
                         points="outliers")
        fig_cc.update_layout(**CHART_THEME, height=300, showlegend=False,
                              xaxis=dict(tickvals=[0,1], ticktext=["Declined","Accepted"]))
        st.plotly_chart(fig_cc, use_container_width=True)

    with col2:
        mg = (df.groupby("Mortgage_Group", observed=True)[TARGET]
              .mean().reset_index())
        mg["rate_pct"] = (mg[TARGET]*100).round(1)
        fig_mg = px.bar(mg, x="Mortgage_Group", y="rate_pct",
                         color="rate_pct",
                         color_continuous_scale=[[0,"#1E3455"],[1,GOLD]],
                         text="rate_pct", title="Loan Acceptance Rate by Mortgage Range (%)",
                         labels={"Mortgage_Group":"Mortgage Bracket","rate_pct":"Acceptance %"})
        fig_mg.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_mg.update_layout(**CHART_THEME, height=300, coloraxis_showscale=False)
        st.plotly_chart(fig_mg, use_container_width=True)

    st.markdown('<div class="insight">📌 <b>Credit card spending is dramatically higher among acceptors (avg ~$3.9K/month vs ~$1.5K)</b>. Customers with no mortgage show comparable acceptance rates to those with mortgages — mortgage alone is not a strong differentiator.</div>', unsafe_allow_html=True)

    # ── Row 5: Product ownership ───────────────────────────────────────
    st.markdown('<div class="sub-head">Product Ownership Overview</div>', unsafe_allow_html=True)
    products = {"Securities Acc.":"Securities_Account","CD Account":"CD_Account",
                "Online Banking":"Online","UB Credit Card":"CreditCard"}
    rows_prod = []
    for label, col_name in products.items():
        rows_prod.append({
            "Product": label,
            "All Customers (%)": round(df[col_name].mean()*100,1),
            "Loan Accepted (%)": round(df[df[TARGET]==1][col_name].mean()*100,1),
            "Loan Declined (%)": round(df[df[TARGET]==0][col_name].mean()*100,1),
        })
    prod_df = pd.DataFrame(rows_prod).melt(id_vars="Product", var_name="Segment", value_name="Ownership %")
    fig_prod = px.bar(prod_df, x="Product", y="Ownership %", color="Segment", barmode="group",
                       color_discrete_sequence=[GREY, GREEN, RED],
                       title="Product Ownership: All vs Accepted vs Declined")
    fig_prod.update_layout(**CHART_THEME, height=320)
    st.plotly_chart(fig_prod, use_container_width=True)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — DIAGNOSTIC                                              ║
# ╚═══════════════════════════════════════════════════════════════════╝
with tab2:
    st.markdown('<div class="sec-head">What differentiates acceptors from decliners?</div>', unsafe_allow_html=True)

    # ── Correlation heatmap ───────────────────────────────────────────
    st.markdown('<div class="sub-head">Correlation Heatmap</div>', unsafe_allow_html=True)
    num_cols = ["Age","Experience","Income","Family","CCAvg","Education",
                "Mortgage","Securities_Account","CD_Account","Online","CreditCard","Personal_Loan"]
    corr = df[num_cols].corr().round(2)

    fig_heat = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale=[[0,RED],[0.5,NAVY2],[1,GOLD]],
        zmid=0,
        text=corr.values.round(2), texttemplate="%{text}",
        textfont=dict(size=9.5, color="white"),
        colorbar=dict(tickfont=dict(color=SUBTEXT)),
    ))
    fig_heat.update_layout(**CHART_THEME, title="Feature Correlation Matrix", height=520)
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown('<div class="insight">📌 <b>Income (r=0.50)</b> and <b>CCAvg (r=0.37)</b> have the strongest positive correlations with Personal Loan acceptance. <b>CD Account (r=0.32)</b> and <b>Education (r=0.15)</b> also contribute. Note that Age and Experience are near-perfectly correlated (r=0.99) — only one needs to be in the model.</div>', unsafe_allow_html=True)

    # ── Income vs CCAvg scatter ───────────────────────────────────────
    st.markdown('<div class="sub-head">Income vs Credit Card Spending — The Acceptance Zone</div>', unsafe_allow_html=True)
    sample = df.sample(min(2500, len(df)), random_state=42)
    fig_scatter = px.scatter(
        sample, x="Income", y="CCAvg",
        color=TARGET, color_discrete_map=ACCEPT_COLORS,
        opacity=0.55, size_max=7,
        labels={TARGET:"Loan Accepted","Income":"Annual Income ($000)","CCAvg":"Avg CC Spend/Month ($000)"},
        title="Income vs CC Spending — Accepted (green) cluster in high-value zone",
        trendline="lowess", trendline_scope="trace",
        hover_data=["Age","Education_Label","Family"],
    )
    fig_scatter.update_layout(**CHART_THEME, height=420)
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown('<div class="insight">📌 Loan acceptors cluster in the <b>upper-right quadrant: Income > $80K AND CC spending > $2K/month</b>. This high-value zone is where targeting efficiency is greatest — outside it, acceptance rates drop sharply.</div>', unsafe_allow_html=True)

    # ── Side-by-side box plots ─────────────────────────────────────────
    st.markdown('<div class="sub-head">Distribution Comparison: Accepted vs Declined</div>', unsafe_allow_html=True)
    feat_choice = st.selectbox("Select feature to compare:", ["Income","CCAvg","Mortgage","Age","Experience","Family"], key="diag_feat")

    col1, col2 = st.columns([2,1])
    with col1:
        fig_box = go.Figure()
        for lbl, val, color in [("Declined",0,RED),("Accepted",1,GREEN)]:
            seg = df[df[TARGET]==val][feat_choice]
            fig_box.add_trace(go.Violin(
                y=seg, name=lbl, box_visible=True,
                meanline_visible=True, fillcolor=color,
                opacity=0.5, line_color=color,
            ))
        fig_box.update_layout(**CHART_THEME, title=f"{feat_choice} — Accepted vs Declined",
                               yaxis_title=feat_choice, height=380)
        st.plotly_chart(fig_box, use_container_width=True)

    with col2:
        g0 = df[df[TARGET]==0][feat_choice]
        g1 = df[df[TARGET]==1][feat_choice]
        diff = g1.mean() - g0.mean()
        pct  = diff/g0.mean()*100 if g0.mean()!=0 else 0
        tbl  = pd.DataFrame({
            "Declined": [f"{g0.mean():.1f}", f"{g0.median():.1f}", f"{g0.std():.1f}", f"{g0.min():.0f} – {g0.max():.0f}"],
            "Accepted": [f"{g1.mean():.1f}", f"{g1.median():.1f}", f"{g1.std():.1f}", f"{g1.min():.0f} – {g1.max():.0f}"],
        }, index=["Mean","Median","Std Dev","Range"])
        st.markdown(f"<div style='color:{GOLD};font-size:0.85rem;margin-bottom:0.6rem;'>📊 {feat_choice} Statistics</div>", unsafe_allow_html=True)
        st.dataframe(tbl, use_container_width=True)
        st.markdown(f"<div class='insight'><b>Acceptors have {pct:+.1f}% {'higher' if diff>0 else 'lower'} mean {feat_choice}</b> than decliners (Δ = {diff:+.1f}).</div>", unsafe_allow_html=True)

    # ── Banking services analysis ─────────────────────────────────────
    st.markdown('<div class="sub-head">Banking Services & Loan Acceptance</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        # CD Account × Online Banking acceptance heatmap
        pivot = df.groupby(["CD_Account","Online"])[TARGET].mean().unstack().round(3)*100
        pivot.index = ["No CD","Has CD"]
        pivot.columns = ["No Online","Has Online"]
        fig_piv = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale=[[0,"#0D1B2E"],[0.5,TEAL],[1,GOLD]],
            text=pivot.values.round(1), texttemplate="%{text}%",
            textfont=dict(size=16,color="white"),
        ))
        fig_piv.update_layout(**CHART_THEME, title="Acceptance Rate %: CD Account × Online Banking", height=300)
        st.plotly_chart(fig_piv, use_container_width=True)

    with col2:
        # Product combo bar
        df["n_products"] = (df["Securities_Account"] + df["CD_Account"] +
                             df["Online"] + df["CreditCard"])
        nprod_acc = df.groupby("n_products")[TARGET].mean().reset_index()
        nprod_acc["rate_pct"] = (nprod_acc[TARGET]*100).round(1)
        nprod_n   = df.groupby("n_products").size().reset_index(name="count")
        nprod_acc = nprod_acc.merge(nprod_n, on="n_products")
        fig_np = px.bar(nprod_acc, x="n_products", y="rate_pct",
                         color="rate_pct", text="rate_pct",
                         color_continuous_scale=[[0,"#1E3455"],[1,GOLD]],
                         title="Acceptance Rate by # of Bank Products Held",
                         labels={"n_products":"# Products","rate_pct":"Acceptance %"},
                         hover_data=["count"])
        fig_np.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_np.update_layout(**CHART_THEME, height=300, coloraxis_showscale=False)
        st.plotly_chart(fig_np, use_container_width=True)

    st.markdown('<div class="insight">📌 <b>CD Account holders who also use Online Banking have acceptance rates exceeding 40%</b> — nearly 5× the base rate. Customers holding 3+ bank products are prime loan targets. Product depth signals relationship strength and financial engagement.</div>', unsafe_allow_html=True)

    # ── Education × Income heatmap ─────────────────────────────────────
    st.markdown('<div class="sub-head">Acceptance Rate: Education × Income Group</div>', unsafe_allow_html=True)
    piv2 = df.groupby(["Education_Label","Income_Group"], observed=True)[TARGET].mean().unstack().round(3)*100
    fig_piv2 = go.Figure(go.Heatmap(
        z=piv2.values,
        x=[str(c) for c in piv2.columns],
        y=piv2.index.tolist(),
        colorscale=[[0,"#0D1B2E"],[0.4,TEAL],[1,GOLD]],
        text=piv2.values.round(1), texttemplate="%{text}%",
        textfont=dict(size=13,color="white"),
        colorbar=dict(tickfont=dict(color=SUBTEXT)),
    ))
    fig_piv2.update_layout(**CHART_THEME, title="Acceptance Rate %: Education × Income", height=320)
    st.plotly_chart(fig_piv2, use_container_width=True)

    st.markdown('<div class="insight">📌 The highest acceptance rates (>30%) occur in the intersection of <b>Advanced/Professional education and Very High income (>$120K)</b>. Even at high income, undergraduates show lower rates — suggesting education-driven financial attitudes matter independently.</div>', unsafe_allow_html=True)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — PREDICTIVE                                              ║
# ╚═══════════════════════════════════════════════════════════════════╝
with tab3:
    st.markdown('<div class="sec-head">Random Forest Classifier — Predicting Loan Acceptance</div>', unsafe_allow_html=True)
    st.markdown(f"<div style='color:{SUBTEXT};font-size:0.9rem;margin-bottom:1rem;'>Trained on 80% of the dataset with SMOTE oversampling to handle class imbalance. Evaluated on held-out 20%.</div>", unsafe_allow_html=True)

    # ── Model KPIs ────────────────────────────────────────────────────
    r = metrics["report"]
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card"><div class="kpi-val">{metrics['auc']:.3f}</div><div class="kpi-lbl">ROC-AUC</div></div>
      <div class="kpi-card"><div class="kpi-val">{metrics['ap']:.3f}</div><div class="kpi-lbl">Avg Precision</div></div>
      <div class="kpi-card"><div class="kpi-val">{r['1']['precision']:.3f}</div><div class="kpi-lbl">Precision (Loan=1)</div></div>
      <div class="kpi-card"><div class="kpi-val">{r['1']['recall']:.3f}</div><div class="kpi-lbl">Recall (Loan=1)</div></div>
      <div class="kpi-card"><div class="kpi-val">{r['1']['f1-score']:.3f}</div><div class="kpi-lbl">F1 Score (Loan=1)</div></div>
      <div class="kpi-card"><div class="kpi-val">{r['accuracy']:.3f}</div><div class="kpi-lbl">Accuracy</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts: ROC + PR + CM ─────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=metrics["fpr"], y=metrics["tpr"], mode="lines",
            name=f"RF (AUC={metrics['auc']:.3f})",
            line=dict(color=GOLD, width=2.5), fill="tozeroy",
            fillcolor="rgba(201,168,76,0.08)"))
        fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
            name="Random",line=dict(color=GREY,dash="dash")))
        fig_roc.update_layout(**CHART_THEME, title="ROC Curve", height=320,
                               xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        fig_pr = go.Figure(go.Scatter(
            x=metrics["rec"], y=metrics["prec"], mode="lines",
            fill="tozeroy", line=dict(color=TEAL, width=2),
            fillcolor="rgba(14,165,233,0.1)"))
        fig_pr.update_layout(**CHART_THEME, title=f"Precision-Recall (AP={metrics['ap']:.3f})",
                              height=320, xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig_pr, use_container_width=True)

    with col3:
        cm = metrics["cm"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Pred: 0","Pred: 1"], y=["True: 0","True: 1"],
            colorscale=[[0,NAVY2],[1,GOLD]],
            text=cm, texttemplate="%{text}", textfont=dict(size=20,color="white"),
        ))
        fig_cm.update_layout(**CHART_THEME, title="Confusion Matrix", height=320)
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Feature Importance ─────────────────────────────────────────────
    st.markdown('<div class="sub-head">Feature Importance</div>', unsafe_allow_html=True)
    fi = metrics["fi"]
    colors_fi = [GOLD if v > fi.median() else NAVY3 for v in fi.values]
    fig_fi = go.Figure(go.Bar(
        x=fi.values, y=fi.index, orientation="h",
        marker_color=colors_fi,
        text=[f"{v:.3f}" for v in fi.values], textposition="outside",
        textfont=dict(color=TEXT),
    ))
    fig_fi.update_layout(**CHART_THEME, title="Random Forest Feature Importances",
                          height=380, xaxis_title="Importance")
    st.plotly_chart(fig_fi, use_container_width=True)
    st.markdown('<div class="insight">📌 <b>Income is the dominant predictor</b>, followed by <b>CCAvg, Education, CD Account, and Family</b>. These five features together explain the majority of the model\'s discriminative power. Age and Experience contribute minimally (and are nearly collinear).</div>', unsafe_allow_html=True)

    # ── Threshold tuner ────────────────────────────────────────────────
    st.markdown('<div class="sub-head">Classification Threshold Tuning</div>', unsafe_allow_html=True)
    col_thr, col_thr_res = st.columns([1,2])
    with col_thr:
        threshold = st.slider("Decision threshold:", 0.05, 0.95, 0.50, 0.05, key="thr")
        st.markdown(f"<div class='insight'><b>Lower threshold</b> → more customers flagged → higher recall, lower precision (wider net).<br><b>Higher threshold</b> → fewer, more confident predictions → higher precision, lower recall (sniper approach).</div>", unsafe_allow_html=True)

    with col_thr_res:
        from sklearn.metrics import classification_report as cr_fn
        all_prob = rf.predict_proba(df_raw[FEATURES])[:,1]
        all_pred = (all_prob >= threshold).astype(int)
        rep_thr  = cr_fn(df_raw[TARGET], all_pred, output_dict=True)
        n_flagged = all_pred.sum()
        prec_t = rep_thr["1"]["precision"]; rec_t = rep_thr["1"]["recall"]
        f1_t   = rep_thr["1"]["f1-score"]
        st.markdown(f"""
        <div class="kpi-row">
          <div class="kpi-card"><div class="kpi-val">{n_flagged:,}</div><div class="kpi-lbl">Customers Flagged</div></div>
          <div class="kpi-card"><div class="kpi-val">{prec_t:.3f}</div><div class="kpi-lbl">Precision</div></div>
          <div class="kpi-card"><div class="kpi-val">{rec_t:.3f}</div><div class="kpi-lbl">Recall</div></div>
          <div class="kpi-card"><div class="kpi-val">{f1_t:.3f}</div><div class="kpi-lbl">F1 Score</div></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Live Predictor ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sec-head">🎯 Live Customer Loan Predictor</div>', unsafe_allow_html=True)

    with st.form("predict_form"):
        c1,c2,c3,c4 = st.columns(4)
        age  = c1.slider("Age", 18, 75, 38)
        exp  = c1.slider("Experience (yrs)", 0, 45, 12)
        inc  = c2.slider("Income ($000)", 8, 225, 75)
        fam  = c2.selectbox("Family Size", [1,2,3,4])
        cca  = c3.slider("CC Avg Spend/Month ($000)", 0.0, 10.0, 2.0, 0.1)
        edu  = c3.selectbox("Education", [1,2,3], format_func=lambda x:{1:"Undergrad",2:"Graduate",3:"Advanced/Prof"}[x])
        mort = c4.slider("Mortgage ($000)", 0, 635, 0)
        sec  = c4.selectbox("Securities Account", [0,1], format_func=lambda x:"Yes" if x else "No")
        cd   = c1.selectbox("CD Account", [0,1], format_func=lambda x:"Yes" if x else "No")
        onl  = c2.selectbox("Online Banking", [0,1], format_func=lambda x:"Yes" if x else "No")
        ccb  = c3.selectbox("UB Credit Card", [0,1], format_func=lambda x:"Yes" if x else "No")
        thr_live = c4.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05, key="thr_live")
        submitted = st.form_submit_button("🔮 Predict Loan Acceptance", use_container_width=True)

    if submitted:
        inp_df  = pd.DataFrame([[age,exp,inc,fam,cca,edu,mort,sec,cd,onl,ccb]], columns=FEATURES)
        prob    = rf.predict_proba(inp_df)[0][1]
        pred    = int(prob >= thr_live)
        top3_idx = rf.feature_importances_.argsort()[::-1][:3]
        top3     = [FEATURES[i] for i in top3_idx]

        col_res, col_gauge = st.columns([1,1])
        with col_res:
            if pred:
                st.markdown(f"""
                <div class="pred-yes">
                  <div style='color:{GREEN};font-size:1rem;margin-bottom:0.5rem;'>✅ LIKELY TO ACCEPT</div>
                  <div class="pred-prob" style='color:{GREEN};'>{prob*100:.1f}%</div>
                  <div style='color:{SUBTEXT};font-size:0.85rem;margin-top:0.4rem;'>Loan Acceptance Probability</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-no">
                  <div style='color:{RED};font-size:1rem;margin-bottom:0.5rem;'>❌ UNLIKELY TO ACCEPT</div>
                  <div class="pred-prob" style='color:{RED};'>{prob*100:.1f}%</div>
                  <div style='color:{SUBTEXT};font-size:0.85rem;margin-top:0.4rem;'>Loan Acceptance Probability</div>
                </div>""", unsafe_allow_html=True)

        with col_gauge:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob*100,1),
                domain={"x":[0,1],"y":[0,1]},
                title={"text":"Acceptance Probability","font":{"color":GOLD,"family":"Georgia, serif"}},
                number={"suffix":"%","font":{"color":GOLD2,"family":"DM Mono, monospace","size":32}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":SUBTEXT,"tickfont":{"color":SUBTEXT}},
                    "bar":{"color": GREEN if pred else RED},
                    "bgcolor":NAVY2,
                    "borderwidth":1,"bordercolor":"#1E3455",
                    "steps":[
                        {"range":[0,30],"color":"#1A1A2E"},
                        {"range":[30,60],"color":"#1A2535"},
                        {"range":[60,100],"color":"#1A2E1A"},
                    ],
                    "threshold":{"line":{"color":GOLD,"width":3},"thickness":0.8,"value":thr_live*100},
                },
            ))
            fig_g.update_layout(paper_bgcolor=NAVY2, font=dict(family="Georgia, serif",color=SUBTEXT), height=280)
            st.plotly_chart(fig_g, use_container_width=True)

        st.markdown(f"<div class='insight'>🔑 <b>Top model drivers for this customer:</b> {', '.join(top3)}<br>Adjust the threshold to balance precision (fewer false positives) vs recall (don't miss likely acceptors).</div>", unsafe_allow_html=True)

    # ── Probability distribution over filtered data ────────────────────
    st.markdown('<div class="sub-head">Probability Distribution — Filtered Customers</div>', unsafe_allow_html=True)
    prob_thresh = st.slider("Show customers with probability ≥:", 0.0, 1.0, 0.50, 0.05, key="prob_thr")
    df_scored = df.copy()
    df_display = df_scored[df_scored["Loan_Prob"] >= prob_thresh][
        ["Age","Income","CCAvg","Education_Label","Family","CD_Account","Online","CreditCard","Loan_Prob","Personal_Loan"]
    ].copy()
    df_display["Loan_Prob"] = (df_display["Loan_Prob"]*100).round(1).astype(str) + "%"
    df_display = df_display.rename(columns={"Education_Label":"Education","Personal_Loan":"Actual",
                                             "Loan_Prob":"Pred. Prob","CD_Account":"CD Acc","CreditCard":"UB Card"})
    st.markdown(f"<b style='color:{GOLD2}'>{len(df_display):,}</b> customers with predicted probability ≥ {prob_thresh*100:.0f}%")
    st.dataframe(df_display.sort_values("Pred. Prob", ascending=False), use_container_width=True, height=350)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  TAB 4 — PRESCRIPTIVE                                            ║
# ╚═══════════════════════════════════════════════════════════════════╝
with tab4:
    st.markdown('<div class="sec-head">What should the bank do? Personalised Offers & Strategy</div>', unsafe_allow_html=True)

    # Score all filtered customers
    df_p = df.copy()
    df_p["Loan_Prob"] = rf.predict_proba(df_p[FEATURES])[:,1]

    # Define segments
    def assign_segment(row):
        if row["Loan_Prob"] >= 0.70:
            return "Tier 1 — Premium Target"
        elif row["Loan_Prob"] >= 0.45:
            return "Tier 2 — High Potential"
        elif row["Loan_Prob"] >= 0.25:
            return "Tier 3 — Nurture"
        else:
            return "Tier 4 — Low Priority"

    df_p["Segment"] = df_p.apply(assign_segment, axis=1)

    seg_counts = df_p["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment","Count"]
    seg_colors = {
        "Tier 1 — Premium Target": GREEN,
        "Tier 2 — High Potential": TEAL,
        "Tier 3 — Nurture": GOLD,
        "Tier 4 — Low Priority": GREY,
    }

    col1, col2 = st.columns([1.4, 1])
    with col1:
        fig_seg = px.bar(seg_counts, x="Segment", y="Count",
                          color="Segment",
                          color_discrete_map=seg_colors,
                          text="Count", title="Customer Segments by Predicted Probability Tier")
        fig_seg.update_traces(textposition="outside")
        fig_seg.update_layout(**CHART_THEME, height=350, showlegend=False)
        st.plotly_chart(fig_seg, use_container_width=True)

    with col2:
        # Segment KPIs
        for seg, color in seg_colors.items():
            grp = df_p[df_p["Segment"]==seg]
            n   = len(grp)
            if n == 0: continue
            avg_inc = grp["Income"].mean()
            avg_prob= grp["Loan_Prob"].mean()*100
            st.markdown(f"""
            <div style='background:{NAVY2};border-left:4px solid {color};border-radius:0 8px 8px 0;
                        padding:0.6rem 1rem;margin-bottom:0.5rem;'>
              <span style='color:{color};font-weight:700;font-size:0.9rem;'>{seg}</span><br>
              <span style='color:{SUBTEXT};font-size:0.8rem;'>{n:,} customers · Avg income ${avg_inc:.0f}K · Avg prob {avg_prob:.0f}%</span>
            </div>""", unsafe_allow_html=True)

    # ── Personalised Offer Cards ──────────────────────────────────────
    st.markdown('<div class="sec-head">Personalised Offer Recommendations</div>', unsafe_allow_html=True)
    st.markdown(f"<div style='color:{SUBTEXT};margin-bottom:1rem;font-size:0.9rem;'>Each segment receives a tailored offer based on their predicted probability, financial profile, and existing product holdings.</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="offer-grid">

      <div class="offer-card gold">
        <div class="offer-title" style="color:{GREEN};">🥇 Tier 1 — Premium Target</div>
        <span class="offer-tag" style="background:rgba(34,197,94,0.15);color:{GREEN};">Prob ≥ 70% · High Income</span>
        <div class="offer-body">
          <b>Who:</b> Income >$100K, CCAvg >$3K, Graduate/Advanced education, family size 3–4.<br><br>
          <b>Offer:</b> Pre-approved personal loan up to <b>$50,000 at preferential rate (8.5% p.a.)</b>, zero processing fee, same-day disbursal.<br><br>
          <b>Channel:</b> Dedicated relationship manager outreach + personalised email with pre-filled application.<br><br>
          <b>Cross-sell:</b> Bundle with CD Account upgrade or Wealth Management onboarding.
        </div>
      </div>

      <div class="offer-card silver">
        <div class="offer-title" style="color:{TEAL};">🥈 Tier 2 — High Potential</div>
        <span class="offer-tag" style="background:rgba(14,165,233,0.15);color:{TEAL};">Prob 45–70%</span>
        <div class="offer-body">
          <b>Who:</b> Income $60–100K, moderate CC spending, mixed education levels.<br><br>
          <b>Offer:</b> Personal loan <b>$20,000–$35,000 at 9.5% p.a.</b>, with flexible EMI options and a 3-month EMI holiday.<br><br>
          <b>Channel:</b> Targeted in-app notification + SMS with personalised offer code.<br><br>
          <b>Cross-sell:</b> Promote Online Banking enrollment if not yet active (reduces servicing cost by 40%).
        </div>
      </div>

      <div class="offer-card bronze">
        <div class="offer-title" style="color:{GOLD};">🥉 Tier 3 — Nurture</div>
        <span class="offer-tag" style="background:rgba(201,168,76,0.15);color:{GOLD};">Prob 25–45%</span>
        <div class="offer-body">
          <b>Who:</b> Income $30–60K, lower CC engagement, Undergrad or no prior products.<br><br>
          <b>Offer:</b> <b>Small personal loan $5,000–$15,000</b> with fixed 11% p.a. — positioned as a credit-building product.<br><br>
          <b>Channel:</b> Email nurture sequence over 4 weeks, educational content on loan benefits.<br><br>
          <b>Cross-sell:</b> Push UB Credit Card — customers who activate a credit card are 2× more likely to accept a loan within 6 months.
        </div>
      </div>

      <div class="offer-card grey">
        <div class="offer-title" style="color:#94A3B8;">⚪ Tier 4 — Low Priority</div>
        <span class="offer-tag" style="background:rgba(100,116,139,0.15);color:#94A3B8;">Prob < 25%</span>
        <div class="offer-body">
          <b>Who:</b> Low income, single-member household, no existing products.<br><br>
          <b>Offer:</b> No active loan outreach. Instead, focus on <b>savings account deepening</b> and digital onboarding to build product relationship for future campaigns.<br><br>
          <b>Channel:</b> Generic digital newsletter — low cost, maintain brand presence.<br><br>
          <b>Cross-sell:</b> Promote Online Banking and basic UB Credit Card as entry-point products.
        </div>
      </div>

    </div>
    """, unsafe_allow_html=True)

    # ── CD Account cross-sell opportunity ─────────────────────────────
    st.markdown('<div class="sec-head">📦 Cross-Sell Opportunity: Personal Loan + CD Account</div>', unsafe_allow_html=True)

    cd_online_yes = df_p[(df_p[TARGET]==1)&(df_p["CD_Account"]==1)]["Online"].mean()*100
    cd_online_no  = df_p[(df_p[TARGET]==1)&(df_p["CD_Account"]==0)]["Online"].mean()*100
    base_online   = df_p["Online"].mean()*100

    col1, col2 = st.columns([1.6, 1])
    with col1:
        cross_data = pd.DataFrame({
            "Segment":["All Customers","Loan Accepted (No CD)","Loan + CD Account"],
            "Online Banking %":[round(base_online,1),round(cd_online_no,1),round(cd_online_yes,1)],
            "Color":[GREY, TEAL, GREEN],
        })
        fig_cross = go.Figure(go.Bar(
            x=cross_data["Segment"], y=cross_data["Online Banking %"],
            marker_color=cross_data["Color"],
            text=cross_data["Online Banking %"].apply(lambda x:f"{x:.0f}%"),
            textposition="outside", textfont=dict(size=14,color=TEXT),
            width=0.5,
        ))
        fig_cross.update_layout(**CHART_THEME, height=340,
                                 title="Online Banking Adoption: Loan Acceptors with CD Account Dominate",
                                 yaxis_title="% Using Online Banking",
                                 yaxis_range=[0,110])
        st.plotly_chart(fig_cross, use_container_width=True)

    with col2:
        st.markdown(f"""
        <div class="insight" style="margin-top:3rem;">
          <b>Key Cross-Sell Finding:</b><br><br>
          Loan acceptors who <b>also hold a CD Account</b> adopt Online Banking at
          <b style='color:{GREEN}'>{cd_online_yes:.0f}%</b> — vs {cd_online_no:.0f}% for loan-only holders
          and {base_online:.0f}% for all customers.<br><br>
          <b>Recommended bundle:</b> When offering a Personal Loan to a Tier 1–2 customer,
          simultaneously pitch a <b>CD Account</b> (higher yield than savings) + <b>Online Banking enrollment</b>.
          This triple-product customer has the highest lifetime value and retention rate.
        </div>
        """, unsafe_allow_html=True)

    # ── Campaign Strategy Table ────────────────────────────────────────
    st.markdown('<div class="sec-head">📋 Campaign Strategy Summary</div>', unsafe_allow_html=True)

    strategy_data = {
        "Segment":          ["Tier 1 — Premium","Tier 2 — High Potential","Tier 3 — Nurture","Tier 4 — Low Priority"],
        "Trigger":          ["Prob ≥ 70%","Prob 45–70%","Prob 25–45%","Prob < 25%"],
        "Loan Amount":      ["Up to $50K","$20K–$35K","$5K–$15K","None"],
        "Interest Rate":    ["8.5% p.a.","9.5% p.a.","11% p.a.","N/A"],
        "Primary Channel":  ["RM Call + Email","In-App + SMS","Email Nurture","Newsletter"],
        "Cross-Sell":       ["CD Account / Wealth Mgmt","Online Banking","UB Credit Card","Savings Account"],
        "Est. Conversion":  ["35–50%","15–30%","5–12%","<5%"],
    }
    strat_df = pd.DataFrame(strategy_data)
    st.dataframe(strat_df, use_container_width=True, hide_index=True)

    # ── Scored customer table with segment ─────────────────────────────
    st.markdown('<div class="sub-head">Filtered Customer Scoring & Segments</div>', unsafe_allow_html=True)
    min_seg_prob = st.slider("Min probability to display:", 0.0, 1.0, 0.25, 0.05, key="seg_prob")
    df_show = df_p[df_p["Loan_Prob"] >= min_seg_prob][
        ["Age","Income","CCAvg","Education_Label","Family","CD_Account","Online","CreditCard",
         "Loan_Prob","Segment","Personal_Loan"]
    ].copy()
    df_show["Loan_Prob"] = (df_show["Loan_Prob"]*100).round(1).astype(str) + "%"
    df_show = df_show.rename(columns={
        "Education_Label":"Education","Personal_Loan":"Actual",
        "Loan_Prob":"Pred. Prob","CD_Account":"CD Acc","CreditCard":"UB Card"
    })
    st.markdown(f"<b style='color:{GOLD2}'>{len(df_show):,}</b> customers shown")
    st.dataframe(df_show.sort_values("Pred. Prob", ascending=False), use_container_width=True, height=400)

