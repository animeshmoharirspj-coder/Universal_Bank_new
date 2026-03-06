# 🏦 Universal Bank — Personal Loan Intelligence Dashboard

A production-grade **Streamlit** analytics dashboard covering all four analytics types to understand which customers are most likely to accept a personal loan offer.

## 🚀 Deploy on Streamlit Community Cloud

1. Push this repository (with the CSV included) to **GitHub**
2. Go to [share.streamlit.io](https://share.streamlit.io) → **"Deploy a public app"**
3. Select your repo · set **Main file path** → `app.py`
4. Click **Deploy** — done!

> ⚠️ Make sure `UniversalBank.csv` is in the **root of the repository** alongside `app.py`.

## 📂 File Structure

```
/
├── app.py               ← Main Streamlit dashboard (single-file)
├── helpers.py           ← Shared data loader, model trainer, colour palette
├── UniversalBank.csv    ← Dataset (5,000 Universal Bank customers)
├── requirements.txt     ← Python dependencies
└── README.md
```

## 📊 Dashboard Sections

| Tab | Analytics Type | Key Content |
|-----|---------------|-------------|
| 📊 Descriptive | *What happened?* | Interactive donut with drill-down (Education/Income/Family/Age), demographic histograms, product ownership, acceptance rates by segment |
| 🔍 Diagnostic | *Why did it happen?* | Correlation heatmap, Income×CCAvg scatter, violin plots, Education×Income acceptance heatmap, CD×Online banking matrix |
| 🤖 Predictive | *What will happen?* | Random Forest (ROC-AUC >0.97), PR curve, confusion matrix, feature importances, threshold tuner, live predictor with gauge |
| 💡 Prescriptive | *What should we do?* | 4-tier customer segmentation, personalised loan offers per tier, cross-sell opportunity analysis, campaign strategy table, scored customer export |

## 🎛️ Sidebar Filters
- Income range slider
- Education level selector
- Family size selector
- Age range slider

All charts and analyses update dynamically with filters.

## 🛠️ Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Dashboard framework |
| `pandas` / `numpy` | Data processing |
| `scikit-learn` | Random Forest, metrics |
| `imbalanced-learn` | SMOTE oversampling |
| `plotly` | Interactive charts |
| `mlxtend` | Association rules |

---

*Data: Hypothetical Universal Bank dataset — 5,000 customers*
