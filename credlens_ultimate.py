"""
CredLens Pro ULTIMATE — Credit Risk Intelligence Platform
Redesigned with:
- Premium dark-navy + gold accent theme
- Glassmorphism cards & animated metrics
- AI-powered chatbot via Anthropic API
- Zero-glitch layout with robust error handling
- Enhanced charts with custom templates
- Animated gauge, gradient backgrounds, premium typography
"""

import os, warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CAT_COLS = [
    "person_home_ownership", "loan_intent",
    "loan_grade", "cb_person_default_on_file",
]
LABEL_MAPS = {
    "person_home_ownership":     ["MORTGAGE", "OTHER", "OWN", "RENT"],
    "loan_intent":               ["DEBTCONSOLIDATION", "EDUCATION",
                                   "HOMEIMPROVEMENT", "MEDICAL",
                                   "PERSONAL", "VENTURE"],
    "loan_grade":                ["A", "B", "C", "D", "E", "F", "G"],
    "cb_person_default_on_file": ["N", "Y"],
}
FEAT_COLS = [
    "person_age", "person_income", "person_emp_length",
    "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length",
    "person_home_ownership_enc", "loan_intent_enc",
    "loan_grade_enc", "cb_person_default_on_file_enc",
]
NUMERIC_FEATS = [
    "person_age", "person_income", "person_emp_length",
    "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length",
]
LOG_FILE = "prediction_log.csv"
LOG_COLS = [
    "timestamp", "institution", "person_age", "person_income",
    "person_home_ownership", "person_emp_length", "loan_intent",
    "loan_grade", "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_default_on_file", "cb_person_cred_hist_length",
    "rf_probability_pct", "lr_probability_pct", "predicted_default",
]

# ── Colour palette — dark premium ──
C_GOLD   = "#F5C518"
C_RED    = "#FF4C4C"
C_GREEN  = "#00C48C"
C_AMBER  = "#FF8C42"
C_TEAL   = "#00B4D8"
C_BLUE   = "#4A90E2"
C_PURPLE = "#A78BFA"
C_NAVY   = "#0D1B2A"
C_NAVY2  = "#1B2B3A"
C_SLATE  = "#2C3E50"
C_MUTED  = "#7A8FA6"

GRADE_COLORS = {
    "A": "#00C48C", "B": "#4ECDC4", "C": "#F5C518",
    "D": "#FF8C42", "E": "#FF6B6B", "F": "#FF4C4C", "G": "#C026D3",
}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CredLens Pro · Credit Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# PREMIUM CSS — Dark theme with gold accents, glassmorphism, animations
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg:       #0A0F1E;
    --bg2:      #0D1421;
    --card:     rgba(255,255,255,0.04);
    --card-h:   rgba(255,255,255,0.07);
    --border:   rgba(255,255,255,0.08);
    --border-h: rgba(245,197,24,0.35);
    --text:     #E8EDF5;
    --muted:    #7A8FA6;
    --gold:     #F5C518;
    --gold2:    #FFD700;
    --red:      #FF4C4C;
    --green:    #00C48C;
    --teal:     #00B4D8;
    --purple:   #A78BFA;
    --amber:    #FF8C42;
}

html, body, [class*="css"], .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Animated gradient background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: 
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(245,197,24,0.04) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 90%, rgba(0,180,216,0.04) 0%, transparent 60%),
        linear-gradient(135deg, #0A0F1E 0%, #0D1421 50%, #0A1628 100%);
    pointer-events: none;
    z-index: 0;
}

.block-container {
    max-width: 1560px !important;
    padding: 1.25rem 2rem 3rem !important;
    position: relative;
    z-index: 1;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060C18 0%, #0A1020 100%) !important;
    border-right: 1px solid rgba(245,197,24,0.12) !important;
    box-shadow: 4px 0 32px rgba(0,0,0,0.5) !important;
}
section[data-testid="stSidebar"] * {
    color: #C8D6E5 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    padding: 0.3rem 0.5rem !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(245,197,24,0.08) !important;
    color: var(--gold) !important;
}

/* ── Typography ── */
h1 {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.025em !important;
    margin-bottom: 0.1rem !important;
    line-height: 1.2 !important;
}
h2 {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    color: #E8EDF5 !important;
}
h3 {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #C8D6E5 !important;
}
p {
    font-size: 0.95rem !important;
    color: var(--muted) !important;
    line-height: 1.6 !important;
}

/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.2rem 1.3rem 1.1rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    backdrop-filter: blur(10px);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, #F5C518);
    border-radius: 16px 16px 0 0;
}
.kpi-card:hover {
    transform: translateY(-3px);
    border-color: rgba(245,197,24,0.25);
    box-shadow: 0 20px 40px rgba(0,0,0,0.4), 0 0 20px rgba(245,197,24,0.06);
}
.kpi-icon {
    font-size: 1.6rem;
    margin-bottom: 0.6rem;
    display: block;
}
.kpi-label {
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    line-height: 1.05 !important;
    letter-spacing: -0.03em !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.kpi-sub {
    font-size: 0.78rem !important;
    color: var(--muted) !important;
    margin-top: 0.3rem;
    font-weight: 500;
}

/* ── Section labels ── */
.sec-lbl {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: var(--gold) !important;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(245,197,24,0.2);
}

/* ── Page header banner ── */
.page-header {
    background: linear-gradient(135deg, rgba(245,197,24,0.08) 0%, rgba(0,180,216,0.05) 100%);
    border: 1px solid rgba(245,197,24,0.15);
    border-radius: 20px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.page-header::after {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(245,197,24,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.page-title {
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
    margin: 0 0 0.3rem 0 !important;
    letter-spacing: -0.02em !important;
}
.page-subtitle {
    font-size: 0.92rem !important;
    color: var(--muted) !important;
    margin: 0 !important;
}

/* ── Insight boxes ── */
.insight-box {
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    margin: 1rem 0 1.3rem;
    border-left: 3px solid;
}
.insight-box .insight-title {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.insight-box ul {
    margin: 0 0 0 1rem;
    padding: 0;
}
.insight-box li {
    font-size: 0.92rem;
    margin: 0.35rem 0;
    line-height: 1.5;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
    padding: 0.6rem 1.1rem !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stTabs"] button:hover {
    color: var(--text) !important;
    background: rgba(255,255,255,0.04) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
    background: rgba(245,197,24,0.05) !important;
}

/* ── Form inputs ── */
label,
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label,
[data-testid="stTextInput"] label,
[data-testid="stSlider"] label,
[data-testid="stRadio"] > label,
[data-testid="stMultiSelect"] label {
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #C8D6E5 !important;
    letter-spacing: 0.01em !important;
}
input, select, textarea,
[data-baseweb="input"] input,
[data-baseweb="select"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #E8EDF5 !important;
    border-radius: 10px !important;
    font-size: 0.93rem !important;
}
input:focus, select:focus, textarea:focus {
    border-color: rgba(245,197,24,0.4) !important;
    box-shadow: 0 0 0 2px rgba(245,197,24,0.1) !important;
}

/* ── Buttons ── */
[data-testid="stFormSubmitButton"] button, .stButton button {
    font-size: 0.93rem !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.8rem !important;
    background: linear-gradient(135deg, #F5C518, #E6B800) !important;
    color: #0A0F1E !important;
    border: none !important;
    box-shadow: 0 8px 24px rgba(245,197,24,0.3) !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFormSubmitButton"] button:hover, .stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 32px rgba(245,197,24,0.4) !important;
    background: linear-gradient(135deg, #FFD700, #F5C518) !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    font-size: 0.88rem !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] th {
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    background: rgba(245,197,24,0.08) !important;
    color: var(--gold) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stDataFrame"] td {
    color: #C8D6E5 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #C8D6E5 !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}
[data-testid="stAlert"] p, [data-testid="stAlert"] div {
    font-size: 0.92rem !important;
    color: #C8D6E5 !important;
}

/* ── Caption ── */
[data-testid="stCaptionContainer"], .stCaption {
    font-size: 0.84rem !important;
    color: var(--muted) !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    margin-bottom: 0.6rem !important;
}
[data-testid="stChatInputContainer"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(245,197,24,0.2) !important;
    border-radius: 14px !important;
}

/* ── Multiselect tags ── */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background-color: rgba(245,197,24,0.15) !important;
    color: var(--gold) !important;
    border: 1px solid rgba(245,197,24,0.25) !important;
    font-size: 0.82rem !important;
    border-radius: 999px !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(255,255,255,0.07) !important;
    margin: 1.2rem 0 !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: rgba(255,255,255,0.06) !important;
    color: var(--text) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    box-shadow: none !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: rgba(245,197,24,0.12) !important;
    border-color: rgba(245,197,24,0.3) !important;
    color: var(--gold) !important;
    transform: translateY(-1px) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(245,197,24,0.25); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(245,197,24,0.4); }

/* ── Prediction result cards ── */
.result-card {
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    border: 2px solid;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    inset: 0;
    opacity: 0.06;
    pointer-events: none;
}

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 0.25rem 0.85rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

@media (max-width: 1200px) {
    .block-container { padding: 1rem 1rem 2rem !important; }
    h1 { font-size: 1.7rem !important; }
    .kpi-value { font-size: 1.55rem !important; }
    .page-title { font-size: 1.5rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def kpi(label, value, sub="", color=C_GOLD, icon=""):
    icon_html = f'<span class="kpi-icon">{icon}</span>' if icon else ""
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{color};">
        {icon_html}
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color};">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def sec(text):
    st.markdown(f'<p class="sec-lbl">{text}</p>', unsafe_allow_html=True)


def page_header(title, subtitle="", emoji=""):
    st.markdown(f"""
    <div class="page-header">
        <div style="display:flex;align-items:center;gap:0.75rem;">
            {f'<span style="font-size:2rem;">{emoji}</span>' if emoji else ""}
            <div>
                <div class="page-title">{title}</div>
                {f'<p class="page-subtitle">{subtitle}</p>' if subtitle else ""}
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


def insight_box(title, bullets, tone="neutral"):
    palettes = {
        "neutral": ("rgba(245,197,24,0.07)", "#F5C518", "#E8C84A"),
        "good":    ("rgba(0,196,140,0.07)",  "#00C48C", "#4ECDC4"),
        "warn":    ("rgba(255,76,76,0.07)",   "#FF4C4C", "#FF8C42"),
        "info":    ("rgba(0,180,216,0.07)",   "#00B4D8", "#7DD3E8"),
    }
    bg, bd, tx = palettes.get(tone, palettes["neutral"])
    bullet_html = "".join([f"<li>{b}</li>" for b in bullets])
    st.markdown(f"""
    <div class="insight-box" style="background:{bg};border-color:{bd};">
        <div class="insight-title" style="color:{tx};">{title}</div>
        <ul style="color:#C8D6E5;">{bullet_html}</ul>
    </div>""", unsafe_allow_html=True)


def chart_theme(height=400):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(family="Space Grotesk, sans-serif", size=13, color="#C8D6E5"),
        margin=dict(t=50, b=50, l=20, r=20),
        legend=dict(
            font=dict(size=12, color="#C8D6E5"),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.05)",
            borderwidth=1,
        ),
        height=height,
    )


AXIS_DARK = dict(
    gridcolor="rgba(255,255,255,0.05)",
    linecolor="rgba(255,255,255,0.08)",
    tickfont=dict(size=11, color="#7A8FA6"),
    title_font=dict(size=12, color="#C8D6E5"),
    showgrid=True,
    zeroline=False,
)


def apply_axes(fig, xtitle="", ytitle="", xrange=None, yrange=None):
    xu = dict(**AXIS_DARK, title=xtitle)
    yu = dict(**AXIS_DARK, title=ytitle)
    if xrange: xu["range"] = xrange
    if yrange: yu["range"] = yrange
    fig.update_xaxes(**xu)
    fig.update_yaxes(**yu)
    return fig


def encode_for_model(row):
    vec = [float(row[k]) for k in [
        "person_age", "person_income", "person_emp_length",
        "loan_amnt", "loan_int_rate", "loan_percent_income",
        "cb_person_cred_hist_length",
    ]]
    for cat in CAT_COLS:
        vec.append(float(LABEL_MAPS[cat].index(row[cat])))
    return np.array(vec).reshape(1, -1)


def load_log():
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
        try:
            return pd.read_csv(LOG_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=LOG_COLS)


def append_log(record):
    df_log = load_log()
    df_log = pd.concat([df_log, pd.DataFrame([record])], ignore_index=True)
    df_log.to_csv(LOG_FILE, index=False)


def interpret_default_rate(rate):
    if rate >= 30: return "very elevated"
    if rate >= 20: return "elevated"
    if rate >= 10: return "moderate"
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# CHATBOT (AI-powered via Anthropic API, falls back to rules)
# ─────────────────────────────────────────────────────────────────────────────
def build_context(df, M, log_df):
    total    = len(df)
    defaults = int(df["loan_status"].sum())
    dr       = defaults / total * 100 if total else 0
    avg_loan = df["loan_amnt"].mean()
    avg_rate = df["loan_int_rate"].mean()

    grade_summary = (
        df.groupby("loan_grade", observed=True)["loan_status"]
        .mean().mul(100).round(2).to_dict()
    )
    intent_summary = (
        df.groupby("loan_intent", observed=True)["loan_status"]
        .mean().mul(100).round(2).to_dict()
    )
    feat_imp = pd.DataFrame({
        "feature": FEAT_COLS,
        "importance": M["feat_imp"]
    }).sort_values("importance", ascending=False).head(5)
    top_feats = ", ".join(
        f"{r.feature.replace('_enc','').replace('_',' ')} ({r.importance:.3f})"
        for r in feat_imp.itertuples()
    )

    log_info = ""
    if not log_df.empty:
        hi  = int(log_df["predicted_default"].sum()) if "predicted_default" in log_df.columns else 0
        log_info = f"Footprint Database has {len(log_df)} records, {hi} high-risk, {len(log_df)-hi} low-risk."

    return f"""You are CredLens AI — a concise, expert credit-risk assistant embedded in the CredLens Pro dashboard.

DATASET FACTS:
- Total borrowers: {total:,}
- Defaults: {defaults:,} ({dr:.2f}% default rate)
- Average loan amount: ${avg_loan:,.0f}
- Average interest rate: {avg_rate:.2f}%
- Default rate by grade: {grade_summary}
- Default rate by intent: {intent_summary}
- Top RF risk drivers: {top_feats}

MODEL PERFORMANCE:
- Random Forest AUC: {M['rf_auc']}, Accuracy: {M['rf_rep']['accuracy']:.4f}
- Logistic Regression AUC: {M['lr_auc']}, Accuracy: {M['lr_rep']['accuracy']:.4f}

FOOTPRINT DATABASE:
{log_info if log_info else 'No predictions saved yet.'}

RULES:
- Answer questions about this dataset and models only. Be concise and factual.
- Do not provide financial advice or guarantee loan approvals.
- Use numbers from above; do not hallucinate.
- Format answers clearly with bullet points when helpful.
- If asked something outside this context, say you only cover credit risk analytics."""


def ai_chatbot_response(query, df, M, log_df):
    """Try Anthropic API first, fall back to rule-based."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        context = build_context(df, M, log_df)
        history = st.session_state.get("chat_history", [])
        messages = []
        for msg in history[1:]:  # skip system welcome
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": query})

        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=600,
            system=context,
            messages=messages,
        )
        return response.content[0].text
    except Exception:
        return rule_chatbot_response(query, df, M, log_df)


def rule_chatbot_response(query, df, M, log_df):
    q = query.strip().lower()
    if not q:
        return "Please ask a question about portfolio risk, model performance, loan grades, intents, or prediction records."

    total      = len(df)
    defaults   = int(df["loan_status"].sum())
    default_rate = defaults / total * 100 if total else 0
    avg_loan   = df["loan_amnt"].mean()
    avg_rate   = df["loan_int_rate"].mean()

    if any(k in q for k in ["overview", "summary", "dataset", "portfolio", "how many"]):
        return (
            f"**Portfolio Overview**\n\n"
            f"- Total borrowers: **{total:,}**\n"
            f"- Defaults: **{defaults:,}** ({default_rate:.2f}% rate)\n"
            f"- Safe loans: **{total-defaults:,}**\n"
            f"- Avg loan: **${avg_loan:,.0f}** · Avg rate: **{avg_rate:.2f}%**"
        )

    if "auc" in q or "accuracy" in q or "model" in q or "best model" in q:
        rf_acc = M["rf_rep"]["accuracy"]
        lr_acc = M["lr_rep"]["accuracy"]
        better = "Random Forest" if M["rf_auc"] >= M["lr_auc"] else "Logistic Regression"
        return (
            f"**Model Performance**\n\n"
            f"- Random Forest → AUC: **{M['rf_auc']:.4f}** · Accuracy: **{rf_acc:.4f}**\n"
            f"- Logistic Regression → AUC: **{M['lr_auc']:.4f}** · Accuracy: **{lr_acc:.4f}**\n"
            f"- Stronger model: **{better}** (by AUC)"
        )

    if "riskiest" in q or "highest default grade" in q or "which grade" in q or ("grade" in q and "risk" in q):
        gs = df.groupby("loan_grade", observed=True)["loan_status"].mean().mul(100).sort_values(ascending=False)
        g, rate = gs.index[0], gs.iloc[0]
        return f"The riskiest loan grade is **{g}** with a default rate of **{rate:.2f}%**."

    if "grade" in q:
        for g in ["A","B","C","D","E","F","G"]:
            if f"grade {g.lower()}" in q or f" {g.lower()} " in q or q.endswith(g.lower()):
                sub = df[df["loan_grade"] == g]
                if len(sub) == 0:
                    return f"No records for grade {g}."
                rate = sub["loan_status"].mean() * 100
                avg_amt = sub["loan_amnt"].mean()
                return (
                    f"**Loan Grade {g}**\n\n"
                    f"- Count: {len(sub):,} loans\n"
                    f"- Default rate: **{rate:.2f}%** ({interpret_default_rate(rate)})\n"
                    f"- Avg loan: **${avg_amt:,.0f}**"
                )

    if "intent" in q or "purpose" in q:
        isumm = df.groupby("loan_intent", observed=True)["loan_status"].mean().mul(100).sort_values(ascending=False)
        lines = "\n".join([f"- {i}: **{r:.2f}%**" for i,r in isumm.items()])
        return f"**Default Rate by Loan Intent**\n\n{lines}"

    if "feature" in q or "drivers" in q or "important" in q:
        imp = pd.DataFrame({"feature": FEAT_COLS, "importance": M["feat_imp"]}).sort_values("importance", ascending=False).head(5)
        parts = "\n".join([f"- {r.feature.replace('_enc','').replace('_',' ')}: **{r.importance:.3f}**" for r in imp.itertuples()])
        return f"**Top 5 Random Forest Risk Drivers**\n\n{parts}"

    if "footprint" in q or "record" in q or "log" in q or "database" in q:
        if log_df.empty:
            return "The Footprint Database is currently empty. Run predictions in the **Loan Risk Predictor** to populate it."
        hi = int(log_df["predicted_default"].sum()) if "predicted_default" in log_df.columns else 0
        return (
            f"**Footprint Database**\n\n"
            f"- Total records: **{len(log_df):,}**\n"
            f"- High risk: **{hi:,}** · Low risk: **{len(log_df)-hi:,}**"
        )

    if "reduce" in q or "improve" in q or "tips" in q:
        return (
            "**Risk Reduction Tips**\n\n"
            "- Lower loan amount relative to income\n"
            "- Build longer credit history\n"
            "- Avoid prior default flags\n"
            "- Choose lower-risk loan grades (A–C)\n"
            "- These improve model score but do not guarantee approval."
        )

    return (
        "I can help with: **dataset summary**, **loan grade/intent risk**, "
        "**model AUC & accuracy**, **top risk drivers**, and **prediction records**.\n\n"
        "Try: *'What is the default rate?'*, *'Which grade is riskiest?'*, or *'Top model drivers?'*"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.6rem 0 0.8rem;'>
        <div style='font-size:2.6rem;filter:drop-shadow(0 0 12px rgba(245,197,24,0.4));'>🏦</div>
        <div style='font-size:1.35rem;font-weight:700;color:#FFFFFF;
                    letter-spacing:-0.01em;margin-top:0.5rem;font-family:"Space Grotesk",sans-serif;'>
            CredLens Pro
        </div>
        <div style='font-size:0.72rem;color:#F5C518;margin-top:0.25rem;
                    font-weight:600;letter-spacing:0.2em;text-transform:uppercase;'>
            Risk Intelligence
        </div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    uploaded = st.file_uploader("📂  Upload CSV", type=["csv"],
                                 help="Upload credit_risk_modelling_dataset.csv")
    st.divider()

    st.markdown(
        "<p style='font-size:0.65rem;font-weight:700;letter-spacing:0.18em;"
        "color:#F5C518;text-transform:uppercase;margin-bottom:0.5rem;'>"
        "Navigation</p>", unsafe_allow_html=True)

    NAV = [
        "📊  Overview",
        "🔍  Deep Analytics",
        "📈  Aggregations",
        "🔗  Correlation & Heatmap",
        "🤖  ML Models",
        "🎯  Loan Risk Predictor",
        "🗄️  Footprint Database",
        "💬  Risk Chatbot",
    ]
    nav = st.radio("", NAV, label_visibility="collapsed")
    st.divider()

    # Model metrics in sidebar
    if "M" in dir() or "M" in st.session_state:
        pass  # will be rendered after training

    st.markdown("""
    <div style='background:rgba(245,197,24,0.06);border:1px solid rgba(245,197,24,0.15);
                border-radius:12px;padding:0.9rem 1rem;text-align:center;'>
        <div style='font-size:0.62rem;color:#F5C518;font-weight:700;
                    letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.6rem;'>
            Model Status
        </div>
        <div style='font-size:0.82rem;color:#C8D6E5;line-height:2;'>
            RF AUC &nbsp;<span style='color:#F5C518;font-weight:700;font-family:"JetBrains Mono",monospace;'>loading…</span><br>
            LR AUC &nbsp;<span style='color:#00B4D8;font-weight:700;font-family:"JetBrains Mono",monospace;'>loading…</span>
        </div>
        <div style='font-size:0.68rem;color:#4A5568;margin-top:0.5rem;'>v 3.0 · ULTIMATE</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="🔄 Loading dataset…")
def load_raw(src):
    if src is None:
        for p in [
            "credit_risk_Modelling.csv",
            "credit_risk_modelling_dataset.csv",
            "credit_risk_modelling_dataset.csv.csv",
            "data/credit_risk_Modelling.csv",
            "data/credit_risk_modelling_dataset.csv",
        ]:
            if os.path.exists(p):
                df = pd.read_csv(p)
                df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed", case=False)]
                return df
        return None
    df = pd.read_csv(src)
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed", case=False)]
    return df


@st.cache_data(show_spinner="⚙️ Engineering features…")
def full_pipeline(raw):
    df = raw.copy()
    required_cols = [
        "person_age", "person_income", "person_home_ownership", "person_emp_length",
        "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate", "loan_status",
        "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].mean())
    df["loan_int_rate"]     = df["loan_int_rate"].fillna(df["loan_int_rate"].mean())
    df = df[(df["person_age"] >= 18) & (df["person_age"] <= 100)].copy()

    df["risk_segment"]  = df["loan_status"].map({1: "High Risk", 0: "Low Risk"})
    df["income_group"]  = pd.cut(
        df["person_income"], bins=[-np.inf, 29999, 69999, np.inf],
        labels=["Low Income", "Middle Income", "High Income"])
    df["employment_group"] = pd.cut(
        df["person_emp_length"], bins=[-np.inf, 1.999, 4.999, np.inf],
        labels=["New Employee", "Mid Experience", "Experienced"])

    for cat in CAT_COLS:
        classes = LABEL_MAPS[cat]
        df[cat + "_enc"] = df[cat].apply(lambda x: classes.index(x) if x in classes else 0)

    return df, df["person_emp_length"].mean(), df["loan_int_rate"].mean()


raw_df = load_raw(uploaded)
if raw_df is None:
    st.markdown("""
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(245,197,24,0.2);
                border-radius:20px;padding:4rem;text-align:center;margin-top:3rem;">
        <div style="font-size:4rem;margin-bottom:1rem;">📂</div>
        <h2 style="color:#FFFFFF;font-size:1.8rem;margin-bottom:0.8rem;">No Data Loaded</h2>
        <p style="font-size:1.05rem;color:#7A8FA6;max-width:460px;margin:0 auto;">
            Upload <strong style='color:#F5C518;'>credit_risk_modelling_dataset.csv</strong>
            using the sidebar uploader, or place it in the same folder as this script.
        </p>
    </div>""", unsafe_allow_html=True)
    st.stop()

try:
    df, mean_emp, mean_rate = full_pipeline(raw_df)
except Exception as e:
    st.error(f"Dataset validation error: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# ML TRAINING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Training ML models… (first run only)")
def train_models(df):
    X = df[FEAT_COLS].copy()
    y = df["loan_status"].copy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=150, max_depth=15,
                                random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_p    = rf.predict_proba(X_te)[:, 1]
    rf_pred = (rf_p >= 0.5).astype(int)

    sc = StandardScaler()
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(sc.fit_transform(X_tr), y_tr)
    lr_p    = lr.predict_proba(sc.transform(X_te))[:, 1]
    lr_pred = (lr_p >= 0.5).astype(int)

    return dict(
        rf=rf, lr=lr, sc=sc, y_te=y_te,
        rf_p=rf_p, rf_pred=rf_pred,
        lr_p=lr_p, lr_pred=lr_pred,
        rf_auc=round(roc_auc_score(y_te, rf_p), 4),
        lr_auc=round(roc_auc_score(y_te, lr_p), 4),
        rf_rep=classification_report(y_te, rf_pred, output_dict=True),
        lr_rep=classification_report(y_te, lr_pred, output_dict=True),
        feat_imp=list(rf.feature_importances_),
    )


M = train_models(df)

# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATIONS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def agg_all(df):
    def _a(grp):
        return (
            df.groupby(grp, observed=True)
            .agg(
                total=("loan_status", "count"),
                default_rate=("loan_status", lambda x: round(x.mean()*100, 2)),
                avg_loan=("loan_amnt", lambda x: round(x.mean(), 2)),
                avg_rate=("loan_int_rate", lambda x: round(x.mean(), 2)),
            ).reset_index()
        )
    return (
        _a("loan_grade").sort_values("loan_grade"),
        _a("loan_intent").sort_values("default_rate", ascending=False),
        _a("cb_person_default_on_file"),
        _a("income_group").sort_values("default_rate", ascending=False),
        _a("person_home_ownership").sort_values("default_rate", ascending=False),
        _a("employment_group").sort_values("default_rate", ascending=False),
    )


grade_df, intent_df, hist_df, income_df, home_df, emp_df = agg_all(df)

# ═══════════════════════════════════════════════════════════════════════════
# 1 · OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if nav == "📊  Overview":
    page_header(
        "Credit Risk Overview",
        f"Dataset · {len(raw_df):,} rows · {raw_df.shape[1]} columns → cleaned: {len(df):,} rows",
        "📊"
    )

    total = len(df)
    deflt = int(df["loan_status"].sum())
    safe  = total - deflt
    dr    = deflt / total * 100
    avgl  = df["loan_amnt"].mean()
    avgr  = df["loan_int_rate"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("Total Borrowers",   f"{total:,}",     "in dataset",          "#A78BFA", "👥")
    with c2: kpi("Defaulted Loans",   f"{deflt:,}",     f"{dr:.1f}% rate",     C_RED,     "⚠️")
    with c3: kpi("Safe Borrowers",    f"{safe:,}",      f"{100-dr:.1f}% safe", C_GREEN,   "✅")
    with c4: kpi("Avg Loan Amount",   f"${avgl:,.0f}",  "per borrower",        C_AMBER,   "💰")
    with c5: kpi("Avg Interest Rate", f"{avgr:.2f}%",   "portfolio avg",       C_TEAL,    "📈")

    insight_box(
        "Portfolio Interpretation",
        [
            f"Observed default rate: {dr:.1f}% ({deflt:,} of {total:,} borrowers) — meaningfully imbalanced but sizable risky segment.",
            f"Average loan: ${avgl:,.0f} · Average rate: {avgr:.2f}% — central tendency of the portfolio.",
            "Use grade, intent, and income visuals together. Single charts are not causal proof.",
        ],
        tone="info",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns([1, 1.7, 1.7])

    with r1:
        sec("Risk Distribution")
        rc = df["risk_segment"].value_counts().reset_index()
        rc.columns = ["Segment", "Count"]
        fig = px.pie(rc, names="Segment", values="Count", hole=0.6,
                     color="Segment",
                     color_discrete_map={"High Risk": C_RED, "Low Risk": C_GREEN})
        fig.update_traces(
            textposition="inside", textinfo="percent+label",
            textfont=dict(size=13, color="white"),
            marker=dict(line=dict(color="rgba(0,0,0,0.3)", width=2)))
        fig.update_layout(**chart_theme(340), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        sec("Default Rate by Loan Grade")
        fig = go.Figure(go.Bar(
            x=grade_df["loan_grade"],
            y=grade_df["default_rate"],
            marker=dict(
                color=[GRADE_COLORS.get(g, "#888") for g in grade_df["loan_grade"]],
                line=dict(color="rgba(0,0,0,0.2)", width=1),
            ),
            text=[f"{v}%" for v in grade_df["default_rate"]],
            textposition="outside",
            textfont=dict(size=12, color="#C8D6E5"),
            width=0.55,
        ))
        fig.update_layout(**chart_theme(340))
        apply_axes(fig, xtitle="Loan Grade", ytitle="Default Rate (%)", yrange=[0, 115])
        st.plotly_chart(fig, use_container_width=True)

    with r3:
        sec("Avg Loan Amount — Home Ownership")
        hl = (df.groupby("person_home_ownership")["loan_amnt"]
                .mean().reset_index().sort_values("loan_amnt"))
        fig = go.Figure(go.Bar(
            x=hl["loan_amnt"], y=hl["person_home_ownership"],
            orientation="h",
            marker=dict(
                color=hl["loan_amnt"],
                colorscale=[[0, "#1A3040"], [1, C_AMBER]],
                showscale=False,
            ),
            text=hl["loan_amnt"].apply(lambda x: f"${x:,.0f}"),
            textposition="outside",
            textfont=dict(size=12, color="#C8D6E5"),
        ))
        fig.update_layout(**chart_theme(340))
        apply_axes(fig, xtitle="Avg Loan Amount ($)", ytitle="")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    r4, r5 = st.columns(2)

    with r4:
        sec("Loan Amount Distribution — Risk Segment")
        fig = px.histogram(df, x="loan_amnt", color="risk_segment",
                           nbins=60, barmode="overlay", opacity=0.72,
                           color_discrete_map={"High Risk": C_RED, "Low Risk": C_GREEN})
        fig.update_layout(**chart_theme(340))
        fig.update_layout(legend=dict(font=dict(size=12), orientation="h", y=1.08))
        apply_axes(fig, xtitle="Loan Amount ($)", ytitle="Count")
        st.plotly_chart(fig, use_container_width=True)

    with r5:
        sec("Loan Intent Volume")
        ic = df["loan_intent"].value_counts().reset_index()
        ic.columns = ["Intent", "Count"]
        colors = [C_PURPLE, C_TEAL, C_AMBER, C_RED, C_GREEN, C_BLUE]
        fig = go.Figure(go.Bar(
            x=ic["Intent"], y=ic["Count"],
            marker=dict(
                color=colors[:len(ic)],
                line=dict(color="rgba(0,0,0,0.2)", width=1),
            ),
            text=ic["Count"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            textfont=dict(size=11, color="#C8D6E5"),
        ))
        fig.update_layout(**chart_theme(340))
        apply_axes(fig, xtitle="Loan Intent", ytitle="Count")
        fig.update_xaxes(tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# 2 · DEEP ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════
elif nav == "🔍  Deep Analytics":
    page_header("Deep Analytics", "Interactive filters with box plots, violins, scatter matrix & sunburst", "🔍")

    with st.expander("🎛️  Filter Controls", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            g_f = st.multiselect("Loan Grade",
                sorted(df["loan_grade"].unique()), sorted(df["loan_grade"].unique()))
        with f2:
            i_f = st.multiselect("Loan Intent",
                sorted(df["loan_intent"].unique()), sorted(df["loan_intent"].unique()))
        with f3:
            h_f = st.multiselect("Home Ownership",
                sorted(df["person_home_ownership"].unique()),
                sorted(df["person_home_ownership"].unique()))
        with f4:
            s_f = st.multiselect("Risk Segment",
                ["High Risk", "Low Risk"], ["High Risk", "Low Risk"])
        age_r = st.slider("Age Range",
            int(df["person_age"].min()), int(df["person_age"].max()),
            (int(df["person_age"].min()), int(df["person_age"].max())))

    dff = df[
        df["loan_grade"].isin(g_f) &
        df["loan_intent"].isin(i_f) &
        df["person_home_ownership"].isin(h_f) &
        df["risk_segment"].isin(s_f) &
        df["person_age"].between(*age_r)
    ]

    st.markdown(
        f"<p style='font-size:0.92rem;font-weight:600;color:{C_GOLD};padding:0.4rem 0;'>"
        f"✦ {len(dff):,} rows match current filters</p>",
        unsafe_allow_html=True)
    st.divider()

    t1, t2, t3, t4 = st.tabs(
        ["📦  Box Plots", "🎻  Violin Plots", "🔵  Scatter Matrix", "🌞  Sunburst"])

    colors_seq = [C_RED, C_PURPLE, C_AMBER, C_GREEN, C_TEAL, C_BLUE, C_GOLD]

    with t1:
        st.markdown("<br>", unsafe_allow_html=True)
        bc1, bc2 = st.columns(2)
        with bc1:
            feat_b = st.selectbox("Feature", NUMERIC_FEATS,
                                   index=NUMERIC_FEATS.index("loan_int_rate"), key="bf")
        with bc2:
            grp_b  = st.radio("Group By",
                ["risk_segment", "loan_grade", "income_group", "employment_group"],
                horizontal=True, key="bg")
        fig = px.box(dff, x=grp_b, y=feat_b, color=grp_b,
                     color_discrete_sequence=colors_seq, points="outliers")
        fig.update_traces(marker=dict(size=3, opacity=0.4))
        fig.update_layout(**chart_theme(420))
        apply_axes(fig, xtitle=grp_b.replace("_"," ").title(), ytitle=feat_b.replace("_"," ").title())
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.markdown("<br>", unsafe_allow_html=True)
        vc1, vc2 = st.columns(2)
        with vc1:
            feat_v = st.selectbox("Feature", NUMERIC_FEATS, key="vf")
        with vc2:
            grp_v  = st.radio("Group By",
                ["risk_segment", "loan_grade", "income_group"],
                horizontal=True, key="vg")
        fig = px.violin(dff, x=grp_v, y=feat_v, color=grp_v,
                        color_discrete_sequence=colors_seq,
                        box=True, points=False)
        fig.update_layout(**chart_theme(420))
        apply_axes(fig, xtitle=grp_v.replace("_"," ").title(), ytitle=feat_v.replace("_"," ").title())
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        st.markdown("<br>", unsafe_allow_html=True)
        sm_f = st.multiselect("Features for Scatter Matrix",
            NUMERIC_FEATS, default=["loan_int_rate","loan_percent_income","loan_amnt"])
        if len(sm_f) >= 2:
            samp = dff.sample(min(3000, len(dff)), random_state=42)
            fig = px.scatter_matrix(
                samp, dimensions=sm_f, color="risk_segment",
                color_discrete_map={"High Risk": C_RED, "Low Risk": C_GREEN},
                opacity=0.4)
            fig.update_traces(diagonal_visible=False, showupperhalf=False,
                              marker=dict(size=3))
            fig.update_layout(**chart_theme(580))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least 2 features.")

    with t4:
        st.markdown("<br>", unsafe_allow_html=True)
        ss = dff.sample(min(10000, len(dff)), random_state=1)
        fig = px.sunburst(ss,
                          path=["income_group", "loan_grade", "risk_segment"],
                          color="loan_grade",
                          color_discrete_map={**GRADE_COLORS, "(?)": "#333"})
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=13, color="#E8EDF5"),
            height=520,
            margin=dict(t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# 3 · AGGREGATIONS
# ═══════════════════════════════════════════════════════════════════════════
elif nav == "📈  Aggregations":
    page_header("Aggregations & Drill-Downs", "Compare default rates, volumes, and averages across dimensions", "📈")

    drill = st.multiselect(
        "🔎  Drill-down — filter by Loan Grade",
        sorted(df["loan_grade"].unique()), sorted(df["loan_grade"].unique()))
    dff = df[df["loan_grade"].isin(drill)]
    gd, ind, hid, ind2, hmd, epd = agg_all(dff)

    def dual(df_a, xcol):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        bar_colors = ([GRADE_COLORS.get(g, C_PURPLE) for g in df_a[xcol]]
                      if xcol == "loan_grade" else [C_PURPLE] * len(df_a))
        fig.add_trace(go.Bar(
            x=df_a[xcol], y=df_a["total"],
            name="Total Loans",
            marker=dict(color=bar_colors, opacity=0.8,
                        line=dict(color="rgba(0,0,0,0.2)", width=1)),
            text=df_a["total"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            textfont=dict(size=11, color="#C8D6E5"),
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=df_a[xcol], y=df_a["default_rate"],
            name="Default Rate (%)", mode="lines+markers",
            line=dict(color=C_RED, width=2.5),
            marker=dict(size=8, color=C_RED, line=dict(color=C_NAVY, width=2)),
        ), secondary_y=True)
        fig.update_yaxes(title_text="Total Loans",
                         title_font=dict(size=12, color="#C8D6E5"),
                         tickfont=dict(size=11, color="#7A8FA6"),
                         gridcolor="rgba(255,255,255,0.05)",
                         secondary_y=False)
        fig.update_yaxes(title_text="Default Rate (%)",
                         title_font=dict(size=12, color=C_RED),
                         tickfont=dict(size=11, color=C_RED),
                         gridcolor="rgba(0,0,0,0)",
                         secondary_y=True)
        fig.update_xaxes(tickfont=dict(size=12, color="#7A8FA6"),
                         title_font=dict(size=12))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
            font=dict(size=13, color="#C8D6E5"),
            legend=dict(font=dict(size=12), orientation="h", y=1.12,
                        bgcolor="rgba(0,0,0,0)"),
            height=400,
        )
        return fig

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "Loan Grade", "Loan Intent", "Default History",
        "Income Group", "Home Ownership", "Employment"])

    with t1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(dual(gd, "loan_grade"), use_container_width=True)
        st.dataframe(gd, hide_index=True, use_container_width=True)

    with t2:
        st.markdown("<br>", unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=ind["default_rate"], y=ind["loan_intent"],
            orientation="h",
            marker=dict(
                color=ind["default_rate"],
                colorscale=[[0,"#1A4030"],[0.5,"#3A4020"],[1,"#4A1010"]],
                showscale=False,
                line=dict(color="rgba(0,0,0,0.2)", width=1),
            ),
            text=ind["default_rate"].apply(lambda x: f"{x}%"),
            textposition="outside",
            textfont=dict(size=12, color="#C8D6E5"),
        ))
        fig.update_layout(**chart_theme(380))
        apply_axes(fig, xtitle="Default Rate (%)", ytitle="")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ind, hide_index=True, use_container_width=True)

    with t3:
        st.markdown("<br>", unsafe_allow_html=True)
        hid["label"] = hid["cb_person_default_on_file"].map(
            {"Y": "Prior Default (Y)", "N": "No Prior Default (N)"})
        fig = go.Figure(go.Bar(
            x=hid["label"], y=hid["default_rate"],
            marker=dict(color=[C_RED, C_GREEN][:len(hid)],
                        line=dict(color="rgba(0,0,0,0.2)", width=1)),
            text=hid["default_rate"].apply(lambda x: f"{x}%"),
            textposition="outside",
            textfont=dict(size=14, color="#C8D6E5"),
            width=0.45,
        ))
        fig.update_layout(**chart_theme(380), showlegend=False)
        apply_axes(fig, xtitle="Prior Default History",
                   ytitle="Default Rate (%)", yrange=[0, 52])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(hid, hide_index=True, use_container_width=True)

    with t4:
        st.markdown("<br>", unsafe_allow_html=True)
        ig_colors = {"Low Income": C_RED, "Middle Income": C_AMBER, "High Income": C_GREEN}
        fig = go.Figure([
            go.Bar(
                x=[row["income_group"]], y=[row["default_rate"]],
                name=str(row["income_group"]),
                marker=dict(color=ig_colors.get(str(row["income_group"]), C_PURPLE),
                            line=dict(color="rgba(0,0,0,0.2)", width=1)),
                text=[f"{row['default_rate']}%"],
                textposition="outside",
                textfont=dict(size=14, color="#C8D6E5"),
                width=0.45,
            )
            for _, row in ind2.iterrows()
        ])
        fig.update_layout(**chart_theme(380), showlegend=False)
        apply_axes(fig, xtitle="Income Group",
                   ytitle="Default Rate (%)", yrange=[0, 60])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ind2, hide_index=True, use_container_width=True)

    with t5:
        st.markdown("<br>", unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=hmd["person_home_ownership"], y=hmd["default_rate"],
            marker=dict(
                color=hmd["default_rate"],
                colorscale=[[0,"#0D2820"],[0.5,"#2A2010"],[1,"#3A0808"]],
                showscale=True,
                colorbar=dict(title="Default %",
                              tickfont=dict(size=11, color="#C8D6E5"),
                              title_font=dict(color="#C8D6E5")),
                line=dict(color="rgba(0,0,0,0.2)", width=1),
            ),
            text=hmd["default_rate"].apply(lambda x: f"{x}%"),
            textposition="outside",
            textfont=dict(size=12, color="#C8D6E5"),
        ))
        fig.update_layout(**chart_theme(380))
        apply_axes(fig, xtitle="Home Ownership", ytitle="Default Rate (%)",
                   yrange=[0, 42])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(hmd, hide_index=True, use_container_width=True)

    with t6:
        st.markdown("<br>", unsafe_allow_html=True)
        ep_c = [C_RED, C_AMBER, C_GREEN]
        fig = go.Figure([
            go.Bar(
                x=[row["employment_group"]], y=[row["default_rate"]],
                name=str(row["employment_group"]),
                marker=dict(color=ep_c[i % len(ep_c)],
                            line=dict(color="rgba(0,0,0,0.2)", width=1)),
                text=[f"{row['default_rate']}%"],
                textposition="outside",
                textfont=dict(size=13, color="#C8D6E5"),
                width=0.45,
            )
            for i, (_, row) in enumerate(epd.iterrows())
        ])
        fig.update_layout(**chart_theme(380), showlegend=False)
        apply_axes(fig, xtitle="Employment Group",
                   ytitle="Default Rate (%)", yrange=[0, 40])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(epd, hide_index=True, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# 4 · CORRELATION & HEATMAP
# ═══════════════════════════════════════════════════════════════════════════
elif nav == "🔗  Correlation & Heatmap":
    page_header("Correlation & Heatmap", "Pearson correlation between numeric features and default status", "🔗")

    num_df = df[NUMERIC_FEATS + ["loan_status"]].copy()
    cm_mat = num_df.corr()
    cols   = cm_mat.columns.tolist()
    z      = cm_mat.values.round(3)

    sec("Full Numeric Correlation Heatmap")
    fig = go.Figure(go.Heatmap(
        z=z, x=cols, y=cols,
        colorscale=[[0, C_GREEN], [0.5, "#1A2030"], [1, C_RED]],
        zmid=0, zmin=-1, zmax=1,
        text=z, texttemplate="<b>%{text:.2f}</b>",
        textfont=dict(size=11, color="#E8EDF5"),
        hovertemplate="%{x} × %{y}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color="#C8D6E5"),
        height=520,
        margin=dict(t=20, b=60, l=20, r=20),
    )
    fig.update_xaxes(tickfont=dict(size=10, color="#7A8FA6"), tickangle=-30)
    fig.update_yaxes(tickfont=dict(size=10, color="#7A8FA6"))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    sec("Feature Correlation with loan_status (target)")
    corr_ls = (
        cm_mat["loan_status"].drop("loan_status")
        .reset_index()
        .rename(columns={"index": "Feature", "loan_status": "Correlation"})
        .sort_values("Correlation")
    )
    bar_cols = [C_GREEN if v < 0 else C_RED for v in corr_ls["Correlation"]]
    fig = go.Figure(go.Bar(
        x=corr_ls["Correlation"], y=corr_ls["Feature"],
        orientation="h",
        marker=dict(color=bar_cols, line=dict(color="rgba(0,0,0,0.2)", width=1)),
        text=corr_ls["Correlation"].apply(lambda x: f"{x:+.4f}"),
        textposition="outside",
        textfont=dict(size=11, color="#C8D6E5"),
        width=0.6,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.2)", line_width=1.5)
    fig.update_layout(**chart_theme(400))
    apply_axes(fig, xtitle="Pearson r with loan_status", ytitle="")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    sec("Top Risk Driver Scatters")
    sc1, sc2 = st.columns(2)
    samp = df.sample(min(5000, len(df)), random_state=42)

    with sc1:
        fig = px.scatter(samp, x="loan_percent_income", y="loan_int_rate",
                         color="risk_segment", opacity=0.4,
                         color_discrete_map={"High Risk": C_RED, "Low Risk": C_GREEN})
        fig.update_layout(**chart_theme(380),
                          title=dict(text="Loan % Income vs Interest Rate",
                                     font=dict(size=15, color="#E8EDF5")))
        apply_axes(fig, xtitle="Loan % of Income", ytitle="Interest Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    with sc2:
        fig = px.scatter(samp, x="person_income", y="loan_amnt",
                         color="risk_segment", opacity=0.4,
                         color_discrete_map={"High Risk": C_RED, "Low Risk": C_GREEN})
        fig.update_layout(**chart_theme(380),
                          title=dict(text="Annual Income vs Loan Amount",
                                     font=dict(size=15, color="#E8EDF5")))
        apply_axes(fig, xtitle="Annual Income ($)", ytitle="Loan Amount ($)")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# 5 · ML MODELS
# ═══════════════════════════════════════════════════════════════════════════
elif nav == "🤖  ML Models":
    page_header("ML Model Performance",
                "Random Forest vs Logistic Regression — trained on 80%, evaluated on 20%", "🤖")

    rf_acc = M["rf_rep"]["accuracy"]
    lr_acc = M["lr_rep"]["accuracy"]

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Random Forest AUC",  str(M["rf_auc"]),    "Area under ROC",  C_PURPLE, "🎯")
    with c2: kpi("RF Accuracy",        f"{rf_acc:.3f}",      "on 20% test set", C_GREEN,  "✅")
    with c3: kpi("Logistic Reg. AUC",  str(M["lr_auc"]),    "Area under ROC",  C_AMBER,  "📊")
    with c4: kpi("LR Accuracy",        f"{lr_acc:.3f}",      "on 20% test set", C_TEAL,   "📐")

    st.divider()
    better_model = "Random Forest" if M["rf_auc"] >= M["lr_auc"] else "Logistic Regression"
    insight_box(
        "Model Interpretation",
        [
            f"RF AUC = {M['rf_auc']:.4f} vs LR AUC = {M['lr_auc']:.4f} — {better_model} wins on ranking quality.",
            f"RF accuracy = {rf_acc:.3f} and LR accuracy = {lr_acc:.3f} — read alongside ROC, not in isolation.",
            "These metrics are on the held-out test set. They do not guarantee perfect predictions for future applicants.",
        ],
        tone="warn" if abs(M['rf_auc']-M['lr_auc']) < 0.03 else "good",
    )

    t1, t2, t3, t4 = st.tabs([
        "📉  ROC Curves", "🟦  Confusion Matrix",
        "🌟  Feature Importance", "📐  Precision-Recall"])

    with t1:
        st.markdown("<br>", unsafe_allow_html=True)
        rf_fpr, rf_tpr, _ = roc_curve(M["y_te"], M["rf_p"])
        lr_fpr, lr_tpr, _ = roc_curve(M["y_te"], M["lr_p"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rf_fpr, y=rf_tpr, mode="lines",
                                  name=f"Random Forest  (AUC={M['rf_auc']})",
                                  line=dict(color=C_PURPLE, width=3),
                                  fill="tozeroy", fillcolor="rgba(167,139,250,0.05)"))
        fig.add_trace(go.Scatter(x=lr_fpr, y=lr_tpr, mode="lines",
                                  name=f"Logistic Regression  (AUC={M['lr_auc']})",
                                  line=dict(color=C_AMBER, width=3),
                                  fill="tozeroy", fillcolor="rgba(255,140,66,0.05)"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                  line=dict(dash="dash", color="rgba(255,255,255,0.2)", width=1.5),
                                  showlegend=False))
        fig.update_layout(**chart_theme(460))
        fig.update_layout(legend=dict(font=dict(size=13), x=0.42, y=0.06))
        apply_axes(fig, xtitle="False Positive Rate", ytitle="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.markdown("<br>", unsafe_allow_html=True)
        ms = st.radio("Select Model", ["Random Forest", "Logistic Regression"],
                      horizontal=True, key="cm_m")
        preds = M["rf_pred"] if ms == "Random Forest" else M["lr_pred"]
        accent = C_PURPLE if ms == "Random Forest" else C_AMBER
        cm_data = confusion_matrix(M["y_te"], preds)
        labels  = ["Non-Default (0)", "Default (1)"]
        fig = ff.create_annotated_heatmap(
            cm_data, x=labels, y=labels,
            colorscale=[[0,"rgba(255,255,255,0.03)"],[1,accent]],
            showscale=False,
            font_colors=["#E8EDF5","#E8EDF5"],
        )
        for ann in fig.layout.annotations:
            ann.font = dict(size=22, color="#FFFFFF")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=13, color="#C8D6E5"),
            height=400, margin=dict(t=40, b=50),
        )
        fig.update_xaxes(title_text="Predicted",
                         title_font=dict(size=13, color="#C8D6E5"),
                         tickfont=dict(size=12, color="#7A8FA6"))
        fig.update_yaxes(title_text="Actual",
                         title_font=dict(size=13, color="#C8D6E5"),
                         tickfont=dict(size=12, color="#7A8FA6"))
        st.plotly_chart(fig, use_container_width=True)

        rep = M["rf_rep"] if ms == "Random Forest" else M["lr_rep"]
        r1, r2, r3 = st.columns(3)
        with r1: kpi("Precision (Default)", f"{rep['1']['precision']:.3f}", "Class 1", accent)
        with r2: kpi("Recall (Default)",    f"{rep['1']['recall']:.3f}",    "Class 1", C_TEAL)
        with r3: kpi("F1-Score (Default)",  f"{rep['1']['f1-score']:.3f}",  "Class 1", C_GREEN)

    with t3:
        st.markdown("<br>", unsafe_allow_html=True)
        imp_df = pd.DataFrame({"Feature": FEAT_COLS, "Importance": M["feat_imp"]}) \
                   .sort_values("Importance")
        imp_df["Feature"] = imp_df["Feature"].str.replace("_enc","").str.replace("_"," ").str.title()

        gradient_colors = [
            f"rgba(167,139,250,{0.3 + 0.7*(i/len(imp_df))})"
            for i in range(len(imp_df))
        ]
        fig = go.Figure(go.Bar(
            x=imp_df["Importance"], y=imp_df["Feature"],
            orientation="h",
            marker=dict(color=gradient_colors,
                        line=dict(color="rgba(0,0,0,0.2)", width=1)),
            text=imp_df["Importance"].apply(lambda x: f"{x:.4f}"),
            textposition="outside",
            textfont=dict(size=11, color="#C8D6E5"),
        ))
        fig.update_layout(**chart_theme(440))
        apply_axes(fig, xtitle="Feature Importance (Gini)", ytitle="")
        st.plotly_chart(fig, use_container_width=True)

    with t4:
        st.markdown("<br>", unsafe_allow_html=True)
        rf_p2, rf_r2, _ = precision_recall_curve(M["y_te"], M["rf_p"])
        lr_p2, lr_r2, _ = precision_recall_curve(M["y_te"], M["lr_p"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rf_r2, y=rf_p2, mode="lines",
                                  name="Random Forest",
                                  line=dict(color=C_PURPLE, width=3),
                                  fill="tozeroy", fillcolor="rgba(167,139,250,0.05)"))
        fig.add_trace(go.Scatter(x=lr_r2, y=lr_p2, mode="lines",
                                  name="Logistic Regression",
                                  line=dict(color=C_AMBER, width=3)))
        fig.update_layout(**chart_theme(440))
        fig.update_layout(legend=dict(font=dict(size=13)))
        apply_axes(fig, xtitle="Recall", ytitle="Precision")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# 6 · LOAN RISK PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════
elif nav == "🎯  Loan Risk Predictor":
    page_header("Loan Risk Predictor",
                "Fill applicant details — both models score the applicant and save to Footprint Database",
                "🎯")

    with st.form("pred_form"):
        st.markdown("""
        <div style="background:rgba(245,197,24,0.06);border-radius:14px;padding:1rem 1.5rem;
                    border-left:4px solid #F5C518;margin-bottom:1.5rem;">
            <p style="font-size:0.9rem;font-weight:700;color:#F5C518;margin:0;letter-spacing:0.05em;">
                🏢 &nbsp; INSTITUTION &amp; APPLICANT DETAILS
            </p>
        </div>""", unsafe_allow_html=True)

        tc1, tc2 = st.columns([1, 3])
        with tc1:
            institution = st.text_input("Institution Name", value="Bank A")

        st.markdown("<br>", unsafe_allow_html=True)
        pc1, pc2, pc3 = st.columns(3)

        with pc1:
            st.markdown("**👤 Personal Info**")
            p_age  = st.number_input("Age (years)", 18, 100, 30)
            p_inc  = st.number_input("Annual Income ($)", 1000, 5_000_000, 55000, step=1000)
            p_home = st.selectbox("Home Ownership", LABEL_MAPS["person_home_ownership"])
            p_emp  = st.number_input("Employment Length (yrs)", 0.0, 50.0, 4.0, step=0.5)

        with pc2:
            st.markdown("**📋 Loan Details**")
            l_int  = st.selectbox("Loan Intent",  LABEL_MAPS["loan_intent"])
            l_grd  = st.selectbox("Loan Grade",   LABEL_MAPS["loan_grade"])
            l_amt  = st.number_input("Loan Amount ($)", 500, 500_000, 10000, step=500)
            l_rate = st.number_input("Interest Rate (%)", 1.0, 35.0, 11.0, step=0.1)

        with pc3:
            st.markdown("**🏦 Credit Profile**")
            l_pct  = st.number_input("Loan % of Income", 0.01, 1.0, 0.18, step=0.01)
            l_dfl  = st.selectbox("Prior Default on File",
                                   LABEL_MAPS["cb_person_default_on_file"],
                                   format_func=lambda x: "Yes (Y)" if x=="Y" else "No (N)")
            l_crd  = st.number_input("Credit History (yrs)", 1, 50, 5)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "🚀  Predict Default Risk", use_container_width=True)

    if submitted:
        applicant = {
            "person_age": p_age, "person_income": p_inc,
            "person_home_ownership": p_home, "person_emp_length": p_emp,
            "loan_intent": l_int, "loan_grade": l_grd,
            "loan_amnt": l_amt, "loan_int_rate": l_rate,
            "loan_percent_income": l_pct,
            "cb_person_default_on_file": l_dfl,
            "cb_person_cred_hist_length": l_crd,
        }
        X_in    = encode_for_model(applicant)
        X_in_sc = M["sc"].transform(X_in)
        rf_prob = float(M["rf"].predict_proba(X_in)[0, 1])
        lr_prob = float(M["lr"].predict_proba(X_in_sc)[0, 1])
        avg_pr  = (rf_prob + lr_prob) / 2
        pred    = 1 if avg_pr >= 0.5 else 0

        st.divider()

        if pred == 1:
            st.markdown(f"""
            <div style="background:rgba(255,76,76,0.08);border:2px solid {C_RED};
                        border-radius:18px;padding:2rem;text-align:center;
                        box-shadow:0 0 40px rgba(255,76,76,0.15);">
                <div style="font-size:3rem;margin-bottom:0.5rem;">⚠️</div>
                <div style="font-size:1.8rem;font-weight:700;color:{C_RED};
                            letter-spacing:-0.02em;margin-bottom:0.4rem;">
                    HIGH DEFAULT RISK
                </div>
                <div style="font-size:1.05rem;color:#FF8080;font-weight:500;">
                    Ensemble probability: <strong style='font-family:"JetBrains Mono"'>{avg_pr*100:.1f}%</strong>
                    &nbsp;·&nbsp; Predicted to <strong>DEFAULT</strong>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:rgba(0,196,140,0.08);border:2px solid {C_GREEN};
                        border-radius:18px;padding:2rem;text-align:center;
                        box-shadow:0 0 40px rgba(0,196,140,0.12);">
                <div style="font-size:3rem;margin-bottom:0.5rem;">✅</div>
                <div style="font-size:1.8rem;font-weight:700;color:{C_GREEN};
                            letter-spacing:-0.02em;margin-bottom:0.4rem;">
                    LOW DEFAULT RISK
                </div>
                <div style="font-size:1.05rem;color:#4ECDC4;font-weight:500;">
                    Ensemble probability: <strong style='font-family:"JetBrains Mono"'>{avg_pr*100:.1f}%</strong>
                    &nbsp;·&nbsp; Predicted to <strong>REPAY</strong>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        kc1, kc2, kc3 = st.columns(3)
        with kc1: kpi("Random Forest",       f"{rf_prob*100:.1f}%", "Default probability", C_PURPLE, "🌲")
        with kc2: kpi("Logistic Regression", f"{lr_prob*100:.1f}%", "Default probability", C_AMBER,  "📊")
        with kc3: kpi("Ensemble Average",    f"{avg_pr*100:.1f}%",  "Final score",
                        C_RED if pred else C_GREEN, "🎯")

        insight_box(
            "Prediction Interpretation",
            [
                f"Random Forest: {rf_prob*100:.1f}% default probability · Logistic Regression: {lr_prob*100:.1f}%",
                f"Ensemble average: {avg_pr*100:.1f}% → threshold 50% → {'HIGH RISK' if pred==1 else 'LOW RISK'}",
                "This is a model-based risk signal for screening support, not a guarantee.",
            ],
            tone="warn" if pred == 1 else "good",
        )

        # Premium gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_pr * 100,
            number={"suffix": "%",
                    "font": {"size": 48, "color": "#E8EDF5",
                             "family": "JetBrains Mono"}},
            title={"text": "<b>Default Probability Score</b>",
                   "font": {"size": 16, "color": "#C8D6E5"}},
            delta={"reference": 50,
                   "increasing": {"color": C_RED},
                   "decreasing": {"color": C_GREEN},
                   "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100],
                         "tickwidth": 1,
                         "tickcolor": "#3A4A5A",
                         "tickfont": {"size": 12, "color": "#7A8FA6"}},
                "bar":  {"color": C_RED if avg_pr >= 0.5 else C_GREEN,
                         "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  30], "color": "rgba(0,196,140,0.12)"},
                    {"range": [30, 60], "color": "rgba(245,197,24,0.08)"},
                    {"range": [60,100], "color": "rgba(255,76,76,0.12)"},
                ],
                "threshold": {
                    "line": {"color": C_GOLD, "width": 2},
                    "thickness": 0.75, "value": 50},
            },
        ))
        fig.update_layout(
            height=360, paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Grotesk, sans-serif", color="#C8D6E5"),
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        append_log({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "institution": institution,
            **applicant,
            "rf_probability_pct":  round(rf_prob * 100, 2),
            "lr_probability_pct":  round(lr_prob * 100, 2),
            "predicted_default":   pred,
        })
        st.success("✅  Prediction saved — navigate to **🗄️ Footprint Database** to view all records.")

# ═══════════════════════════════════════════════════════════════════════════
# 7 · FOOTPRINT DATABASE
# ═══════════════════════════════════════════════════════════════════════════
elif nav == "🗄️  Footprint Database":
    page_header("Footprint Database",
                "Every applicant scored through the Predictor is stored here permanently",
                "🗄️")

    log_df = load_log()

    if log_df.empty:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                    border-radius:18px;padding:4rem;text-align:center;margin-top:2rem;">
            <div style="font-size:4rem;">🗄️</div>
            <h2 style="color:#FFFFFF;margin:1rem 0 0.8rem;">No Records Yet</h2>
            <p style="font-size:1rem;color:#7A8FA6;max-width:440px;margin:0 auto;">
                Go to <strong style='color:#F5C518;'>🎯 Loan Risk Predictor</strong> and assess your
                first applicant. Records appear here automatically.
            </p>
        </div>""", unsafe_allow_html=True)
        st.stop()

    tot  = len(log_df)
    hi   = int((log_df["predicted_default"]==1).sum()) if "predicted_default" in log_df.columns else 0
    lo   = tot - hi
    rate = hi / tot * 100 if tot else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi("Total Assessed", f"{tot:,}",     "all time",          C_PURPLE, "📋")
    with k2: kpi("High Risk",      f"{hi:,}",      "predicted default", C_RED,    "⚠️")
    with k3: kpi("Low Risk",       f"{lo:,}",      "predicted safe",    C_GREEN,  "✅")
    with k4: kpi("Risk Rate",      f"{rate:.1f}%", "of assessed",       C_AMBER,  "📊")

    insight_box(
        "Database Interpretation",
        [
            f"Footprint Database: {tot:,} assessed applicants, {hi:,} flagged high risk.",
            f"Assessed risk rate: {rate:.1f}% — reflects model output on your pipeline, not the raw dataset.",
            "Use this to monitor your assessed pipeline over time and across institutions.",
        ],
        tone="info",
    )

    if tot >= 2:
        st.divider()
        fa1, fa2 = st.columns(2)
        with fa1:
            sec("Risk Split of Assessed Applicants")
            rc = (log_df["predicted_default"]
                  .map({1:"High Risk",0:"Low Risk"})
                  .value_counts().reset_index())
            rc.columns = ["Segment","Count"]
            fig = px.pie(rc, names="Segment", values="Count", hole=0.55,
                         color="Segment",
                         color_discrete_map={"High Risk":C_RED,"Low Risk":C_GREEN})
            fig.update_traces(textfont=dict(size=13, color="white"),
                              marker=dict(line=dict(color="rgba(0,0,0,0.3)", width=2)))
            fig.update_layout(**chart_theme(320), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        with fa2:
            sec("Assessments Over Time")
            if "timestamp" in log_df.columns:
                log_df["date"] = pd.to_datetime(log_df["timestamp"]).dt.date
                tl = log_df.groupby("date").size().reset_index(name="Assessments")
                fig = px.area(tl, x="date", y="Assessments",
                               color_discrete_sequence=[C_PURPLE], markers=True)
                fig.update_traces(line=dict(width=2.5),
                                  fillcolor="rgba(167,139,250,0.12)")
                fig.update_layout(**chart_theme(320))
                apply_axes(fig, xtitle="Date", ytitle="Assessments")
                st.plotly_chart(fig, use_container_width=True)

        if "rf_probability_pct" in log_df.columns and tot >= 3:
            sec("RF Probability Distribution")
            fig = px.histogram(log_df, x="rf_probability_pct",
                               color="predicted_default",
                               color_discrete_map={1:C_RED, 0:C_GREEN},
                               nbins=20, barmode="overlay", opacity=0.75)
            fig.update_layout(**chart_theme(320))
            apply_axes(fig, xtitle="RF Default Probability (%)", ytitle="Count")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    sec("All Assessment Records")

    view = log_df.copy()
    vf1, vf2 = st.columns([2,1])
    with vf1:
        if "institution" in log_df.columns:
            inst_opts = sorted(log_df["institution"].dropna().unique())
            inst_f = st.multiselect("Filter by Institution", inst_opts, default=inst_opts)
            view = view[view["institution"].isin(inst_f)]
    with vf2:
        if "predicted_default" in log_df.columns:
            rf_filter = st.multiselect(
                "Risk Level", [0,1], default=[0,1],
                format_func=lambda x: "Low Risk (0)" if x==0 else "High Risk (1)")
            view = view[view["predicted_default"].isin(rf_filter)]

    st.dataframe(view.reset_index(drop=True), use_container_width=True, height=420)
    st.download_button(
        "⬇️  Export as CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="credlens_footprint.csv",
        mime="text/csv",
    )

# ═══════════════════════════════════════════════════════════════════════════
# 8 · RISK CHATBOT
# ═══════════════════════════════════════════════════════════════════════════
elif nav == "💬  Risk Chatbot":
    page_header("Risk Chatbot",
                "Ask analytical questions about portfolio risk, model performance, grades, intents & predictions",
                "💬")

    # Quick-start prompts
    st.markdown("""
    <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.75rem;margin-bottom:1.5rem;'>
        <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                    border-radius:12px;padding:0.9rem 1rem;'>
            <div style='font-size:0.68rem;font-weight:700;color:#F5C518;letter-spacing:0.12em;
                        text-transform:uppercase;margin-bottom:0.5rem;'>Try asking</div>
            <div style='font-size:0.85rem;color:#C8D6E5;line-height:1.7;'>
                What is the overall default rate?<br>
                Which loan grade is riskiest?
            </div>
        </div>
        <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                    border-radius:12px;padding:0.9rem 1rem;'>
            <div style='font-size:0.68rem;font-weight:700;color:#00B4D8;letter-spacing:0.12em;
                        text-transform:uppercase;margin-bottom:0.5rem;'>Analytics</div>
            <div style='font-size:0.85rem;color:#C8D6E5;line-height:1.7;'>
                Which loan intent has highest default rate?<br>
                What are the top model drivers?
            </div>
        </div>
        <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                    border-radius:12px;padding:0.9rem 1rem;'>
            <div style='font-size:0.68rem;font-weight:700;color:#A78BFA;letter-spacing:0.12em;
                        text-transform:uppercase;margin-bottom:0.5rem;'>Operational</div>
            <div style='font-size:0.85rem;color:#C8D6E5;line-height:1.7;'>
                How many records in Footprint Database?<br>
                How can I reduce default risk?
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    insight_box(
        "Chatbot Guidance",
        [
            "Ask analytical questions about grades, intents, model metrics, and saved prediction records.",
            "Answers are grounded in the current uploaded dataset — explainable and auditable.",
            "Does not browse the internet and does not guarantee loan approval decisions.",
        ],
        tone="info",
    )

    # Starter buttons
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        if st.button("📊 What is the overall default rate?", use_container_width=True):
            st.session_state["starter_question"] = "What is the overall default rate?"
    with col_s2:
        if st.button("🏆 Which loan grade is riskiest?", use_container_width=True):
            st.session_state["starter_question"] = "Which loan grade is riskiest?"
    with col_s3:
        if st.button("🌟 What are the top model drivers?", use_container_width=True):
            st.session_state["starter_question"] = "What are the top model drivers?"

    st.divider()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "Hello. I am the CredLens AI Risk Assistant. "
                    "I can summarize the portfolio, explain model metrics, "
                    "compare loan grades and intents, and review saved prediction records. "
                    "What would you like to explore?"
                ),
            }
        ]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input(
        st.session_state.pop("starter_question", "Ask about credit risk analytics…"))
    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        log_df = load_log()
        with st.spinner("Analyzing…"):
            reply = ai_chatbot_response(user_prompt, df, M, log_df)

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

    st.divider()
    st.info(
        "🔒 This chatbot is analytics-aware, not a generative approval engine. "
        "Its answers are grounded in the current dataset and model outputs — explainable and auditable."
    )
