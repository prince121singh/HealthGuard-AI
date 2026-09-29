import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from core.predictor import predict_risk, get_risk_label
from core.preprocessing import load_data
from core.explainability import explain_prediction
from core.report_generator import generate_html_report


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HealthGuard AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "page": "Dashboard",
    "theme": "Midnight Cyan",
    "prediction_result": None,
    "patient_data": None,
    "explanation_df": None,
    "global_importance": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# THEMES
# ============================================================

THEMES = {
    "Midnight Cyan": {
        "bg": "#050814",
        "surface": "#0B1020",
        "surface2": "#10182B",
        "border": "#24314D",
        "primary": "#22D3EE",
        "secondary": "#6366F1",
        "text": "#F8FAFC",
        "muted": "#94A3B8",
        "success": "#22C55E",
        "warning": "#F59E0B",
        "danger": "#EF4444"
    },
    "Emerald Matrix": {
        "bg": "#04100C",
        "surface": "#081A13",
        "surface2": "#0D241B",
        "border": "#214D3D",
        "primary": "#34D399",
        "secondary": "#14B8A6",
        "text": "#ECFDF5",
        "muted": "#94A3B8",
        "success": "#22C55E",
        "warning": "#F59E0B",
        "danger": "#EF4444"
    },
    "Royal Violet": {
        "bg": "#0A0612",
        "surface": "#140C20",
        "surface2": "#1D1230",
        "border": "#3D2A59",
        "primary": "#C084FC",
        "secondary": "#6366F1",
        "text": "#FAF5FF",
        "muted": "#A8A29E",
        "success": "#22C55E",
        "warning": "#F59E0B",
        "danger": "#EF4444"
    }
}

theme = THEMES[st.session_state.theme]


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    f"""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

:root {{
    --bg: {theme["bg"]};
    --surface: {theme["surface"]};
    --surface2: {theme["surface2"]};
    --border: {theme["border"]};
    --primary: {theme["primary"]};
    --secondary: {theme["secondary"]};
    --text: {theme["text"]};
    --muted: {theme["muted"]};
}}

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background:
        radial-gradient(
            circle at 5% 5%,
            {theme["primary"]}16,
            transparent 25%
        ),
        radial-gradient(
            circle at 95% 10%,
            {theme["secondary"]}18,
            transparent 28%
        ),
        radial-gradient(
            circle at 50% 100%,
            {theme["primary"]}08,
            transparent 35%
        ),
        {theme["bg"]};
    color: {theme["text"]};
}}

/* GRID */

.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: .12;

    background-image:
        linear-gradient(
            {theme["primary"]}12 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            {theme["primary"]}12 1px,
            transparent 1px
        );

    background-size: 45px 45px;
    z-index: 0;
}}

/* MAIN */

.block-container {{
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}}

/* SIDEBAR */

section[data-testid="stSidebar"] {{
    background:
        linear-gradient(
            180deg,
            {theme["surface"]},
            {theme["bg"]}
        );

    border-right: 1px solid {theme["border"]};
}}

section[data-testid="stSidebar"] * {{
    color: {theme["text"]};
}}

.brand {{
    text-align: center;
    padding: 15px 5px 28px;
}}

.brand-icon {{
    width: 76px;
    height: 76px;

    margin: auto;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 24px;

    font-size: 38px;

    background:
        linear-gradient(
            135deg,
            {theme["primary"]},
            {theme["secondary"]}
        );

    box-shadow:
        0 0 35px {theme["primary"]}55;
}}

.brand-name {{
    font-size: 23px;
    font-weight: 800;
    margin-top: 14px;
}}

.brand-sub {{
    font-size: 11px;
    color: {theme["muted"]};
    margin-top: 5px;
    letter-spacing: .8px;
}}

/* HERO */

.hero {{
    position: relative;
    overflow: hidden;

    padding: 38px;

    border-radius: 30px;

    border: 1px solid {theme["border"]};

    background:
        linear-gradient(
            135deg,
            {theme["surface"]}F5,
            {theme["surface2"]}E8
        );

    box-shadow:
        0 30px 90px rgba(0,0,0,.28);
}}

.hero::before {{
    content: "";

    position: absolute;

    width: 330px;
    height: 330px;

    right: -120px;
    top: -160px;

    border-radius: 50%;

    background: {theme["primary"]}18;

    filter: blur(4px);
}}

.hero-kicker {{
    color: {theme["primary"]};

    font-size: 12px;
    font-weight: 800;

    letter-spacing: 2px;
    text-transform: uppercase;
}}

.hero-title {{
    font-size: clamp(38px, 5vw, 66px);

    line-height: 1;

    font-weight: 800;

    margin: 13px 0;
}}

.gradient {{
    background:
        linear-gradient(
            90deg,
            {theme["primary"]},
            {theme["secondary"]}
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.hero-text {{
    max-width: 850px;

    color: {theme["muted"]};

    font-size: 15px;

    line-height: 1.8;
}}

/* SECTION */

.section-title {{
    margin-top: 34px;
    margin-bottom: 18px;

    font-size: 25px;
    font-weight: 800;
}}

/* CARDS */

.card {{
    padding: 22px;

    border-radius: 20px;

    border: 1px solid {theme["border"]};

    background:
        linear-gradient(
            145deg,
            {theme["surface"]},
            {theme["surface2"]}
        );

    box-shadow:
        0 15px 45px rgba(0,0,0,.18);
}}

.card:hover {{
    border-color: {theme["primary"]}66;
}}

.card-title {{
    font-size: 16px;
    font-weight: 700;
}}

.card-text {{
    color: {theme["muted"]};

    font-size: 13px;

    line-height: 1.65;

    margin-top: 8px;
}}

/* METRIC */

.metric {{
    padding: 21px;

    border-radius: 20px;

    border: 1px solid {theme["border"]};

    background: {theme["surface"]};

    min-height: 125px;
}}

.metric-icon {{
    font-size: 24px;
}}

.metric-label {{
    color: {theme["muted"]};

    font-size: 11px;

    margin-top: 10px;

    text-transform: uppercase;

    letter-spacing: .8px;
}}

.metric-value {{
    font-size: 28px;

    font-weight: 800;

    margin-top: 5px;
}}

/* INPUT AREA */

.input-section {{
    padding: 26px;

    border-radius: 24px;

    border: 1px solid {theme["border"]};

    background:
        linear-gradient(
            145deg,
            {theme["surface"]},
            {theme["surface2"]}
        );

    margin-bottom: 20px;
}}

.input-section-title {{
    font-size: 20px;
    font-weight: 800;

    display: flex;
    align-items: center;

    gap: 10px;
}}

.input-section-desc {{
    color: {theme["muted"]};

    font-size: 12px;

    margin-top: 5px;

    margin-bottom: 20px;
}}

/* STREAMLIT LABEL FIX */

label {{
    color: {theme["text"]} !important;
}}

div[data-testid="stWidgetLabel"] p {{
    color: {theme["text"]} !important;

    font-weight: 600 !important;
}}

div[data-testid="stWidgetLabel"] small {{
    color: {theme["muted"]} !important;
}}

[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label {{
    color: {theme["text"]} !important;
}}

/* INPUTS */

div[data-baseweb="input"] {{
    background: {theme["surface2"]} !important;

    border-radius: 12px !important;

    border: 1px solid {theme["border"]} !important;
}}

div[data-baseweb="select"] > div {{
    background: {theme["surface2"]} !important;

    border-radius: 12px !important;

    border: 1px solid {theme["border"]} !important;
}}

input {{
    color: {theme["text"]} !important;
}}

[data-baseweb="select"] * {{
    color: {theme["text"]} !important;
}}

/* BUTTON */

.stButton > button,
.stDownloadButton > button {{
    min-height: 48px;

    border-radius: 13px !important;

    border: 1px solid {theme["border"]} !important;

    background:
        linear-gradient(
            135deg,
            {theme["primary"]}20,
            {theme["secondary"]}20
        ) !important;

    color: {theme["text"]} !important;

    font-weight: 700 !important;

    transition: .2s ease !important;
}}

.stButton > button:hover,
.stDownloadButton > button:hover {{
    transform: translateY(-2px);

    border-color: {theme["primary"]} !important;

    box-shadow:
        0 0 30px {theme["primary"]}22 !important;
}}

/* RESULT */

.result-card {{
    padding: 30px;

    border-radius: 25px;

    border: 1px solid {theme["border"]};

    background: {theme["surface"]};

    text-align: center;
}}

.result-number {{
    font-size: 55px;

    font-weight: 800;

    margin: 10px 0;
}}

.result-label {{
    font-size: 18px;

    font-weight: 700;
}}

.result-sub {{
    color: {theme["muted"]};

    font-size: 12px;

    margin-top: 6px;
}}

/* BADGE */

.badge {{
    display: inline-block;

    padding: 7px 13px;

    border-radius: 999px;

    font-size: 11px;

    font-weight: 700;

    background: {theme["primary"]}15;

    color: {theme["primary"]};

    border: 1px solid {theme["primary"]}35;
}}

/* DISCLAIMER */

.disclaimer {{
    padding: 17px;

    margin-top: 25px;

    border-radius: 15px;

    border: 1px solid {theme["warning"]}35;

    background: {theme["warning"]}0A;

    color: {theme["muted"]};

    font-size: 12px;

    line-height: 1.7;
}}

/* FOOTER */

.footer {{
    text-align: center;

    color: {theme["muted"]};

    font-size: 11px;

    margin-top: 50px;

    padding-top: 25px;

    border-top: 1px solid {theme["border"]};
}}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HELPERS
# ============================================================

def safe_load_data():

    try:
        data = load_data()

        if isinstance(data, tuple):
            data = data[0]

        if data is None:
            return pd.DataFrame()

        return data

    except Exception:
        return pd.DataFrame()


def normalize_prediction(result):

    prediction = None
    probability = None

    if isinstance(result, dict):

        prediction = (
            result.get("prediction")
            or result.get("predicted_class")
            or result.get("class")
        )

        probability = (
            result.get("probability")
            or result.get("risk_probability")
            or result.get("probability_1")
        )

    elif isinstance(result, (list, tuple)):

        if len(result) > 0:
            prediction = result[0]

        if len(result) > 1:
            probability = result[1]

    else:

        prediction = result

    if prediction is None:
        raise ValueError(
            "Prediction was not returned by the model."
        )

    prediction = int(
        np.asarray(prediction).flatten()[0]
    )

    if probability is None:
        probability = 0.5

    probability = float(
        np.asarray(probability).flatten()[0]
    )

    if probability > 1:
        probability /= 100

    probability = max(
        0,
        min(1, probability)
    )

    return prediction, probability


def risk_info(probability):

    if probability < 0.35:
        return (
            "Lower Risk",
            "Your model probability falls in the lower range.",
            theme["success"]
        )

    if probability < 0.65:
        return (
            "Moderate Risk",
            "Your model probability falls in the moderate range.",
            theme["warning"]
        )

    return (
        "Higher Risk",
        "Your model probability falls in the higher range.",
        theme["danger"]
    )


def gauge(probability):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,

            number={
                "suffix": "%",
                "font": {
                    "size": 42,
                    "color": theme["text"]
                }
            },

            title={
                "text": "MODEL PROBABILITY",
                "font": {
                    "size": 13,
                    "color": theme["muted"]
                }
            },

            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": theme["muted"]
                },

                "bar": {
                    "color": theme["primary"]
                },

                "bgcolor": theme["surface"],

                "bordercolor": theme["border"],

                "borderwidth": 1
            }
        )
    )

    fig.update_layout(
        height=300,

        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        ),

        paper_bgcolor="rgba(0,0,0,0)",

        font_color=theme["text"]
    )

    return fig


def shap_chart(data, limit=8):

    chart_df = data.head(limit).copy()

    chart_df = chart_df.sort_values(
        "shap_value"
    )

    fig = px.bar(
        chart_df,
        x="shap_value",
        y="feature",
        orientation="h",
        title="Feature Contribution"
    )

    fig.update_layout(
        height=430,

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font_color=theme["text"],

        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        ),

        xaxis=dict(
            gridcolor=theme["border"]
        )
    )

    return fig


def metric(icon, title, value):

    return f"""
    <div class="metric">

        <div class="metric-icon">
            {icon}
        </div>

        <div class="metric-label">
            {title}
        </div>

        <div class="metric-value">
            {value}
        </div>

    </div>
    """


# ============================================================
# DATA
# ============================================================

df = safe_load_data()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">

            <div class="brand-icon">
                🩺
            </div>

            <div class="brand-name">
                HealthGuard AI
            </div>

            <div class="brand-sub">
                INTELLIGENT HEALTH ANALYTICS
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    pages = [
        "Dashboard",
        "Risk Prediction",
        "SHAP Explainability",
        "Analytics",
        "AI Insights",
        "Reports"
    ]

    page = st.radio(
        "NAVIGATION",
        pages,
        index=pages.index(
            st.session_state.page
        )
    )

    st.session_state.page = page

    st.divider()

    st.markdown("### 🎨 Interface")

    selected_theme = st.selectbox(
        "Visual Theme",
        list(THEMES.keys()),
        index=list(THEMES.keys()).index(
            st.session_state.theme
        )
    )

    if selected_theme != st.session_state.theme:

        st.session_state.theme = selected_theme

        st.rerun()

    st.divider()

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                🧠 AI ENGINE
            </div>

            <div class="card-text">
                Machine learning + SHAP explainability
                + interactive analytics.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-kicker">
                INTELLIGENT HEALTH ANALYTICS
            </div>

            <div class="hero-title">
                HealthGuard <span class="gradient">AI</span>
            </div>

            <div class="hero-text">
                A premium machine-learning dashboard for
                exploring health-risk model predictions,
                explainable AI and dataset analytics.
            </div>

            <br>

            <span class="badge">
                ● AI ENGINE ACTIVE
            </span>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">System Overview</div>',
        unsafe_allow_html=True
    )

    total = len(df)

    features = (
        len(df.columns) - 1
        if len(df.columns)
        else 0
    )

    positive = (
        int(df["target"].sum())
        if "target" in df.columns
        else 0
    )

    negative = total - positive

    columns = st.columns(4)

    cards = [
        ("🗂️", "Dataset Records", total),
        ("🧬", "ML Features", features),
        ("⚠️", "Positive Class", positive),
        ("✓", "Negative Class", negative)
    ]

    for column, item in zip(columns, cards):

        with column:

            st.markdown(
                metric(*item),
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="section-title">What HealthGuard AI Does</div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    content = [
        (
            "🧠",
            "Machine Learning",
            "Processes structured input features and generates a model-based probability estimate."
        ),
        (
            "🔍",
            "Explainable AI",
            "Uses SHAP to show which features contributed to an individual prediction."
        ),
        (
            "📊",
            "Data Analytics",
            "Interactive charts help explore the dataset and model-related patterns."
        )
    ]

    for column, item in zip(
        [a, b, c],
        content
    ):

        with column:

            st.markdown(
                f"""
                <div class="card">

                    <div style="font-size:28px;">
                        {item[0]}
                    </div>

                    <div class="card-title">
                        {item[1]}
                    </div>

                    <div class="card-text">
                        {item[2]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="section-title">Dataset Snapshot</div>',
        unsafe_allow_html=True
    )

    if not df.empty:

        c1, c2 = st.columns(2)

        with c1:

            if "target" in df.columns:

                target = (
                    df["target"]
                    .value_counts()
                    .reset_index()
                )

                target.columns = [
                    "Class",
                    "Count"
                ]

                fig = px.pie(
                    target,
                    names="Class",
                    values="Count",
                    hole=.58,
                    title="Target Distribution"
                )

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color=theme["text"]
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        with c2:

            numeric = df.select_dtypes(
                include=np.number
            ).columns.tolist()

            if numeric:

                selected = st.selectbox(
                    "Explore a feature",
                    numeric
                )

                fig = px.histogram(
                    df,
                    x=selected,
                    color="target"
                    if "target" in df.columns
                    else None,
                    title=f"{selected} Distribution"
                )

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color=theme["text"]
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


# ============================================================
# RISK PREDICTION
# ============================================================

elif page == "Risk Prediction":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-kicker">
                AI RISK ANALYSIS ENGINE
            </div>

            <div class="hero-title">
                Smart <span class="gradient">Risk Prediction</span>
            </div>

            <div class="hero-text">
                Enter the available measurements below.
                Each field includes a simple explanation so
                you know what the model expects.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Patient Information</div>',
        unsafe_allow_html=True
    )

    with st.form("healthguard_prediction"):

        # ====================================================
        # BASIC INFORMATION
        # ====================================================

        st.markdown(
            """
            <div class="input-section">

                <div class="input-section-title">
                    👤 Basic Information
                </div>

                <div class="input-section-desc">
                    Basic demographic information used by the
                    machine-learning model.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=50,
                help="Age in years."
            )

        with c2:

            sex = st.selectbox(
                "Sex",
                ["Female", "Male"],
                help="Sex category used by the dataset."
            )

        with c3:

            chest_pain = st.selectbox(
                "Chest Pain Type",
                [
                    "Typical Angina",
                    "Atypical Angina",
                    "Non-anginal Pain",
                    "Asymptomatic"
                ],
                help=(
                    "Category describing the type of chest "
                    "pain recorded in the dataset."
                )
            )

        # ====================================================
        # VITALS
        # ====================================================

        st.markdown(
            """
            <div class="input-section">

                <div class="input-section-title">
                    ❤️ Vital & Laboratory Features
                </div>

                <div class="input-section-desc">
                    Enter measured values when available.
                    Units are shown directly in the field labels.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            trestbps = st.number_input(
                "Resting Blood Pressure (mmHg)",
                min_value=50,
                max_value=250,
                value=120,
                help=(
                    "Blood pressure measurement recorded "
                    "while resting. Example: 120 mmHg."
                )
            )

        with c2:

            chol = st.number_input(
                "Total Cholesterol (mg/dL)",
                min_value=50,
                max_value=700,
                value=200,
                help=(
                    "Total cholesterol measurement. "
                    "Usually recorded in mg/dL."
                )
            )

        with c3:

            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dL",
                ["No", "Yes"],
                help=(
                    "Dataset indicator showing whether fasting "
                    "blood sugar was above the specified threshold."
                )
            )

        # ====================================================
        # HEART / ECG
        # ====================================================

        st.markdown(
            """
            <div class="input-section">

                <div class="input-section-title">
                    📈 Heart & ECG Features
                </div>

                <div class="input-section-desc">
                    Measurements and categories derived from
                    the dataset's cardiac/ECG-related features.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            restecg = st.selectbox(
                "Resting ECG Result",
                [
                    "Normal",
                    "ST-T Wave Abnormality",
                    "Left Ventricular Hypertrophy"
                ],
                help=(
                    "Category representing the resting ECG "
                    "result in the dataset."
                )
            )

        with c2:

            thalach = st.number_input(
                "Maximum Heart Rate (bpm)",
                min_value=50,
                max_value=250,
                value=150,
                help=(
                    "Maximum heart rate recorded during "
                    "the relevant test. Unit: beats per minute."
                )
            )

        with c3:

            exang = st.selectbox(
                "Exercise-Induced Angina",
                ["No", "Yes"],
                help=(
                    "Indicates whether exercise-induced angina "
                    "was recorded in the dataset."
                )
            )

        c1, c2, c3 = st.columns(3)

        with c1:

            oldpeak = st.number_input(
                "ST Depression (Oldpeak)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help=(
                    "An ECG-related numerical feature called "
                    "oldpeak in the original dataset."
                )
            )

        with c2:

            slope = st.selectbox(
                "ST Segment Slope",
                [
                    "Upsloping",
                    "Flat",
                    "Downsloping"
                ],
                help=(
                    "Category describing the slope of the "
                    "ST segment in the original dataset."
                )
            )

        with c3:

            ca = st.selectbox(
                "Major Vessels Recorded",
                [0, 1, 2, 3],
                help=(
                    "Number of major vessels represented by "
                    "the dataset's ca feature."
                )
            )

        # ====================================================
        # THAL
        # ====================================================

        st.markdown(
            """
            <div class="input-section">

                <div class="input-section-title">
                    🧬 Additional Cardiac Feature
                </div>

                <div class="input-section-desc">
                    Category encoded in the original dataset.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2 = st.columns([1, 2])

        with c1:

            thal = st.selectbox(
                "Thalassemia Category",
                [
                    "Normal",
                    "Fixed Defect",
                    "Reversible Defect"
                ],
                help=(
                    "Categorical thalassemia-related feature "
                    "used by the original dataset."
                )
            )

        with c2:

            st.info(
                "💡 Use measured or dataset-provided values "
                "when available. Avoid guessing medical measurements."
            )

        st.markdown("<br>", unsafe_allow_html=True)

        submitted = st.form_submit_button(
            "🚀 RUN AI HEALTH ANALYSIS",
            use_container_width=True
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    if submitted:

        cp_map = {
            "Typical Angina": 1,
            "Atypical Angina": 2,
            "Non-anginal Pain": 3,
            "Asymptomatic": 4
        }

        restecg_map = {
            "Normal": 0,
            "ST-T Wave Abnormality": 1,
            "Left Ventricular Hypertrophy": 2
        }

        slope_map = {
            "Upsloping": 1,
            "Flat": 2,
            "Downsloping": 3
        }

        thal_map = {
            "Normal": 3,
            "Fixed Defect": 6,
            "Reversible Defect": 7
        }

        input_data = {
            "age": age,
            "sex": 1 if sex == "Male" else 0,
            "cp": cp_map[chest_pain],
            "trestbps": trestbps,
            "chol": chol,
            "fbs": 1 if fbs == "Yes" else 0,
            "restecg": restecg_map[restecg],
            "thalach": thalach,
            "exang": 1 if exang == "Yes" else 0,
            "oldpeak": oldpeak,
            "slope": slope_map[slope],
            "ca": ca,
            "thal": thal_map[thal]
        }

        try:

            with st.spinner(
                "🧠 Running HealthGuard AI..."
            ):

                raw = predict_risk(
                    input_data
                )

                prediction, probability = (
                    normalize_prediction(raw)
                )

                try:

                    risk_label = get_risk_label(
                        prediction,
                        probability
                    )

                except Exception:

                    risk_label, _, = risk_info(
                        probability
                    )

                explanation = explain_prediction(
                    input_data
                )

            st.session_state.prediction_result = {
                "prediction": prediction,
                "probability": probability,
                "risk_label": risk_label
            }

            st.session_state.patient_data = (
                input_data
            )

            st.session_state.explanation_df = (
                explanation
            )

            st.success(
                "✅ AI analysis completed."
            )

        except Exception as error:

            st.error(
                "Prediction could not be completed."
            )

            st.exception(error)

    # ========================================================
    # SHOW RESULT
    # ========================================================

    result = st.session_state.prediction_result

    if result:

        probability = result["probability"]
        prediction = result["prediction"]
        risk_label = result["risk_label"]

        display_label, display_text, display_color = (
            risk_info(probability)
        )

        st.markdown(
            '<div class="section-title">Analysis Result</div>',
            unsafe_allow_html=True
        )

        left, right = st.columns([1, 1])

        with left:

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="badge">
                        AI MODEL OUTPUT
                    </div>

                    <div class="result-number"
                         style="color:{display_color};">

                        {probability * 100:.1f}%

                    </div>

                    <div class="result-label">
                        {display_label}
                    </div>

                    <div class="result-sub">
                        {display_text}
                    </div>

                    <br>

                    <div class="result-sub">
                        Model class: {prediction}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with right:

            st.plotly_chart(
                gauge(probability),
                use_container_width=True
            )

        # ====================================================
        # SHAP
        # ====================================================

        explanation = (
            st.session_state.explanation_df
        )

        if explanation is not None:

            st.markdown(
                '<div class="section-title">Why did the model produce this result?</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="card">

                    <div class="card-title">
                        🔍 Explainable AI
                    </div>

                    <div class="card-text">
                        SHAP values show how strongly each
                        feature contributed to the model output.
                        Larger absolute values indicate stronger
                        contribution to this prediction.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            c1, c2 = st.columns([1.35, 1])

            with c1:

                st.plotly_chart(
                    shap_chart(
                        explanation,
                        8
                    ),
                    use_container_width=True
                )

            with c2:

                st.dataframe(
                    explanation[
                        [
                            "feature",
                            "value",
                            "shap_value"
                        ]
                    ].head(8),
                    use_container_width=True,
                    hide_index=True
                )

        st.markdown(
            """
            <div class="disclaimer">

            <strong>Important:</strong>
            HealthGuard AI is an educational machine-learning
            project. The displayed probability is a model output,
            not a medical diagnosis. Medical measurements should
            come from appropriate professional sources.

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# SHAP PAGE
# ============================================================

elif page == "SHAP Explainability":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-kicker">
                EXPLAINABLE ARTIFICIAL INTELLIGENCE
            </div>

            <div class="hero-title">
                SHAP <span class="gradient">Explainability</span>
            </div>

            <div class="hero-text">
                Explore the feature contributions behind
                the most recent model prediction.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    explanation = (
        st.session_state.explanation_df
    )

    if explanation is None:

        st.info(
            "Run a prediction first to generate SHAP explanations."
        )

    else:

        st.plotly_chart(
            shap_chart(
                explanation,
                13
            ),
            use_container_width=True
        )

        st.dataframe(
            explanation,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-kicker">
                DATA EXPLORATION
            </div>

            <div class="hero-title">
                Dataset <span class="gradient">Analytics</span>
            </div>

            <div class="hero-text">
                Explore distributions and patterns in the
                HealthGuard training dataset.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    if df.empty:

        st.warning(
            "Dataset could not be loaded."
        )

    else:

        numeric = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        feature = st.selectbox(
            "Select a feature",
            numeric
        )

        c1, c2 = st.columns(2)

        with c1:

            fig = px.histogram(
                df,
                x=feature,
                color="target"
                if "target" in df.columns
                else None,
                title=f"{feature} Distribution"
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color=theme["text"]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with c2:

            if "target" in df.columns:

                fig = px.box(
                    df,
                    x="target",
                    y=feature,
                    title=f"{feature} vs Target"
                )

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color=theme["text"]
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        st.markdown(
            '<div class="section-title">Dataset Preview</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            df.head(25),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# AI INSIGHTS
# ============================================================

elif page == "AI Insights":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-kicker">
                GLOBAL MODEL INTELLIGENCE
            </div>

            <div class="hero-title">
                AI <span class="gradient">Insights</span>
            </div>

            <div class="hero-text">
                Estimate global feature importance using
                SHAP explanations across a dataset sample.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    if df.empty:

        st.warning(
            "Dataset unavailable."
        )

    else:

        if st.button(
            "🧠 Generate Global Insights",
            use_container_width=True
        ):

            features = [
                column
                for column in df.columns
                if column != "target"
            ]

            sample = df.sample(
                min(15, len(df)),
                random_state=42
            )

            progress = st.progress(0)

            results = []

            for index, (_, row) in enumerate(
                sample.iterrows()
            ):

                try:

                    row_data = {
                        feature: row[feature]
                        for feature in features
                    }

                    explanation = (
                        explain_prediction(
                            row_data
                        )
                    )

                    results.append(
                        explanation[
                            [
                                "feature",
                                "absolute_shap"
                            ]
                        ]
                    )

                except Exception:
                    pass

                progress.progress(
                    (index + 1) / len(sample)
                )

            if results:

                combined = pd.concat(
                    results,
                    ignore_index=True
                )

                importance = (
                    combined
                    .groupby("feature")
                    ["absolute_shap"]
                    .mean()
                    .sort_values(
                        ascending=False
                    )
                    .reset_index()
                )

                importance.columns = [
                    "Feature",
                    "Mean Absolute SHAP"
                ]

                st.session_state.global_importance = (
                    importance
                )

        importance = (
            st.session_state.global_importance
        )

        if importance is not None:

            fig = px.bar(
                importance.head(13),
                x="Mean Absolute SHAP",
                y="Feature",
                orientation="h",
                title="Global Feature Importance"
            )

            fig.update_layout(
                height=500,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color=theme["text"]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                importance,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# REPORTS
# ============================================================

elif page == "Reports":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-kicker">
                PROFESSIONAL REPORT CENTER
            </div>

            <div class="hero-title">
                HealthGuard <span class="gradient">Reports</span>
            </div>

            <div class="hero-text">
                Generate a structured HTML report containing
                the model output and SHAP explanation.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    patient = st.session_state.patient_data
    result = st.session_state.prediction_result
    explanation = st.session_state.explanation_df

    if (
        patient is None
        or result is None
        or explanation is None
    ):

        st.info(
            "No prediction is available yet."
        )

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    📋 How to create a report
                </div>

                <div class="card-text">
                    Open Risk Prediction → enter the
                    available values → run the analysis →
                    return here.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        c1, c2, c3 = st.columns(3)

        with c1:

            st.markdown(
                metric(
                    "🎯",
                    "Prediction",
                    result["prediction"]
                ),
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                metric(
                    "📊",
                    "Probability",
                    f"{result['probability'] * 100:.2f}%"
                ),
                unsafe_allow_html=True
            )

        with c3:

            st.markdown(
                metric(
                    "🛡️",
                    "Risk Category",
                    result["risk_label"]
                ),
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(
            "📄 GENERATE PROFESSIONAL REPORT",
            use_container_width=True
        ):

            try:

                path, html = generate_html_report(
                    patient_data=patient,
                    prediction=result["prediction"],
                    probability=result["probability"],
                    risk_label=result["risk_label"],
                    explanation_df=explanation
                )

                st.success(
                    "✅ Report generated successfully."
                )

                st.download_button(
                    "⬇️ DOWNLOAD HTML REPORT",
                    data=html,
                    file_name=Path(path).name,
                    mime="text/html",
                    use_container_width=True
                )

            except Exception as error:

                st.error(
                    "Report generation failed."
                )

                st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <strong>HEALTHGUARD AI</strong>

        <br>

        Machine Learning • Explainable AI • Data Analytics

        <br><br>

        Educational AI/ML demonstration — not a medical
        diagnostic system.

    </div>
    """,
    unsafe_allow_html=True
)