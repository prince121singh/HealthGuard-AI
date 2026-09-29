import html
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from core.predictor import predict_risk, get_risk_label
from core.preprocessing import load_data
from core.explainability import explain_prediction

st.set_page_config(page_title="HealthGuard AI", page_icon="🫀", layout="wide",
                   initial_sidebar_state="expanded")

# ============================================================
# CONSTANTS  (single source of truth for encodings)
# Must match the encoding used when the model was TRAINED.
# UCI: cp 1-4, slope 1-3, thal 3/6/7.  Kaggle: cp 0-3, slope 0-2, thal 0-3.
# ============================================================
CP = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 4}
SLOPE = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
THAL = {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}
ECG = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
ALLOWED = {"cp": set(CP.values()), "slope": set(SLOPE.values()), "thal": set(THAL.values())}

DEFAULTS = dict(age=50, sex="Male", cp="Asymptomatic", trestbps=130, chol=220, fbs=False,
                restecg="Normal", thalach=150, exang=False, oldpeak=1.0, slope="Flat",
                ca=0, thal="Normal")

I18N = {
    "English": {
        "Studio": "Live Studio", "Compare": "Compare", "Explain": "Explain", "Data": "Data",
        "Models": "Models", "Report": "Report", "About": "About & Limits",
        "headline": "See how each number moves the model.",
        "sub": "Change any value and the model output updates instantly. This is a learning tool, not a diagnosis.",
        "familiar": "Input familiarity", "percentile": "Where you sit in the dataset",
    },
    "हिन्दी": {
        "Studio": "लाइव स्टूडियो", "Compare": "तुलना", "Explain": "व्याख्या", "Data": "डेटा",
        "Models": "मॉडल", "Report": "रिपोर्ट", "About": "जानकारी और सीमाएँ",
        "headline": "देखिए हर संख्या मॉडल को कैसे बदलती है।",
        "sub": "कोई भी मान बदलें, मॉडल का आउटपुट तुरंत बदलेगा। यह सीखने का टूल है, निदान नहीं।",
        "familiar": "इनपुट कितना जाना-पहचाना है", "percentile": "डेटासेट में आपकी स्थिति",
    },
}

THEMES = {
    "Ink": dict(bg="#0a0e14", bg2="#10161f", card="rgba(20,28,40,.78)", solid="#141c28",
                text="#eef2f7", muted="#8a99ab", border="rgba(150,175,205,.16)",
                ok="#4fd1a5", mid="#f2b84b", bad="#ff5d73", accent="#7aa2ff", shadow="rgba(0,0,0,.35)"),
    "Paper": dict(bg="#f3f1ec", bg2="#faf9f6", card="rgba(255,255,255,.9)", solid="#ffffff",
                  text="#1b2430", muted="#657386", border="rgba(40,60,80,.14)",
                  ok="#0a8f6a", mid="#b57a00", bad="#cf3a4f", accent="#2f5fd0", shadow="rgba(30,45,60,.10)"),
    "Lagoon": dict(bg="#06161a", bg2="#0a2228", card="rgba(12,40,47,.8)", solid="#0d2a31",
                   text="#eaf7f6", muted="#88aeb0", border="rgba(110,220,210,.16)",
                   ok="#5eead4", mid="#fcd34d", bad="#fb7185", accent="#67b7ff", shadow="rgba(0,0,0,.35)"),
}

# ============================================================
# STATE
# ============================================================
for k, v in {"theme": "Ink", "lang": "English", "page": "Studio", "profile": dict(DEFAULTS),
             "history": [], "explain_on": True}.items():
    st.session_state.setdefault(k, v)

T = THEMES[st.session_state.theme]
L = I18N[st.session_state.lang]
esc = lambda v: html.escape(str(v))


def css():
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Instrument+Sans:wght@400;500;600;700&display=swap');
:root {{ --bg:{T['bg']}; --bg2:{T['bg2']}; --card:{T['card']}; --solid:{T['solid']}; --text:{T['text']};
  --muted:{T['muted']}; --border:{T['border']}; --ok:{T['ok']}; --mid:{T['mid']}; --bad:{T['bad']};
  --accent:{T['accent']}; --shadow:{T['shadow']}; }}
html, body, .stApp, [data-testid="stSidebar"] * {{ font-family:'Instrument Sans',system-ui,sans-serif; }}
.stApp {{ background: linear-gradient(160deg, var(--bg), var(--bg2)); color:var(--text); }}
.block-container {{ max-width:1440px; padding-top:1.2rem; padding-bottom:4rem; }}
[data-testid="stSidebar"] {{ background:var(--solid); border-right:1px solid var(--border); }}
h1,h2,h3,h4,label,.stMarkdown p {{ color:var(--text); }}
h1,h2,h3 {{ font-family:'Fraunces',Georgia,serif !important; letter-spacing:-.4px; }}
.muted {{ color:var(--muted); }}
.hero {{ padding:8px 4px 18px; }}
.hero h1 {{ font-size:clamp(2rem,3.6vw,3.2rem); line-height:1.05; margin:0; max-width:820px; }}
.hero p {{ color:var(--muted); max-width:640px; margin:10px 0 0; line-height:1.6; }}
.panel {{ background:var(--card); border:1px solid var(--border); border-radius:18px; padding:20px;
  box-shadow:0 10px 34px var(--shadow); backdrop-filter:blur(14px); }}
.tile {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:14px 16px; }}
.tile b {{ display:block; font-size:1.35rem; margin-top:2px; }}
.tile span {{ color:var(--muted); font-size:.78rem; }}
.note {{ border-left:3px solid var(--mid); padding:10px 14px; background:color-mix(in srgb,var(--mid) 9%,transparent);
  border-radius:0 10px 10px 0; color:var(--muted); font-size:.84rem; line-height:1.55; }}
.flag {{ border-left:3px solid var(--bad); padding:8px 12px; margin-top:8px; border-radius:0 10px 10px 0;
  background:color-mix(in srgb,var(--bad) 9%,transparent); color:var(--muted); font-size:.82rem; }}
.pill {{ display:inline-block; padding:5px 12px; border-radius:999px; font-weight:600; font-size:.8rem; }}
.monitor {{ position:relative; text-align:center; padding:8px 0 4px; }}
.ecg {{ fill:none; stroke-width:2.4; stroke-linecap:round; stroke-linejoin:round;
  stroke-dasharray:420; animation:sweep var(--beat,1s) linear infinite; }}
@keyframes sweep {{ from {{ stroke-dashoffset:420; }} to {{ stroke-dashoffset:0; }} }}
.heart {{ transform-origin:center; animation:beat var(--beat,1s) ease-in-out infinite; }}
@keyframes beat {{ 0%,100% {{ transform:scale(1); }} 12% {{ transform:scale(1.09); }} 24% {{ transform:scale(1); }} 36% {{ transform:scale(1.05); }} }}
.ring {{ transition:stroke-dasharray .6s ease, stroke .4s ease; }}
.stButton > button[kind="primary"], .stDownloadButton > button {{ border:0; color:#fff;
  background:linear-gradient(135deg,var(--accent),var(--ok)); border-radius:12px; font-weight:600; }}
[data-testid="stMetric"] {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:12px 14px; }}
button:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}

/* ---- 3D / depth layer ---- */
.stApp {{ perspective:1400px; }}
.orb {{ position:fixed; border-radius:50%; filter:blur(60px); opacity:.35; z-index:0; pointer-events:none; }}
.orb1 {{ width:420px; height:420px; top:-160px; left:-120px; background:radial-gradient(circle,var(--accent),transparent 70%);
  animation:drift1 22s ease-in-out infinite; }}
.orb2 {{ width:360px; height:360px; bottom:-140px; right:-100px; background:radial-gradient(circle,var(--ok),transparent 70%);
  animation:drift2 26s ease-in-out infinite; }}
@keyframes drift1 {{ 0%,100% {{ transform:translate(0,0) scale(1); }} 50% {{ transform:translate(60px,40px) scale(1.15); }} }}
@keyframes drift2 {{ 0%,100% {{ transform:translate(0,0) scale(1); }} 50% {{ transform:translate(-50px,-30px) scale(1.1); }} }}

.hero h1 {{ background:linear-gradient(100deg,var(--text) 30%,var(--accent) 55%,var(--text) 80%);
  background-size:220% auto; -webkit-background-clip:text; background-clip:text; color:transparent;
  animation:sheen 7s linear infinite; }}
@keyframes sheen {{ to {{ background-position:-220% center; }} }}

.panel, .tile {{ transform-style:preserve-3d; transition:transform .35s ease, box-shadow .35s ease, border-color .35s ease; }}
.panel:hover {{ transform:translateY(-4px) rotateX(2deg) rotateY(-1.5deg); box-shadow:0 22px 55px var(--shadow),
  0 0 0 1px color-mix(in srgb,var(--accent) 30%,transparent); }}
.tile {{ position:relative; overflow:hidden; }}
.tile:hover {{ transform:translateY(-3px) scale(1.015); border-color:color-mix(in srgb,var(--accent) 45%,var(--border)); }}
.tile::after {{ content:""; position:absolute; inset:0; background:linear-gradient(120deg,transparent,color-mix(in srgb,var(--accent) 14%,transparent),transparent);
  transform:translateX(-120%); transition:transform .6s ease; }}
.tile:hover::after {{ transform:translateX(120%); }}
.heart3d {{ border-radius:18px; overflow:hidden; }}
@media (prefers-reduced-motion:reduce) {{ .ecg,.heart {{ animation:none; }} .ecg {{ stroke-dashoffset:0; }}
  .orb1,.orb2,.hero h1 {{ animation:none; }} .panel:hover,.tile:hover {{ transform:none; }} }}
</style>
<div class="orb orb1"></div><div class="orb orb2"></div>""", unsafe_allow_html=True)


css()

# ============================================================
# DATA / MODEL LOADERS
# ============================================================
@st.cache_data(show_spinner=False)
def get_dataset():
    return load_data()


@st.cache_data(show_spinner=False)
def get_model_results():
    p = "models/model_results.pkl"
    if not os.path.exists(p):
        return {}
    try:
        import joblib
        r = joblib.load(p)
        return r if isinstance(r, dict) else {}
    except Exception:
        return {}


@st.cache_data(show_spinner=False, max_entries=512)
def cached_predict(key):
    return predict_risk(dict(key))


@st.cache_data(show_spinner=False, max_entries=128)
def cached_explain(key):
    try:
        return explain_prediction(dict(key)), None
    except Exception as e:
        return None, str(e)


try:
    df = get_dataset()
except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.stop()

model_results = get_model_results()

# Auto-detect encoding from the CSV the model was trained on.
# Kaggle heart.csv starts cp/slope at 0; the original UCI file starts at 1.
# NOTE: the Kaggle thal mapping below is the commonly used one - verify it against your CSV source.
ENCODING = "kaggle" if ("cp" in df.columns and int(df["cp"].min()) == 0) else "uci"
if ENCODING == "kaggle":
    CP = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    SLOPE = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    THAL = {"Normal": 2, "Fixed Defect": 1, "Reversible Defect": 3}
    ALLOWED = {"cp": set(CP.values()), "slope": set(SLOPE.values()), "thal": {0, 1, 2, 3}}

PRESETS = {
    "Custom": None,
    "Healthy 30-year-old": dict(age=30, sex="Female", cp="Non-anginal Pain", trestbps=115, chol=180,
                                thalach=185, oldpeak=0.0, slope="Upsloping", exang=False, ca=0, thal="Normal"),
    "60-year-old, high BP": dict(age=60, sex="Male", cp="Asymptomatic", trestbps=165, chol=260,
                                 thalach=125, oldpeak=2.4, slope="Flat", exang=True, ca=1, thal="Reversible Defect"),
}
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal"]
NICE = {"age": "Age", "sex": "Sex", "cp": "Chest pain", "trestbps": "Resting BP", "chol": "Cholesterol",
        "fbs": "Fasting sugar", "restecg": "Resting ECG", "thalach": "Max heart rate",
        "exang": "Exercise angina", "oldpeak": "ST depression", "slope": "ST slope",
        "ca": "Major vessels", "thal": "Thalassemia"}


def encoding_problems():
    """Compare dataset values with the encodings this app sends to the model."""
    out = []
    for col, ok in ALLOWED.items():
        if col in df.columns:
            seen = set(pd.Series(df[col]).dropna().astype(int).unique())
            if not seen.issubset(ok):
                out.append(f"`{col}` in dataset has values {sorted(seen)}, app sends {sorted(ok)}")
    return out


def to_model_input(p):
    return {"age": p["age"], "sex": 1 if p["sex"] == "Male" else 0, "cp": CP[p["cp"]],
            "trestbps": p["trestbps"], "chol": p["chol"], "fbs": int(p["fbs"]),
            "restecg": ECG[p["restecg"]], "thalach": p["thalach"], "exang": int(p["exang"]),
            "oldpeak": float(p["oldpeak"]), "slope": SLOPE[p["slope"]], "ca": int(p["ca"]),
            "thal": THAL[p["thal"]]}


def run(p):
    key = tuple(sorted(to_model_input(p).items()))
    return key, cached_predict(key)


def tone(prob):
    return T["ok"] if prob < .33 else T["mid"] if prob < .66 else T["bad"]


def band(prob):
    return "Lower model output" if prob < .33 else "Moderate model output" if prob < .66 else "Elevated model output"


def familiarity(p):
    """Share of numeric inputs inside the 5th-95th percentile of training data."""
    cols = [c for c in ["age", "trestbps", "chol", "thalach", "oldpeak"] if c in df.columns]
    inside, flags = 0, []
    for c in cols:
        v = p[c]
        lo, hi = df[c].quantile(.05), df[c].quantile(.95)
        if lo <= v <= hi:
            inside += 1
        elif v < df[c].min() or v > df[c].max():
            flags.append(f"{NICE[c]} = {v} is outside the whole training range "
                         f"({df[c].min():g}–{df[c].max():g}); the model is extrapolating.")
    return (inside / len(cols) if cols else 1.0), flags


def monitor_svg(prob, hr):
    c = tone(prob)
    beat = max(0.4, min(2.0, 60 / max(hr, 40)))
    dash = int(max(0.0, min(1.0, prob)) * 553)  # circumference of r=88 circle
    return f"""
<div class="monitor" style="--beat:{beat:.2f}s">
<svg viewBox="0 0 320 240" width="100%" style="max-width:380px" role="img" aria-label="Model output {prob*100:.0f} percent">
  <circle cx="160" cy="112" r="88" fill="none" stroke="{T['border']}" stroke-width="10"/>
  <circle class="ring" cx="160" cy="112" r="88" fill="none" stroke="{c}" stroke-width="10" stroke-linecap="round"
     stroke-dasharray="{dash} 553" transform="rotate(-90 160 112)"/>
  <g class="heart"><path d="M160 150 C116 118 122 80 146 80 C155 80 160 87 160 92 C160 87 165 80 174 80 C198 80 204 118 160 150Z" fill="{c}" opacity=".9"/></g>
  <text x="160" y="182" text-anchor="middle" font-family="Fraunces,serif" font-size="30" fill="{T['text']}">{prob*100:.0f}%</text>
  <path class="ecg" stroke="{c}" d="M10 218 L80 218 L92 218 L100 200 L108 232 L116 190 L124 218 L170 218 L182 218 L190 204 L198 226 L206 218 L310 218"/>
</svg></div>"""


def heart_3d(prob, hr, height=300):
    """Rotating, beating 3D heart rendered with three.js (WebGL). Color follows risk tone."""
    c = tone(prob)
    bpm = max(40, min(180, hr))
    period_ms = int(60000 / bpm)
    return components.html(f"""
<div id="wrap" style="width:100%;height:{height}px"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function() {{
  const wrap = document.getElementById('wrap');
  const w = wrap.clientWidth || 320, h = {height};
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, w/h, 0.1, 100);
  camera.position.set(0, 0.3, 5.4);
  const renderer = new THREE.WebGLRenderer({{ antialias:true, alpha:true }});
  renderer.setSize(w, h);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  wrap.appendChild(renderer.domElement);

  // Build a heart silhouette and extrude it into a 3D mesh.
  const shape = new THREE.Shape();
  const x0 = 0, y0 = 0;
  shape.moveTo(x0, y0);
  shape.bezierCurveTo(x0, y0 - 0.6, x0 - 1.1, y0 - 0.6, x0 - 1.1, y0 + 0.25);
  shape.bezierCurveTo(x0 - 1.1, y0 + 1.05, x0 - 0.5, y0 + 1.5, x0, y0 + 2.0);
  shape.bezierCurveTo(x0 + 0.5, y0 + 1.5, x0 + 1.1, y0 + 1.05, x0 + 1.1, y0 + 0.25);
  shape.bezierCurveTo(x0 + 1.1, y0 - 0.6, x0, y0 - 0.6, x0, y0);
  const geo = new THREE.ExtrudeGeometry(shape, {{ depth:0.7, bevelEnabled:true, bevelThickness:0.12,
    bevelSize:0.1, bevelSegments:6, curveSegments:24 }});
  geo.center();
  geo.rotateZ(Math.PI);
  geo.rotateX(-0.15);

  const material = new THREE.MeshStandardMaterial({{ color:new THREE.Color('{c}'), roughness:0.32,
    metalness:0.18, emissive:new THREE.Color('{c}'), emissiveIntensity:0.18 }});
  const mesh = new THREE.Mesh(geo, material);
  scene.add(mesh);

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const key = new THREE.PointLight(0xffffff, 1.1); key.position.set(3, 4, 5); scene.add(key);
  const rim = new THREE.PointLight('{c}', 0.9); rim.position.set(-4, -2, -3); scene.add(rim);

  const period = {period_ms};
  const start = performance.now();
  function animate(t) {{
    const dt = (t - start) / 1000;
    const beatPhase = ((t - start) % period) / period;
    const pulse = 1 + 0.09 * Math.exp(-8 * beatPhase) * Math.sin(beatPhase * Math.PI * 6);
    mesh.scale.set(pulse, pulse, pulse);
    mesh.rotation.y = dt * 0.55;
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }}
  requestAnimationFrame(animate);
}})();
</script>
""", height=height)


def layout(fig, h=360):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=48, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=T["text"], family="Instrument Sans"),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor=T["border"]); fig.update_yaxes(gridcolor=T["border"])
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def tile(label, value, note=""):
    st.markdown(f'<div class="tile"><span>{esc(label)}</span><b>{esc(value)}</b><span>{esc(note)}</span></div>',
                unsafe_allow_html=True)


def disclaimer(text="Educational machine-learning demo. Not a diagnosis, not medical advice."):
    st.markdown(f'<div class="note">⚠️ {esc(text)}</div>', unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
PAGES = ["Studio", "Compare", "Explain", "Data", "Models", "Report", "About"]
with st.sidebar:
    st.markdown("### 🫀 HealthGuard AI")
    st.caption("Health-risk analytics workbench")
    st.session_state.page = st.radio("Navigate", PAGES, format_func=lambda k: L[k],
                                     index=PAGES.index(st.session_state.page), label_visibility="collapsed")
    st.divider()
    lang = st.selectbox("Language / भाषा", list(I18N), index=list(I18N).index(st.session_state.lang))
    theme = st.selectbox("Theme", list(THEMES), index=list(THEMES).index(st.session_state.theme))
    if lang != st.session_state.lang or theme != st.session_state.theme:
        st.session_state.lang, st.session_state.theme = lang, theme
        st.rerun()
    st.divider()
    ok_model = os.path.exists("models/model_results.pkl")
    try:
        import shap  # noqa: F401
        ok_shap = True
    except Exception:
        ok_shap = False
    st.caption(f"{'🟢' if len(df) else '🔴'} Dataset · {len(df):,} rows")
    st.caption(f"{'🟢' if ok_model else '🟠'} Training metrics file")
    st.caption(f"{'🟢' if ok_shap else '🔴'} SHAP library")
    st.caption(f"🧬 Detected encoding: {ENCODING.upper()}")
    probs = encoding_problems()
    if probs:
        st.error("Encoding mismatch detected. Predictions may be wrong.")

if probs:
    with st.expander("⚠️ Encoding mismatch details", expanded=True):
        for m in probs:
            st.write("•", m)
        st.caption("Fix the dictionaries in CONSTANTS so they match the data the model was trained on.")

page = st.session_state.page
P = st.session_state.profile

# ============================================================
# LIVE STUDIO
# ============================================================
if page == "Studio":
    st.markdown(f'<div class="hero"><h1>{esc(L["headline"])}</h1><p>{esc(L["sub"])}</p></div>',
                unsafe_allow_html=True)
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        pa, pb = st.columns([2, 1])
        choice = pa.selectbox("Start from a sample profile", list(PRESETS), key="preset_choice")
        if pb.button("Apply", use_container_width=True) and PRESETS[choice]:
            st.session_state.profile = {**DEFAULTS, **PRESETS[choice]}
            st.rerun()
        t1, t2, t3, t4 = st.tabs(["Profile", "Vitals & labs", "Exercise test", "Other"])
        with t1:
            a, b = st.columns(2)
            P["age"] = a.slider("Age (years)", 1, 100, P["age"])
            P["sex"] = b.radio("Sex", ["Female", "Male"], index=["Female", "Male"].index(P["sex"]), horizontal=True)
            P["cp"] = st.selectbox("Chest pain type", list(CP), index=list(CP).index(P["cp"]))
        with t2:
            P["trestbps"] = st.slider("Resting blood pressure (mmHg)", 70, 220, P["trestbps"])
            P["chol"] = st.slider("Serum cholesterol (mg/dL)", 100, 600, P["chol"])
            a, b = st.columns(2)
            P["fbs"] = a.toggle("Fasting sugar > 120 mg/dL", value=P["fbs"])
            P["restecg"] = b.selectbox("Resting ECG", list(ECG), index=list(ECG).index(P["restecg"]))
        with t3:
            P["thalach"] = st.slider("Maximum heart rate (bpm)", 60, 220, P["thalach"])
            P["oldpeak"] = st.slider("ST depression (oldpeak)", 0.0, 7.0, float(P["oldpeak"]), 0.1)
            a, b = st.columns(2)
            P["slope"] = a.selectbox("ST segment slope", list(SLOPE), index=list(SLOPE).index(P["slope"]))
            P["exang"] = b.toggle("Exercise-induced angina", value=P["exang"])
        with t4:
            a, b = st.columns(2)
            P["ca"] = a.selectbox("Major vessels (0–3)", [0, 1, 2, 3], index=P["ca"])
            P["thal"] = b.selectbox("Thalassemia", list(THAL), index=list(THAL).index(P["thal"]))
        if st.button("Reset to defaults"):
            st.session_state.profile = dict(DEFAULTS)
            st.rerun()

    try:
        key, res = run(P)
        prob = res.get("probability")
        prob = 0.5 if prob is None else float(prob)
        label = get_risk_label(res.get("prediction"), res.get("probability"))
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    fam, flags = familiarity(P)
    st.session_state.last = dict(key=key, res=res, prob=prob, label=label)

    with right:
        c = tone(prob)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        use_3d = st.toggle("3D view", value=True, help="Turn off for a lighter, static gauge")
        st.markdown('<div class="heart3d">', unsafe_allow_html=True)
        if use_3d:
            heart_3d(prob, P["thalach"], height=280)
            st.markdown(f'<div style="text-align:center;font-family:Fraunces,serif;font-size:2.1rem;'
                        f'color:{c};margin-top:-6px">{prob*100:.0f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(monitor_svg(prob, P["thalach"]), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="text-align:center"><div style="font-family:Fraunces,serif;font-size:1.5rem">'
                    f'{esc(label)}</div><span class="pill" style="color:{c};background:{c}22">{band(prob)}</span></div>',
                    unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")
        a, b = st.columns(2)
        with a:
            tile(L["familiar"], f"{fam*100:.0f}%", "of numeric inputs are typical for the training data")
        with b:
            tile("Model class", str(int(res.get("prediction", 0))), "binary classifier output")
        for f in flags:
            st.markdown(f'<div class="flag">{esc(f)}</div>', unsafe_allow_html=True)
        st.write("")
        if st.button("Save to session history", type="primary", use_container_width=True):
            st.session_state.history.insert(0, dict(time=datetime.now().strftime("%H:%M:%S"),
                                                    output=label, probability=prob))
            st.session_state.history = st.session_state.history[:10]
            st.toast("Saved to this session only. Nothing is stored on a server.")

    st.markdown(f"#### {L['percentile']}")
    cols = st.columns(5)
    for col, f in zip(cols, ["age", "trestbps", "chol", "thalach", "oldpeak"]):
        if f in df.columns:
            pct = float((df[f] < P[f]).mean() * 100)
            with col:
                tile(NICE[f], f"{pct:.0f}th", f"percentile · you: {P[f]}")
    st.write("")
    disclaimer()

# ============================================================
# COMPARE
# ============================================================
elif page == "Compare":
    st.markdown(f'<div class="hero"><h1>{esc(L["Compare"])}: today vs. a different scenario</h1>'
                '<p>Change a few values and see how the model output shifts. This shows how the model behaves, '
                'not what would happen to a real person.</p></div>', unsafe_allow_html=True)
    Q = dict(P)
    a, b, c, d = st.columns(4)
    Q["trestbps"] = a.slider("Resting BP", 70, 220, P["trestbps"], key="cmp_bp")
    Q["chol"] = b.slider("Cholesterol", 100, 600, P["chol"], key="cmp_ch")
    Q["thalach"] = c.slider("Max heart rate", 60, 220, P["thalach"], key="cmp_hr")
    Q["oldpeak"] = d.slider("ST depression", 0.0, 7.0, float(P["oldpeak"]), 0.1, key="cmp_op")
    _, r0 = run(P); _, r1 = run(Q)
    p0, p1 = float(r0["probability"] or 0), float(r1["probability"] or 0)
    m1, m2, m3 = st.columns(3)
    m1.metric("Current profile", f"{p0*100:.1f}%")
    m2.metric("Scenario", f"{p1*100:.1f}%", f"{(p1-p0)*100:+.1f} pts", delta_color="inverse")
    m3.metric("Change", "Lower" if p1 < p0 else "Higher" if p1 > p0 else "Same")
    fig = go.Figure(go.Bar(x=["Current", "Scenario"], y=[p0 * 100, p1 * 100],
                           marker_color=[tone(p0), tone(p1)], text=[f"{p0*100:.1f}%", f"{p1*100:.1f}%"],
                           textposition="outside"))
    fig.update_yaxes(range=[0, 105], title="Model probability (%)")
    show(layout(fig, 380))
    disclaimer("A change in model output is a statistical association in the training data, not proof that changing a value would change someone's health.")

# ============================================================
# EXPLAIN
# ============================================================
elif page == "Explain":
    st.markdown(f'<div class="hero"><h1>Why did the model say that?</h1>'
                '<p>Each bar pushes the output up or down from the starting point. Bigger bars mean more influence on the model.</p></div>',
                unsafe_allow_html=True)
    last = st.session_state.get("last")
    if not last:
        st.info("Open Live Studio first so there is a profile to explain.")
    else:
        with st.spinner("Computing SHAP values…"):
            exp, err = cached_explain(last["key"])
        if exp is None:
            st.warning(f"SHAP explanation unavailable: {err}")
        else:
            top = exp.reindex(exp["shap_value"].abs().sort_values(ascending=False).index).head(8)
            top = top.iloc[::-1]
            fig = go.Figure(go.Waterfall(
                orientation="h", measure=["relative"] * len(top), y=[NICE.get(f, f) for f in top["feature"]],
                x=top["shap_value"], connector={"line": {"color": T["border"]}},
                increasing={"marker": {"color": T["bad"]}}, decreasing={"marker": {"color": T["ok"]}},
                text=[f"{v:+.3f}" for v in top["shap_value"]], textposition="outside"))
            fig.update_layout(title="Top feature contributions (red raises output, green lowers it)")
            show(layout(fig, 460))
            pos = top[top["shap_value"] > 0].sort_values("shap_value", ascending=False)
            neg = top[top["shap_value"] < 0].sort_values("shap_value")
            s = []
            if len(pos):
                r = pos.iloc[0]; s.append(f"**{NICE.get(r['feature'], r['feature'])}** ({r['value']}) raised the output the most.")
            if len(neg):
                r = neg.iloc[0]; s.append(f"**{NICE.get(r['feature'], r['feature'])}** ({r['value']}) lowered it the most.")
            st.markdown(" ".join(s))
            with st.expander("Full attribution table"):
                st.dataframe(exp, use_container_width=True, hide_index=True)
    disclaimer("A model contribution is not medical causation.")

# ============================================================
# DATA
# ============================================================
elif page == "Data":
    st.markdown('<div class="hero"><h1>The data behind the model</h1></div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    with a: tile("Rows", f"{len(df):,}")
    with b: tile("Columns", len(df.columns))
    with c: tile("Missing cells", f"{int(df.isna().sum().sum()):,}")
    with d: tile("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    st.write("")
    t1, t2, t3, t4 = st.tabs(["Target", "Distributions", "Correlation", "Raw data"])
    with t1:
        if "target" in df.columns:
            vc = df["target"].value_counts().sort_index()
            fig = px.pie(names=[f"Class {i}" for i in vc.index], values=vc.values, hole=.65,
                         color_discrete_sequence=[T["ok"], T["bad"]])
            show(layout(fig, 380))
            st.caption("If one class is much larger, accuracy alone can mislead. Check recall and calibration.")
    with t2:
        num = df.select_dtypes(include=np.number).columns.tolist()
        f = st.selectbox("Numeric feature", num)
        show(layout(px.histogram(df, x=f, nbins=30, marginal="box", color_discrete_sequence=[T["accent"]]), 430))
    with t3:
        show(layout(px.imshow(df.select_dtypes(include=np.number).corr(), text_auto=".2f", aspect="auto"), 620))
    with t4:
        st.dataframe(df, use_container_width=True, hide_index=True, height=520)

# ============================================================
# MODELS
# ============================================================
elif page == "Models":
    st.markdown('<div class="hero"><h1>Model lab</h1><p>Metrics saved by the training pipeline.</p></div>',
                unsafe_allow_html=True)
    if not model_results:
        st.info("models/model_results.pkl was not found or could not be read. Run the training script first.")
    else:
        rows = []
        for name, m in model_results.items():
            if isinstance(m, dict):
                rows.append({"Model": str(name), **{k: v for k, v in m.items()
                                                    if isinstance(v, (int, float, np.floating))}})
        if rows:
            mdf = pd.DataFrame(rows)
            st.dataframe(mdf, use_container_width=True, hide_index=True)
            cols = [c for c in mdf.columns if c != "Model"]
            if cols:
                mt = st.selectbox("Compare metric", cols)
                fig = px.bar(mdf, x="Model", y=mt, text=mt, color_discrete_sequence=[T["accent"]])
                fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                show(layout(fig, 400))
        else:
            st.json(model_results)
    disclaimer("These metrics come from held-out data of a small public dataset. They are not evidence of clinical effectiveness.")

# ============================================================
# REPORT
# ============================================================
elif page == "Report":
    st.markdown('<div class="hero"><h1>Report</h1><p>A portable HTML summary of the current profile.</p></div>',
                unsafe_allow_html=True)
    last = st.session_state.get("last")
    if not last:
        st.info("Open Live Studio first to create a profile.")
    else:
        exp, _ = cached_explain(last["key"])
        rows = "".join(f"<tr><td>{esc(NICE[k])}</td><td>{esc(v)}</td></tr>" for k, v in P.items())
        shap_rows = ""
        if exp is not None:
            for _, r in exp.head(8).iterrows():
                shap_rows += (f"<tr><td>{esc(NICE.get(r['feature'], r['feature']))}</td>"
                              f"<td>{esc(r['value'])}</td><td>{float(r['shap_value']):+.4f}</td></tr>")
        shap_html = (f"<h2>Top contributions</h2><table><tr><th>Feature</th><th>Value</th><th>SHAP</th></tr>"
                     f"{shap_rows}</table>") if shap_rows else ""
        report = f"""<!doctype html><html><head><meta charset="utf-8"><title>HealthGuard AI report</title>
<style>body{{font-family:Georgia,serif;background:#f3f1ec;color:#1b2430;padding:40px}}
.c{{max-width:820px;margin:auto;background:#fff;padding:40px;border-radius:16px}}
table{{width:100%;border-collapse:collapse;margin-top:10px;font-family:Arial,sans-serif}}
th,td{{border:1px solid #dde3e8;padding:8px;text-align:left}}th{{background:#f1f4f6}}
.n{{margin-top:28px;padding:14px;background:#fff6dc;border-radius:10px;font-family:Arial,sans-serif;font-size:14px}}</style></head>
<body><div class="c"><h1>HealthGuard AI</h1><p>{datetime.now():%d %B %Y, %I:%M %p}</p>
<h2>{esc(last['label'])}</h2><p>Model probability: <b>{last['prob']*100:.1f}%</b> ({band(last['prob'])})</p>
<h2>Inputs</h2><table><tr><th>Feature</th><th>Value</th></tr>{rows}</table>{shap_html}
<div class="n"><b>Important:</b> Educational ML demonstration. Not a diagnosis or medical advice.
Please consult a qualified healthcare professional.</div></div></body></html>"""
        a, b = st.columns(2)
        with a: tile("Latest output", last["label"])
        with b: tile("Probability", f"{last['prob']*100:.1f}%")
        st.download_button("Download HTML report", report, "healthguard_report.html", "text/html",
                           use_container_width=True)
    if st.session_state.history:
        st.markdown("#### This session")
        h = pd.DataFrame(st.session_state.history)
        h["probability"] = h["probability"].map(lambda x: f"{x*100:.1f}%")
        st.dataframe(h, use_container_width=True, hide_index=True)

# ============================================================
# ABOUT
# ============================================================
elif page == "About":
    st.markdown('<div class="hero"><h1>What this tool can and cannot do</h1></div>', unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("""
**What it is**
- A learning workbench for tabular health-risk models
- A way to see how inputs move a model's output
- An example of explainable ML with SHAP

**Privacy**
- Inputs stay in your browser session
- Nothing is written to a database or file on the server
""")
    with b:
        st.markdown("""
**Known limits**
- Trained on a small public dataset (Cleveland-style, a few hundred rows)
- Not clinically validated and not tested on external populations
- Inputs like vessel count and thalassemia need hospital tests
- SHAP shows what the model used, not what causes disease

**If you feel unwell** or are worried about your heart, see a doctor. This app cannot replace that.
""")
    disclaimer()

st.markdown(f'<div style="margin-top:40px;text-align:center;color:{T["muted"]};font-size:.76rem">'
            'HealthGuard AI · Python · Streamlit · scikit-learn · SHAP · Plotly · Educational use only</div>',
            unsafe_allow_html=True)