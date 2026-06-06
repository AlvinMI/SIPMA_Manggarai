"""
SIPMA — Sistem Informasi Prediksi Muka Air
Pintu Air Manggarai · Jakarta
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import io
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# ════════════════════════════════════════════
# 0. PAGE CONFIG  ← harus baris pertama Streamlit
# ════════════════════════════════════════════
st.set_page_config(
    page_title="SIPMA · Prediksi Muka Air",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",   # ← KUNCI: selalu expanded saat load
)

# ════════════════════════════════════════════
# 1. SESSION STATE
# ════════════════════════════════════════════
_defaults = {
    "dark_mode": True,
    "page"     : "🏠 Beranda",
    "result"   : None,
    "history"  : [],
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ════════════════════════════════════════════
# 2. PALET WARNA
# ════════════════════════════════════════════
DK = dict(
    bg="#080c12", surface="#0e1420", surface2="#141b26",
    border="#1e2a3a", border2="#172030",
    text="#e8f0fe", subtext="#c5d3e8", muted="#6b7fa3", faint="#3d4f6b",
    accent="#4d9fff", accent2="#1a6fd4",
    green="#2ecc71",  green_bg="#071a0f",
    yellow="#f0c040", yellow_bg="#1a1500",
    orange="#ff8c42", orange_bg="#1a0d00",
    red="#ff5c7a",    red_bg="#1a0510",
    sidebar="#060a10",
    chart_bg="#0b1018",
)
LT = dict(
    bg="#f0f4fb", surface="#ffffff", surface2="#e8eef7",
    border="#c8d4e8", border2="#dde6f5",
    text="#0d1421", subtext="#1e2d47", muted="#5a6e92", faint="#8fa3c4",
    accent="#1a6fd4", accent2="#0a4fa0",
    green="#1a8a3a",  green_bg="#e0fce9",
    yellow="#8a6000", yellow_bg="#fff8d0",
    orange="#c44a00", orange_bg="#fff0e0",
    red="#c0003a",    red_bg="#ffe0ea",
    sidebar="#e4ecf8",
    chart_bg="#f8faff",
)
T = DK if st.session_state.dark_mode else LT

# ════════════════════════════════════════════
# 3. CSS — FIX SIDEBAR + NAVBAR + LAYOUT
# ════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap');

/* ── Reset & base ───────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; }}
html, body {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: {T['bg']} !important; color: {T['text']} !important; }}
#MainMenu, footer, header {{ visibility: hidden; }}
h1,h2,h3,h4,h5,h6,p,span,div,label {{ color: inherit; }}

/* ── KRITIS: hapus padding atas bawaan Streamlit ── */
.block-container {{
    padding-top: 0.6rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}}

/* ── KRITIS: Fix sidebar agar SELALU tampil & bisa dibuka kembali ── */
/* Jangan sembunyikan / override tombol collapse sidebar sama sekali */
[data-testid="stSidebar"] {{
    background: {T['sidebar']} !important;
    border-right: 1px solid {T['border']} !important;
    min-width: 270px !important;
}}
[data-testid="stSidebar"] > div:first-child {{
    background: {T['sidebar']} !important;
}}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {{
    color: {T['subtext']} !important;
}}
/* Tombol collapse bawaan — JANGAN disentuh, biarkan Streamlit handle */

/* ── Header SIPMA ───────────────────────── */
.sipma-header {{
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 0.4rem 0 0.7rem;
    border-bottom: 1px solid {T['border']};
    margin-bottom: 0.6rem;
}}
.sipma-logo {{
    font-size: 1.9rem;
    line-height: 1;
    flex-shrink: 0;
    filter: drop-shadow(0 0 12px rgba(77,159,255,0.6));
}}
.sipma-title {{
    font-size: 1.4rem;
    font-weight: 800;
    font-family: 'Syne', sans-serif;
    color: {T['text']} !important;
    letter-spacing: -.02em;
    line-height: 1.1;
}}
.sipma-sub {{
    font-size: .7rem;
    color: {T['muted']} !important;
    letter-spacing: .04em;
    margin-top: .15rem;
}}
/* Badge status di header */
.sipma-badge {{
    margin-left: auto;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 4px;
}}
.badge-pill {{
    font-size: .62rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    padding: .2rem .55rem;
    border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
}}
.badge-online {{ background: {T['green_bg']}; color: {T['green']}; border: 1px solid {T['green']}33; }}
.badge-time   {{ background: {T['surface']}; color: {T['muted']}; border: 1px solid {T['border']}; }}

/* ── Navbar ─────────────────────────────── */
.nav-outer {{
    background: {T['surface']};
    border: 1px solid {T['border']};
    border-radius: 12px;
    padding: 4px;
    margin-bottom: 1rem;
    display: flex;
    gap: 3px;
}}

/* ── Metric cards ───────────────────────── */
[data-testid="stMetric"] {{
    background: {T['surface']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
    position: relative;
    overflow: hidden;
}}
[data-testid="stMetric"]::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {T['accent']}, {T['accent2']});
    border-radius: 12px 12px 0 0;
}}
[data-testid="stMetricLabel"] div {{
    font-size: .65rem !important;
    font-weight: 700 !important;
    letter-spacing: .1em !important;
    text-transform: uppercase !important;
    color: {T['muted']} !important;
}}
[data-testid="stMetricValue"] div {{
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    color: {T['text']} !important;
}}
[data-testid="stMetricDelta"] div {{
    font-size: .76rem !important;
    font-weight: 600 !important;
}}

/* ── Tombol ─────────────────────────────── */
div.stButton > button {{
    background: {T['surface2']} !important;
    color: {T['subtext']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: .82rem !important;
    transition: all .15s !important;
    font-family: 'Inter', sans-serif !important;
}}
div.stButton > button:hover {{
    background: {T['border']} !important;
    color: {T['text']} !important;
    border-color: {T['muted']} !important;
}}
div.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {T['accent']}, {T['accent2']}) !important;
    color: #fff !important;
    border: none !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 14px rgba(77,159,255,0.35) !important;
}}
div.stButton > button[kind="primary"]:hover {{
    box-shadow: 0 6px 20px rgba(77,159,255,0.5) !important;
    transform: translateY(-1px);
}}

/* ── Download button ────────────────────── */
div.stDownloadButton > button {{
    background: {T['surface']} !important;
    color: {T['accent']} !important;
    border: 1.5px solid {T['accent']}55 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all .15s !important;
}}
div.stDownloadButton > button:hover {{
    background: {T['accent']}15 !important;
    border-color: {T['accent']} !important;
}}

/* ── File uploader ──────────────────────── */
[data-testid="stFileUploader"] {{
    background: {T['surface']} !important;
    border: 1.5px dashed {T['border']} !important;
    border-radius: 10px !important;
    padding: .4rem !important;
    transition: border-color .2s !important;
}}
[data-testid="stFileUploader"]:hover {{
    border-color: {T['accent']}88 !important;
}}
[data-testid="stFileUploader"] * {{ color: {T['subtext']} !important; }}
[data-testid="stFileUploader"] button {{
    background: {T['accent']} !important;
    color: #fff !important;
    border: none !important;
}}

/* ── Number input ───────────────────────── */
input[type="number"], .stNumberInput input {{
    background: {T['surface']} !important;
    color: {T['text']} !important;
    border-color: {T['border']} !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .85rem !important;
}}

/* ── Checkbox ───────────────────────────── */
[data-testid="stCheckbox"] span {{
    color: {T['subtext']} !important;
    font-size: .84rem !important;
}}

/* ── Divider ────────────────────────────── */
hr {{ border: none; border-top: 1px solid {T['border2']}; margin: .8rem 0; }}

/* ── Dataframe ──────────────────────────── */
[data-testid="stDataFrame"] iframe {{
    border-radius: 10px;
    border: 1px solid {T['border']} !important;
}}

/* ── Custom cards ───────────────────────── */
.info-card {{
    background: {T['surface']};
    border: 1px solid {T['border']};
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin: .5rem 0;
}}
.info-card .card-label {{
    font-size: .62rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: {T['muted']};
    margin-bottom: .65rem;
    font-family: 'Syne', sans-serif;
}}
.info-card ul {{
    margin: 0; padding-left: 1.1rem;
    color: {T['subtext']};
    font-size: .84rem;
    line-height: 2;
}}

/* ── EWS cards ──────────────────────────── */
.ews {{
    border-radius: 12px;
    padding: 1rem 1.3rem;
    margin: .5rem 0;
    border-left: 4px solid;
    position: relative;
    overflow: hidden;
}}
.ews::after {{
    content: '';
    position: absolute;
    top: -20px; right: -20px;
    width: 80px; height: 80px;
    border-radius: 50%;
    opacity: .06;
    background: currentColor;
}}
.ews h4 {{ margin: 0 0 .35rem; font-size: .95rem; font-weight: 700; font-family: 'Syne', sans-serif; }}
.ews p  {{ margin: 0; font-size: .83rem; color: {T['subtext']}; line-height: 1.55; }}
.ews-normal  {{ background:{T['green_bg']};  border-color:{T['green']};  }}
.ews-waspada {{ background:{T['yellow_bg']}; border-color:{T['yellow']}; }}
.ews-kritis  {{ background:{T['orange_bg']}; border-color:{T['orange']}; }}
.ews-bahaya  {{ background:{T['red_bg']};    border-color:{T['red']};    }}

/* ── Compare table ──────────────────────── */
.cmp-tbl {{ width:100%; border-collapse:collapse; font-size:.84rem; }}
.cmp-tbl th {{
    background:{T['surface2']}; padding:.5rem .9rem;
    text-align:left; font-size:.62rem; font-weight:700;
    letter-spacing:.1em; text-transform:uppercase; color:{T['muted']};
    border-bottom: 1px solid {T['border']};
    font-family: 'Syne', sans-serif;
}}
.cmp-tbl td {{
    padding:.55rem .9rem;
    border-bottom:1px solid {T['border2']};
    color:{T['subtext']};
    font-size:.84rem;
}}
.cmp-tbl tr:last-child td {{ border-bottom:none; }}
.cmp-tbl tr:hover td {{ background: {T['surface2']}44; }}
.val-best {{ color:{T['green']} !important; font-weight:700; font-family:'IBM Plex Mono',monospace; }}
.val-norm {{ font-family:'IBM Plex Mono',monospace; color:{T['subtext']}; }}

/* ── Progress bar stat ──────────────────── */
.stat-bar-wrap {{
    margin: .3rem 0 .8rem;
}}
.stat-bar-label {{
    display: flex;
    justify-content: space-between;
    font-size: .72rem;
    color: {T['muted']};
    margin-bottom: .25rem;
}}
.stat-bar-track {{
    height: 5px;
    background: {T['border']};
    border-radius: 99px;
    overflow: hidden;
}}
.stat-bar-fill {{
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, {T['accent']}, {T['green']});
    transition: width .6s cubic-bezier(.4,0,.2,1);
}}

/* ── Empty state ────────────────────────── */
.empty {{
    text-align:center; padding:3.5rem 1rem;
}}
.empty .icon {{ font-size:2.8rem; margin-bottom:.75rem; filter: opacity(.6); }}
.empty h3 {{ color:{T['text']}; font-size:1.05rem; font-weight:700; margin:0 0 .4rem; font-family:'Syne',sans-serif; }}
.empty p  {{ font-size:.84rem; color:{T['muted']}; margin:0; line-height:1.6; }}

/* ── Section label ──────────────────────── */
.sec-label {{
    font-size:.62rem; font-weight:700; letter-spacing:.12em;
    text-transform:uppercase; color:{T['muted']};
    margin: .8rem 0 .4rem;
    font-family: 'Syne', sans-serif;
}}

/* ── Laporan table ──────────────────────── */
.rpt-tbl {{ width:100%; border-collapse:collapse; }}
.rpt-tbl td {{ padding:.32rem 0; vertical-align: top; }}
.rpt-tbl td:first-child {{
    color:{T['muted']};
    font-size:.83rem;
    width:42%;
    padding-right:.5rem;
}}
.rpt-tbl td:last-child {{
    color:{T['subtext']};
    font-weight:600;
    font-size:.85rem;
}}
.rpt-mono {{ font-family:'IBM Plex Mono',monospace; }}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════
# 4. KONSTANTA
# ════════════════════════════════════════════
EVAL = {
    "LSTM": {"RMSE": 25.4271, "MAE": 12.8734, "R2": 0.7932},
    "RF"  : {"RMSE": 26.5742, "MAE": 13.6658, "R2": 0.7735},
}
SIAGA_DEF = [
    {"lvl":1,"label":"BAHAYA", "css":"bahaya", "icon":"🔴","min":950,"max":9999,"color":DK["red"]    if st.session_state.dark_mode else LT["red"]},
    {"lvl":2,"label":"KRITIS", "css":"kritis", "icon":"🟠","min":850,"max":949, "color":DK["orange"]  if st.session_state.dark_mode else LT["orange"]},
    {"lvl":3,"label":"WASPADA","css":"waspada","icon":"🟡","min":750,"max":849, "color":DK["yellow"]  if st.session_state.dark_mode else LT["yellow"]},
    {"lvl":4,"label":"NORMAL", "css":"normal", "icon":"🟢","min":0,  "max":749, "color":DK["green"]   if st.session_state.dark_mode else LT["green"]},
]
INSTRUKSI = {
    "NORMAL" : "Kondisi aman. Lanjutkan pemantauan rutin setiap jam. Tidak ada tindakan darurat.",
    "WASPADA": "Tingkatkan pemantauan menjadi 30 menit sekali. Siapkan tim siaga & periksa kapasitas pompa.",
    "KRITIS" : "Aktifkan protokol darurat. Koordinasi BPBD DKI. Pertimbangkan pembukaan pintu air tambahan segera.",
    "BAHAYA" : "DARURAT PENUH — Aktifkan semua pompa & buka seluruh pintu air. Koordinasi evakuasi kawasan rawan banjir segera!",
}
PAGES = ["🏠 Beranda", "🤖 LSTM", "🌲 Random Forest", "📜 Riwayat", "📋 Laporan"]

def get_siaga(tma):
    for s in SIAGA_DEF:
        if s["min"] <= tma <= s["max"]:
            return s
    return SIAGA_DEF[-1]

# ════════════════════════════════════════════
# 5. LOAD MODEL
# ════════════════════════════════════════════
@st.cache_resource(show_spinner="⚙️ Memuat model…")
def load_models():
    m = {}
    try:
        m["lstm"]        = load_model("models/model_lstm_manggarai.h5")
        m["scaler_lstm"] = joblib.load("models/scaler_tma.sav")
        m["lstm_ok"]     = True
    except Exception as e:
        m["lstm_ok"] = False; m["lstm_err"] = str(e)
    try:
        m["rf"]          = joblib.load("models/model_rf_manggarai.sav")
        m["scaler_rf"]   = joblib.load("models/scaler_rf.sav")
        m["rf_ok"]       = True
    except Exception as e:
        m["rf_ok"]  = False; m["rf_err"] = str(e)
    return m

MDL = load_models()

# ════════════════════════════════════════════
# 6. FUNGSI PREDIKSI
# ════════════════════════════════════════════
def sanitize(raw):
    arr = np.array(raw, dtype=float).flatten()[:24].reshape(-1,1)
    if np.median(arr) > 2000:
        arr /= 10.0
    return arr

def safe_inv(scaler, val):
    r = float(scaler.inverse_transform([[val]])[0][0])
    if r > 2000: r /= 10.0
    return None if (r < 0 or r > 2000) else r

def pred_lstm(arr):
    if not MDL.get("lstm_ok"): return None
    try:
        sc = MDL["scaler_lstm"]
        p  = MDL["lstm"].predict(sc.transform(arr).reshape(1,24,1), verbose=0)
        return safe_inv(sc, float(p[0][0]))
    except: return None

def pred_rf(arr):
    if not MDL.get("rf_ok"): return None
    try:
        sc = MDL["scaler_rf"]
        p  = MDL["rf"].predict(sc.transform(arr).reshape(1,-1))
        return safe_inv(sc, float(p[0]))
    except: return None

# ════════════════════════════════════════════
# 7. SIDEBAR  ← tidak menyentuh tombol collapse
# ════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding:.6rem 0 .5rem; text-align:center;">
        <div style="font-size:2rem; filter:drop-shadow(0 0 10px {T['accent']}66);">🌊</div>
        <div style="font-size:1.2rem; font-weight:800; font-family:'Syne',sans-serif;
                    color:{T['accent']}; letter-spacing:-.01em; margin-top:.2rem;">SIPMA</div>
        <div style="font-size:.58rem; color:{T['muted']}; letter-spacing:.14em;
                    text-transform:uppercase; margin-top:.1rem;">Prediksi Muka Air</div>
        <div style="font-size:.58rem; color:{T['faint']}; margin-top:.1rem;">
            Pintu Air Manggarai · Jakarta</div>
    </div>
    """, unsafe_allow_html=True)

    # Status model
    lstm_ok = MDL.get("lstm_ok", False)
    rf_ok   = MDL.get("rf_ok", False)
    ok_c  = T["green"]  if lstm_ok else T["red"]
    ok_c2 = T["green"]  if rf_ok   else T["red"]
    st.markdown(f"""
    <div style="display:flex;gap:6px;justify-content:center;margin:.4rem 0 .6rem;flex-wrap:wrap;">
        <span style="font-size:.6rem;font-weight:700;padding:.18rem .5rem;
                     border-radius:20px;background:{ok_c}22;color:{ok_c};
                     border:1px solid {ok_c}44;font-family:'IBM Plex Mono',monospace;">
            {'✓' if lstm_ok else '✗'} LSTM
        </span>
        <span style="font-size:.6rem;font-weight:700;padding:.18rem .5rem;
                     border-radius:20px;background:{ok_c2}22;color:{ok_c2};
                     border:1px solid {ok_c2}44;font-family:'IBM Plex Mono',monospace;">
            {'✓' if rf_ok else '✗'} RF
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)

    # Theme + Reset
    c1, c2 = st.columns(2)
    if c1.button("☀️ Light" if st.session_state.dark_mode else "🌙 Dark",
                 key="theme_btn", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    if c2.button("🔄 Reset", key="reset_btn", use_container_width=True):
        st.session_state.result = None
        st.session_state.page   = "🏠 Beranda"
        st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Input method
    st.markdown(f"<div class='sec-label' style='color:{T['muted']};'>📥 Sumber Data TMA</div>",
                unsafe_allow_html=True)
    method = st.radio("input_method", ["📂 Upload File", "✏️ Input Manual"],
                      label_visibility="collapsed")
    data_series = []

    if "Upload" in method:
        uploaded = st.file_uploader(
            "CSV / Excel — min. 24 baris", type=["csv","xlsx","xls"],
            label_visibility="visible"
        )
        if uploaded:
            try:
                if uploaded.name.endswith(("xlsx","xls")):
                    df_raw = pd.read_excel(uploaded)
                else:
                    df_raw = pd.read_csv(uploaded, sep=None, engine="python")
                num_df = df_raw.apply(pd.to_numeric, errors="coerce")
                vals   = num_df[num_df.count().idxmax()].dropna().values
                if len(vals) < 24:
                    st.error(f"Hanya {len(vals)} data valid. Butuh ≥ 24.")
                else:
                    data_series = vals[-24:].tolist()
                    st.success(f"✅ {len(vals)} data dimuat · ambil 24 terakhir.")
            except Exception as e:
                st.error(f"Gagal membaca: {e}")
    else:
        st.caption("Isi TMA (cm) — urutan terlama → terbaru")
        cols_in = st.columns(2)
        for i in range(24):
            v = cols_in[i%2].number_input(
                f"J-{24-i}", value=550.0, step=1.0,
                key=f"mi_{i}", label_visibility="visible"
            )
            data_series.append(v)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"<div class='sec-label' style='color:{T['muted']};'>⚙️ Tampilan Grafik</div>",
                unsafe_allow_html=True)
    show_lstm = st.checkbox("Tampilkan prediksi LSTM", value=True, key="ck_lstm")
    show_rf   = st.checkbox("Tampilkan prediksi RF",   value=True, key="ck_rf")
    show_band = st.checkbox("Tampilkan interval ±5%",  value=True, key="ck_band")

    st.markdown("<hr/>", unsafe_allow_html=True)
    run = st.button("▶  Jalankan Analisis", key="run_btn", type="primary", use_container_width=True)

    st.markdown(f"""
    <div style="font-size:.58rem;color:{T['faint']};text-align:center;margin-top:.5rem;
                font-family:'IBM Plex Mono',monospace;">
        {datetime.now().strftime('%d %b %Y  %H:%M')}
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════
# 8. PROSES PREDIKSI
# ════════════════════════════════════════════
if run:
    if len(data_series) != 24:
        st.warning("Data belum lengkap (butuh tepat 24 nilai).", icon="⚠️")
    else:
        arr     = sanitize(np.array(data_series))
        last_cm = float(arr[-1][0])
        with st.spinner("🔄 Menghitung prediksi…"):
            r_l = pred_lstm(arr)
            r_r = pred_rf(arr)
        if r_l is None and r_r is None:
            st.error("Kedua model gagal. Periksa file model & scaler.", icon="❌")
        else:
            ts      = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            primary = r_l if r_l is not None else r_r
            sg      = get_siaga(primary)
            st.session_state.result = dict(
                arr=arr, last_cm=last_cm,
                lstm=r_l, rf=r_r, ts=ts, siaga=sg
            )
            st.session_state.history.append({
                "Waktu"         : ts,
                "TMA (cm)"      : round(last_cm, 1),
                "LSTM t+6 (cm)" : round(r_l, 2) if r_l else "—",
                "RF t+6 (cm)"   : round(r_r, 2) if r_r else "—",
                "Status"        : f"{sg['icon']} Siaga {sg['lvl']} · {sg['label']}",
            })
            st.session_state.history = st.session_state.history[-30:]
            st.session_state.page    = "🏠 Beranda"
            st.rerun()

# ════════════════════════════════════════════
# 9. HEADER  (nempel ke atas, minimal space)
# ════════════════════════════════════════════
res = st.session_state.result
sg_now = res["siaga"] if res else None

st.markdown(f"""
<div class="sipma-header">
    <div class="sipma-logo">🌊</div>
    <div>
        <div class="sipma-title">SIPMA
            <span style="font-weight:400;color:{T['muted']};font-size:.88rem;"> · Manggarai</span>
        </div>
        <div class="sipma-sub">Sistem Informasi Prediksi Muka Air · Komparatif LSTM &amp; Random Forest</div>
    </div>
    <div class="sipma-badge">
        <span class="badge-pill badge-online">● Sistem Aktif</span>
        <span class="badge-pill badge-time">{datetime.now().strftime('%H:%M · %d %b')}</span>
        {f'<span class="badge-pill" style="background:{sg_now["color"]}22;color:{sg_now["color"]};border:1px solid {sg_now["color"]}44;">{sg_now["icon"]} Siaga {sg_now["lvl"]}</span>' if sg_now else ''}
    </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════
# 10. NAVBAR (langsung di bawah header, rapat)
# ════════════════════════════════════════════
nav_cols = st.columns(len(PAGES))
for i, (col, lbl) in enumerate(zip(nav_cols, PAGES)):
    active = (st.session_state.page == lbl)
    # Override style per kolom
    st.markdown(f"""
    <style>
    div[data-testid="stHorizontalBlock"] > div:nth-child({i+1}) > div:first-child > div:first-child > button {{
        background: {'linear-gradient(135deg,'+T['accent']+','+T['accent2']+')' if active else T['surface']} !important;
        color: {'#ffffff' if active else T['muted']} !important;
        border: 1px solid {'transparent' if active else T['border']} !important;
        border-radius: 8px !important;
        font-weight: {'700' if active else '500'} !important;
        font-size: .77rem !important;
        width: 100% !important;
        padding: .38rem !important;
        box-shadow: {'0 3px 10px rgba(77,159,255,0.35)' if active else 'none'} !important;
        font-family: 'Inter', sans-serif !important;
    }}
    </style>
    """, unsafe_allow_html=True)
    if col.button(lbl, key=f"nav_{i}", use_container_width=True):
        st.session_state.page = lbl
        st.rerun()

st.markdown("<hr style='margin:.2rem 0 .8rem;'/>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# 11. HELPER — GRAFIK (upgrade)
# ════════════════════════════════════════════
def build_chart(arr, r_l=None, r_r=None, sl=True, sr=True, band=True):
    hist = arr.flatten()
    tmax = max(hist) * 1.20
    tmin = max(0, min(hist) * 0.82)

    fig = go.Figure()

    # Band siaga (subtle)
    bands = [
        (950, tmax, T["red"],    "Siaga 1 · Bahaya"),
        (850, 950,  T["orange"], "Siaga 2 · Kritis"),
        (750, 850,  T["yellow"], "Siaga 3 · Waspada"),
        (tmin,750,  T["green"],  "Siaga 4 · Normal"),
    ]
    for y0, y1, col, nm in bands:
        lo, hi = max(y0,tmin), min(y1,tmax)
        if hi > lo:
            fig.add_hrect(y0=lo, y1=hi, fillcolor=col, opacity=0.05,
                          layer="below", line_width=0,
                          annotation_text=nm, annotation_position="right",
                          annotation_font=dict(size=8.5, color=col))
        if tmin < y0 < tmax:
            fig.add_hline(y=y0, line_dash="dot", line_color=col, line_width=1.2,
                          annotation_text=f"  {y0} cm",
                          annotation_font=dict(size=8.5, color=col),
                          annotation_position="left")

    x_hist = list(range(24))
    x_pred = 29  # t+6 jam

    # Smoothed area fill historis
    fig.add_trace(go.Scatter(
        x=x_hist, y=hist.tolist(),
        fill="tozeroy", fillcolor=f"rgba(77,159,255,0.05)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))

    # Garis historis utama
    fig.add_trace(go.Scatter(
        x=x_hist, y=hist.tolist(), name="Historis 24 Jam",
        line=dict(color=T["accent"], width=2.5),
        mode="lines",
        hovertemplate="<b>J-%{x}</b>  TMA: <b>%{y:.1f} cm</b><extra></extra>",
    ))

    # Titik terakhir historis
    fig.add_trace(go.Scatter(
        x=[23], y=[hist[-1]],
        mode="markers", showlegend=False,
        marker=dict(color=T["accent"], size=9,
                    line=dict(color=T["chart_bg"], width=2)),
        hoverinfo="skip",
    ))

    # LSTM prediction
    if r_l is not None and sl:
        # Confidence band ±5%
        if band:
            lo_l, hi_l = r_l*0.95, r_l*1.05
            fig.add_trace(go.Scatter(
                x=[23, x_pred, x_pred, 23],
                y=[hist[-1], hi_l, lo_l, hist[-1]],
                fill="toself", fillcolor=f"rgba(255,92,122,0.10)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ))
        # Garis proyeksi
        fig.add_trace(go.Scatter(
            x=[23, x_pred], y=[hist[-1], r_l],
            mode="lines", showlegend=False,
            line=dict(color=T["red"], width=1.8, dash="dash"), hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[x_pred], y=[r_l],
            name=f"LSTM  {r_l:.1f} cm",
            mode="markers+text",
            marker=dict(color=T["red"], size=14, symbol="diamond",
                        line=dict(color=T["chart_bg"], width=2)),
            text=[f"  {r_l:.1f} cm"], textposition="middle right",
            textfont=dict(size=11, color=T["red"], family="IBM Plex Mono"),
            hovertemplate="<b>LSTM t+6</b>: %{y:.2f} cm<extra></extra>",
        ))

    # RF prediction
    if r_r is not None and sr:
        if band:
            lo_r, hi_r = r_r*0.95, r_r*1.05
            fig.add_trace(go.Scatter(
                x=[23, x_pred, x_pred, 23],
                y=[hist[-1], hi_r, lo_r, hist[-1]],
                fill="toself", fillcolor=f"rgba(240,192,64,0.10)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=[23, x_pred], y=[hist[-1], r_r],
            mode="lines", showlegend=False,
            line=dict(color=T["yellow"], width=1.8, dash="dash"), hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[x_pred], y=[r_r],
            name=f"RF  {r_r:.1f} cm",
            mode="markers+text",
            marker=dict(color=T["yellow"], size=14, symbol="diamond",
                        line=dict(color=T["chart_bg"], width=2)),
            text=[f"  {r_r:.1f} cm"], textposition="middle right",
            textfont=dict(size=11, color=T["yellow"], family="IBM Plex Mono"),
            hovertemplate="<b>RF t+6</b>: %{y:.2f} cm<extra></extra>",
        ))

    # Garis pembatas historis / prediksi
    fig.add_vline(
        x=23.5,
        line=dict(color=T["faint"], width=1.2, dash="dashdot"),
        annotation_text="  ← Historis  |  Prediksi →",
        annotation_font=dict(size=9, color=T["faint"]),
        annotation_position="top right",
    )

    # Tick labels
    tick_vals = list(range(0, 24, 3)) + [x_pred]
    tick_text = [f"J-{24-i}" for i in range(0,24,3)] + ["t+6"]

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=T["chart_bg"],
        font=dict(family="Inter", color=T["subtext"], size=11),
        margin=dict(l=10, r=110, t=18, b=52),
        height=400,
        legend=dict(
            bgcolor=T["surface"] + "cc",
            bordercolor=T["border"], borderwidth=1,
            x=0.01, y=0.99,
            font=dict(size=11, color=T["subtext"], family="IBM Plex Mono"),
        ),
        xaxis=dict(
            title=dict(text="Waktu", font=dict(color=T["muted"], size=11)),
            gridcolor=T["border2"],
            tickvals=tick_vals, ticktext=tick_text,
            tickfont=dict(color=T["subtext"], size=10),
            linecolor=T["border"], zerolinecolor=T["border2"],
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(text="TMA (cm)", font=dict(color=T["muted"], size=11)),
            gridcolor=T["border2"],
            tickfont=dict(color=T["subtext"], size=10),
            ticksuffix=" cm", range=[tmin, tmax],
            linecolor=T["border"], zerolinecolor=T["border2"],
            showgrid=True,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=T["surface"] + "ee",
            bordercolor=T["border"],
            font=dict(color=T["text"], size=11, family="IBM Plex Mono"),
        ),
    )
    return fig

# ════════════════════════════════════════════
# 12. HELPER — KOMPONEN UI
# ════════════════════════════════════════════
def ews_card(tma):
    s = get_siaga(tma)
    st.markdown(f"""
    <div class="ews ews-{s['css']}">
        <h4 style="color:{s['color']};">{s['icon']}  Siaga {s['lvl']} — {s['label']}</h4>
        <p>{INSTRUKSI[s['label']]}</p>
    </div>""", unsafe_allow_html=True)

def eval_tbl():
    E = EVAL
    def win(k, hi=False):
        return "LSTM" if (E["LSTM"][k] > E["RF"][k]) == hi else "RF"
    wr, wm, w2 = win("RMSE"), win("MAE"), win("R2", hi=True)
    def cls(model, winner):
        return "val-best" if model == winner else "val-norm"
    # R² sebagai progress bar
    r2_l = E['LSTM']['R2'] * 100
    r2_r = E['RF']['R2']   * 100
    st.markdown(f"""
    <table class="cmp-tbl">
    <thead><tr><th>Metrik</th><th>LSTM</th><th>Random Forest</th><th>Terbaik</th></tr></thead>
    <tbody>
    <tr><td>RMSE ↓ (cm)</td>
        <td class="{cls('LSTM',wr)}">{E['LSTM']['RMSE']:.4f}</td>
        <td class="{cls('RF',wr)}">{E['RF']['RMSE']:.4f}</td>
        <td style="color:{T['green']};font-weight:700;font-size:.78rem;">🏆 {wr}</td></tr>
    <tr><td>MAE ↓ (cm)</td>
        <td class="{cls('LSTM',wm)}">{E['LSTM']['MAE']:.4f}</td>
        <td class="{cls('RF',wm)}">{E['RF']['MAE']:.4f}</td>
        <td style="color:{T['green']};font-weight:700;font-size:.78rem;">🏆 {wm}</td></tr>
    <tr><td>R² ↑</td>
        <td class="{cls('LSTM',w2)}">{E['LSTM']['R2']:.4f}</td>
        <td class="{cls('RF',w2)}">{E['RF']['R2']:.4f}</td>
        <td style="color:{T['green']};font-weight:700;font-size:.78rem;">🏆 {w2}</td></tr>
    </tbody></table>
    <div class="stat-bar-wrap" style="margin-top:.8rem;">
        <div class="stat-bar-label"><span>R² LSTM</span><span>{r2_l:.2f}%</span></div>
        <div class="stat-bar-track"><div class="stat-bar-fill" style="width:{r2_l}%;background:linear-gradient(90deg,{T['accent']},{T['green']});"></div></div>
    </div>
    <div class="stat-bar-wrap">
        <div class="stat-bar-label"><span>R² RF</span><span>{r2_r:.2f}%</span></div>
        <div class="stat-bar-track"><div class="stat-bar-fill" style="width:{r2_r}%;background:linear-gradient(90deg,{T['yellow']},{T['orange']});"></div></div>
    </div>
    <p style="font-size:.65rem;color:{T['faint']};margin:.4rem 0 0;">Data uji: TMA Manggarai 2016–2020</p>
    """, unsafe_allow_html=True)

def export_btns(res):
    if not res: return
    df = pd.DataFrame([{
        "Waktu Analisis"   : res["ts"],
        "TMA Terakhir (cm)": res["last_cm"],
        "LSTM t+6 (cm)"    : res["lstm"] or "N/A",
        "RF t+6 (cm)"      : res["rf"]   or "N/A",
        "Delta LSTM (cm)"  : round(res["lstm"]-res["last_cm"],2) if res["lstm"] else "N/A",
        "Delta RF (cm)"    : round(res["rf"]-res["last_cm"],2)   if res["rf"]   else "N/A",
        "Status Siaga"     : f"Siaga {res['siaga']['lvl']} {res['siaga']['label']}",
    }])
    ts_str = datetime.now().strftime("%Y%m%d_%H%M")
    c1, c2 = st.columns(2)
    c1.download_button(
        "⬇ Export CSV", data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"sipma_{ts_str}.csv", mime="text/csv", use_container_width=True
    )
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Prediksi")
        if st.session_state.history:
            pd.DataFrame(st.session_state.history).to_excel(w, index=False, sheet_name="Riwayat")
    c2.download_button(
        "⬇ Export Excel", data=buf.getvalue(),
        file_name=f"sipma_{ts_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

def empty_state(icon="🌊", title="Sistem Siap",
                msg="Lengkapi data di sidebar lalu tekan <b>▶ Jalankan Analisis</b>."):
    st.markdown(f"""
    <div class="empty">
        <div class="icon">{icon}</div>
        <h3>{title}</h3>
        <p>{msg}</p>
    </div>""", unsafe_allow_html=True)

def stat_mini(label, val, color=None):
    c = color or T["accent"]
    st.markdown(f"""
    <div style="background:{T['surface']};border:1px solid {T['border']};
                border-radius:10px;padding:.8rem 1rem;text-align:center;">
        <div style="font-size:.62rem;font-weight:700;letter-spacing:.1em;
                    text-transform:uppercase;color:{T['muted']};font-family:'Syne',sans-serif;">{label}</div>
        <div style="font-size:1.45rem;font-weight:700;font-family:'IBM Plex Mono',monospace;
                    color:{c};margin-top:.2rem;">{val}</div>
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════
# 13. HALAMAN: BERANDA
# ════════════════════════════════════════════
page = st.session_state.page

if page == "🏠 Beranda":
    st.markdown(f"<p style='color:{T['muted']};font-size:.83rem;margin:-.4rem 0 .9rem;'>Komparatif prediksi t+6 jam · LSTM vs Random Forest</p>", unsafe_allow_html=True)

    if not res:
        # Info panel saat belum ada hasil
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="info-card">
                <div class="card-label">Cara Penggunaan</div>
                <ul>
                    <li>Pilih metode input di <b>sidebar kiri</b></li>
                    <li>Upload file CSV/Excel <b>≥ 24 baris</b> data TMA, atau isi manual</li>
                    <li>Atur tampilan grafik sesuai kebutuhan</li>
                    <li>Tekan <b>▶ Jalankan Analisis</b></li>
                </ul>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="info-card">
                <div class="card-label">Ambang Batas Siaga (Manggarai)</div>
                <ul>
                    <li><b style="color:{T['green']};">🟢 Normal</b>  — TMA &lt; 750 cm</li>
                    <li><b style="color:{T['yellow']};">🟡 Waspada</b> — 750–849 cm</li>
                    <li><b style="color:{T['orange']};">🟠 Kritis</b>  — 850–949 cm</li>
                    <li><b style="color:{T['red']};">🔴 Bahaya</b>  — ≥ 950 cm</li>
                </ul>
            </div>""", unsafe_allow_html=True)
        empty_state()
    else:
        L, R, last, sg = res["lstm"], res["rf"], res["last_cm"], res["siaga"]
        primary = L if L is not None else R

        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💧 TMA Saat Ini", f"{last:.1f} cm", help="Nilai jam terakhir dari input")
        c2.metric("🤖 LSTM (t+6)", f"{L:.2f} cm" if L else "N/A",
                  delta=f"{L-last:+.2f} cm" if L else None,
                  delta_color="inverse" if (L and L > last) else "normal")
        c3.metric("🌲 RF (t+6)", f"{R:.2f} cm" if R else "N/A",
                  delta=f"{R-last:+.2f} cm" if R else None,
                  delta_color="inverse" if (R and R > last) else "normal")
        c4.metric(f"{sg['icon']} Status", f"Siaga {sg['lvl']}",
                  delta=sg["label"], delta_color="off")

        # Grafik
        st.markdown(f"<div class='sec-label'>Visualisasi Tren & Proyeksi 6 Jam</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(
            build_chart(res["arr"], L, R, show_lstm, show_rf, show_band),
            use_container_width=True, config={"displayModeBar": False}
        )

        # Status + Eval
        ca, cb = st.columns([1, 1])
        with ca:
            st.markdown(f"<div class='sec-label'>Status & Instruksi Petugas</div>",
                        unsafe_allow_html=True)
            ews_card(primary)
        with cb:
            st.markdown(f"<div class='sec-label'>Performa Model</div>",
                        unsafe_allow_html=True)
            eval_tbl()

        # Kesimpulan komparatif
        st.markdown("<hr/>", unsafe_allow_html=True)
        better = "LSTM" if EVAL["LSTM"]["RMSE"] < EVAL["RF"]["RMSE"] else "RF"
        diff_txt = (f"Selisih prediksi: <b style='font-family:IBM Plex Mono;color:{T['accent']};'>"
                    f"{abs(L-R):.2f} cm</b>. ") if (L and R) else ""
        delta_pct = (abs(L-R)/((L+R)/2)*100) if (L and R) else 0
        st.markdown(f"""
        <div class="info-card" style="border-left:3px solid {T['accent']};">
            <div class="card-label">📌 Kesimpulan Komparatif</div>
            <p style="color:{T['subtext']};font-size:.86rem;margin:0;line-height:1.65;">
                {diff_txt}{"Deviasi antar-model: <b style='font-family:IBM Plex Mono;'>"+f"{delta_pct:.1f}%</b>. " if (L and R) else ""}
                Model <b style="color:{T['green']};">{better}</b> lebih direkomendasikan
                (RMSE {EVAL[better]['RMSE']:.4f} cm · R² {EVAL[better]['R2']:.4f}).
                Proyeksi 6 jam ke depan: <b style="color:{sg['color']};">Siaga {sg['lvl']} — {sg['label']}</b>.
            </p>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"<div class='sec-label'>Export Hasil</div>", unsafe_allow_html=True)
        export_btns(res)

# ════════════════════════════════════════════
# 14. HALAMAN: LSTM
# ════════════════════════════════════════════
elif page == "🤖 LSTM":
    st.markdown(f"<p style='color:{T['muted']};font-size:.83rem;margin:-.4rem 0 .75rem;'>Long Short-Term Memory · Deep Learning</p>", unsafe_allow_html=True)
    m = EVAL["LSTM"]
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE (cm)", f"{m['RMSE']:.4f}", help="Root Mean Squared Error — semakin kecil semakin akurat")
    c2.metric("MAE (cm)",  f"{m['MAE']:.4f}",  help="Mean Absolute Error — rata-rata kesalahan absolut")
    c3.metric("R² Score",  f"{m['R2']:.4f}",   help="Koefisien determinasi — mendekati 1.0 = sangat baik")
    st.markdown(f"""
    <div class="info-card">
        <div class="card-label">Arsitektur & Karakteristik</div>
        <ul>
            <li>Input: sliding window <b>24 jam</b> TMA historis → output prediksi <b>t+6 jam</b></li>
            <li>Unggul menangkap pola sekuensial, tren, dan dependensi temporal jangka panjang</li>
            <li>R² 0.7932 → model menjelaskan <b>79.32%</b> variansi data aktual</li>
            <li style="color:{T['green']};font-weight:600;">✔ Model terbaik — RMSE & MAE lebih rendah dari RF</li>
        </ul>
    </div>""", unsafe_allow_html=True)

    if not res:
        empty_state()
    elif not res["lstm"]:
        st.error("Model LSTM tidak tersedia. Pastikan file model ada di folder `models/`.", icon="❌")
        if MDL.get("lstm_err"):
            st.code(MDL["lstm_err"], language="text")
    else:
        L, last = res["lstm"], res["last_cm"]
        sg_l = get_siaga(L)
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediksi LSTM (t+6)", f"{L:.2f} cm")
        c2.metric("Delta vs Sekarang",   f"{L-last:+.2f} cm",
                  delta_color="inverse" if L > last else "normal")
        c3.metric("Status",              f"Siaga {sg_l['lvl']} — {sg_l['label']}")
        st.markdown(f"<div class='sec-label'>Visualisasi Tren LSTM</div>", unsafe_allow_html=True)
        st.plotly_chart(
            build_chart(res["arr"], L, None, True, False, show_band),
            use_container_width=True, config={"displayModeBar": False}
        )
        ews_card(L)
        st.markdown("<hr/>", unsafe_allow_html=True)
        export_btns(res)

# ════════════════════════════════════════════
# 15. HALAMAN: RANDOM FOREST
# ════════════════════════════════════════════
elif page == "🌲 Random Forest":
    st.markdown(f"<p style='color:{T['muted']};font-size:.83rem;margin:-.4rem 0 .75rem;'>Ensemble Decision Trees · Machine Learning</p>", unsafe_allow_html=True)
    m = EVAL["RF"]
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE (cm)", f"{m['RMSE']:.4f}")
    c2.metric("MAE (cm)",  f"{m['MAE']:.4f}")
    c3.metric("R² Score",  f"{m['R2']:.4f}")
    st.markdown(f"""
    <div class="info-card">
        <div class="card-label">Arsitektur & Karakteristik</div>
        <ul>
            <li>Input: flat vector <b>1×24</b> TMA historis → output prediksi <b>t+6 jam</b></li>
            <li>Robust terhadap outlier & noise data lapangan, tidak perlu scaling ketat</li>
            <li>R² 0.7735 → model menjelaskan <b>77.35%</b> variansi data aktual</li>
            <li style="color:{T['muted']};font-weight:600;">Runner-up — RMSE & MAE sedikit lebih tinggi dari LSTM</li>
        </ul>
    </div>""", unsafe_allow_html=True)

    if not res:
        empty_state()
    elif not res["rf"]:
        st.error("Model RF tidak tersedia. Pastikan file model ada di folder `models/`.", icon="❌")
        if MDL.get("rf_err"):
            st.code(MDL["rf_err"], language="text")
    else:
        R, last = res["rf"], res["last_cm"]
        sg_r = get_siaga(R)
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediksi RF (t+6)", f"{R:.2f} cm")
        c2.metric("Delta vs Sekarang", f"{R-last:+.2f} cm",
                  delta_color="inverse" if R > last else "normal")
        c3.metric("Status",            f"Siaga {sg_r['lvl']} — {sg_r['label']}")
        st.markdown(f"<div class='sec-label'>Visualisasi Tren RF</div>", unsafe_allow_html=True)
        st.plotly_chart(
            build_chart(res["arr"], None, R, False, True, show_band),
            use_container_width=True, config={"displayModeBar": False}
        )
        ews_card(R)
        st.markdown("<hr/>", unsafe_allow_html=True)
        export_btns(res)

# ════════════════════════════════════════════
# 16. HALAMAN: RIWAYAT
# ════════════════════════════════════════════
elif page == "📜 Riwayat":
    st.markdown(f"<p style='color:{T['muted']};font-size:.83rem;margin:-.4rem 0 .75rem;'>Sesi aktif · maksimal 30 entri terakhir</p>", unsafe_allow_html=True)
    if not st.session_state.history:
        empty_state("📜", "Belum Ada Riwayat",
                    "Jalankan minimal satu analisis terlebih dahulu.")
    else:
        df_h = pd.DataFrame(st.session_state.history[::-1])
        # Ringkasan cepat
        total = len(st.session_state.history)
        statuses = [h["Status"] for h in st.session_state.history]
        bahaya_cnt = sum(1 for s in statuses if "BAHAYA" in s)
        kritis_cnt = sum(1 for s in statuses if "KRITIS" in s)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Analisis", total)
        c2.metric("🔴 Bahaya", bahaya_cnt)
        c3.metric("🟠 Kritis", kritis_cnt)
        st.markdown(f"<div class='sec-label'>Log Analisis</div>", unsafe_allow_html=True)
        st.dataframe(df_h, use_container_width=True, hide_index=True)
        col_exp, col_del = st.columns([3,1])
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df_h.to_excel(w, index=False, sheet_name="Riwayat")
        col_exp.download_button(
            "⬇ Export Riwayat (.xlsx)", data=buf.getvalue(),
            file_name=f"sipma_riwayat_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        if col_del.button("🗑 Hapus Semua", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ════════════════════════════════════════════
# 17. HALAMAN: LAPORAN
# ════════════════════════════════════════════
elif page == "📋 Laporan":
    st.markdown(f"<p style='color:{T['muted']};font-size:.83rem;margin:-.4rem 0 .75rem;'>Ringkasan hasil analisis siap dilaporkan</p>", unsafe_allow_html=True)
    if not res:
        empty_state("📋", "Belum Ada Data", "Jalankan analisis terlebih dahulu.")
    else:
        L, R, last, sg = res["lstm"], res["rf"], res["last_cm"], res["siaga"]
        better  = "LSTM" if EVAL["LSTM"]["RMSE"] < EVAL["RF"]["RMSE"] else "RF"
        primary = L if (better == "LSTM" and L) else R

        rows = [
            ("Lokasi",              "Pintu Air Manggarai, Jakarta"),
            ("Waktu Analisis",      res["ts"]),
            ("TMA Terakhir",        f"{last:.1f} cm"),
            ("Prediksi LSTM (t+6)", f"{L:.2f} cm" if L else "N/A"),
            ("Prediksi RF (t+6)",   f"{R:.2f} cm" if R else "N/A"),
            ("Selisih Prediksi",    f"{abs(L-R):.2f} cm" if (L and R) else "N/A"),
            ("Model Rekomendasi",   f"{better} — RMSE {EVAL[better]['RMSE']:.4f} cm"),
        ]
        rows_html = "".join([
            f'<tr><td>{k}</td><td class="rpt-mono">{v}</td></tr>'
            for k, v in rows
        ])
        st.markdown(f"""
        <div class="info-card">
            <div class="card-label">📄 Laporan Prediksi Muka Air — {res['ts']}</div>
            <table class="rpt-tbl">{rows_html}
            <tr>
                <td>Status Siaga</td>
                <td style="color:{sg['color']};font-weight:700;">
                    {sg['icon']} Siaga {sg['lvl']} — {sg['label']}
                </td>
            </tr>
            </table>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"<div class='sec-label'>Instruksi Petugas</div>", unsafe_allow_html=True)
        ews_card(primary if primary else last)

        # Mini stats
        if L and R:
            st.markdown(f"<div class='sec-label'>Ringkasan Statistik Input</div>",
                        unsafe_allow_html=True)
            arr_flat = res["arr"].flatten()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Min TMA", f"{arr_flat.min():.1f} cm")
            c2.metric("Max TMA", f"{arr_flat.max():.1f} cm")
            c3.metric("Rata-rata", f"{arr_flat.mean():.1f} cm")
            c4.metric("Std Dev", f"{arr_flat.std():.1f} cm")

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown(f"<div class='sec-label'>Export Laporan</div>", unsafe_allow_html=True)
        export_btns(res)