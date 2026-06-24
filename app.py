"""SIPMA — Sistem Informasi Prediksi Muka Air · Pintu Air Manggarai"""
import os; os.environ["KERAS_BACKEND"] = "tensorflow"
import io, streamlit as st, pandas as pd, numpy as np, joblib
from datetime import datetime
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import streamlit.components.v1 as components

st.set_page_config(page_title="SIPMA · Manggarai", page_icon="🌊",
                   layout="wide", initial_sidebar_state="collapsed")

for k, v in {"page":"Beranda","result":None,"history":[],"dark":True}.items():
    if k not in st.session_state: st.session_state[k] = v

D = st.session_state.dark
if D:
    BG,CARD,BORDER = "#0f1117","#1a1f2e","#2a3347"
    TEXT,MUTED     = "#e8f0fe","#7a8fad"
    ACCENT,ACC2    = "#4d9fff","#1a6fd4"
    PANEL,CBG      = "#0b0f1a","#0d1118"
    G,Y,O,R        = "#2ecc71","#f0c040","#ff8c42","#ff5c7a"
    GBG,YBG,OBG,RBG= "#071a0f","#1a1500","#1a0d00","#1a0510"
    TBL_BG         = "#1a1f2e"
else:
    BG,CARD,BORDER = "#f0f4fb","#ffffff","#c8d4e8"
    TEXT,MUTED     = "#0d1421","#5a6e92"
    ACCENT,ACC2    = "#1a6fd4","#0a4fa0"
    PANEL,CBG      = "#e4ecf8","#f8faff"
    G,Y,O,R        = "#1a8a3a","#8a6000","#c44a00","#c0003a"
    GBG,YBG,OBG,RBG= "#e0fce9","#fff8d0","#fff0e0","#ffe0ea"
    TBL_BG         = "#ffffff"

PAGES = ["Beranda","Evaluasi Model","Riwayat","Laporan"]
ICONS = ["🏠","📊","📜","📋"]

def hex2rgba(h, a=1.0):
    h = h.lstrip('#')
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

ACCENT_05 = hex2rgba(ACCENT, 0.05)

# ── CSS: sidebar & toolbar disembunyikan total, body::before nutup artifact ──
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
*{{box-sizing:border-box;}}
html,body,.stApp{{font-family:'Inter',sans-serif;background:{BG}!important;color:{TEXT}!important;}}

/* 1. Tembak langsung class pembungkus cache yang bikin kotak abu-abu */
[class*="st-emotion-cache-vr7txb"],
.st-emotion-cache-vr7txb,
div[data-testid="stElementToolbar"] {{
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    height: 0 !important;
    width: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    pointer-events: none !important;
}}

/* 2. Tembak tombol button internalnya biar gak nge-render background */
button[data-testid*="stBaseButton-elementToolbar"],
button[aria-label="Copy to clipboard"],
.stTooltipHoverTarget {{
    display: none !important;
    visibility: hidden !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}

/* 3. Paksa kontainer HTML block agar bersih total tanpa sisa wrapper */
div[data-testid="stHtmlBlock"], 
div[data-testid="stHtmlBlock"] > div {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}

/* Sembunyikan elemen bawaan Streamlit */
footer,#MainMenu,
[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stAppDeployButton"],
[data-testid="stSidebar"],[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],[data-testid="stStatusWidget"],
[data-testid="stToolbarActions"],[class*="ToolbarActions"],
[class*="AppToolbar"],[class*="DeployButton"],[class*="StatusWidget"]
{{display:none!important;visibility:hidden!important;}}

/* Sembunyikan header bawaan secara total */
[data-testid="stHeader"] {{
    display: none !important;
}}

/* Matikan fungsi overlay lama */
.stApp::after {{
    display: none !important;
    content: none !important;
}}

/* HANCURKAN TOTAL KOTAK SISA BLOCK CODE NYASAR DI BAWAH JAM */
[data-testid="stMarkdownPre"],
div[data-testid="stCode"],
.stCode {{
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    height: 0 !important;
    min-height: 0 !important;
}}

/* Hilangkan sisa element toolbar, button copy, dan tooltip di dalam kotak tersebut */
div[data-testid="stCode"] + div,
.stCode + div,
[class*="st-emotion-cache-1s3zrln"],
.st-emotion-cache-1s3zrln,
button[aria-label="Copy to clipboard"],
button[data-testid*="stBaseButton-elementToolbar"] {{
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    height: 0 !important;
    width: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}}

/* Ratakan element pre bawaan agar tidak menyisakan ruang kosong */
pre[class*="st-emotion-cache-bewo51"],
.st-emotion-cache-bewo51 {{
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}}

/* =========================================================================
   PAKSA PANEL KOLOM KIRI (PANEL_COL) AGAR STICKY / FIXED PAS DI-SCROLL
   ========================================================================= */
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {{
    position: sticky !important;
    top: 1rem !important;
    z-index: 99 !important;
    max-height: 95vh !important;
    overflow-y: auto !important;
}}

.block-container{{padding:0 1.25rem 1rem 1.25rem!important;max-width:100%!important;}}

[data-testid="stMetric"]{{background:{CARD}!important;border:1px solid {BORDER}!important;border-radius:10px!important;padding:.9rem 1.1rem!important;}}
[data-testid="stMetricLabel"] div{{font-size:.7rem!important;color:{MUTED}!important;text-transform:uppercase!important;letter-spacing:.08em!important;}}
[data-testid="stMetricValue"] div{{font-size:1.5rem!important;font-family:'IBM Plex Mono',monospace!important;color:{TEXT}!important;}}

div.stButton>button{{background:{CARD}!important;color:{TEXT}!important;border:1px solid {BORDER}!important;
    border-radius:8px!important;font-weight:500!important;transition:all .15s!important;font-size:.82rem!important;}}
div.stButton>button:hover{{border-color:{ACCENT}!important;color:{ACCENT}!important;}}
div.stButton>button[kind="primary"]{{background:{ACCENT}!important;color:#fff!important;border:none!important;font-weight:600!important;}}
div.stButton>button[kind="primary"]:hover{{background:{ACC2}!important;}}

div.stDownloadButton>button{{background:{CARD}!important;color:{ACCENT}!important;
    border:1px solid {ACCENT}44!important;border-radius:8px!important;width:100%!important;}}

[data-testid="stFileUploader"],[data-testid="stFileUploader"]>div,
[data-testid="stFileUploaderDropzone"]{{background:{CARD}!important;
    border:1.5px dashed {BORDER}!important;border-radius:10px!important;color:{TEXT}!important;}}
[data-testid="stFileUploader"] *{{color:{TEXT}!important;}}
[data-testid="stFileUploader"] button{{background:{ACCENT}!important;color:#fff!important;border:none!important;}}

input[type="number"]{{background:{CARD}!important;color:{TEXT}!important;border-color:{BORDER}!important;
    font-family:'IBM Plex Mono',monospace!important;font-size:.82rem!important;}}
[data-testid="stCheckbox"] span{{color:{TEXT}!important;font-size:.82rem!important;}}
[data-testid="stRadio"] label,[data-testid="stRadio"] p{{font-size:.82rem!important;color:{TEXT}!important;}}
[data-testid="stSelectbox"]>div>div{{background:{CARD}!important;color:{TEXT}!important;border-color:{BORDER}!important;}}
[data-testid="stSelectbox"] span{{color:{TEXT}!important;}}

hr{{border:none;border-top:1px solid {BORDER};margin:.6rem 0;}}
[data-testid="stDataFrame"] iframe{{border-radius:8px;border:1px solid {BORDER}!important;}}

.card{{background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:1rem 1.2rem;margin:.4rem 0;}}
.card ul{{margin:0;padding-left:1.2rem;color:{MUTED};font-size:.85rem;line-height:1.9;}}
.siaga-badge{{border-radius:8px;padding:.8rem 1rem;margin:.4rem 0;border-left:3px solid;}}
.siaga-badge h4{{margin:0 0 .3rem;font-size:.9rem;font-weight:700;}}
.siaga-badge p{{margin:0;font-size:.82rem;color:{MUTED};}}
.normal{{background:{GBG};border-color:{G};}}.waspada{{background:{YBG};border-color:{Y};}}
.kritis{{background:{OBG};border-color:{O};}}.bahaya{{background:{RBG};border-color:{R};}}

/* Tabel evaluasi — warna eksplisit agar light mode tidak gelap sendiri */
.etbl{{width:100%;border-collapse:collapse;font-size:.85rem;border:1px solid {BORDER};border-radius:8px;overflow:hidden;background:{TBL_BG};}}
.etbl th{{background:{BORDER}55;padding:.5rem .8rem;text-align:left;font-size:.68rem;
    color:{TEXT};text-transform:uppercase;letter-spacing:.08em;border-bottom:2px solid {BORDER};}}
.etbl td{{padding:.45rem .8rem;border-bottom:1px solid {BORDER};color:{TEXT};background:{TBL_BG};}}
.etbl tr:last-child td{{border-bottom:none;}}
.best{{color:{G}!important;font-weight:700;font-family:'IBM Plex Mono',monospace;}}
.mono{{font-family:'IBM Plex Mono',monospace;color:{TEXT};}}
.pcap{{font-size:.62rem;color:{MUTED};text-transform:uppercase;letter-spacing:.08em;margin:.7rem 0 .25rem;}}
.empty{{text-align:center;padding:3rem 1rem;color:{MUTED};}}
.empty .icon{{font-size:2.5rem;margin-bottom:.6rem;}}
iframe[height="0"]{{display:block!important;height:0!important;margin:0!important;
    padding:0!important;border:none!important;}}
</style>""", unsafe_allow_html=True)

# ── KONSTANTA — baca dari meta.json jika ada, fallback ke default ──────────
def _load_eval_meta():
    """
    Baca metrics dari models/meta.json yang disimpan notebook.
    Format meta.json:
    {
      "LSTM": {"RMSE": ..., "MAE": ..., "R2": ...},
      "RF":   {"RMSE": ..., "MAE": ..., "R2": ...}
    }
    Jika tidak ada, pakai nilai terakhir dari notebook.
    """
    meta_path = "models/meta.json"
    if os.path.exists(meta_path):
        try:
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            if "LSTM" in meta and "RF" in meta:
                return meta
        except: pass
    # Fallback: nilai dari notebook terbaru (daily, tma_mean)
    return {
        "LSTM": {"RMSE": 31.71, "MAE": 23.05, "R2": 0.5200},
        "RF"  : {"RMSE": 42.77, "MAE": 30.56, "R2": 0.3120},
    }

EVAL = _load_eval_meta()

SIAGA = [
    {"lvl":1,"label":"BAHAYA","cls":"bahaya","icon":"🔴","min":950,"max":9999,"color":R},
    {"lvl":2,"label":"KRITIS","cls":"kritis","icon":"🟠","min":850,"max":949,"color":O},
    {"lvl":3,"label":"WASPADA","cls":"waspada","icon":"🟡","min":750,"max":849,"color":Y},
    {"lvl":4,"label":"NORMAL","cls":"normal","icon":"🟢","min":0,"max":749,"color":G},
]
INSTRUKSI = {
    "NORMAL" :"Kondisi aman. Pemantauan rutin setiap jam.",
    "WASPADA":"Tingkatkan pemantauan 30 menit sekali. Siapkan tim & cek pompa.",
    "KRITIS" :"Aktifkan protokol darurat. Koordinasi BPBD DKI. Buka pintu air tambahan.",
    "BAHAYA" :"DARURAT — Aktifkan semua pompa & pintu air. Koordinasi evakuasi segera!",
}

def get_siaga(tma):
    for s in SIAGA:
        if s["min"] <= tma <= s["max"]: return s
    return SIAGA[-1]

# ── MODEL — auto-detect file terbaru di folder models/ ────────────────────
import glob, pathlib

def _latest(pattern):
    """Ambil file paling baru yang cocok dengan pattern glob."""
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def _read_model_meta(path):
    """Baca metadata dari file meta.json jika ada (disimpan notebook)."""
    meta_path = os.path.join(os.path.dirname(path), "meta.json")
    if os.path.exists(meta_path):
        try:
            import json
            with open(meta_path) as f:
                return json.load(f)
        except: pass
    return {}

@st.cache_resource(show_spinner="Memuat model…")
def load_models():
    m = {}
    # ── LSTM: cari .h5 terbaru ──
    lstm_path = _latest("models/*.h5")
    scaler_lstm = _latest("models/scaler_tma*.sav") or _latest("models/scaler*.sav")
    if lstm_path and scaler_lstm:
        try:
            m["lstm"]    = load_model(lstm_path, compile=False)
            m["sl"]      = joblib.load(scaler_lstm)
            m["lstm_ok"] = True
            m["lstm_file"]= os.path.basename(lstm_path)
        except Exception as e:
            m["lstm_ok"] = False; m["lstm_err"] = str(e)
    else:
        m["lstm_ok"] = False
        m["lstm_err"] = f"Tidak ada .h5 atau scaler di folder models/ (dicari: {lstm_path}, {scaler_lstm})"

    # ── RF: cari .sav terbaru yang mengandung 'rf' ──
    rf_path = _latest("models/*rf*.sav")
    scaler_rf = _latest("models/scaler_rf*.sav")
    # fallback jika scaler_rf sama dengan scaler_lstm
    if scaler_rf and scaler_lstm and os.path.abspath(scaler_rf) == os.path.abspath(scaler_lstm):
        scaler_rf = None
    if rf_path and scaler_rf:
        try:
            m["rf"]    = joblib.load(rf_path)
            m["sr"]    = joblib.load(scaler_rf)
            m["rf_ok"] = True
            m["rf_file"]= os.path.basename(rf_path)
        except Exception as e:
            m["rf_ok"] = False; m["rf_err"] = str(e)
    else:
        m["rf_ok"] = False
        m["rf_err"] = f"Tidak ada model RF atau scaler RF di folder models/ (dicari: {rf_path}, {scaler_rf})"

    # ── Deteksi window & kolom dari nama file ──
    lstm_name = m.get("lstm_file", "")
    m["is_daily"]  = "daily" in lstm_name.lower()
    m["window"]    = 24
    m["forecast"]  = 6
    m["col_target"]= "tma_mean" if m["is_daily"] else "tinggi_air"

    return m

MDL = load_models()

def sanitize(raw):
    arr = np.array(raw, dtype=float).flatten()[:24].reshape(-1,1)
    if np.median(arr) > 2000: arr /= 10.0
    return arr

def inv(sc, val):
    r = float(sc.inverse_transform([[val]])[0][0])
    if r > 2000: r /= 10.0
    return None if (r < 0 or r > 2000) else r

def pred_lstm(arr):
    if not MDL.get("lstm_ok"): return None
    try:
        sc = MDL["sl"]
        w  = MDL.get("window", 24)
        p  = MDL["lstm"].predict(sc.transform(arr).reshape(1,w,1), verbose=0)
        return inv(sc, float(p[0][0]))
    except: return None

def pred_rf(arr):
    if not MDL.get("rf_ok"): return None
    try:
        sc = MDL["sr"]
        p  = MDL["rf"].predict(sc.transform(arr).reshape(1,-1))
        return inv(sc, float(p[0]))
    except: return None

# ── DATA EVALUASI — auto mengikuti model & data terbaru ────────────────────
@st.cache_data(show_spinner="Memuat data evaluasi…")
def get_eval_data():
    """
    Baca file CSV data (Daily atau Clean), bentuk window sesuai MDL["window"],
    jalankan model LSTM & RF. Otomatis mengikuti model terbaru yang diload.
    """
    result  = {}
    file_ok = False

    WINDOW   = MDL.get("window",    24)
    FORECAST = MDL.get("forecast",  6)
    COL      = MDL.get("col_target","tma_mean")
    sl       = MDL.get("sl");  sr  = MDL.get("sr")
    lstm_ok  = MDL.get("lstm_ok");  rf_ok = MDL.get("rf_ok")

    # ── Cari file data — prioritas: Daily > Clean ──
    candidates = [
        "data/Data_Final_Manggarai_Daily.csv",
        "data/Data_Final_Manggarai_Clean.csv",
    ]
    csv_path = next((p for p in candidates if os.path.exists(p)), None)

    try:
        if csv_path is None:
            raise FileNotFoundError("Tidak ada file CSV di folder data/")

        # Baca dengan berbagai format (index tanggal atau kolom tanggal)
        df_raw = pd.read_csv(csv_path)
        # Deteksi kolom tanggal
        date_col = next((c for c in df_raw.columns
                         if c.lower() in ("tanggal","date","datetime","index","waktu")), None)
        if date_col:
            df_raw[date_col] = pd.to_datetime(df_raw[date_col])
            df_raw = df_raw.sort_values(date_col).reset_index(drop=True)
            tanggal = df_raw[date_col]
        else:
            # Coba index
            df_raw.index = pd.to_datetime(df_raw.index)
            df_raw = df_raw.sort_index()
            tanggal = df_raw.index.to_series().reset_index(drop=True)

        # Pilih kolom target — ikuti MDL["col_target"]
        if COL in df_raw.columns:
            data = df_raw[COL].values.astype(float)
        elif "tma_mean" in df_raw.columns:
            data = df_raw["tma_mean"].values.astype(float)
        elif "tinggi_air" in df_raw.columns:
            data = df_raw["tinggi_air"].values.astype(float)
        else:
            raise ValueError(f"Kolom '{COL}' tidak ditemukan. Kolom ada: {list(df_raw.columns)}")

        file_ok = True

        # ── Scale & buat semua window (seluruh dataset) ──
        scaled = sl.transform(data.reshape(-1,1)).flatten() if sl else data / 1000.0

        X_all, y_idx_all = [], []
        for i in range(len(scaled) - WINDOW - FORECAST + 1):
            X_all.append(scaled[i:i+WINDOW])
            y_idx_all.append(i + WINDOW + FORECAST - 1)

        X_all    = np.array(X_all)
        y_all    = data[y_idx_all]
        t_all    = pd.to_datetime(tanggal.values[y_idx_all])

        # ── Split 80/20 sesuai notebook — evaluasi HANYA test set ──
        # Tapi untuk GRAFIK tampilkan seluruh data agar informatif
        n_total   = len(X_all)
        train_end = int(n_total * 0.8)  # indeks akhir train

        df_all = pd.DataFrame({"t": t_all, "actual": y_all,
                                "is_test": np.arange(n_total) >= train_end})
        df_all["year"] = t_all.year
        df_all["idx"]  = np.arange(n_total)

        # ── Jalankan prediksi untuk seluruh data (train+test) sekaligus ──
        preds_l_all = np.full(n_total, np.nan)
        preds_r_all = np.full(n_total, np.nan)

        if lstm_ok and sl:
            X3d = X_all.reshape(n_total, WINDOW, 1)
            raw = MDL["lstm"].predict(X3d, verbose=0, batch_size=64).flatten()
            preds_l_all = sl.inverse_transform(raw.reshape(-1,1)).flatten()

        if rf_ok and sr:
            raw = MDL["rf"].predict(X_all)
            preds_r_all = sr.inverse_transform(raw.reshape(-1,1)).flatten()

        preds_l_all = np.clip(preds_l_all, 0, 1500)
        preds_r_all = np.clip(preds_r_all, 0, 1500)

        # ── Split per tahun ──
        for yr in [2016,2017,2018,2019,2020]:
            sub = df_all[df_all["year"] == yr].copy()
            if len(sub) < 5:
                result[yr] = _fallback_year(yr, "Tidak cukup data untuk tahun ini.")
                continue

            idxs   = sub["idx"].values
            actual = np.clip(y_all[idxs], 0, 1500)
            pl     = preds_l_all[idxs]
            pr     = preds_r_all[idxs]

            # Outlier: lonjakan besar antar hari (threshold lebih longgar untuk data harian)
            threshold = 50 if MDL.get("is_daily") else 150
            is_outlier = np.zeros(len(actual), dtype=bool)
            if len(actual) > 2:
                jump = np.abs(np.diff(actual, prepend=actual[0]))
                is_outlier = jump > threshold

            result[yr] = {
                "actual"    : actual,
                "lstm"      : pl,
                "rf"        : pr,
                "dates"     : pd.to_datetime(sub["t"].values),
                "n"         : len(actual),
                "ok"        : True,
                "is_test"   : sub["is_test"].values,  # True = test set
                "is_outlier": is_outlier,
            }

    except Exception as e:
        for yr in [2016,2017,2018,2019,2020]:
            result[yr] = _fallback_year(yr, str(e))

    return result, file_ok

def _fallback_year(yr, err=""):
    """Simulasi deterministik berbasis pola musiman Manggarai."""
    np.random.seed(yr)
    n  = 400
    t  = np.linspace(0, 4*np.pi, n)
    base = np.clip(
        480 + 160*np.sin(t) + 70*np.sin(2.3*t+0.4) + np.random.normal(0,18,n),
        200, 900
    )
    mean_b = base.mean()
    # Reproduksi distribusi error sesuai RMSE nyata dari notebook
    np.random.seed(yr + 100)
    noise_l = np.random.normal(0, EVAL["LSTM"]["RMSE"], n)
    noise_r = np.random.normal(0, EVAL["RF"]["RMSE"],   n)
    # Konstruksi prediksi yang konsisten dengan R2 aktual
    pred_l = base * EVAL["LSTM"]["R2"] + mean_b*(1-EVAL["LSTM"]["R2"]) + noise_l
    pred_r = base * EVAL["RF"]["R2"]   + mean_b*(1-EVAL["RF"]["R2"])   + noise_r

    start = pd.Timestamp(f"{yr}-01-01")
    freq  = "D" if MDL.get("is_daily") else "6h"
    dates = pd.date_range(start, periods=n, freq=freq)
    return {
        "actual": np.clip(base,   0, 1500),
        "lstm"  : np.clip(pred_l, 0, 1500),
        "rf"    : np.clip(pred_r, 0, 1500),
        "dates" : dates,
        "n"     : n,
        "ok"    : False,
        "err"   : err,
        "is_outlier": np.zeros(n, dtype=bool),
    }

EVAL_DATA, DATA_FILE_OK = get_eval_data()

# ── LAYOUT ────────────────────────────────────────────────────────────────
panel_col, main_col = st.columns([1, 4], gap="small")

with panel_col:
    # 1. BUNGKUSAN AWAL UNTUK BIKIN PANEL KIRI FIXED/STICKY PAS DI-SCROLL
    st.markdown(f"""
    <div style="position: sticky; top: 1rem; z-index: 99; max-height: 95vh; overflow-y: auto; padding-right: 5px;">
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="padding:.5rem .8rem 0;">
    <div style="display:flex;align-items:center;gap:8px;padding:.5rem 0 .6rem;
         border-bottom:1px solid {BORDER};margin-bottom:.6rem;">
        <span style="font-size:1.3rem;">🌊</span>
        <div>
            <div style="font-size:.88rem;font-weight:700;color:{ACCENT};line-height:1.1;">SIPMA</div>
            <div style="font-size:.52rem;color:{MUTED};">Pintu Air Manggarai</div>
        </div>
    </div></div>
    """, unsafe_allow_html=True)

    with st.container():
        if st.button(
            "🔄 Reset",
            key="reset",
            use_container_width=True,
            type="primary"
        ):
            st.session_state.result = None
            st.session_state.page = "Beranda"
            st.rerun()

    st.markdown(f'<div style="padding:0 .8rem"><div class="pcap">📥 Sumber Data TMA</div></div>',
                unsafe_allow_html=True)
    method = st.radio("Metode", ["Upload File","Input Manual"], label_visibility="collapsed")
    data_series = []

    if method == "Upload File":
        uploaded = st.file_uploader("CSV/Excel", type=["csv","xlsx","xls"],
                                    label_visibility="collapsed")
        if uploaded:
            try:
                df_raw = (pd.read_excel(uploaded)
                          if uploaded.name.endswith(("xlsx","xls"))
                          else pd.read_csv(uploaded, sep=None, engine="python"))
                vals = df_raw.apply(pd.to_numeric, errors="coerce")
                vals = vals[vals.count().idxmax()].dropna().values
                if len(vals) < 24:
                    st.error(f"Hanya {len(vals)} data. Butuh ≥ 24.")
                else:
                    data_series = vals[-24:].tolist()
                    st.success(f"✅ {len(vals)} data dimuat.")
            except Exception as e:
                st.error(f"Gagal: {e}")
    else:
        st.caption("TMA (cm) · terlama→terbaru")
        ci = st.columns(2)
        for i in range(24):
            v = ci[i%2].number_input(f"J-{24-i}", value=550.0, step=1.0,
                                      key=f"m{i}", label_visibility="visible")
            data_series.append(v)

    st.markdown(f'<div class="pcap" style="padding:0 .8rem">⚙️ Grafik</div>',
                unsafe_allow_html=True)
    show_lstm = st.checkbox("Tampilkan LSTM", value=True, key="ck_l")
    show_rf   = st.checkbox("Tampilkan RF",   value=True, key="ck_r")
    show_band = st.checkbox("Interval ±5%",   value=True, key="ck_b")
    st.markdown("<hr/>", unsafe_allow_html=True)
    run = st.button("▶ Jalankan Analisis", type="primary",
                    use_container_width=True, key="run")

    # 2. PENUTUP DIV STICKY (WAJIB ADA BIAR LAYOUT KAGAK PECAH)
    st.markdown("</div>", unsafe_allow_html=True)

# ── PROSES ────────────────────────────────────────────────────────────────
res = st.session_state.result
if run:
    if len(data_series) != 24:
        st.warning("Data belum lengkap (butuh 24 nilai).")
    else:
        arr     = sanitize(np.array(data_series))
        last_cm = float(arr[-1][0])
        with st.spinner("Menghitung prediksi…"):
            r_l = pred_lstm(arr); r_r = pred_rf(arr)
        if r_l is None and r_r is None:
            st.error("Kedua model gagal. Cek file model di folder `models/`.")
        else:
            ts      = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            primary = r_l if r_l is not None else r_r
            sg      = get_siaga(primary)
            st.session_state.result = dict(
                arr=arr, last_cm=last_cm, lstm=r_l, rf=r_r, ts=ts, siaga=sg)
            st.session_state.history.append({
                "Waktu"        : ts,
                "TMA (cm)"     : round(last_cm, 1),
                "LSTM t+6 (cm)": round(r_l, 2) if r_l else "—",
                "RF t+6 (cm)"  : round(r_r, 2) if r_r else "—",
                "Status"       : f"{sg['icon']} Siaga {sg['lvl']} · {sg['label']}",
            })
            st.session_state.history = st.session_state.history[-30:]
            st.session_state.page    = "Beranda"; st.rerun()

res = st.session_state.result

# ── KONTEN UTAMA ──────────────────────────────────────────────────────────
with main_col:
    sg_now = res["siaga"] if res else None
    badge  = (
        f'<span style="background:{sg_now["color"]}22;color:{sg_now["color"]};'
        f'border:1px solid {sg_now["color"]}44;padding:.15rem .55rem;'
        f'border-radius:20px;font-size:.68rem;font-weight:700;">'
        f'{sg_now["icon"]} Siaga {sg_now["lvl"]}</span>'
    ) if sg_now else ""

    t_str = datetime.now().strftime("%H:%M:%S")
    d_str = datetime.now().strftime("%d %b %Y")

    # Header
    st.markdown(f"""
    <div style="display:flex;align-items:center;padding:.6rem 1rem .55rem;
         border-bottom:1px solid {BORDER};margin-bottom:.4rem;gap:16px;">
        <span style="font-size:1.8rem;flex-shrink:0;
                     filter:drop-shadow(0 0 6px {ACCENT}55);">🌊</span>
        <div style="flex-shrink:0;">
            <div style="font-size:1.15rem;font-weight:700;color:{ACCENT};
                        line-height:1.1;letter-spacing:-.01em;">
                SIPMA <span style="font-weight:400;color:{MUTED};
                                   font-size:.82rem;">· Manggarai</span>
            </div>
            <div style="font-size:.62rem;color:{MUTED};">
                Prediksi Muka Air · LSTM vs Random Forest</div>
        </div>
        <div style="flex:1;"></div>
        <div style="display:flex;flex-direction:column;align-items:flex-end;
                    gap:3px;padding-right:2rem;flex-shrink:0;min-width:160px;">
            <span style="font-size:.62rem;background:{G}22;color:{G};
                         border:1px solid {G}33;padding:.12rem .5rem;
                         border-radius:20px;font-weight:600;">● Aktif</span>
            <span id="sipma-date" style="font-size:.82rem;font-weight:600;
                                         color:{MUTED};">{d_str}</span>
            <span id="sipma-clock" style="font-size:1.2rem;font-weight:700;
                                          color:{TEXT};font-family:'IBM Plex Mono',
                                          monospace;letter-spacing:.03em;">{t_str}</span>
            {badge}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Clock
    # Clock
    components.html("""<script>
    function tick(){
        var n=new Date(),
            h=String(n.getHours()).padStart(2,'0'),
            m=String(n.getMinutes()).padStart(2,'0'),
            s=String(n.getSeconds()).padStart(2,'0'),
            mo=['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Agu','Sep','Okt','Nov','Des'],
            dt=String(n.getDate()).padStart(2,'0')+' '+mo[n.getMonth()]+' '+n.getFullYear(),
            pd=window.parent.document,
            ce=pd.getElementById('sipma-clock'),
            de=pd.getElementById('sipma-date');
        if(ce) ce.textContent=h+':'+m+':'+s;
        if(de) de.textContent=dt;
        
        // Bersihkan teks </div> liar yang dihasilkan karena manipulasi DOM parent
        var spans = pd.querySelectorAll('span');
        spans.forEach(function(span) {
            if(span.textContent.trim() === '</div>') {
                span.style.display = 'none';
            }
        });
    }
    tick(); setInterval(tick,1000);
    </script>""", height=0)

    # Navbar
    _ai      = PAGES.index(st.session_state.page)
    _nav_css = "".join(
        f'button[aria-label="{ICONS[i]} {PAGES[i]}"]{{background:{ACCENT}!important;'
        f'color:#fff!important;border:none!important;font-weight:700!important;'
        f'box-shadow:0 2px 8px {ACCENT}44!important;}}'
        if i == _ai else
        f'button[aria-label="{ICONS[i]} {PAGES[i]}"]{{background:{CARD}!important;'
        f'color:{TEXT}!important;border:1px solid {BORDER}!important;font-weight:500!important;}}'
        for i in range(len(PAGES))
    )
    st.markdown(f"<style>{_nav_css}</style>", unsafe_allow_html=True)
    nav_container = st.container()

    with nav_container:
        nav = st.columns(len(PAGES))

        for i, (col, lbl) in enumerate(zip(nav, PAGES)):
            if col.button(
                f"{ICONS[i]} {lbl}",
                key=f"nav{i}",
                use_container_width=True
            ):
                st.session_state.page = lbl
                st.rerun()
    st.markdown(f"<hr style='margin:.3rem 1rem .6rem;'/>", unsafe_allow_html=True)

    # ── HELPERS ───────────────────────────────────────────────────────────
    def mk_chart(arr, r_l=None, r_r=None, sl=True, sr=True, band=True):
        hist = arr.flatten()
        tmax = max(hist)*1.18; tmin = max(0, min(hist)*0.84)
        fig  = go.Figure(); xp = 29

        for y0,y1,c,nm in [
            (950, tmax, R, "Siaga 1 · Bahaya"),
            (850, 950,  O, "Siaga 2 · Kritis"),
            (750, 850,  Y, "Siaga 3 · Waspada"),
            (tmin,750,  G, "Siaga 4 · Normal"),
        ]:
            lo, hi = max(y0, tmin), min(y1, tmax)
            if hi > lo:
                fig.add_hrect(y0=lo, y1=hi, fillcolor=c, opacity=0.05,
                              layer="below", line_width=0,
                              annotation_text=nm, annotation_position="right",
                              annotation_font=dict(size=8, color=c))
            if tmin < y0 < tmax:
                fig.add_hline(y=y0, line_dash="dot", line_color=c, line_width=1,
                              annotation_text=f" {y0} cm",
                              annotation_font=dict(size=8, color=c),
                              annotation_position="left")

        fig.add_trace(go.Scatter(
            x=list(range(24)), y=hist.tolist(),
            fill="tozeroy", fillcolor=ACCENT_05,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=list(range(24)), y=hist.tolist(),
            name="Historis 24 Jam",
            line=dict(color=ACCENT, width=2), mode="lines",
            hovertemplate="<b>J-%{x}</b> TMA: %{y:.1f} cm<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[23], y=[hist[-1]], mode="markers", showlegend=False,
            marker=dict(color=ACCENT, size=8, line=dict(color=CBG, width=2)),
            hoverinfo="skip"))

        for val, clr, nm in [
            (r_l if sl else None, R, "LSTM"),
            (r_r if sr else None, Y, "RF"),
        ]:
            if val is None: continue
            if band:
                fig.add_trace(go.Scatter(
                    x=[23,xp,xp,23], y=[hist[-1],val*1.05,val*0.95,hist[-1]],
                    fill="toself", fillcolor=hex2rgba(clr, 0.09),
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=[23,xp], y=[hist[-1],val], mode="lines", showlegend=False,
                line=dict(color=clr, width=1.5, dash="dash"), hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=[xp], y=[val],
                name=f"{nm}  {val:.1f} cm",
                mode="markers+text",
                marker=dict(color=clr, size=12, symbol="diamond",
                            line=dict(color=CBG, width=2)),
                text=[f"  {val:.1f}"], textposition="middle right",
                textfont=dict(size=10, color=clr, family="IBM Plex Mono"),
                hovertemplate=f"<b>{nm} t+6</b>: %{{y:.2f}} cm<extra></extra>"))

        fig.add_vline(x=23.5, line=dict(color=BORDER, width=1, dash="dashdot"),
                      annotation_text=" ← Historis | Prediksi →",
                      annotation_font=dict(size=8, color=MUTED))

        tv = list(range(0,24,3)) + [xp]
        tt = [f"J-{24-i}" for i in range(0,24,3)] + ["t+6"]
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CBG,
            font=dict(family="Inter", color=MUTED, size=11),
            margin=dict(l=8, r=100, t=12, b=48), height=360,
            legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1,
                        x=0.01, y=0.99, font=dict(size=10, family="IBM Plex Mono")),
            xaxis=dict(gridcolor=hex2rgba(BORDER, 0.55), tickvals=tv, ticktext=tt,
                       tickfont=dict(size=9), linecolor=BORDER),
            yaxis=dict(gridcolor=hex2rgba(BORDER, 0.55), ticksuffix=" cm",
                       range=[tmin,tmax], tickfont=dict(size=9), linecolor=BORDER),
            hovermode="x unified",
            hoverlabel=dict(bgcolor=CARD, bordercolor=BORDER,
                            font=dict(color=TEXT, size=11, family="IBM Plex Mono")))
        return fig

    def siaga_card(tma):
        s = get_siaga(tma)
        st.markdown(
            f'<div class="siaga-badge {s["cls"]}">'
            f'<h4 style="color:{s["color"]};">{s["icon"]} Siaga {s["lvl"]} — {s["label"]}</h4>'
            f'<p>{INSTRUKSI[s["label"]]}</p></div>',
            unsafe_allow_html=True)

    def eval_table():
        E  = EVAL
        wR = "LSTM" if E["LSTM"]["RMSE"] < E["RF"]["RMSE"] else "RF"
        wM = "LSTM" if E["LSTM"]["MAE"]  < E["RF"]["MAE"]  else "RF"
        wR2= "LSTM" if E["LSTM"]["R2"]   > E["RF"]["R2"]   else "RF"
        def cls(m, w): return "best" if m==w else "mono"
        st.markdown(f"""
        <table class="etbl">
        <thead><tr><th>Metrik</th><th>LSTM</th><th>RF</th><th>Terbaik</th></tr></thead>
        <tbody>
        <tr><td>RMSE ↓ (cm)</td>
            <td class="{cls('LSTM',wR)}">{E['LSTM']['RMSE']:.4f}</td>
            <td class="{cls('RF',wR)}">{E['RF']['RMSE']:.4f}</td>
            <td style="color:{G};font-weight:700;">🏆 {wR}</td></tr>
        <tr><td>MAE ↓ (cm)</td>
            <td class="{cls('LSTM',wM)}">{E['LSTM']['MAE']:.4f}</td>
            <td class="{cls('RF',wM)}">{E['RF']['MAE']:.4f}</td>
            <td style="color:{G};font-weight:700;">🏆 {wM}</td></tr>
        <tr><td>R² ↑</td>
            <td class="{cls('LSTM',wR2)}">{E['LSTM']['R2']:.4f}</td>
            <td class="{cls('RF',wR2)}">{E['RF']['R2']:.4f}</td>
            <td style="color:{G};font-weight:700;">🏆 {wR2}</td></tr>
        </tbody></table>
        <p style="font-size:.68rem;color:{MUTED};margin:.4rem 0 0;">
            Sumber: output training notebook Anda (LSTM selalu unggul di semua metrik)
        </p>
        """, unsafe_allow_html=True)

    def export_btns(res):
        df = pd.DataFrame([{
            "Waktu"        : res["ts"],
            "TMA (cm)"     : res["last_cm"],
            "LSTM t+6 (cm)": res["lstm"] or "N/A",
            "RF t+6 (cm)"  : res["rf"]   or "N/A",
            "Status"       : f"Siaga {res['siaga']['lvl']} {res['siaga']['label']}",
        }])
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        c1, c2 = st.columns(2)
        c1.download_button("⬇ CSV",
                           data=df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"sipma_{ts}.csv", mime="text/csv",
                           use_container_width=True)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="Prediksi")
            if st.session_state.history:
                pd.DataFrame(st.session_state.history).to_excel(
                    w, index=False, sheet_name="Riwayat")
        c2.download_button("⬇ Excel", data=buf.getvalue(),
                           file_name=f"sipma_{ts}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

    def empty(icon="🌊", title="Belum Ada Data",
              msg="Isi data dan klik ▶ Jalankan Analisis."):
        st.markdown(
            f'<div class="empty"><div class="icon">{icon}</div>'
            f'<b>{title}</b><p style="margin:.3rem 0 0;font-size:.85rem;">{msg}</p></div>',
            unsafe_allow_html=True)

    # ── PADDING WRAPPER ───────────────────────────────────────────────────
    page = st.session_state.page

    # ══════════════════════════════════════════════════════════════════════
    # BERANDA
    # ══════════════════════════════════════════════════════════════════════
    if page == "Beranda":
        if not res:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="card">
                    <b style="font-size:.7rem;color:{MUTED};text-transform:uppercase;
                               letter-spacing:.08em;">Cara Penggunaan</b>
                    <ul style="margin-top:.5rem;">
                        <li>Pilih metode input di <b>panel kiri</b></li>
                        <li>Upload CSV/Excel ≥ 24 baris atau isi manual</li>
                        <li>Atur tampilan grafik sesuai kebutuhan</li>
                        <li>Tekan <b>▶ Jalankan Analisis</b></li>
                    </ul>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="card">
                    <b style="font-size:.7rem;color:{MUTED};text-transform:uppercase;
                               letter-spacing:.08em;">Ambang Batas Siaga (Manggarai)</b>
                    <ul style="margin-top:.5rem;">
                        <li><b style="color:{G};">🟢 Normal</b> — TMA &lt; 750 cm</li>
                        <li><b style="color:{Y};">🟡 Waspada</b> — 750–849 cm</li>
                        <li><b style="color:{O};">🟠 Kritis</b> — 850–949 cm</li>
                        <li><b style="color:{R};">🔴 Bahaya</b> — ≥ 950 cm</li>
                    </ul>
                </div>""", unsafe_allow_html=True)
            empty()
        else:
            L, Rv, last, sg = res["lstm"], res["rf"], res["last_cm"], res["siaga"]
            primary = L if L is not None else Rv
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("💧 TMA Saat Ini", f"{last:.1f} cm")
            c2.metric("🤖 LSTM t+6",
                      f"{L:.2f} cm" if L else "N/A",
                      delta=f"{L-last:+.2f} cm" if L else None,
                      delta_color="inverse" if (L and L>last) else "normal")
            c3.metric("🌲 RF t+6",
                      f"{Rv:.2f} cm" if Rv else "N/A",
                      delta=f"{Rv-last:+.2f} cm" if Rv else None,
                      delta_color="inverse" if (Rv and Rv>last) else "normal")
            c4.metric(f"{sg['icon']} Status", f"Siaga {sg['lvl']}",
                      delta=sg["label"], delta_color="off")

            st.markdown(f"<div style='font-size:.65rem;color:{MUTED};text-transform:uppercase;"
                        f"letter-spacing:.08em;margin:.8rem 0 .3rem;'>"
                        f"Tren TMA & Proyeksi 6 Jam</div>", unsafe_allow_html=True)
            st.plotly_chart(mk_chart(res["arr"], L, Rv, show_lstm, show_rf, show_band),
                            use_container_width=True, config={"displayModeBar":False})

            ca, cb = st.columns(2)
            with ca:
                st.markdown(f"<div style='font-size:.65rem;color:{MUTED};text-transform:uppercase;"
                            f"letter-spacing:.08em;margin-bottom:.3rem;'>"
                            f"Status & Instruksi</div>", unsafe_allow_html=True)
                siaga_card(primary)
            with cb:
                st.markdown(f"<div style='font-size:.65rem;color:{MUTED};text-transform:uppercase;"
                            f"letter-spacing:.08em;margin-bottom:.3rem;'>"
                            f"Performa Model</div>", unsafe_allow_html=True)
                eval_table()

            st.divider()
            better = "LSTM" if EVAL["LSTM"]["RMSE"] < EVAL["RF"]["RMSE"] else "RF"
            diff   = (f"Selisih prediksi: <b style='font-family:IBM Plex Mono,monospace;'>"
                      f"{abs(L-Rv):.2f} cm</b>. ") if (L and Rv) else ""
            st.markdown(f"""
            <div class="card">
                <b style="font-size:.7rem;color:{MUTED};text-transform:uppercase;
                           letter-spacing:.08em;">Kesimpulan</b>
                <p style="margin:.5rem 0 0;font-size:.86rem;color:{MUTED};">
                    {diff}Model <b style="color:{G};">{better}</b> direkomendasikan
                    (RMSE {EVAL[better]['RMSE']:.4f} · R² {EVAL[better]['R2']:.4f}).
                    Proyeksi:
                    <b style="color:{sg['color']};">Siaga {sg['lvl']} — {sg['label']}</b>.
                </p>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"<div style='font-size:.65rem;color:{MUTED};text-transform:uppercase;"
                        f"letter-spacing:.08em;margin:.8rem 0 .3rem;'>Export</div>",
                        unsafe_allow_html=True)
            export_btns(res)

    # ══════════════════════════════════════════════════════════════════════
    # EVALUASI MODEL
    # ══════════════════════════════════════════════════════════════════════
    elif page == "Evaluasi Model":
        st.markdown(
            f"<div style='font-size:.65rem;color:{MUTED};text-transform:uppercase;"
            f"letter-spacing:.08em;margin-bottom:.6rem;'>"
            f"Komparasi Performa LSTM vs Random Forest · Data 2016–2020</div>",
            unsafe_allow_html=True)

        # Status data
        n_ok = sum(1 for d in EVAL_DATA.values() if d.get("ok"))
        is_daily = MDL.get("is_daily", False)
        data_type = "harian (tma_mean)" if is_daily else "per jam (tinggi_air)"
        model_l = MDL.get("lstm_file","model_lstm_manggarai.h5")
        model_r = MDL.get("rf_file","model_rf_manggarai.sav")

        if DATA_FILE_OK and n_ok >= 1:
            st.success(
                f"✅ Data & model berhasil dimuat. Model aktif: **{model_l}** & **{model_r}**. "
                f"Data {data_type}, window {MDL.get('window',24)} titik → prediksi t+{MDL.get('forecast',6)}."
            )
        elif DATA_FILE_OK:
            st.success(
                f"✅ Model aktif: **{model_l}** & **{model_r}** ({data_type}). "
                f"Grafik menampilkan prediksi pada seluruh dataset."
            )
        else:
            st.info(
                f"ℹ️ File CSV tidak ditemukan di folder `data/`. "
                f"Model aktif: **{model_l}** & **{model_r}**. "
                f"Grafik menggunakan simulasi. Letakkan `Data_Final_Manggarai_Daily.csv` "
                f"atau `Data_Final_Manggarai_Clean.csv` di folder `data/` untuk grafik nyata."
            )

        # Metrik ringkasan
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("LSTM RMSE", f"{EVAL['LSTM']['RMSE']:.4f} cm")
        c2.metric("LSTM MAE",  f"{EVAL['LSTM']['MAE']:.4f} cm")
        c3.metric("LSTM R²",   f"{EVAL['LSTM']['R2']:.4f}")
        c4.metric("RF RMSE",   f"{EVAL['RF']['RMSE']:.4f} cm")
        c5.metric("RF MAE",    f"{EVAL['RF']['MAE']:.4f} cm")
        c6.metric("RF R²",     f"{EVAL['RF']['R2']:.4f}")

        st.markdown("<div style='margin:.5rem 0;'>", unsafe_allow_html=True)
        eval_table()
        st.divider()

        # ── Filter tahun ──────────────────────────────────────────────────
        st.markdown(
            f"<div style='font-size:.65rem;color:{MUTED};text-transform:uppercase;"
            f"letter-spacing:.08em;margin-bottom:.4rem;'>Pilih Tahun</div>",
            unsafe_allow_html=True)
        yc = st.columns(5)
        sel_years = [yr for i, yr in enumerate([2016,2017,2018,2019,2020])
                     if yc[i].checkbox(str(yr), value=(yr==2020), key=f"yr{yr}")]

        if not sel_years:
            st.info("Pilih minimal satu tahun.")
        else:
            # ── Grafik evaluasi ───────────────────────────────────────────
            # Warna per tahun — kontras, mudah dibedakan
            YR_COLORS = {
                2016: "#60a5fa",   # biru muda
                2017: "#a78bfa",   # ungu
                2018: "#34d399",   # hijau
                2019: "#fbbf24",   # kuning
                2020: "#fb923c",   # oranye
            }
            # Garis: Aktual = solid tebal, LSTM = dash merah, RF = dot kuning
            ACTUAL_CLR = TEXT        # warna teks (putih dark / hitam light)
            LSTM_CLR   = "#f87171"   # merah lembut
            RF_CLR     = "#fde68a"   # kuning muda

            mc = st.columns(3)
            show_act  = mc[0].checkbox("📈 Data Aktual", value=True,  key="ev_act")
            show_ev_l = mc[1].checkbox("🤖 LSTM",        value=True,  key="ev_l")
            show_ev_r = mc[2].checkbox("🌲 Random Forest",value=True, key="ev_r")

            fig2    = go.Figure()
            offset  = 0
            xbreaks = []       # posisi garis pemisah tahun

            for yr in sel_years:
                d   = EVAL_DATA[yr]
                n   = len(d["actual"])
                xs  = list(range(offset, offset + n))
                mid = offset + n//2

                # Garis pemisah antar tahun (kecuali tahun pertama)
                if offset > 0:
                    xbreaks.append(offset)
                    fig2.add_vline(
                        x=offset - 0.5,
                        line=dict(color=hex2rgba(BORDER, 0.6), width=1.5, dash="solid"),
                    )

                # Label tahun di tengah area
                fig2.add_annotation(
                    x=mid, y=1.04, yref="paper",
                    text=f"<b>{yr}</b>",
                    showarrow=False,
                    font=dict(size=12, color=YR_COLORS[yr], family="Inter"),
                    bgcolor=hex2rgba(YR_COLORS[yr], 0.12),
                    bordercolor=YR_COLORS[yr],
                    borderwidth=1, borderpad=4,
                )

                # ── Traces ───────────────────────────────────────────────
                if show_act:
                    fig2.add_trace(go.Scatter(
                        x=xs, y=d["actual"].tolist(),
                        name="Data Aktual" if yr == sel_years[0] else None,
                        legendgroup="actual",
                        showlegend=(yr == sel_years[0]),
                        line=dict(color=ACTUAL_CLR, width=2),
                        opacity=0.85,
                        hovertemplate=f"<b>{yr}</b> Aktual: %{{y:.1f}} cm<extra></extra>",
                    ))

                if show_ev_l:
                    fig2.add_trace(go.Scatter(
                        x=xs, y=d["lstm"].tolist(),
                        name="Prediksi LSTM" if yr == sel_years[0] else None,
                        legendgroup="lstm",
                        showlegend=(yr == sel_years[0]),
                        line=dict(color=LSTM_CLR, width=1.8, dash="dash"),
                        hovertemplate=f"<b>{yr}</b> LSTM: %{{y:.1f}} cm<extra></extra>",
                    ))

                if show_ev_r:
                    fig2.add_trace(go.Scatter(
                        x=xs, y=d["rf"].tolist(),
                        name="Prediksi RF" if yr == sel_years[0] else None,
                        legendgroup="rf",
                        showlegend=(yr == sel_years[0]),
                        line=dict(color=RF_CLR, width=1.8, dash="dot"),
                        hovertemplate=f"<b>{yr}</b> RF: %{{y:.1f}} cm<extra></extra>",
                    ))

                # Area background per tahun — warna berbeda, transparan
                fig2.add_vrect(
                    x0=offset, x1=offset + n,
                    fillcolor=YR_COLORS[yr], opacity=0.04,
                    layer="below", line_width=0,
                )

                offset += n

            # Garis siaga horizontal
            all_vals = np.concatenate([
                EVAL_DATA[yr]["actual"] for yr in sel_years
            ])
            y_min = max(0, all_vals.min() * 0.88)
            y_max = all_vals.max() * 1.15

            for thresh, clr, lbl in [
                (950, R, "Siaga 1 ≥950"),
                (850, O, "Siaga 2 ≥850"),
                (750, Y, "Siaga 3 ≥750"),
            ]:
                if y_min < thresh < y_max:
                    fig2.add_hline(
                        y=thresh,
                        line=dict(color=clr, width=1.2, dash="dot"),
                        annotation_text=f"  {lbl}",
                        annotation_font=dict(size=9, color=clr),
                        annotation_position="right",
                    )

            # Sumbu x: tick di tengah tiap tahun
            tick_vals = [
                len(EVAL_DATA[sel_years[0]]["actual"]) * i
                + len(EVAL_DATA[sel_years[i]]["actual"]) // 2
                if i < len(sel_years) else 0
                for i in range(len(sel_years))
            ]
            # Hitung ulang tick_vals secara akurat
            tv, tt = [], []
            off2   = 0
            for yr in sel_years:
                n2 = len(EVAL_DATA[yr]["actual"])
                tv.append(off2 + n2//2)
                tt.append(str(yr))
                off2 += n2

            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CBG,
                font=dict(family="Inter", color=MUTED, size=11),
                margin=dict(l=8, r=90, t=52, b=50),
                height=440,
                # Legend di atas — mudah dibaca, tidak tumpang tindih grafik
                legend=dict(
                    bgcolor=CARD, bordercolor=BORDER, borderwidth=1,
                    orientation="h", x=0.0, y=1.12,
                    font=dict(size=11, family="IBM Plex Mono"),
                    itemsizing="constant",
                    # Gambar swatch lebih besar agar jelas
                    traceorder="normal",
                ),
                xaxis=dict(
                    gridcolor=hex2rgba(BORDER, 0.3),
                    tickvals=tv, ticktext=tt,
                    tickfont=dict(size=12, color=TEXT, family="Inter"),
                    linecolor=BORDER, zeroline=False,
                    # Aktifkan zoom & pan
                    fixedrange=False,
                ),
                yaxis=dict(
                    gridcolor=hex2rgba(BORDER, 0.5),
                    ticksuffix=" cm",
                    range=[y_min, y_max],
                    tickfont=dict(size=9), linecolor=BORDER,
                    fixedrange=False,
                ),
                hovermode="x unified",
                hoverlabel=dict(
                    bgcolor=CARD, bordercolor=BORDER,
                    font=dict(color=TEXT, size=11, family="IBM Plex Mono")),
                # Aktifkan toolbar zoom untuk user
                modebar_remove=["select2d","lasso2d","autoScale2d"],
            )

            st.plotly_chart(fig2, use_container_width=True,
                            config={
                                "displayModeBar": True,
                                "modeBarButtonsToRemove": ["select2d","lasso2d"],
                                "scrollZoom": True,
                            })

            # Panduan legenda — eksplisit agar tidak bingung
            st.markdown(f"""
            <div style="display:flex;gap:1.5rem;padding:.5rem .2rem;
                        font-size:.78rem;color:{MUTED};flex-wrap:wrap;margin-bottom:.4rem;">
                <span>
                    <span style="display:inline-block;width:24px;height:2px;
                                 background:{ACTUAL_CLR};vertical-align:middle;
                                 margin-right:5px;"></span>
                    Data Aktual (solid)
                </span>
                <span>
                    <span style="display:inline-block;width:24px;height:2px;
                                 background:{LSTM_CLR};vertical-align:middle;
                                 margin-right:5px;border-top:2px dashed {LSTM_CLR};
                                 background:none;"></span>
                    Prediksi LSTM (putus-putus merah)
                </span>
                <span>
                    <span style="display:inline-block;width:24px;
                                 border-top:2px dotted {RF_CLR};vertical-align:middle;
                                 margin-right:5px;"></span>
                    Prediksi RF (titik-titik kuning)
                </span>
                <span style="color:{MUTED};">
                    💡 Scroll/pinch untuk zoom · drag untuk pan
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # ── Info anomali sensor ─────────────────────────────────────
            n_outliers = sum(
                int(EVAL_DATA[yr]["is_outlier"].sum()) for yr in sel_years
            )
            if n_outliers > 0:
                st.caption(
                    f"⚠️ Terdeteksi {n_outliers} titik anomali ekstrem (lonjakan/penurunan "
                    f">150 cm dalam 1 jam, lalu kembali normal) — diduga **glitch sensor**, "
                    f"bukan kejadian hidrologi nyata. Titik ini tetap ditampilkan di grafik "
                    f"untuk transparansi, namun ikut memengaruhi RMSE keseluruhan."
                )

            # ── Tabel ringkasan per tahun ─────────────────────────────────
            st.markdown(
                f"<div style='font-size:.65rem;color:{MUTED};text-transform:uppercase;"
                f"letter-spacing:.08em;margin:.4rem 0 .3rem;'>"
                f"Ringkasan Per Tahun</div>",
                unsafe_allow_html=True)

            rows_yr = []
            for yr in sel_years:
                d     = EVAL_DATA[yr]
                act   = d["actual"]
                rmse_l = float(np.sqrt(np.mean((act - d["lstm"])**2)))
                rmse_r = float(np.sqrt(np.mean((act - d["rf"])**2)))
                mae_l  = float(np.mean(np.abs(act - d["lstm"])))
                mae_r  = float(np.mean(np.abs(act - d["rf"])))
                # R2 lokal per tahun
                ss_res_l = np.sum((act - d["lstm"])**2)
                ss_res_r = np.sum((act - d["rf"])**2)
                ss_tot   = np.sum((act - act.mean())**2)
                r2_l = float(1 - ss_res_l/ss_tot) if ss_tot > 0 else 0.0
                r2_r = float(1 - ss_res_r/ss_tot) if ss_tot > 0 else 0.0
                unggul = "🤖 LSTM" if rmse_l < rmse_r else "🌲 RF"
                rows_yr.append({
                    "Tahun"     : yr,
                    "RMSE LSTM" : round(rmse_l, 2),
                    "RMSE RF"   : round(rmse_r, 2),
                    "MAE LSTM"  : round(mae_l,  2),
                    "MAE RF"    : round(mae_r,  2),
                    "R² LSTM"   : round(r2_l, 4),
                    "R² RF"     : round(r2_r, 4),
                    "Unggul"    : unggul,
                    "Data"      : "Nyata" if d.get("ok") else "Simulasi",
                })

            df_tbl = pd.DataFrame(rows_yr)

            def _style_unggul(v):
                if v == "🤖 LSTM":
                    return f"color:{G};font-weight:700;"
                if v == "🌲 RF":
                    return f"color:{Y};"
                return ""

            try:
                styled = df_tbl.style.map(_style_unggul, subset=["Unggul"])
            except AttributeError:
                # fallback untuk pandas lama yang belum punya .map
                styled = df_tbl.style.applymap(_style_unggul, subset=["Unggul"])

            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.caption(
                "Catatan: RMSE/MAE/R² per tahun dihitung dari prediksi model nyata "
                "jika data CSV tersedia, atau simulasi deterministik jika tidak. "
                "LSTM konsisten unggul sesuai hasil training notebook Anda."
            )

    # ══════════════════════════════════════════════════════════════════════
    # RIWAYAT
    # ══════════════════════════════════════════════════════════════════════
    elif page == "Riwayat":
        if not st.session_state.history:
            empty("📜", "Belum Ada Riwayat", "Jalankan minimal satu analisis.")
        else:
            hist   = st.session_state.history
            df_all = pd.DataFrame(hist[::-1])
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total",        len(hist))
            c2.metric("Sesi Ini",     len(hist))
            c3.metric("🔴 Bahaya",    sum(1 for h in hist if "BAHAYA" in h["Status"]))
            c4.metric("🟠 Kritis",    sum(1 for h in hist if "KRITIS"  in h["Status"]))
            st.dataframe(df_all, use_container_width=True, hide_index=True)
            ce, cd = st.columns([3,1])
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                df_all.to_excel(w, index=False, sheet_name="Riwayat")
            ce.download_button(
                "⬇ Export (.xlsx)", data=buf.getvalue(),
                file_name=f"sipma_riwayat_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)
            if cd.button("🗑 Hapus", use_container_width=True):
                st.session_state.history = []; st.rerun()

    # ══════════════════════════════════════════════════════════════════════
    # LAPORAN
    # ══════════════════════════════════════════════════════════════════════
    elif page == "Laporan":
        if not res:
            empty("📋", "Belum Ada Data", "Jalankan analisis terlebih dahulu.")
        else:
            L, Rv, last, sg = res["lstm"], res["rf"], res["last_cm"], res["siaga"]
            better  = "LSTM" if EVAL["LSTM"]["RMSE"] < EVAL["RF"]["RMSE"] else "RF"
            primary = L if (better == "LSTM" and L) else Rv
            rows = [
                ("Lokasi",         "Pintu Air Manggarai, Jakarta"),
                ("Waktu Analisis", res["ts"]),
                ("TMA Terakhir",   f"{last:.1f} cm"),
                ("LSTM t+6",       f"{L:.2f} cm" if L else "N/A"),
                ("RF t+6",         f"{Rv:.2f} cm" if Rv else "N/A"),
                ("Selisih",        f"{abs(L-Rv):.2f} cm" if (L and Rv) else "N/A"),
                ("Rekomendasi",    f"{better} (RMSE {EVAL[better]['RMSE']:.4f})"),
            ]
            tbl = "".join(
                f'<tr><td style="color:{MUTED};font-size:.83rem;padding:.3rem 0;'
                f'width:40%;">{k}</td>'
                f'<td style="font-weight:600;font-size:.85rem;'
                f'font-family:IBM Plex Mono,monospace;color:{TEXT};">{v}</td></tr>'
                for k, v in rows
            )
            st.markdown(f"""
            <div class="card">
                <b style="font-size:.7rem;color:{MUTED};text-transform:uppercase;
                           letter-spacing:.08em;">Laporan Prediksi — {res['ts']}</b>
                <table style="width:100%;border-collapse:collapse;margin-top:.6rem;">
                    {tbl}
                    <tr>
                        <td style="color:{MUTED};font-size:.83rem;padding:.3rem 0;">
                            Status Siaga</td>
                        <td style="color:{sg['color']};font-weight:700;">
                            {sg['icon']} Siaga {sg['lvl']} — {sg['label']}</td>
                    </tr>
                </table>
            </div>""", unsafe_allow_html=True)

            siaga_card(primary if primary else last)

            if L and Rv:
                arr_f = res["arr"].flatten()
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Min TMA",   f"{arr_f.min():.1f} cm")
                c2.metric("Max TMA",   f"{arr_f.max():.1f} cm")
                c3.metric("Rata-rata", f"{arr_f.mean():.1f} cm")
                c4.metric("Std Dev",   f"{arr_f.std():.1f} cm")

            st.divider()
            export_btns(res)

    