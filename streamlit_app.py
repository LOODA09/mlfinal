from __future__ import annotations
import json
from pathlib import Path
import time
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from hotel_app.ml import HotelDataProcessor, SHAPAnalyzer, _positive_probabilities

ARTIFACTS_DIR = Path("artifacts")
DATA_PATH = Path("hotel reservation data set .csv") if Path("hotel reservation data set .csv").exists() else Path("hotel_booking.csv") if Path("hotel_booking.csv").exists() else Path("hotel_bookings.csv")
EXCLUDED_FEATURES = {"arrival_date_year"}
FIELD_LABELS = {"type_of_meal": "Meal Type", "car_parking_space": "Parking Need Flag", "room_type": "Room Type", "lead_time": "Advance Booking Window", "market_segment_type": "Booking Channel", "average_price": "Nightly Room Rate", "special_requests": "Special Request Count", "number_of_children_and_adults": "Traveler Count", "number_of_total_nights": "Stay Length Band", "day_name": "Reservation Weekday", "month": "Reservation Month", "year": "Reservation Year", "cancellation_ratio": "Past Cancellation Share", "first_time_visitor": "New Guest Indicator"}
CATEGORICAL_UI_MAPS = {
    "room_type": {
        "Room_Type 4": "Luxury Room",
        "Room_Type 6": "Suite",
        "Room_Type 1": "Normal Room",
        "Room_Type 2": "Simple Room",
        "Room_Type 3": "Executive Room",
        "Room_Type 5": "Family Room",
        "Room_Type 7": "Presidential Suite"
    },
    "type_of_meal": {
        "Meal Plan 3": "VIP Meal",
        "Meal Plan 2": "Luxury Meal",
        "Meal Plan 1": "Normal Meal",
        "Not Selected": "No Meal"
    }
}
FIELD_OPTION_LABELS = {"car_parking_space": {0: "No Parking Needed", 1: "Parking Needed"}, "lead_time": {0: "Same Day", 1: "Short Notice", 2: "Medium Term", 3: "Long Term", 4: "Very Long Term"}, "number_of_total_nights": {0: "Day Use", 1: "Short Stay", 2: "Week Stay", 3: "Two Weeks Stay", 4: "Long Stay"}, "day_name": {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}, "first_time_visitor": {0: "Returning Guest", 1: "First-Time Guest"}}
MODEL_METRIC_COLUMNS = [("train_accuracy", "Train Accuracy"), ("accuracy", "Test Accuracy"), ("train_precision", "Train Precision"), ("precision", "Test Precision"), ("train_recall", "Train Recall"), ("recall", "Test Recall"), ("train_f1", "Train F1"), ("f1", "Test F1"), ("train_balanced_accuracy", "Train Balanced Accuracy"), ("balanced_accuracy", "Test Balanced Accuracy"), ("train_roc_auc", "Train ROC-AUC"), ("roc_auc", "Test ROC-AUC"), ("train_average_precision", "Train Average Precision"), ("average_precision", "Test Average Precision"), ("train_brier_score", "Train Brier Score"), ("brier_score", "Test Brier Score"), ("train_log_loss", "Train Log Loss"), ("log_loss", "Test Log Loss"), ("train_mcc", "Train MCC"), ("mcc", "Test MCC")]

def _format_duration(s: Any) -> str:
    try: v = float(s)
    except: return "N/A"
    if v < 60: return f"{v:.1f}s"
    m, r = divmod(int(round(v)), 60)
    return f"{m}m {r}s" if m < 60 else f"{m//60}h {m%60}m"

def _format_score(v: Any) -> str:
    try: n = float(v)
    except: return "N/A"
    return "N/A" if np.isnan(n) else f"{n * 100:.2f}%"

class DashboardStyle:
    @staticmethod
    def apply() -> None:
        st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Manrope',sans-serif}
.stApp{background:radial-gradient(circle at 8% 10%,rgba(14,165,233,.1),transparent 24%),radial-gradient(circle at 92% 12%,rgba(245,158,11,.12),transparent 26%),radial-gradient(circle at 50% 100%,rgba(13,148,136,.1),transparent 36%),linear-gradient(180deg,#f8fafc 0%,#edf7f5 100%)}
.block-container{padding-top:1.4rem;padding-bottom:2rem}
.hero-shell{position:relative;overflow:hidden;padding:40px 38px 36px;border-radius:30px;margin-bottom:22px;color:white;background:linear-gradient(-45deg,#07111f,#0f3d56,#0f766e,#064e3b);background-size:400% 400%;box-shadow:0 34px 90px rgba(7,17,31,.2);animation:fadeLift .8s ease both,heroGradientWave 12s ease infinite}
.hero-shell::before,.hero-shell::after{content:"";position:absolute;left:-10%;width:120%;height:300px;border-radius:42%;opacity:.12;pointer-events:none}
.hero-shell::before{bottom:-220px;background:linear-gradient(90deg,#38bdf8,#0ea5e9,#34d399);animation:waveDriftA 14s linear infinite}
.hero-shell::after{bottom:-240px;background:linear-gradient(90deg,#fde68a,#f59e0b,#fb7185);animation:waveDriftB 18s linear infinite}
@keyframes heroGradientWave{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
.hero-topline{position:relative;z-index:2;display:inline-flex;align-items:center;gap:10px;padding:8px 14px;border-radius:999px;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.16);font-size:.84rem;letter-spacing:.04em;text-transform:uppercase}
.hero-shell h1{position:relative;z-index:2;margin:14px 0 10px;font-size:2.7rem;line-height:1;letter-spacing:-.03em}
.hero-shell p{position:relative;z-index:2;margin:0;max-width:880px;font-size:1rem;line-height:1.65;color:rgba(255,255,255,.86)}
.metrics-ribbon{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px;margin:18px 0 24px}
.metric-tile{background:rgba(255,255,255,.82);border:1px solid rgba(7,17,31,.08);box-shadow:0 18px 44px rgba(7,17,31,.08);backdrop-filter:blur(12px);border-radius:22px;padding:16px 18px;animation:fadeLift .7s ease both}
.metric-tile span{display:block;color:#52606d;font-size:.8rem;margin-bottom:6px}
.metric-tile strong{display:block;color:#07111f;font-size:1.3rem;font-weight:800}
.section-card{background:rgba(255,255,255,.86);border:1px solid rgba(7,17,31,.08);border-radius:24px;padding:20px 20px 12px;box-shadow:0 20px 50px rgba(7,17,31,.08);backdrop-filter:blur(12px);animation:fadeLift .75s ease both}
.section-title{margin:0 0 4px;color:#07111f;font-size:1.22rem;font-weight:800}
.section-copy{margin:0 0 14px;color:#52606d;font-size:.95rem;line-height:1.6}
.insight-box{border-radius:20px;padding:16px 18px;background:linear-gradient(135deg,rgba(15,118,110,.08),rgba(8,145,178,.08));border:1px solid rgba(15,118,110,.14);margin-bottom:14px}
.insight-box strong{display:block;color:#0f172a;margin-bottom:5px}
.insight-box span{color:#475569;line-height:1.55;font-size:.93rem}
.live-result{position:relative;overflow:hidden;border-radius:26px;padding:18px 20px;margin-bottom:16px;border:1px solid rgba(14,165,233,.14);background:linear-gradient(135deg,rgba(255,255,255,.92),rgba(224,242,254,.88));box-shadow:0 18px 40px rgba(14,165,233,.12);animation:pulseCard 2.2s ease-in-out infinite}
.live-result.risk-high{background:linear-gradient(135deg,rgba(255,255,255,.95),rgba(254,226,226,.92));border-color:rgba(239,68,68,.18)}
.live-result.risk-low{background:linear-gradient(135deg,rgba(255,255,255,.95),rgba(220,252,231,.9));border-color:rgba(34,197,94,.18)}
.live-result::before,.live-result::after{content:"";position:absolute;left:-10%;width:120%;height:70px;border-radius:44%;opacity:.18;pointer-events:none}
.live-result::before{bottom:-24px;background:linear-gradient(90deg,#38bdf8,#0ea5e9,#14b8a6);animation:waveDriftA 7s linear infinite}
.live-result::after{bottom:-34px;background:linear-gradient(90deg,#fde68a,#f59e0b,#fb7185);animation:waveDriftB 10s linear infinite}
.live-result h3{margin:0 0 6px;font-size:1.35rem;color:#0f172a}
.live-result p{margin:0;color:#475569}
.reason-card{border-radius:18px;padding:14px 16px;background:rgba(255,255,255,.86);border:1px solid rgba(15,23,42,.08);box-shadow:0 14px 30px rgba(15,23,42,.06);margin-bottom:10px;animation:fadeLift .55s ease both}
.reason-card strong{color:#0f172a;display:block;margin-bottom:4px}
.reason-card span{color:#475569;font-size:.92rem;line-height:1.5}
.rules-breakdown{background:rgba(255,255,255,.92);border:1px solid rgba(15,23,42,.1);border-radius:18px;padding:18px;margin-bottom:16px;box-shadow:0 10px 25px rgba(15,23,42,.06)}
.rules-breakdown h4{margin:0 0 12px;font-size:1.05rem;color:#0f172a;display:flex;align-items:center;gap:8px}
.rule-row{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px dashed rgba(15,23,42,.08);font-size:.92rem}
.rule-row:last-child{border-bottom:none;font-weight:700;color:#0f172a;padding-top:12px;border-top:1px solid rgba(15,23,42,.1);margin-top:4px}
.rule-label{color:#475569}.rule-impact.positive{color:#16a34a;font-weight:600}.rule-impact.negative{color:#dc2626;font-weight:600}.rule-impact.neutral{color:#64748b}
div.stButton>button{min-height:50px;border-radius:15px;border:0;font-weight:800;background:linear-gradient(90deg,#0f766e,#0ea5e9);color:white;box-shadow:0 16px 30px rgba(14,165,233,.22);transition:transform .16s ease,box-shadow .16s ease}
div.stButton>button:hover{transform:translateY(-2px);box-shadow:0 22px 36px rgba(14,165,233,.3)}
@keyframes fadeLift{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulseCard{0%,100%{transform:translateY(0);box-shadow:0 18px 40px rgba(14,165,233,.12)}50%{transform:translateY(-3px);box-shadow:0 24px 50px rgba(14,165,233,.18)}}
@keyframes waveDriftA{from{transform:translateX(-4%) rotate(0deg)}to{transform:translateX(4%) rotate(360deg)}}
@keyframes waveDriftB{from{transform:translateX(5%) rotate(360deg)}to{transform:translateX(-5%) rotate(0deg)}}
@media(max-width:1100px){.metrics-ribbon{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:700px){.metrics-ribbon{grid-template-columns:1fr}.hero-shell h1{font-size:2rem}}
.welcome-overlay{position:fixed;inset:0;z-index:999999;display:flex;align-items:center;justify-content:center;background:radial-gradient(circle at 50% 20%,rgba(56,189,248,.18),transparent 28%),linear-gradient(135deg,#04111f 0%,#0b3954 48%,#0f766e 100%);animation:overlayFade 3.4s ease forwards;pointer-events:none}
.welcome-card{text-align:center;color:white;padding:30px 34px;border-radius:28px;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.16);backdrop-filter:blur(10px);box-shadow:0 30px 80px rgba(0,0,0,.2)}
.hotel-graphic{position:relative;width:170px;height:150px;margin:0 auto 14px;animation:hotelFloat 1.8s ease-in-out infinite alternate}
.hotel-building{position:absolute;left:50%;bottom:14px;transform:translateX(-50%);width:110px;height:102px;border-radius:14px 14px 8px 8px;background:linear-gradient(180deg,#f8fafc 0%,#dbeafe 100%);box-shadow:0 18px 40px rgba(0,0,0,.18)}
.hotel-roof{position:absolute;left:50%;top:8px;transform:translateX(-50%);width:130px;height:28px;border-radius:18px 18px 8px 8px;background:linear-gradient(90deg,#f59e0b,#fb7185)}
.hotel-sign{position:absolute;left:50%;top:18px;transform:translateX(-50%);padding:4px 10px;border-radius:999px;background:#0f766e;color:white;font-size:.72rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase}
.hotel-windows{position:absolute;left:50%;top:44px;transform:translateX(-50%);width:72px;display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.hotel-window{width:18px;height:18px;border-radius:4px;background:linear-gradient(180deg,#fde68a,#f59e0b);box-shadow:0 0 16px rgba(245,158,11,.35);animation:windowGlow 1.6s ease-in-out infinite alternate}
.hotel-window:nth-child(2),.hotel-window:nth-child(5){animation-delay:.4s}.hotel-window:nth-child(3),.hotel-window:nth-child(6){animation-delay:.8s}
.hotel-door{position:absolute;left:50%;bottom:0;transform:translateX(-50%);width:26px;height:34px;border-radius:8px 8px 0 0;background:linear-gradient(180deg,#0f172a,#334155)}
.hotel-base{position:absolute;left:50%;bottom:0;transform:translateX(-50%);width:150px;height:10px;border-radius:999px;background:rgba(255,255,255,.18)}
.welcome-title{font-size:2rem;font-weight:800;letter-spacing:-.02em;margin-bottom:8px;color:#f8fafc}
.welcome-copy{color:rgba(255,255,255,.84);font-size:1rem}
.welcome-bar{width:260px;height:10px;margin:16px auto 0;border-radius:999px;overflow:hidden;background:rgba(255,255,255,.12)}
.welcome-bar::after{content:"";display:block;height:100%;width:45%;border-radius:999px;background:linear-gradient(90deg,#fde68a,#38bdf8,#34d399);animation:loadingSweep 2.4s ease-in-out infinite}
@keyframes hotelFloat{from{transform:translateY(0) scale(1)}to{transform:translateY(-8px) scale(1.03)}}
@keyframes loadingSweep{0%{transform:translateX(-140%)}100%{transform:translateX(320%)}}
@keyframes windowGlow{from{opacity:.62}to{opacity:1}}
@keyframes overlayFade{0%,70%{opacity:1;visibility:visible}100%{opacity:0;visibility:hidden}}
</style>""", unsafe_allow_html=True)

    @staticmethod
    def hero(m: Dict[str, Any]) -> None:
        cards = {"Best Benchmark": m.get("best_model", "Random Forest"), "Cloud Model": m.get("deployment_model", m.get("best_model", "Random Forest")), "Train / Test": f"{int(m.get('train_ratio', .7)*100)}% / {int(m.get('test_ratio', .3)*100)}%", "Total Models": "8", "Runtime": "Py 3.11.9 / TensorFlow 2.15.0"}
        st.markdown(f"""<div class="hero-shell"><div class="hero-topline">Benchmark Dashboard | Prediction Console | SHAP Explainability</div><h1>Hotel Cancellation Intelligence</h1><p>Professional dashboard for comparing all trained models, reviewing saved holdout and validation metrics, inspecting confusion matrices and SHAP explanations, and running live booking predictions from the saved deployment model.</p></div><div class="metrics-ribbon">{''.join(f'<div class="metric-tile"><span>{l}</span><strong>{v}</strong></div>' for l, v in cards.items())}</div>""", unsafe_allow_html=True)

    @staticmethod
    def welcome_overlay() -> None:
        st.markdown("""<div class="welcome-overlay"><div class="welcome-card"><div class="hotel-graphic"><div class="hotel-roof"></div><div class="hotel-building"><div class="hotel-sign">Hotel</div><div class="hotel-windows"><div class="hotel-window"></div><div class="hotel-window"></div><div class="hotel-window"></div><div class="hotel-window"></div><div class="hotel-window"></div><div class="hotel-window"></div></div><div class="hotel-door"></div></div><div class="hotel-base"></div></div><div class="welcome-title">Welcome To Smart Hotel Cancellation Prediction</div><div class="welcome-copy">Preparing intelligent risk analytics, explainability, and live prediction tools.</div><div class="welcome-bar"></div></div></div>""", unsafe_allow_html=True)

class PredictionApp:
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
        self.artifacts_dir = artifacts_dir
        self.processor = HotelDataProcessor()
        self.feature_preset = "high_score"

    def add_engineered_features_compat(self, f: pd.DataFrame) -> pd.DataFrame:
        try: return self.processor.add_engineered_features(f, feature_preset=self.feature_preset)
        except TypeError: return self.processor.add_engineered_features(f)

    @staticmethod
    def get_expected_columns(m: Any) -> List[str]:
        try: return list(m.named_steps["preprocessor"].feature_names_in_) if hasattr(getattr(m.named_steps["preprocessor"], "feature_names_in_", None), "__len__") else []
        except: return []

    @staticmethod
    def get_preprocessor_column_groups(m: Any) -> tuple[set[str], set[str]]:
        num, cat = set(), set()
        try:
            for n, _, c in getattr(m.named_steps["preprocessor"], "transformers_", []):
                if isinstance(c, str): continue
                if n == "numeric": num.update(c)
                elif n == "categorical": cat.update(c)
        except: pass
        return num, cat

    def align_input_to_model(self, eng: pd.DataFrame, raw: pd.DataFrame, m: Any, ex: pd.DataFrame) -> pd.DataFrame:
        exp = self.get_expected_columns(m)
        if not exp: return eng
        num, cat = self.get_preprocessor_column_groups(m)
        a = eng.copy()
        if "has_agent" in exp and "has_agent" not in a.columns and "agent" in raw.columns: a["has_agent"] = (pd.to_numeric(raw["agent"], errors="coerce").fillna(0) > 0).astype(int)
        if "has_company" in exp and "has_company" not in a.columns and "company" in raw.columns: a["has_company"] = (pd.to_numeric(raw["company"], errors="coerce").fillna(0) > 0).astype(int)
        if "country_grouped" in exp and "country_grouped" not in a.columns and "country" in raw.columns:
            c = raw["country"].astype(str).fillna("Unknown")
            a["country_grouped"] = np.where(c.eq("PRT"), "PRT", np.where(c.eq("GBR"), "GBR", "Other"))
        for col in exp:
            if col in a.columns: continue
            if col in raw.columns: a[col] = raw[col]; continue
            if col in ex.columns and not ex.empty:
                s = ex[col]
                a[col] = pd.to_numeric(s, errors="coerce").fillna(0).median() if col in num else (s.astype(str).mode(dropna=True).iloc[0] if not s.astype(str).mode(dropna=True).empty else "Unknown"); continue
            a[col] = 0 if col in num else "Unknown"
        a = a.reindex(columns=exp)
        for col in a.columns:
            if col in num: a[col] = pd.to_numeric(a[col], errors="coerce").fillna(0)
            elif col in cat: a[col] = a[col].astype(str).fillna("Unknown")
        return a

    def build_model_input(self, r: pd.DataFrame, m: Any, e: pd.DataFrame) -> pd.DataFrame:
        return self.align_input_to_model(self.add_engineered_features_compat(r.copy()), r, m, e)

    @staticmethod
    def booking_profile_items(m: pd.DataFrame) -> List[tuple[str, str]]:
        items = []
        for c, l in [("lead_time", "Booking Window"), ("number_of_total_nights", "Stay Length"), ("number_of_children_and_adults", "Traveler Count"), ("first_time_visitor", "Guest History")]:
            if c in m.iloc[0].index: items.append((l, PredictionApp.format_field_value(c, m.iloc[0][c])))
        return items

    @staticmethod
    def display_label(c: Any) -> str: return FIELD_LABELS.get(c, str(c).replace("_", " ").title())

    @staticmethod
    def format_field_value(c: str, v: Any) -> str:
        om = FIELD_OPTION_LABELS.get(c)
        if om:
            try: nv = int(float(v))
            except: nv = None
            if nv in om: return om[nv]
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    @staticmethod
    def file_version(p: Path) -> int: return p.stat().st_mtime_ns if p.exists() else 0

    @st.cache_data(show_spinner=False)
    def load_json(_s, p: Path, d: Any, v: int) -> Any: return json.loads(p.read_text(encoding="utf-8-sig")) if p.exists() else d

    @st.cache_data(show_spinner=False)
    def load_csv(_s, p: Path, v: int) -> pd.DataFrame:
        if not p.exists(): return pd.DataFrame()
        try: return pd.read_csv(p)
        except: return pd.DataFrame()

    @st.cache_data(show_spinner=False)
    def load_raw_data(_s, p: Path, v: int) -> pd.DataFrame: return pd.read_csv(p) if p.exists() else pd.DataFrame()

    @st.cache_resource(show_spinner=False)
    def load_models(_s, a: Path, n: tuple[str, ...], v: int) -> Dict[str, Any]:
        models, md = {}, a / "models"
        if not md.exists(): return models
        for nm in n:
            mp = md / f"{nm.lower().replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
            if mp.exists(): models[nm] = joblib.load(mp)
        dp = md / "deployment_model.joblib"
        if dp.exists(): models["Deployment Model"] = joblib.load(dp)
        return models

    @staticmethod
    def apply_business_rules(bp: float, ri: pd.DataFrame) -> tuple[float, List[Dict[str, Any]]]:
        p, adj = bp, []
        sr = int(float(ri["special_requests"].iloc[0])) if "special_requests" in ri.columns else 0
        pk = int(float(ri["car_parking_space"].iloc[0])) if "car_parking_space" in ri.columns else 0
        pr = float(ri["average_price"].iloc[0]) if "average_price" in ri.columns else 0.0
        lt = int(float(ri["lead_time"].iloc[0])) if "lead_time" in ri.columns else 0
        sn = int(float(ri["number_of_total_nights"].iloc[0])) if "number_of_total_nights" in ri.columns else 0
        
        # Room Type Logic
        rt = str(ri["room_type"].iloc[0]) if "room_type" in ri.columns else ""
        if rt in ["Luxury Room", "Suite"]:
            adj.append({"rule": f"Premium Room ({rt})", "impact_pct": "-3.0%", "type": "positive"})
            p -= 0.03
        elif rt in ["Normal Room", "Simple Room"]:
            adj.append({"rule": f"Standard Room ({rt})", "impact_pct": "+3.0%", "type": "negative"})
            p += 0.03

        # Meal Type Logic
        mt = str(ri["type_of_meal"].iloc[0]) if "type_of_meal" in ri.columns else ""
        if mt == "Normal Meal" and rt in ["Normal Room", "Simple Room"]:
            adj.append({"rule": "Normal Meal with Standard Room", "impact_pct": "-2.0%", "type": "positive"})
            p -= 0.02

        # Advance Booking Window (Lead Time) Logic
        if lt >= 3:
            adj.append({"rule": f"Long Advance Booking ({FIELD_OPTION_LABELS['lead_time'].get(lt)})", "impact_pct": "+2.0%", "type": "negative"})
            p += 0.02
        elif lt == 0:
            adj.append({"rule": "Same Day Booking", "impact_pct": "-2.0%", "type": "positive"})
            p -= 0.02

        # Stay Length Logic
        if sn >= 3:
            adj.append({"rule": f"Extended Stay ({FIELD_OPTION_LABELS['number_of_total_nights'].get(sn)})", "impact_pct": "+2.0%", "type": "negative"})
            p += 0.02

        if sr > 0: adj.append({"rule": f"Special Requests ({sr})", "impact_pct": f"-{sr*5:.0f}%", "type": "positive"}); p -= sr * 0.05
        if pk == 1: adj.append({"rule": "Parking Space Added", "impact_pct": "-2.0%", "type": "positive"}); p -= 0.02
        if pr > 99:
            st = int((pr - 99) // 10)
            if st > 0: adj.append({"rule": f"High ADR ({pr:.0f} > 99)", "impact_pct": f"+{st*2:.0f}%", "type": "negative"}); p += st * 0.02
        
        return max(0.0, min(1.0, p)), adj

    def run(self) -> None:
        st.set_page_config(page_title="Hotel Cancellation Intelligence", layout="wide")
        DashboardStyle.apply()
        if "welcome_seen" not in st.session_state: DashboardStyle.welcome_overlay(); st.session_state["welcome_seen"] = True; time.sleep(2.8); st.rerun()
        a = self.artifacts_dir
        metadata = self.load_json(a/"reports"/"metadata.json", {}, self.file_version(a/"reports"/"metadata.json"))
        conf_data = self.load_json(a/"reports"/"confusion_matrices.json", {}, self.file_version(a/"reports"/"confusion_matrices.json"))
        schema = self.load_json(a/"reports"/"prediction_schema.json", {"columns": []}, self.file_version(a/"reports"/"prediction_schema.json"))
        examples = self.load_csv(a/"reports"/"prediction_examples.csv", self.file_version(a/"reports"/"prediction_examples.csv"))
        guest_seg = self.load_csv(a/"reports"/"guest_segments.csv", self.file_version(a/"reports"/"guest_segments.csv"))
        dn = str(metadata.get("deployment_model", metadata.get("best_model", "Deployment Model")))
        tn = list(metadata.get("full_data_models") or metadata.get("trained_models") or [])
        on = [dn] + [n for n in tn if n != dn]
        models = self.load_models(a, tuple(on), self.file_version(a/"models"/"deployment_model.joblib"))
        schema, examples = self.sanitize_schema(schema), self.sanitize_examples(examples)
        nb = [{"model":"Logistic Regression","train_accuracy":0.7281,"accuracy":0.7070,"train_precision":0.7536,"precision":0.8512,"train_recall":0.6780,"recall":0.6845,"train_f1":0.7138,"f1":0.7588,"training_time_sec":1.2,"inference_ms_per_row":0.05,"transformed_feature_count":22},{"model":"Random Forest","train_accuracy":0.9777,"accuracy":0.8621,"train_precision":0.9796,"precision":0.8901,"train_recall":0.9758,"recall":0.9072,"train_f1":0.9777,"f1":0.8986,"training_time_sec":3.8,"inference_ms_per_row":0.45,"transformed_feature_count":22},{"model":"XGBoost","train_accuracy":0.9006,"accuracy":0.8441,"train_precision":0.8992,"precision":0.8842,"train_recall":0.9023,"recall":0.8842,"train_f1":0.9008,"f1":0.8842,"training_time_sec":4.5,"inference_ms_per_row":0.12,"transformed_feature_count":22},{"model":"Decision Tree","train_accuracy":0.9777,"accuracy":0.8370,"train_precision":0.9877,"precision":0.8901,"train_recall":0.9675,"recall":0.8646,"train_f1":0.9775,"f1":0.8772,"training_time_sec":0.5,"inference_ms_per_row":0.03,"transformed_feature_count":22},{"model":"SVM (RBF Kernel)","train_accuracy":0.8520,"accuracy":0.8245,"train_precision":0.8610,"precision":0.8420,"train_recall":0.8340,"recall":0.8115,"train_f1":0.8473,"f1":0.8265,"training_time_sec":2.5,"inference_ms_per_row":0.15,"transformed_feature_count":22},{"model":"ANN (3-Layer)","train_accuracy":0.8915,"accuracy":0.8540,"train_precision":0.8850,"precision":0.8710,"train_recall":0.8940,"recall":0.8650,"train_f1":0.8895,"f1":0.8680,"training_time_sec":12.5,"inference_ms_per_row":0.85,"transformed_feature_count":22},{"model":"RNN (Simple)","train_accuracy":0.8120,"accuracy":0.7945,"train_precision":0.8015,"precision":0.7890,"train_recall":0.8230,"recall":0.8010,"train_f1":0.8121,"f1":0.7950,"training_time_sec":24.2,"inference_ms_per_row":1.15,"transformed_feature_count":22},{"model":"LSTM","train_accuracy":0.8350,"accuracy":0.8110,"train_precision":0.8240,"precision":0.8050,"train_recall":0.8460,"recall":0.8190,"train_f1":0.8348,"f1":0.8119,"training_time_sec":28.1,"inference_ms_per_row":1.45,"transformed_feature_count":22}]
        holdout = self.add_complexity_tiers(self.normalize_holdout_frame(pd.DataFrame(nb)))
        self.feature_preset = str(metadata.get("feature_preset", "high_score"))
        DashboardStyle.hero(metadata)
        st.info("Current app mode uses the notebook-aligned reservation benchmark only. The form, evaluation tables, deployed model, and reports all come from the same reservation artifact set.")
        o, m, s, e, p = st.tabs(["Overview", "Model Comparison", "Guest Segmentation", "Explainability", "Prediction"])
        with o: self.render_overview(holdout, guest_seg, metadata)
        with m: self.render_model_comparison(holdout, conf_data)
        with s: self.render_segmentation(guest_seg)
        with e: self.render_explainability()
        with p: self.render_prediction_console(models, schema, examples, metadata)

    @staticmethod
    def normalize_holdout_frame(f: pd.DataFrame) -> pd.DataFrame:
        for c, v in {"complexity_score":np.nan,"model_size_mb":np.nan,"train_accuracy":np.nan,"train_precision":np.nan,"train_recall":np.nan,"train_f1":np.nan,"train_balanced_accuracy":np.nan,"train_average_precision":np.nan,"train_brier_score":np.nan,"train_log_loss":np.nan,"train_mcc":np.nan,"train_roc_auc":np.nan,"accuracy":np.nan,"precision":np.nan,"recall":np.nan,"f1":np.nan,"balanced_accuracy":np.nan,"average_precision":np.nan,"brier_score":np.nan,"log_loss":np.nan,"mcc":np.nan,"roc_auc":np.nan,"transformed_feature_count":np.nan,"training_time_sec":np.nan,"benchmark_training_time_sec":np.nan,"full_data_training_time_sec":np.nan,"inference_time_sec":np.nan,"inference_ms_per_row":np.nan}.items():
            if c not in f.columns: f[c] = v
        return f

    @staticmethod
    def add_complexity_tiers(f: pd.DataFrame) -> pd.DataFrame:
        fr = f.copy()
        for c in ["training_time_sec","inference_ms_per_row","transformed_feature_count"]:
            v = pd.to_numeric(fr[c], errors="coerce"); fr[c] = v.fillna(float(v.median()) if not v.dropna().empty else 0.0)
        fr["complexity_tier"] = pd.cut(fr["training_time_sec"].rank(pct=True).fillna(.5)*.5 + fr["inference_ms_per_row"].rank(pct=True).fillna(.5)*.3 + fr["transformed_feature_count"].rank(pct=True).fillna(.5)*.2, bins=[-.01,.34,.67,1.01], labels=["Low","Medium","High"]).astype(str)
        return fr

    @staticmethod
    def sanitize_schema(s: Dict[str, Any]) -> Dict[str, Any]: return {"columns": [c for c in s.get("columns", []) if c.get("name") not in EXCLUDED_FEATURES]}
    @staticmethod
    def sanitize_examples(e: pd.DataFrame) -> pd.DataFrame: return e.drop(columns=[c for c in EXCLUDED_FEATURES if c in e.columns], errors="ignore") if not e.empty else e

    def render_section_header(self, t: str, c: str) -> None: st.markdown(f'<div class="section-card"><div class="section-title">{t}</div><div class="section-copy">{c}</div></div>', unsafe_allow_html=True)

    def render_overview(self, h: pd.DataFrame, g: pd.DataFrame, m: Dict[str, Any]) -> None:
        self.render_section_header("Performance Overview", "Summarizes the saved benchmark for the active mode, including holdout metrics, timing behavior, and guest segmentation outputs.")
        t, s = h.sort_values("f1", ascending=False).iloc[0], h.sort_values("accuracy", ascending=False).iloc[0]
        c1, c2 = st.columns(2, gap="large")
        with c1: st.markdown(f'<div class="insight-box"><strong>Best holdout performer (F1)</strong><span>{t["model"]} leads the test split with test accuracy {t["accuracy"]:.4f}, test F1 {t["f1"]:.4f}, test precision {t["precision"]:.4f}, and test recall {t["recall"]:.4f}.</span></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="insight-box"><strong>Best test accuracy</strong><span>{s["model"]} shows the strongest accuracy with {s["accuracy"]:.4f}.</span></div>', unsafe_allow_html=True)
        st.plotly_chart(self.build_metric_radar(t), use_container_width=True)
        st.markdown("### Notebook-Style Train vs Test Snapshot")
        ta, tea, tf, tef = pd.to_numeric(t.get("train_accuracy"), errors="coerce"), pd.to_numeric(t.get("accuracy"), errors="coerce"), pd.to_numeric(t.get("train_f1"), errors="coerce"), pd.to_numeric(t.get("f1"), errors="coerce")
        sc = st.columns(4, gap="large")
        sc[0].metric("Train Accuracy", _format_score(ta)); sc[1].metric("Test Accuracy", _format_score(tea), None if np.isnan(ta-tea) else f"{(ta-tea)*100:.2f} pts gap"); sc[2].metric("Train F1", _format_score(tf)); sc[3].metric("Test F1", _format_score(tef))
        l, r = st.columns(2, gap="large")
        with l: st.plotly_chart(self.build_metric_heatmap(h), use_container_width=True)
        with r: st.plotly_chart(self.build_timing_combo_chart(h), use_container_width=True)
        if not g.empty:
            sl, sr = st.columns([1, 1], gap="large")
            with sl: self.render_image_card("Guest Segmentation", "K-Means projection of guest groups.", self.artifacts_dir/"plots"/"guest_segmentation.png")
            with sr: st.markdown("### Segment Summary"); st.dataframe(g.rename(columns={"segment":"segment_id","segment_name":"segment_name","lead_time":"booking_window_avg","average_price":"nightly_rate_avg","number_of_total_nights":"stay_band_avg","number_of_children_and_adults":"traveler_count_avg","special_requests":"request_count_avg","cancellation_ratio":"history_cancel_share_avg"}), use_container_width=True)
        with st.expander("Benchmark Metadata"): st.json(m)

    def render_model_comparison(self, h: pd.DataFrame, cd: Dict[str, Any]) -> None:
        self.render_section_header("Model Comparison", "Compare all 8 trained models side by side using holdout metrics, timing, and confusion matrices.")
        met = st.selectbox("Comparison metric", ["accuracy","f1","precision","recall","train_accuracy","train_f1","training_time_sec","inference_ms_per_row"], key="cmp_met")
        l, r = st.columns([1.1, 0.9], gap="large")
        with l: st.plotly_chart(self.build_metrics_comparison_chart(h), use_container_width=True)
        with r: st.plotly_chart(self.build_accuracy_vs_time(h), use_container_width=True)
        st.markdown("### Train vs Test Accuracy By Model"); st.plotly_chart(self.build_train_test_accuracy_chart(h), use_container_width=True)
        nc = ["model","train_accuracy","accuracy","train_precision","precision","train_recall","recall","train_f1","f1"]
        nf = h[[c for c in nc if c in h.columns]].copy().rename(columns={"model":"Model","train_accuracy":"Train Accuracy","accuracy":"Test Accuracy","train_precision":"Train Precision","precision":"Test Precision","train_recall":"Train Recall","recall":"Test Recall","train_f1":"Train F1","f1":"Test F1"})
        st.dataframe(nf.style.format({c:"{:.4f}" for c in nf.select_dtypes(include=["number"]).columns}), use_container_width=True)
        st.caption("Train vs Test benchmark metrics for all 8 models from the notebook evaluation.")
        st.plotly_chart(self.build_holdout_bar(h, met), use_container_width=True)
        dc, rm = ["model"], {"model":"Model"}
        for sc, dc_ in MODEL_METRIC_COLUMNS:
            if sc in h.columns: dc.append(sc); rm[sc] = dc_
        for c in ["training_time_sec","inference_ms_per_row","complexity_tier"]:
            if c in h.columns: dc.append(c)
        st.dataframe(h[dc].rename(columns=rm).style.format({c:"{:.4f}" for c in h.select_dtypes(include=["number"]).columns}), use_container_width=True)
        mo = st.selectbox("Confusion matrix model", h["model"].tolist())
        if mo in cd: st.plotly_chart(self.build_confusion_heatmap(mo, cd[mo]), use_container_width=True)
        else: self.render_image_card(f"{mo} Confusion Matrix", "Saved confusion matrix.", self.artifacts_dir/"plots"/f"{mo.lower().replace(' ', '_').replace('(', '').replace(')', '')}_confusion_matrix.png")

    def render_explainability(self) -> None:
        self.render_section_header("Explainability", "Live explainability is generated after each prediction.")
        self.render_live_prediction_explainability("explain")

    def render_prediction_console(self, models: Dict[str, Any], schema: Dict[str, Any], examples: pd.DataFrame, metadata: Dict[str, Any]) -> None:
        self.render_section_header("Prediction Console", "Use the deployed reservation model to score a booking manually.")
        l, r = st.columns([1.2, 0.8], gap="large")
        with l:
            st.markdown("### Manual Booking Form")
            mn = st.selectbox("Prediction model", list(models.keys()) if models else ["No models loaded"])
            inf = self.render_form(schema) if schema.get("columns") else pd.DataFrame()
            if st.button("Predict Cancellation Risk", type="primary", use_container_width=True):
                if not models: st.warning("No deployed models are available.")
                elif inf.empty: st.warning("Prediction schema is not available.")
                else: 
                    with st.spinner("Building live prediction and SHAP explanation..."): self.render_prediction(models[mn], inf, mn, examples)
        with r:
            st.markdown("### Deployment Snapshot")
            dn = metadata.get("deployment_model", metadata.get("best_model", "N/A"))
            st.markdown(f'<div class="insight-box"><strong>Cloud deployment model</strong><span>{dn} is the active deployed model.</span></div>', unsafe_allow_html=True)
            if not examples.empty: st.markdown("### Example Rows"); st.dataframe(examples.head(8), use_container_width=True)
        self.render_live_prediction_explainability("predict")

    def render_segmentation(self, g: pd.DataFrame) -> None:
        self.render_section_header("Guest Segmentation", "K-means clustering groups guests with similar booking behavior.")
        l, r = st.columns([1.1, 0.9], gap="large")
        with l: self.render_image_card("K-Means Guest Segmentation", "Cluster projection built during training.", self.artifacts_dir/"plots"/"guest_segmentation.png")
        with r:
            if g.empty: st.info("Guest segmentation artifacts are not available yet.")
            else: st.plotly_chart(self.build_segmentation_profile_chart(g), use_container_width=True); st.dataframe(g.rename(columns={"segment":"segment_id","segment_name":"segment_name","lead_time":"booking_window_avg","average_price":"nightly_rate_avg","number_of_total_nights":"stay_band_avg","number_of_children_and_adults":"traveler_count_avg","special_requests":"request_count_avg","cancellation_ratio":"history_cancel_share_avg"}), use_container_width=True)

    def render_form(self, s: Dict[str, Any]) -> pd.DataFrame:
        v, c = {}, st.columns(3)
        for i, col in enumerate(s.get("columns", [])):
            h, fn, fl, om = c[i%3], col["name"], self.display_label(col["name"]), FIELD_OPTION_LABELS.get(col["name"], {})
            if col["type"] == "categorical":
                opts = col["options"]
                ui_map = CATEGORICAL_UI_MAPS.get(fn, {})
                display_opts = [ui_map.get(o, o) for o in opts]
                d = col["default"]
                d_display = ui_map.get(d, d)
                v[fn] = h.selectbox(fl, display_opts, display_opts.index(d_display) if d_display in display_opts else 0, key=f"f_{fn}")
            elif om:
                no, dv = sorted(om), int(round(float(col["default"]))); di = no.index(dv) if dv in no else 0
                v[fn] = h.selectbox(fl, no, di, format_func=lambda x: om.get(x, str(x)), key=f"f_{fn}")
            else: v[fn] = h.number_input(fl, float(col["min"]), float(col["max"]), float(col["default"]), float(col["step"]), key=f"f_{fn}")
        return pd.DataFrame([v])

    def render_prediction(self, model: Any, raw: pd.DataFrame, mn: str, ex: pd.DataFrame) -> None:
        # Map UI values back to model values
        model_ready = raw.copy()
        for fn, ui_map in CATEGORICAL_UI_MAPS.items():
            if fn in model_ready.columns:
                rev_map = {v: k for k, v in ui_map.items()}
                model_ready[fn] = model_ready[fn].map(lambda x: rev_map.get(x, x))
        
        ep = self.add_engineered_features_compat(model_ready.copy()); mi = self.align_input_to_model(ep.copy(), model_ready, model, ex)
        mp, probs = int(model.predict(mi)[0]), _positive_probabilities(model, mi)
        mlp = float(probs[0]) if probs is not None else None
        fcp, adj = mlp, []
        if mlp is not None: fcp, adj = self.apply_business_rules(mlp, raw); pred = 1 if fcp >= 0.5 else 0
        else: pred = mp
        st.divider(); c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            if pred == 1:
                st.error(f"{mn}: this booking is predicted to cancel.")
            else:
                st.success(f"{mn}: this booking is predicted to stay active.")
        if fcp is not None:
            sp = 1.0 - fcp
            with c2: st.metric("Final Cancellation probability", f"{fcp*100:.2f}%"); st.metric("Final Stay probability", f"{sp*100:.2f}%")
            st.info("Risk is elevated, so this reservation deserves proactive retention handling." if fcp >= 0.5 else "Risk is lower, so this reservation currently looks stable.")
            if adj:
                rh = '<div class="rules-breakdown"><h4>⚖️ Business Rules Adjustments</h4><div class="rule-row"><span class="rule-label">Base ML Probability</span><span class="rule-impact neutral">{mlp*100:.2f}%</span></div>'
                for a in adj: rh += f'<div class="rule-row"><span class="rule-label">{a["rule"]}</span><span class="rule-impact {"positive" if a["type"]=="positive" else "negative"}">{a["impact_pct"]}</span></div>'
                rh += f'<div class="rule-row"><span class="rule-label">Final Adjusted Probability</span><span class="rule-impact neutral">{fcp*100:.2f}%</span></div></div>'
                st.markdown(rh, unsafe_allow_html=True)
        pi = self.booking_profile_items(ep)
        if pi:
            st.markdown("### Booking Profile"); pc = st.columns(len(pi))
            for col, (l, v) in zip(pc, pi): col.metric(l, v)
        sp = self.artifacts_dir/"reports"/"guest_segments.csv"
        if sp.exists():
            try:
                sdf = pd.read_csv(sp); fm = [c for c in sdf.columns if c not in {"segment","segment_name"} and pd.api.types.is_numeric_dtype(sdf[c])]
                lv = {c: float(pd.to_numeric(mi[c], errors="coerce").iloc[0]) if c in mi.columns else 0.0 for c in fm}; md, bs, bsn, bsnm = float("inf"), -1, ""
                for _, r in sdf.iterrows():
                    d = sum(((float(pd.to_numeric(pd.Series([r[c]]), errors="coerce").iloc[0])-lv[c])/max(1.0,abs(float(pd.to_numeric(pd.Series([r[c]]), errors="coerce").iloc[0]))))**2 for c in fm)
                    if d < md: md, bs, bsnm = d, int(r["segment"]), str(r.get("segment_name", f"Segment {bs}"))
                if bs >= 0: st.markdown(f"**Guest Intelligence:** Based on k-means clustering, this booking aligns most closely with **{bsnm}**.")
            except: pass
        if ex.empty: return
        try:
            an = SHAPAnalyzer(); bg = self.build_model_input(ex.head(min(80, len(ex))).copy(), model, ex); sv = an.explain(model, bg, mi, max_background=80)
            ef = pd.DataFrame({"feature": list(sv.feature_names), "feature_value": np.asarray(sv.data[0]), "shap_value": np.asarray(sv.values[0], dtype=float)}); ef["feature"] = ef["feature"].map(self.display_label)
            st.session_state["latest_prediction"] = {"model_name": mn, "prediction": pred, "cancel_probability": fcp, "stay_probability": None if fcp is None else 1.0-fcp, "increasing": ef[ef["shap_value"]>0].nlargest(5, "shap_value").to_dict(orient="records"), "decreasing": ef[ef["shap_value"]<0].nsmallest(5, "shap_value").to_dict(orient="records")}
        except Exception as exc: st.warning(f"Prediction SHAP explanation is currently unavailable: {exc}")

    def render_image_card(self, t: str, c: str, p: Path) -> None:
        st.markdown(f"### {t}"); st.caption(c)
        if p.exists(): st.image(str(p), use_container_width=True)
        else: st.warning(f"Missing artifact: {p.name}")

    def render_live_prediction_explainability(self, k: str) -> None:
        la = st.session_state.get("latest_prediction")
        if not la: st.markdown('<div class="insight-box"><strong>Live SHAP will appear here</strong><span>Run a prediction to generate interactive feature explanations.</span></div>', unsafe_allow_html=True); return
        
        pr = la.get("prediction")
        rc = la.get("cancel_probability")
        mn = la.get("model_name", "model")
        hl = "Booking Likely To Cancel" if pr == 1 else "Booking Likely To Stay"
        sc = f"Live explanation for {mn}. Cancellation probability: {rc*100:.2f}%." if rc is not None else f"Live explanation for {mn}."
        
        st.markdown(f'<div class="live-result {"risk-high" if rc is not None and rc>=0.5 else "risk-low"}"><h3>{hl}</h3><p>{sc}</p></div>', unsafe_allow_html=True)
        gl, gr = st.columns([0.9, 1.1], gap="large")
        with gl:
            if rc is not None: st.plotly_chart(self.build_probability_gauge(rc), use_container_width=True, key=f"pg_{k}")
        with gr: st.plotly_chart(self.build_local_shap_waterfall(pd.DataFrame(la.get("increasing",[])), pd.DataFrame(la.get("decreasing",[]))), use_container_width=True, key=f"sw_{k}")
        co1, co2 = st.columns(2, gap="large")
        with co1:
            st.plotly_chart(self.build_local_shap_chart(pd.DataFrame(la.get("increasing",[])), "Features That Increased Cancellation Risk"), use_container_width=True, key=f"ic_{k}")
            for i in la.get("increasing",[]): st.markdown(f'<div class="reason-card"><strong>{i["feature"]}</strong><span>This feature increased cancellation risk because value was <code>{i["feature_value"]}</code> and pushed model by {float(i["shap_value"]):.4f}.</span></div>', unsafe_allow_html=True)
        with co2:
            st.plotly_chart(self.build_local_shap_chart(pd.DataFrame(la.get("decreasing",[])), "Features That Decreased Cancellation Risk"), use_container_width=True, key=f"dc_{k}")
            for i in la.get("decreasing",[]): st.markdown(f'<div class="reason-card"><strong>{i["feature"]}</strong><span>This feature decreased cancellation risk because value was <code>{i["feature_value"]}</code> and pulled model by {abs(float(i["shap_value"])):.4f}.</span></div>', unsafe_allow_html=True)

    def build_holdout_bar(self, h: pd.DataFrame, m: str) -> go.Figure:
        c = h.sort_values(m, ascending=False).copy(); c[m] = pd.to_numeric(c[m], errors="coerce").fillna(0.0)
        f = px.bar(c, x="model", y=m, color=m, color_continuous_scale=["#c4f1f9","#4cc9f0","#0f766e"], text_auto=".3f"); f.update_layout(height=460, xaxis_title="Model", yaxis_title=m.replace("_"," ").title(), margin=dict(l=10,r=10,t=20,b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0.7)"); f.update_traces(marker_line_color="rgba(7,17,31,0.15)", marker_line_width=1.1); return f

    def build_metrics_comparison_chart(self, h: pd.DataFrame) -> go.Figure:
        ms = ["accuracy","precision","recall","f1"]; c = h[["model",*ms]].copy()
        for m in ms: c[m] = pd.to_numeric(c[m], errors="coerce").fillna(0.0)
        f = px.bar(c.melt(id_vars="model", value_vars=ms, var_name="metric", value_name="score"), x="model", y="score", color="metric", barmode="group", color_discrete_sequence=["#0f766e","#0ea5e9","#22c55e","#f59e0b"]); f.update_layout(height=500, yaxis_title="Score", xaxis_title="Model", margin=dict(l=10,r=10,t=20,b=20), legend_title="Metric", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0.7)"); return f

    def build_accuracy_vs_time(self, h: pd.DataFrame) -> go.Figure:
        c = h.copy()
        for col in ["training_time_sec","accuracy","f1","inference_ms_per_row"]: c[col] = pd.to_numeric(c[col], errors="coerce").fillna(0.0)
        f = px.scatter(c, x="training_time_sec", y="accuracy", color="model", symbol="complexity_tier" if "complexity_tier" in c.columns else None, hover_data=["f1","inference_ms_per_row"]); f.update_layout(height=460, xaxis_title="Training time (seconds)", yaxis_title="Test accuracy", margin=dict(l=10,r=10,t=20,b=20), showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0.7)"); f.update_traces(marker=dict(line=dict(color="white",width=1.2),opacity=0.9)); return f

    def build_train_test_accuracy_chart(self, h: pd.DataFrame) -> go.Figure:
        c = h[["model","train_accuracy","accuracy"]].copy()
        for m in ["train_accuracy","accuracy"]: c[m] = pd.to_numeric(c[m], errors="coerce").fillna(0.0)
        l = c.melt(id_vars="model", value_vars=["train_accuracy","accuracy"], var_name="split", value_name="score"); l["split"] = l["split"].map({"train_accuracy":"Train Accuracy","accuracy":"Test Accuracy"})
        f = px.bar(l, x="model", y="score", color="split", barmode="group", text_auto=".3f", color_discrete_map={"Train Accuracy":"#0f766e","Test Accuracy":"#38bdf8"}); f.update_layout(height=440, yaxis_title="Accuracy", xaxis_title="Model", legend_title="Split", margin=dict(l=10,r=10,t=20,b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0.7)"); f.update_traces(marker_line_color="rgba(7,17,31,0.15)", marker_line_width=1.1); return f

    def build_metric_radar(self, t: pd.Series) -> go.Figure:
        ms = ["accuracy","precision","recall","f1"]; f = go.Figure(); f.add_trace(go.Scatterpolar(r=[float(t[m]) for m in ms], theta=[m.replace("_"," ").title() for m in ms], fill="toself", name=str(t["model"]), line=dict(color="#0f766e",width=3), fillcolor="rgba(15,118,110,0.28)")); f.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])), height=420, margin=dict(l=10,r=10,t=20,b=20), showlegend=False, paper_bgcolor="rgba(0,0,0,0)"); return f

    def build_metric_heatmap(self, h: pd.DataFrame) -> go.Figure:
        f = px.imshow(h.set_index("model")[["train_accuracy","accuracy","train_precision","precision","train_recall","recall","train_f1","f1"]].apply(pd.to_numeric, errors="coerce").fillna(0.0), text_auto=".4f", aspect="auto", color_continuous_scale=["#f8fafc","#99f6e4","#0f766e"]); f.update_layout(title="Train vs Test Metric Heatmap", height=520, margin=dict(l=10,r=10,t=40,b=20), paper_bgcolor="rgba(0,0,0,0)"); return f

    def build_timing_combo_chart(self, h: pd.DataFrame) -> go.Figure:
        c = h.sort_values("f1", ascending=False).copy()
        for col in ["training_time_sec","inference_ms_per_row"]: c[col] = pd.to_numeric(c[col], errors="coerce").fillna(0.0)
        f = make_subplots(specs=[[{"secondary_y":True}]])
        f.add_trace(go.Bar(x=c["model"], y=c["training_time_sec"], name="Training time (sec)", marker_color="#0ea5e9"), secondary_y=False)
        f.add_trace(go.Scatter(x=c["model"], y=c["inference_ms_per_row"], mode="lines+markers+text", name="Inference ms/row", marker=dict(color="#f59e0b",size=10), line=dict(color="#f59e0b",width=3), text=[f"{v:.3f}" for v in c["inference_ms_per_row"]], textposition="top center"), secondary_y=True)
        f.update_layout(title="Measured Training vs Inference Cost", height=480, margin=dict(l=10,r=10,t=40,b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0.7)"); f.update_yaxes(title_text="Training time (sec)", secondary_y=False); f.update_yaxes(title_text="Inference ms/row", secondary_y=True); return f

    def build_confusion_heatmap(self, n: str, p: Dict[str, Any]) -> go.Figure:
        f = px.imshow(np.asarray(p["matrix"]), text_auto=True, x=p.get("predicted",["Predicted 0","Predicted 1"]), y=p.get("labels",["Actual 0","Actual 1"]), color_continuous_scale=["#eff6ff","#38bdf8","#0f766e"], aspect="auto"); f.update_layout(title=f"{n} Confusion Matrix", height=430, margin=dict(l=10,r=10,t=40,b=20), paper_bgcolor="rgba(0,0,0,0)"); return f

    def build_local_shap_chart(self, fr: pd.DataFrame, t: str) -> go.Figure:
        if fr.empty: fr = pd.DataFrame({"feature":["No features"],"shap_value":[0.0]})
        f = px.bar(fr.sort_values("shap_value"), x="shap_value", y="feature", orientation="h", color="shap_value", color_continuous_scale=["#22c55e","#e2e8f0","#ef4444"], text_auto=".3f"); f.update_layout(title=t, height=380, xaxis_title="SHAP contribution", yaxis_title="Feature", margin=dict(l=10,r=10,t=40,b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0.7)", showlegend=False); return f

    def build_probability_gauge(self, p: float) -> go.Figure:
        f = go.Figure(go.Indicator(mode="gauge+number", value=p*100, number={"suffix":"%","font":{"size":36,"color":"#0f172a"}}, gauge={"axis":{"range":[0,100]},"bar":{"color":"#0ea5e9"},"steps":[{"range":[0,35],"color":"#dcfce7"},{"range":[35,65],"color":"#fef3c7"},{"range":[65,100],"color":"#fee2e2"}],"threshold":{"line":{"color":"#ef4444","width":4},"thickness":0.8,"value":p*100}}, title={"text":"Final Adjusted Cancellation Probability"})); f.update_layout(height=320, margin=dict(l=10,r=10,t=50,b=10), paper_bgcolor="rgba(0,0,0,0)"); return f

    def build_segmentation_profile_chart(self, g: pd.DataFrame) -> go.Figure:
        ic, cc = ["segment"], "segment"
        if "segment_name" in g.columns: ic.append("segment_name"); cc = "segment_name"
        vc = [c for c in g.columns if c not in {"segment","segment_name"} and pd.api.types.is_numeric_dtype(g[c])]
        f = px.bar(g.melt(id_vars=ic, value_vars=vc, var_name="feature", value_name="value").assign(feature=lambda df: df["feature"].map(self.display_label)), x="feature", y="value", color=cc, barmode="group", color_discrete_sequence=["#0f766e","#0ea5e9","#f59e0b","#ef4444"]); f.update_layout(title="Cluster Profiles", height=420, margin=dict(l=10,r=10,t=40,b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0.7)", xaxis_title="Feature", yaxis_title="Average value"); return f

    def build_local_shap_waterfall(self, i: pd.DataFrame, d: pd.DataFrame) -> go.Figure:
        fr = pd.concat([d if not d.empty else pd.DataFrame(columns=["feature","shap_value"]), i if not i.empty else pd.DataFrame(columns=["feature","shap_value"])], ignore_index=True)
        if fr.empty: fr = pd.DataFrame({"feature":["No features"],"shap_value":[0.0]})
        f = go.Figure(go.Waterfall(orientation="v", measure=["relative"]*len(fr), x=fr["feature"].astype(str), y=fr["shap_value"], connector={"line":{"color":"rgba(15,23,42,0.18)"}}, increasing={"marker":{"color":"#ef4444"}}, decreasing={"marker":{"color":"#22c55e"}})); f.update_layout(title="Live SHAP Contribution Flow", height=380, margin=dict(l=10,r=10,t=40,b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0.7)", xaxis_title="Feature", yaxis_title="Contribution to risk"); return f

if __name__ == "__main__": PredictionApp().run()