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


ARTIFACTS_DIR = Path("artifacts_hotel_booking") if Path("artifacts_hotel_booking").exists() else Path("artifacts")
DATA_PATH = Path("hotel_booking.csv") if Path("hotel_booking.csv").exists() else Path("hotel_bookings.csv")
EXCLUDED_FEATURES = {"arrival_date_year"}


def _format_duration(seconds: Any) -> str:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "N/A"
    if value < 60:
        return f"{value:.1f}s"
    minutes, remainder = divmod(int(round(value)), 60)
    if minutes < 60:
        return f"{minutes}m {remainder}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


class DashboardStyle:
    @staticmethod
    def apply() -> None:
        st.markdown(
            """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

            html, body, [class*="css"] {
                font-family: 'Manrope', sans-serif;
            }

            .stApp {
                background:
                    radial-gradient(circle at 8% 10%, rgba(14, 165, 233, 0.10), transparent 24%),
                    radial-gradient(circle at 92% 12%, rgba(245, 158, 11, 0.12), transparent 26%),
                    radial-gradient(circle at 50% 100%, rgba(13, 148, 136, 0.10), transparent 36%),
                    linear-gradient(180deg, #f8fafc 0%, #edf7f5 100%);
            }

            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 2rem;
            }

            .hero-shell {
                position: relative;
                overflow: hidden;
                padding: 40px 38px 36px;
                border-radius: 30px;
                margin-bottom: 22px;
                color: white;
                background: linear-gradient(-45deg, #07111f, #0f3d56, #0f766e, #064e3b);
                background-size: 400% 400%;
                box-shadow: 0 34px 90px rgba(7, 17, 31, 0.20);
                animation: fadeLift .8s ease both, heroGradientWave 12s ease infinite;
            }

            .hero-shell::before,
            .hero-shell::after {
                content: "";
                position: absolute;
                left: -10%;
                width: 120%;
                height: 300px;
                border-radius: 42%;
                opacity: .12;
                pointer-events: none;
            }

            .hero-shell::before {
                bottom: -220px;
                background: linear-gradient(90deg, #38bdf8, #0ea5e9, #34d399);
                animation: waveDriftA 14s linear infinite;
            }

            .hero-shell::after {
                bottom: -240px;
                background: linear-gradient(90deg, #fde68a, #f59e0b, #fb7185);
                animation: waveDriftB 18s linear infinite;
            }

            @keyframes heroGradientWave {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

            .hero-topline {
                position: relative;
                z-index: 2;
                display: inline-flex;
                align-items: center;
                gap: 10px;
                padding: 8px 14px;
                border-radius: 999px;
                background: rgba(255,255,255,.10);
                border: 1px solid rgba(255,255,255,.16);
                font-size: .84rem;
                letter-spacing: .04em;
                text-transform: uppercase;
            }

            .hero-shell h1 {
                position: relative;
                z-index: 2;
                margin: 14px 0 10px;
                font-size: 2.7rem;
                line-height: 1;
                letter-spacing: -.03em;
            }

            .hero-shell p {
                position: relative;
                z-index: 2;
                margin: 0;
                max-width: 880px;
                font-size: 1rem;
                line-height: 1.65;
                color: rgba(255,255,255,.86);
            }

            .metrics-ribbon {
                display: grid;
                grid-template-columns: repeat(5, minmax(0, 1fr));
                gap: 14px;
                margin: 18px 0 24px;
            }

            .metric-tile {
                background: rgba(255,255,255,.82);
                border: 1px solid rgba(7, 17, 31, 0.08);
                box-shadow: 0 18px 44px rgba(7, 17, 31, 0.08);
                backdrop-filter: blur(12px);
                border-radius: 22px;
                padding: 16px 18px;
                animation: fadeLift .7s ease both;
            }

            .metric-tile span {
                display: block;
                color: #52606d;
                font-size: .8rem;
                margin-bottom: 6px;
            }

            .metric-tile strong {
                display: block;
                color: #07111f;
                font-size: 1.3rem;
                font-weight: 800;
            }

            .section-card {
                background: rgba(255,255,255,.86);
                border: 1px solid rgba(7, 17, 31, 0.08);
                border-radius: 24px;
                padding: 20px 20px 12px;
                box-shadow: 0 20px 50px rgba(7, 17, 31, 0.08);
                backdrop-filter: blur(12px);
                animation: fadeLift .75s ease both;
            }

            .section-title {
                margin: 0 0 4px;
                color: #07111f;
                font-size: 1.22rem;
                font-weight: 800;
            }

            .section-copy {
                margin: 0 0 14px;
                color: #52606d;
                font-size: .95rem;
                line-height: 1.6;
            }

            .insight-box {
                border-radius: 20px;
                padding: 16px 18px;
                background: linear-gradient(135deg, rgba(15,118,110,.08), rgba(8,145,178,.08));
                border: 1px solid rgba(15,118,110,.14);
                margin-bottom: 14px;
            }

            .insight-box strong {
                display: block;
                color: #0f172a;
                margin-bottom: 5px;
            }

            .insight-box span {
                color: #475569;
                line-height: 1.55;
                font-size: .93rem;
            }

            .live-result {
                position: relative;
                overflow: hidden;
                border-radius: 26px;
                padding: 18px 20px;
                margin-bottom: 16px;
                border: 1px solid rgba(14,165,233,.14);
                background: linear-gradient(135deg, rgba(255,255,255,.92), rgba(224,242,254,.88));
                box-shadow: 0 18px 40px rgba(14,165,233,.12);
                animation: pulseCard 2.2s ease-in-out infinite;
            }

            .live-result.risk-high {
                background: linear-gradient(135deg, rgba(255,255,255,.95), rgba(254,226,226,.92));
                border-color: rgba(239,68,68,.18);
            }

            .live-result.risk-low {
                background: linear-gradient(135deg, rgba(255,255,255,.95), rgba(220,252,231,.90));
                border-color: rgba(34,197,94,.18);
            }

            .live-result::before,
            .live-result::after {
                content: "";
                position: absolute;
                left: -10%;
                width: 120%;
                height: 70px;
                border-radius: 44%;
                opacity: .18;
                pointer-events: none;
            }

            .live-result::before {
                bottom: -24px;
                background: linear-gradient(90deg, #38bdf8, #0ea5e9, #14b8a6);
                animation: waveDriftA 7s linear infinite;
            }

            .live-result::after {
                bottom: -34px;
                background: linear-gradient(90deg, #fde68a, #f59e0b, #fb7185);
                animation: waveDriftB 10s linear infinite;
            }

            .live-result h3 {
                margin: 0 0 6px;
                font-size: 1.35rem;
                color: #0f172a;
            }

            .live-result p {
                margin: 0;
                color: #475569;
            }

            .reason-card {
                border-radius: 18px;
                padding: 14px 16px;
                background: rgba(255,255,255,.86);
                border: 1px solid rgba(15,23,42,.08);
                box-shadow: 0 14px 30px rgba(15,23,42,.06);
                margin-bottom: 10px;
                animation: fadeLift .55s ease both;
            }

            .reason-card strong {
                color: #0f172a;
                display: block;
                margin-bottom: 4px;
            }

            .reason-card span {
                color: #475569;
                font-size: .92rem;
                line-height: 1.5;
            }

            div.stButton > button {
                min-height: 50px;
                border-radius: 15px;
                border: 0;
                font-weight: 800;
                background: linear-gradient(90deg, #0f766e, #0ea5e9);
                color: white;
                box-shadow: 0 16px 30px rgba(14, 165, 233, .22);
                transition: transform .16s ease, box-shadow .16s ease;
            }

            div.stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 22px 36px rgba(14, 165, 233, .30);
            }

            @keyframes fadeLift {
                from { opacity: 0; transform: translateY(14px); }
                to { opacity: 1; transform: translateY(0); }
            }

            @keyframes floatA {
                from { transform: translate3d(0, 0, 0); }
                to { transform: translate3d(16px, 18px, 0); }
            }

            @keyframes floatB {
                from { transform: translate3d(0, 0, 0); }
                to { transform: translate3d(-18px, -10px, 0); }
            }

            @media (max-width: 1100px) {
                .metrics-ribbon {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }

            @media (max-width: 700px) {
                .metrics-ribbon {
                    grid-template-columns: 1fr;
                }
                .hero-shell h1 {
                    font-size: 2rem;
                }
            }

            .welcome-overlay {
                position: fixed;
                inset: 0;
                z-index: 999999;
                display: flex;
                align-items: center;
                justify-content: center;
                background:
                    radial-gradient(circle at 50% 20%, rgba(56,189,248,.18), transparent 28%),
                    linear-gradient(135deg, #04111f 0%, #0b3954 48%, #0f766e 100%);
                animation: overlayFade 3.4s ease forwards;
                pointer-events: none;
            }

            .welcome-card {
                text-align: center;
                color: white;
                padding: 30px 34px;
                border-radius: 28px;
                background: rgba(255,255,255,.08);
                border: 1px solid rgba(255,255,255,.16);
                backdrop-filter: blur(10px);
                box-shadow: 0 30px 80px rgba(0,0,0,.20);
            }

            .hotel-graphic {
                position: relative;
                width: 170px;
                height: 150px;
                margin: 0 auto 14px;
                animation: hotelFloat 1.8s ease-in-out infinite alternate;
            }

            .hotel-building {
                position: absolute;
                left: 50%;
                bottom: 14px;
                transform: translateX(-50%);
                width: 110px;
                height: 102px;
                border-radius: 14px 14px 8px 8px;
                background: linear-gradient(180deg, #f8fafc 0%, #dbeafe 100%);
                box-shadow: 0 18px 40px rgba(0,0,0,.18);
            }

            .hotel-roof {
                position: absolute;
                left: 50%;
                top: 8px;
                transform: translateX(-50%);
                width: 130px;
                height: 28px;
                border-radius: 18px 18px 8px 8px;
                background: linear-gradient(90deg, #f59e0b, #fb7185);
            }

            .hotel-sign {
                position: absolute;
                left: 50%;
                top: 18px;
                transform: translateX(-50%);
                padding: 4px 10px;
                border-radius: 999px;
                background: #0f766e;
                color: white;
                font-size: .72rem;
                font-weight: 800;
                letter-spacing: .14em;
                text-transform: uppercase;
            }

            .hotel-windows {
                position: absolute;
                left: 50%;
                top: 44px;
                transform: translateX(-50%);
                width: 72px;
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 8px;
            }

            .hotel-window {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                background: linear-gradient(180deg, #fde68a, #f59e0b);
                box-shadow: 0 0 16px rgba(245,158,11,.35);
                animation: windowGlow 1.6s ease-in-out infinite alternate;
            }

            .hotel-window:nth-child(2),
            .hotel-window:nth-child(5) {
                animation-delay: .4s;
            }

            .hotel-window:nth-child(3),
            .hotel-window:nth-child(6) {
                animation-delay: .8s;
            }

            .hotel-door {
                position: absolute;
                left: 50%;
                bottom: 0;
                transform: translateX(-50%);
                width: 26px;
                height: 34px;
                border-radius: 8px 8px 0 0;
                background: linear-gradient(180deg, #0f172a, #334155);
            }

            .hotel-base {
                position: absolute;
                left: 50%;
                bottom: 0;
                transform: translateX(-50%);
                width: 150px;
                height: 10px;
                border-radius: 999px;
                background: rgba(255,255,255,.18);
            }

            .welcome-title {
                font-size: 2rem;
                font-weight: 800;
                letter-spacing: -.02em;
                margin-bottom: 8px;
                color: #f8fafc;
            }

            .welcome-copy {
                color: rgba(255,255,255,.84);
                font-size: 1rem;
            }

            .welcome-bar {
                width: 260px;
                height: 10px;
                margin: 16px auto 0;
                border-radius: 999px;
                overflow: hidden;
                background: rgba(255,255,255,.12);
            }

            .welcome-bar::after {
                content: "";
                display: block;
                height: 100%;
                width: 45%;
                border-radius: 999px;
                background: linear-gradient(90deg, #fde68a, #38bdf8, #34d399);
                animation: loadingSweep 2.4s ease-in-out infinite;
            }

            @keyframes hotelFloat {
                from { transform: translateY(0px) scale(1); }
                to { transform: translateY(-8px) scale(1.03); }
            }

            @keyframes loadingSweep {
                0% { transform: translateX(-140%); }
                100% { transform: translateX(320%); }
            }

            @keyframes windowGlow {
                from { opacity: .62; }
                to { opacity: 1; }
            }

            @keyframes overlayFade {
                0%, 70% { opacity: 1; visibility: visible; }
                100% { opacity: 0; visibility: hidden; }
            }

            @keyframes pulseCard {
                0%, 100% { transform: translateY(0px); box-shadow: 0 18px 40px rgba(14,165,233,.12); }
                50% { transform: translateY(-3px); box-shadow: 0 24px 50px rgba(14,165,233,.18); }
            }

            @keyframes waveDriftA {
                from { transform: translateX(-4%) rotate(0deg); }
                to { transform: translateX(4%) rotate(360deg); }
            }

            @keyframes waveDriftB {
                from { transform: translateX(5%) rotate(360deg); }
                to { transform: translateX(-5%) rotate(0deg); }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def hero(metadata: Dict[str, Any]) -> None:
        st.markdown(
            f"""
            <div class="hero-shell">
                <div class="hero-topline">Benchmark Dashboard | Prediction Console | SHAP Explainability</div>
                <h1>Hotel Cancellation Intelligence</h1>
                <p>
                    Professional dashboard for comparing all trained models, reviewing honest holdout and
                    5-fold validation metrics, inspecting confusion matrices and SHAP explanations, and
                    running live booking predictions from the saved deployment model.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cards = {
            "Best Benchmark": metadata.get("best_model", "N/A"),
            "Cloud Model": metadata.get("deployment_model", metadata.get("best_model", "N/A")),
            "Train / Test": f"{int(metadata.get('train_ratio', 0.7) * 100)}% / {int(metadata.get('test_ratio', 0.3) * 100)}%",
            "Cross-Validation": f"{metadata.get('cross_validation_folds', 'N/A')}-fold",
            "Pipeline Wall Clock": _format_duration(metadata.get("total_pipeline_wall_clock_sec")),
            "Runtime": f"Py {metadata.get('python_version', 'N/A')} / TF {metadata.get('tensorflow_version', 'N/A') or 'off'}",
        }
        html = "".join(
            f'<div class="metric-tile"><span>{label}</span><strong>{value}</strong></div>'
            for label, value in cards.items()
        )
        st.markdown(f'<div class="metrics-ribbon">{html}</div>', unsafe_allow_html=True)

    @staticmethod
    def welcome_overlay() -> None:
        st.markdown(
            """
            <div class="welcome-overlay">
                <div class="welcome-card">
                    <div class="hotel-graphic">
                        <div class="hotel-roof"></div>
                        <div class="hotel-building">
                            <div class="hotel-sign">Hotel</div>
                            <div class="hotel-windows">
                                <div class="hotel-window"></div>
                                <div class="hotel-window"></div>
                                <div class="hotel-window"></div>
                                <div class="hotel-window"></div>
                                <div class="hotel-window"></div>
                                <div class="hotel-window"></div>
                            </div>
                            <div class="hotel-door"></div>
                        </div>
                        <div class="hotel-base"></div>
                    </div>
                    <div class="welcome-title">Welcome To Smart Hotel Cancellation Prediction</div>
                    <div class="welcome-copy">Preparing intelligent risk analytics, explainability, and live prediction tools.</div>
                    <div class="welcome-bar"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


class PredictionApp:
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
        self.artifacts_dir = artifacts_dir
        self.processor = HotelDataProcessor()

    @staticmethod
    def file_version(path: Path) -> int:
        if not path.exists():
            return 0
        return path.stat().st_mtime_ns

    @st.cache_data(show_spinner=False)
    def load_json(_self, path: Path, default: Any, version: int) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8-sig"))

    @st.cache_data(show_spinner=False)
    def load_csv(_self, path: Path, version: int) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    @st.cache_data(show_spinner=False)
    def load_raw_data(_self, path: Path, version: int) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    @st.cache_resource(show_spinner=False)
    def load_models(_self, artifacts_dir: Path, version: int) -> Dict[str, Any]:
        models: Dict[str, Any] = {}
        models_dir = artifacts_dir / "models"
        if not models_dir.exists():
            return models
        deployment_path = models_dir / "deployment_model.joblib"
        if deployment_path.exists():
            models["Deployment Model"] = joblib.load(deployment_path)
        return models

    def run(self) -> None:
        st.set_page_config(page_title="Hotel Cancellation Intelligence", layout="wide")
        DashboardStyle.apply()
        if "welcome_seen" not in st.session_state:
            DashboardStyle.welcome_overlay()
            st.session_state["welcome_seen"] = True
            time.sleep(2.8)
            st.rerun()

        metadata_path = self.artifacts_dir / "reports" / "metadata.json"
        confusion_path = self.artifacts_dir / "reports" / "confusion_matrices.json"
        schema_path = self.artifacts_dir / "reports" / "prediction_schema.json"
        examples_path = self.artifacts_dir / "reports" / "prediction_examples.csv"
        holdout_path = self.artifacts_dir / "reports" / "holdout_summary.csv"
        cv_path = self.artifacts_dir / "reports" / "cross_validation_results.csv"
        segments_path = self.artifacts_dir / "reports" / "guest_segments.csv"
        model_path = self.artifacts_dir / "models" / "deployment_model.joblib"

        metadata = self.load_json(metadata_path, {}, self.file_version(metadata_path))
        confusion_data = self.load_json(confusion_path, {}, self.file_version(confusion_path))
        schema = self.load_json(schema_path, {"columns": []}, self.file_version(schema_path))
        examples = self.load_csv(examples_path, self.file_version(examples_path))
        holdout = self.load_csv(holdout_path, self.file_version(holdout_path))
        cv_results = self.load_csv(cv_path, self.file_version(cv_path))
        guest_segments = self.load_csv(segments_path, self.file_version(segments_path))
        raw_data = self.load_raw_data(DATA_PATH, self.file_version(DATA_PATH))
        models = self.load_models(self.artifacts_dir, self.file_version(model_path))

        if holdout.empty or not schema.get("columns"):
            st.error("Saved training artifacts are missing. Run terminal training and redeploy the app.")
            return

        schema = self.sanitize_schema(schema)
        examples = self.sanitize_examples(examples)
        holdout = self.normalize_holdout_frame(holdout)
        cv_results = self.normalize_cv_frame(cv_results)
        holdout = self.add_complexity_tiers(holdout)

        DashboardStyle.hero(metadata)

        overview_tab, models_tab, segment_tab, explain_tab, predict_tab = st.tabs(
            ["Overview", "Model Comparison", "Guest Segmentation", "Explainability", "Prediction"]
        )

        with overview_tab:
            self.render_overview(holdout, cv_results, guest_segments, metadata, raw_data)

        with models_tab:
            self.render_model_comparison(holdout, cv_results, confusion_data)

        with segment_tab:
            self.render_segmentation(guest_segments)

        with explain_tab:
            self.render_explainability()

        with predict_tab:
            self.render_prediction_console(models, schema, examples, metadata)

    @staticmethod
    def normalize_holdout_frame(holdout: pd.DataFrame) -> pd.DataFrame:
        frame = holdout.copy()
        defaults = {
            "complexity_score": np.nan,
            "model_size_mb": np.nan,
            "transformed_feature_count": np.nan,
            "training_time_sec": np.nan,
            "benchmark_training_time_sec": np.nan,
            "full_data_training_time_sec": np.nan,
            "inference_time_sec": np.nan,
            "inference_ms_per_row": np.nan,
        }
        for column, value in defaults.items():
            if column not in frame.columns:
                frame[column] = value
        return frame

    @staticmethod
    def normalize_cv_frame(cv_results: pd.DataFrame) -> pd.DataFrame:
        if cv_results.empty:
            return cv_results
        frame = cv_results.copy()
        for column in ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "roc_auc", "average_precision"]:
            if column not in frame.columns:
                frame[column] = np.nan
        return frame

    @staticmethod
    def add_complexity_tiers(holdout: pd.DataFrame) -> pd.DataFrame:
        frame = holdout.copy()
        required = ["training_time_sec", "inference_ms_per_row", "transformed_feature_count"]
        for column in required:
            values = pd.to_numeric(frame[column], errors="coerce")
            fill_value = float(values.median()) if not values.dropna().empty else 0.0
            frame[column] = values.fillna(fill_value)
        scoring = (
            frame["training_time_sec"].rank(pct=True).fillna(0.5) * 0.5
            + frame["inference_ms_per_row"].rank(pct=True).fillna(0.5) * 0.3
            + frame["transformed_feature_count"].rank(pct=True).fillna(0.5) * 0.2
        )
        frame["complexity_tier"] = pd.cut(
            scoring,
            bins=[-0.01, 0.34, 0.67, 1.01],
            labels=["Low", "Medium", "High"],
        ).astype(str)
        return frame

    @staticmethod
    def sanitize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
        columns = [column for column in schema.get("columns", []) if column.get("name") not in EXCLUDED_FEATURES]
        return {"columns": columns}

    @staticmethod
    def sanitize_examples(examples: pd.DataFrame) -> pd.DataFrame:
        if examples.empty:
            return examples
        return examples.drop(columns=[col for col in EXCLUDED_FEATURES if col in examples.columns], errors="ignore")

    def render_section_header(self, title: str, copy: str) -> None:
        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-title">{title}</div>
                <div class="section-copy">{copy}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def render_overview(
        self,
        holdout: pd.DataFrame,
        cv_results: pd.DataFrame,
        guest_segments: pd.DataFrame,
        metadata: Dict[str, Any],
        raw_data: pd.DataFrame,
    ) -> None:
        self.render_section_header(
            "Performance Overview",
            "This section summarizes the honest test performance, validation consistency, timing behavior, and guest segmentation outputs generated from terminal training.",
        )

        top_model = holdout.sort_values("f1", ascending=False).iloc[0]
        secondary = holdout.sort_values("roc_auc", ascending=False).iloc[0]
        insight_left, insight_right = st.columns(2, gap="large")
        with insight_left:
            st.markdown(
                f"""
                <div class="insight-box">
                    <strong>Best holdout performer</strong>
                    <span>{top_model['model']} leads the honest 30% test split with accuracy {top_model['accuracy']:.4f}, F1 {top_model['f1']:.4f}, and ROC-AUC {top_model['roc_auc']:.4f}.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with insight_right:
            st.markdown(
                f"""
                <div class="insight-box">
                    <strong>Best ranking power</strong>
                    <span>{secondary['model']} shows the strongest class ranking signal with ROC-AUC {secondary['roc_auc']:.4f}. Training and inference times shown below come from the real benchmark run.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if metadata.get("pipeline_wall_clock_note"):
            timing_note = (
                f"{metadata['pipeline_wall_clock_note']} "
                f"Saved pipeline wall clock: {_format_duration(metadata.get('total_pipeline_wall_clock_sec'))}."
            )
        elif metadata.get("total_pipeline_wall_clock_sec") is not None:
            timing_note = (
                f"Total pipeline wall clock for the saved run was {_format_duration(metadata.get('total_pipeline_wall_clock_sec'))}. "
                f"This includes benchmark training, cross-validation, SHAP generation, full-data retraining, and artifact creation."
            )
        else:
            timing_note = (
                "Training time in the tables refers to the saved benchmark run and may not match the total interactive experimentation time."
            )
        st.caption(timing_note)

        left, right = st.columns([1.25, 0.75], gap="large")
        with left:
            st.plotly_chart(self.build_metric_radar(top_model), use_container_width=True)
        with right:
            cv_mean = cv_results[cv_results["fold"].astype(str) == "mean"].sort_values("f1", ascending=False)
            if cv_mean.empty:
                st.info("Cross-validation results are not available for the current artifact set.")
            else:
                st.plotly_chart(self.build_cv_f1_chart(cv_mean), use_container_width=True)

        rnn_reason = metadata.get("skipped_models", {}).get("RNN")
        if rnn_reason:
            st.markdown(
                f"""
                <div class="insight-box">
                    <strong>RNN status</strong>
                    <span>RNN is not included in the current benchmark tables because it was not trainable in this deployment environment. Reason: {rnn_reason}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        chart_left, chart_right = st.columns(2, gap="large")
        with chart_left:
            st.plotly_chart(self.build_metric_heatmap(holdout), use_container_width=True)
        with chart_right:
            st.plotly_chart(self.build_timing_combo_chart(holdout), use_container_width=True)

        if not guest_segments.empty:
            seg_left, seg_right = st.columns([1, 1], gap="large")
            with seg_left:
                self.render_image_card(
                    "Guest Segmentation",
                    "K-Means projection of guest groups built from booking behavior features.",
                    self.artifacts_dir / "plots" / "guest_segmentation.png",
                )
            with seg_right:
                st.markdown("### Segment Summary")
                st.dataframe(guest_segments, use_container_width=True)

        with st.expander("Benchmark Metadata"):
            st.json(metadata)

    def render_model_comparison(
        self,
        holdout: pd.DataFrame,
        cv_results: pd.DataFrame,
        confusion_data: Dict[str, Any],
    ) -> None:
        self.render_section_header(
            "Model Comparison",
            "Compare all trained models side by side using holdout metrics, validation averages, model size, timing, and confusion matrices.",
        )

        metric = st.selectbox(
            "Comparison metric",
            ["f1", "accuracy", "roc_auc", "precision", "recall", "training_time_sec", "inference_ms_per_row"],
            index=0,
            key="comparison_metric",
        )

        left, right = st.columns([1.1, 0.9], gap="large")
        with left:
            st.plotly_chart(self.build_metrics_comparison_chart(holdout), use_container_width=True)
        with right:
            st.plotly_chart(self.build_accuracy_vs_time(holdout), use_container_width=True)

        st.plotly_chart(self.build_holdout_bar(holdout, metric), use_container_width=True)

        styled = holdout.copy()
        styled = styled[
            [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "balanced_accuracy",
                "roc_auc",
                "average_precision",
                "benchmark_training_time_sec",
                "full_data_training_time_sec",
                "training_time_sec",
                "inference_ms_per_row",
                "complexity_tier",
            ]
        ]
        numeric_columns = styled.select_dtypes(include=["number"]).columns
        st.dataframe(
            styled.style.format({column: "{:.4f}" for column in numeric_columns}),
            use_container_width=True,
        )
        st.caption("`benchmark_training_time_sec` is the 70/30 holdout benchmark fit time, `full_data_training_time_sec` is the deployment retrain on the full dataset, and `training_time_sec` is their combined saved run cost. `complexity_tier` is derived from measured training time, inference time, and transformed feature count.")

        cv_mean = cv_results[cv_results["fold"].astype(str) == "mean"].copy()
        if not cv_mean.empty:
            st.markdown("### 5-Fold Validation Means")
            cv_mean = cv_mean[
                [
                    "model",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "balanced_accuracy",
                    "roc_auc",
                    "average_precision",
                ]
            ]
            st.dataframe(
                cv_mean.style.format(
                    {column: "{:.4f}" for column in cv_mean.select_dtypes(include=["number"]).columns}
                ),
                use_container_width=True,
            )
            if "RNN" not in cv_mean["model"].tolist() and "RNN" in holdout["model"].tolist():
                st.caption("RNN appears in the holdout benchmark, but the saved 5-fold validation table does not yet include an RNN row from the TensorFlow runtime.")

        model_options = holdout["model"].tolist()
        selected_confusion = st.selectbox("Confusion matrix model", model_options, index=0)
        if selected_confusion in confusion_data:
            st.plotly_chart(
                self.build_confusion_heatmap(selected_confusion, confusion_data[selected_confusion]),
                use_container_width=True,
            )
        else:
            confusion_name = selected_confusion.lower().replace(" ", "_")
            self.render_image_card(
                f"{selected_confusion} Confusion Matrix",
                "Saved confusion matrix from the holdout evaluation.",
                self.artifacts_dir / "plots" / f"{confusion_name}_confusion_matrix.png",
            )

    def render_explainability(self) -> None:
        self.render_section_header(
            "Explainability",
            "Live explainability is generated after each prediction. The visuals below update from the last booking you scored, so they always reflect that booking's exact features and result.",
        )
        self.render_live_prediction_explainability(section_key="explain")

    def render_prediction_console(
        self,
        models: Dict[str, Any],
        schema: Dict[str, Any],
        examples: pd.DataFrame,
        metadata: Dict[str, Any],
    ) -> None:
        self.render_section_header(
            "Prediction Console",
            "Use the deployed cloud model to score a booking manually. The UI only predicts; all training and benchmarking are loaded from saved terminal artifacts.",
        )
        left, right = st.columns([1.2, 0.8], gap="large")

        with left:
            st.markdown("### Manual Booking Form")
            model_name = st.selectbox("Prediction model", list(models.keys()))
            input_frame = self.render_form(schema)
            if st.button("Predict Cancellation Risk", type="primary", use_container_width=True):
                with st.spinner("Building live prediction and SHAP explanation..."):
                    self.render_prediction(models[model_name], input_frame, model_name, examples)

        with right:
            st.markdown("### Deployment Snapshot")
            deployment_name = metadata.get("deployment_model", metadata.get("best_model", "N/A"))
            st.markdown(
                f"""
                <div class="insight-box">
                    <strong>Cloud deployment model</strong>
                    <span>{deployment_name} is the lightweight model shipped with the app. The benchmark leader may differ when larger ensembles are too heavy for cloud deployment.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if not examples.empty:
                st.markdown("### Example Rows")
                st.dataframe(examples.head(8), use_container_width=True)

        self.render_live_prediction_explainability(section_key="predict")

    def render_segmentation(self, guest_segments: pd.DataFrame) -> None:
        self.render_section_header(
            "Guest Segmentation",
            "K-means clustering groups guests with similar booking behavior. This section shows the saved segmentation plot and the cluster profile table from the training artifacts.",
        )
        left, right = st.columns([1.1, 0.9], gap="large")
        with left:
            self.render_image_card(
                "K-Means Guest Segmentation",
                "Cluster projection built during terminal training from real booking behavior features.",
                self.artifacts_dir / "plots" / "guest_segmentation.png",
            )
        with right:
            if guest_segments.empty:
                st.info("Guest segmentation artifacts are not available yet.")
            else:
                st.plotly_chart(self.build_segmentation_profile_chart(guest_segments), use_container_width=True)
                st.dataframe(guest_segments, use_container_width=True)

    def render_form(self, schema: Dict[str, Any]) -> pd.DataFrame:
        values: Dict[str, Any] = {}
        columns = st.columns(3)
        for index, column in enumerate(schema.get("columns", [])):
            holder = columns[index % 3]
            if column["type"] == "categorical":
                options: List[str] = column["options"]
                default = column["default"]
                default_index = options.index(default) if default in options else 0
                values[column["name"]] = holder.selectbox(
                    column["name"],
                    options=options,
                    index=default_index,
                    key=f"field_{column['name']}",
                )
            else:
                values[column["name"]] = holder.number_input(
                    column["name"],
                    min_value=float(column["min"]),
                    max_value=float(column["max"]),
                    value=float(column["default"]),
                    step=float(column["step"]),
                    key=f"field_{column['name']}",
                )
        return pd.DataFrame([values])

    def render_prediction(
        self,
        model: Any,
        raw_input: pd.DataFrame,
        model_name: str,
        examples: pd.DataFrame,
    ) -> None:
        model_input = self.processor.add_engineered_features(raw_input.copy())

        prediction = int(model.predict(model_input)[0])
        probabilities = _positive_probabilities(model, model_input)
        cancel_probability = float(probabilities[0]) if probabilities is not None else None

        st.divider()
        status_col, metric_col = st.columns([1, 1], gap="large")
        with status_col:
            if prediction == 1:
                st.error(f"{model_name}: this booking is predicted to cancel.")
            else:
                st.success(f"{model_name}: this booking is predicted to stay active.")

        if cancel_probability is not None:
            stay_probability = 1 - cancel_probability
            with metric_col:
                st.metric("Cancellation probability", f"{cancel_probability * 100:.2f}%")
                st.metric("Stay probability", f"{stay_probability * 100:.2f}%")
            risk_text = (
                "Risk is elevated, so this reservation deserves proactive retention handling."
                if cancel_probability >= 0.5
                else "Risk is lower, so this reservation currently looks stable."
            )
            st.info(risk_text)

        # K-Means Segmentation Matching
        segment_path = self.artifacts_dir / "reports" / "guest_segments.csv"
        if segment_path.exists():
            try:
                segments_df = pd.read_csv(segment_path)
                features_to_match = ["lead_time", "adr", "total_nights", "total_guests", "previous_cancellations"]
                
                live_values = {col: float(model_input[col].iloc[0]) if col in model_input.columns else 0.0 for col in features_to_match}
                
                min_dist = float('inf')
                best_segment = -1
                for _, row in segments_df.iterrows():
                    dist = sum((((row[col] - live_values[col]) / max(1.0, row[col])) ** 2) for col in features_to_match if col in row)
                    if dist < min_dist:
                        min_dist = dist
                        best_segment = int(row['segment'])
                
                if best_segment >= 0:
                    st.markdown(f"💡 **Guest Intelligence:** Based on k-means clustering, this booking aligns closely with the behavior profile of **Segment {best_segment}**.")
            except Exception:
                pass

        if examples.empty:
            return

        try:
            analyzer = SHAPAnalyzer()
            background = self.processor.add_engineered_features(
                examples.head(min(80, len(examples))).copy()
            )
            shap_values = analyzer.explain(model, background, model_input, max_background=80)
            feature_names = list(shap_values.feature_names)
            shap_row = np.asarray(shap_values.values[0], dtype=float)
            data_row = np.asarray(shap_values.data[0])
            explanation_frame = pd.DataFrame(
                {
                    "feature": feature_names,
                    "feature_value": data_row,
                    "shap_value": shap_row,
                }
            )
            increasing = explanation_frame[explanation_frame["shap_value"] > 0].nlargest(5, "shap_value")
            decreasing = explanation_frame[explanation_frame["shap_value"] < 0].nsmallest(5, "shap_value")
            st.session_state["latest_prediction"] = {
                "model_name": model_name,
                "prediction": prediction,
                "cancel_probability": cancel_probability,
                "stay_probability": None if cancel_probability is None else 1 - cancel_probability,
                "increasing": increasing.to_dict(orient="records"),
                "decreasing": decreasing.to_dict(orient="records"),
            }
        except Exception as exc:
            st.warning(f"Prediction SHAP explanation is currently unavailable: {exc}")

    def render_image_card(self, title: str, copy: str, path: Path) -> None:
        st.markdown(f"### {title}")
        st.caption(copy)
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            st.warning(f"Missing artifact: {path.name}")

    def render_live_prediction_explainability(self, section_key: str) -> None:
        latest = st.session_state.get("latest_prediction")
        if not latest:
            st.markdown(
                """
                <div class="insight-box">
                    <strong>Live SHAP will appear here</strong>
                    <span>Run a prediction to generate interactive feature explanations showing exactly which features increased cancellation risk and which ones reduced it for that booking.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        probability = latest.get("cancel_probability")
        prediction = latest.get("prediction")
        risk_class = "risk-high" if prediction == 1 else "risk-low"
        headline = "Booking Likely To Cancel" if prediction == 1 else "Booking Likely To Stay"
        subcopy = (
            f"Live explanation for {latest.get('model_name', 'model')}. Cancellation probability: {probability * 100:.2f}%."
            if probability is not None
            else f"Live explanation for {latest.get('model_name', 'model')}."
        )
        st.markdown(
            f"""
            <div class="live-result {risk_class}">
                <h3>{headline}</h3>
                <p>{subcopy}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        gauge_left, gauge_right = st.columns([0.9, 1.1], gap="large")
        with gauge_left:
            if probability is not None:
                st.plotly_chart(
                    self.build_probability_gauge(probability),
                    use_container_width=True,
                    key=f"probability_gauge_{section_key}",
                )
        with gauge_right:
            inc = pd.DataFrame(latest.get("increasing", []))
            dec = pd.DataFrame(latest.get("decreasing", []))
            st.plotly_chart(
                self.build_local_shap_waterfall(inc, dec),
                use_container_width=True,
                key=f"local_shap_waterfall_{section_key}",
            )

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.plotly_chart(
                self.build_local_shap_chart(pd.DataFrame(latest.get("increasing", [])), "Features That Increased Cancellation Risk"),
                use_container_width=True,
                key=f"increase_shap_chart_{section_key}",
            )
            for item in latest.get("increasing", []):
                st.markdown(
                    f"""
                    <div class="reason-card">
                        <strong>{item['feature']}</strong>
                        <span>This feature increased cancellation risk for this booking because the current value was <code>{item['feature_value']}</code> and it pushed the model upward by {float(item['shap_value']):.4f}.</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with col2:
            st.plotly_chart(
                self.build_local_shap_chart(pd.DataFrame(latest.get("decreasing", [])), "Features That Decreased Cancellation Risk"),
                use_container_width=True,
                key=f"decrease_shap_chart_{section_key}",
            )
            for item in latest.get("decreasing", []):
                st.markdown(
                    f"""
                    <div class="reason-card">
                        <strong>{item['feature']}</strong>
                        <span>This feature decreased cancellation risk for this booking because the current value was <code>{item['feature_value']}</code> and it pulled the model downward by {abs(float(item['shap_value'])):.4f}.</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    def build_holdout_bar(self, holdout: pd.DataFrame, metric: str) -> go.Figure:
        chart = holdout.sort_values(metric, ascending=False).copy()
        chart[metric] = pd.to_numeric(chart[metric], errors="coerce").fillna(0.0)
        fig = px.bar(
            chart,
            x="model",
            y=metric,
            color=metric,
            color_continuous_scale=["#c4f1f9", "#4cc9f0", "#0f766e"],
            text_auto=".3f",
        )
        fig.update_layout(
            height=460,
            xaxis_title="Model",
            yaxis_title=metric.replace("_", " ").title(),
            margin=dict(l=10, r=10, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.7)",
        )
        fig.update_traces(marker_line_color="rgba(7,17,31,0.15)", marker_line_width=1.1)
        return fig

    def build_metrics_comparison_chart(self, holdout: pd.DataFrame) -> go.Figure:
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        chart = holdout[["model", *metrics]].copy()
        for metric in metrics:
            chart[metric] = pd.to_numeric(chart[metric], errors="coerce").fillna(0.0)
        long_frame = chart.melt(
            id_vars="model",
            value_vars=metrics,
            var_name="metric",
            value_name="score",
        )
        fig = px.bar(
            long_frame,
            x="model",
            y="score",
            color="metric",
            barmode="group",
            color_discrete_sequence=["#0f766e", "#0ea5e9", "#22c55e", "#f59e0b", "#1d4ed8"],
        )
        fig.update_layout(
            height=500,
            yaxis_title="Score",
            xaxis_title="Model",
            margin=dict(l=10, r=10, t=20, b=20),
            legend_title="Metric",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.7)",
        )
        return fig

    def build_accuracy_vs_time(self, holdout: pd.DataFrame) -> go.Figure:
        chart = holdout.copy()
        chart["training_time_sec"] = pd.to_numeric(chart["training_time_sec"], errors="coerce").fillna(0.0)
        chart["accuracy"] = pd.to_numeric(chart["accuracy"], errors="coerce").fillna(0.0)
        chart["f1"] = pd.to_numeric(chart["f1"], errors="coerce").fillna(0.0)
        chart["roc_auc"] = pd.to_numeric(chart["roc_auc"], errors="coerce").fillna(0.0)
        chart["inference_ms_per_row"] = pd.to_numeric(chart["inference_ms_per_row"], errors="coerce").fillna(0.0)
        fig = px.scatter(
            chart,
            x="training_time_sec",
            y="accuracy",
            color="model",
            symbol="complexity_tier" if "complexity_tier" in chart.columns else None,
            hover_data=["f1", "roc_auc", "inference_ms_per_row"],
        )
        fig.update_layout(
            height=460,
            xaxis_title="Training time (seconds)",
            yaxis_title="Holdout accuracy",
            margin=dict(l=10, r=10, t=20, b=20),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.7)",
        )
        fig.update_traces(marker=dict(line=dict(color="white", width=1.2), opacity=0.9))
        return fig

    def build_cv_f1_chart(self, cv_mean: pd.DataFrame) -> go.Figure:
        fig = px.bar(
            cv_mean,
            x="model",
            y="f1",
            color="f1",
            color_continuous_scale=["#dff7f3", "#34d399", "#0f766e"],
            text_auto=".3f",
        )
        fig.update_layout(
            height=420,
            xaxis_title="Model",
            yaxis_title="Mean 5-fold F1",
            margin=dict(l=10, r=10, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.7)",
        )
        fig.update_traces(marker_line_color="rgba(7,17,31,0.15)", marker_line_width=1.1)
        return fig

    def build_metric_radar(self, top_model: pd.Series) -> go.Figure:
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "balanced_accuracy"]
        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=[float(top_model[metric]) for metric in metrics],
                theta=[metric.replace("_", " ").title() for metric in metrics],
                fill="toself",
                name=str(top_model["model"]),
                line=dict(color="#0f766e", width=3),
                fillcolor="rgba(15, 118, 110, 0.28)",
            )
        )
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=420,
            margin=dict(l=10, r=10, t=20, b=20),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    def build_metric_heatmap(self, holdout: pd.DataFrame) -> go.Figure:
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
        frame = holdout.set_index("model")[metrics].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        fig = px.imshow(
            frame,
            text_auto=".3f",
            aspect="auto",
            color_continuous_scale=["#f8fafc", "#99f6e4", "#0f766e"],
        )
        fig.update_layout(
            title="Metric Comparison Heatmap",
            height=480,
            margin=dict(l=10, r=10, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    def build_timing_combo_chart(self, holdout: pd.DataFrame) -> go.Figure:
        chart = holdout.sort_values("f1", ascending=False).copy()
        chart["training_time_sec"] = pd.to_numeric(chart["training_time_sec"], errors="coerce").fillna(0.0)
        chart["inference_ms_per_row"] = pd.to_numeric(chart["inference_ms_per_row"], errors="coerce").fillna(0.0)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=chart["model"],
                y=chart["training_time_sec"],
                name="Training time (sec)",
                marker_color="#0ea5e9",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=chart["model"],
                y=chart["inference_ms_per_row"],
                mode="lines+markers+text",
                name="Inference ms/row",
                marker=dict(color="#f59e0b", size=10),
                line=dict(color="#f59e0b", width=3),
                text=[f"{value:.3f}" for value in chart["inference_ms_per_row"]],
                textposition="top center",
            ),
            secondary_y=True,
        )
        fig.update_layout(
            title="Measured Training vs Inference Cost",
            height=480,
            margin=dict(l=10, r=10, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.7)",
        )
        fig.update_yaxes(title_text="Training time (sec)", secondary_y=False)
        fig.update_yaxes(title_text="Inference ms/row", secondary_y=True)
        return fig

    def build_confusion_heatmap(self, model_name: str, payload: Dict[str, Any]) -> go.Figure:
        matrix = np.asarray(payload["matrix"])
        fig = px.imshow(
            matrix,
            text_auto=True,
            x=payload.get("predicted", ["Predicted 0", "Predicted 1"]),
            y=payload.get("labels", ["Actual 0", "Actual 1"]),
            color_continuous_scale=["#eff6ff", "#38bdf8", "#0f766e"],
            aspect="auto",
        )
        fig.update_layout(
            title=f"{model_name} Confusion Matrix",
            height=430,
            margin=dict(l=10, r=10, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    def build_global_shap_importance_chart(self, metadata: Dict[str, Any]) -> go.Figure:
        frame = pd.DataFrame(metadata.get("shap_explanations", []))
        if frame.empty:
            frame = pd.DataFrame(
                {"feature": ["No saved SHAP data"], "mean_abs_shap": [0.0]}
            )
        fig = px.bar(
            frame.sort_values("mean_abs_shap", ascending=True),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            color="mean_abs_shap",
            color_continuous_scale=["#dbeafe", "#38bdf8", "#0f766e"],
            text_auto=".3f",
        )
        fig.update_layout(
            height=420,
            xaxis_title="Mean |SHAP|",
            yaxis_title="Feature",
            margin=dict(l=10, r=10, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.7)",
            showlegend=False,
        )
        return fig

    def build_local_shap_chart(self, frame: pd.DataFrame, title: str) -> go.Figure:
        if frame.empty:
            frame = pd.DataFrame({"feature": ["No features"], "shap_value": [0.0]})
        plot_frame = frame.sort_values("shap_value")
        fig = px.bar(
            plot_frame,
            x="shap_value",
            y="feature",
            orientation="h",
            color="shap_value",
            color_continuous_scale=["#22c55e", "#e2e8f0", "#ef4444"],
            text_auto=".3f",
        )
        fig.update_layout(
            title=title,
            height=380,
            xaxis_title="SHAP contribution",
            yaxis_title="Feature",
            margin=dict(l=10, r=10, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.7)",
            showlegend=False,
        )
        return fig

    def build_probability_gauge(self, probability: float) -> go.Figure:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%", "font": {"size": 36, "color": "#0f172a"}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#0ea5e9"},
                    "steps": [
                        {"range": [0, 35], "color": "#dcfce7"},
                        {"range": [35, 65], "color": "#fef3c7"},
                        {"range": [65, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {"line": {"color": "#ef4444", "width": 4}, "thickness": 0.8, "value": probability * 100},
                },
                title={"text": "Cancellation Probability"},
            )
        )
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
        return fig

    def build_segmentation_profile_chart(self, guest_segments: pd.DataFrame) -> go.Figure:
        value_columns = [column for column in guest_segments.columns if column != "segment"]
        frame = guest_segments.melt(id_vars="segment", value_vars=value_columns, var_name="feature", value_name="value")
        fig = px.bar(
            frame,
            x="feature",
            y="value",
            color="segment",
            barmode="group",
            color_discrete_sequence=["#0f766e", "#0ea5e9", "#f59e0b", "#ef4444"],
        )
        fig.update_layout(
            title="Cluster Profiles",
            height=420,
            margin=dict(l=10, r=10, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.7)",
            xaxis_title="Feature",
            yaxis_title="Average value",
        )
        return fig

    def build_local_shap_waterfall(self, increasing: pd.DataFrame, decreasing: pd.DataFrame) -> go.Figure:
        inc = increasing.copy() if not increasing.empty else pd.DataFrame(columns=["feature", "shap_value"])
        dec = decreasing.copy() if not decreasing.empty else pd.DataFrame(columns=["feature", "shap_value"])
        frame = pd.concat([dec, inc], ignore_index=True)
        if frame.empty:
            frame = pd.DataFrame({"feature": ["No features"], "shap_value": [0.0]})
        frame["label"] = frame["feature"].astype(str)
        fig = go.Figure(
            go.Waterfall(
                orientation="v",
                measure=["relative"] * len(frame),
                x=frame["label"],
                y=frame["shap_value"],
                connector={"line": {"color": "rgba(15,23,42,0.18)"}},
                increasing={"marker": {"color": "#ef4444"}},
                decreasing={"marker": {"color": "#22c55e"}},
            )
        )
        fig.update_layout(
            title="Live SHAP Contribution Flow",
            height=380,
            margin=dict(l=10, r=10, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.7)",
            xaxis_title="Feature",
            yaxis_title="Contribution to risk",
        )
        return fig


if __name__ == "__main__":
    PredictionApp().run()
