from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
import streamlit as st

from hotel_cancellation_oop import HotelDataProcessor, _positive_probabilities


ARTIFACTS_DIR = Path("artifacts")


class PredictionStyle:
    @staticmethod
    def apply() -> None:
        st.markdown(
            """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&display=swap');

            html, body, [class*="css"] {
                font-family: 'Manrope', sans-serif;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(13, 148, 136, 0.12), transparent 30%),
                    radial-gradient(circle at 85% 10%, rgba(245, 158, 11, 0.16), transparent 26%),
                    linear-gradient(180deg, #f8fafc 0%, #eef6f5 100%);
            }

            .hero {
                position: relative;
                overflow: hidden;
                padding: 34px 32px;
                border-radius: 28px;
                margin-bottom: 22px;
                color: white;
                background: linear-gradient(135deg, #0f172a 0%, #155e75 52%, #0f766e 100%);
                box-shadow: 0 30px 80px rgba(15, 23, 42, 0.18);
                animation: rise .65s ease both;
            }

            .hero::before, .hero::after {
                content: "";
                position: absolute;
                border-radius: 999px;
                filter: blur(8px);
                opacity: .55;
                animation: drift 10s ease-in-out infinite alternate;
            }

            .hero::before {
                width: 240px;
                height: 240px;
                right: -60px;
                top: -60px;
                background: rgba(250, 204, 21, .22);
            }

            .hero::after {
                width: 180px;
                height: 180px;
                left: -40px;
                bottom: -70px;
                background: rgba(255,255,255,.14);
            }

            .hero h1, .hero p {
                position: relative;
                z-index: 2;
            }

            .hero h1 {
                margin: 0 0 10px;
                font-size: 2.4rem;
                line-height: 1.05;
            }

            .hero p {
                margin: 0;
                max-width: 820px;
                font-size: 1rem;
                color: rgba(255,255,255,.88);
                line-height: 1.6;
            }

            .glass-card {
                background: rgba(255,255,255,.82);
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 22px;
                padding: 18px 18px 10px;
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
                backdrop-filter: blur(10px);
                animation: rise .55s ease both;
            }

            .mini-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 12px;
                margin: 14px 0 12px;
            }

            .mini-card {
                border-radius: 18px;
                padding: 16px;
                background: linear-gradient(180deg, rgba(255,255,255,.94), rgba(241,245,249,.96));
                border: 1px solid rgba(15, 23, 42, 0.08);
            }

            .mini-card span {
                display: block;
                color: #475569;
                font-size: .8rem;
                margin-bottom: 6px;
            }

            .mini-card strong {
                color: #0f172a;
                font-size: 1.1rem;
            }

            div.stButton > button {
                min-height: 48px;
                border-radius: 14px;
                border: 0;
                font-weight: 800;
                background: linear-gradient(90deg, #0f766e, #0891b2);
                color: white;
                transition: transform .15s ease, box-shadow .15s ease;
                box-shadow: 0 18px 28px rgba(8, 145, 178, .2);
            }

            div.stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 22px 32px rgba(8, 145, 178, .28);
            }

            @keyframes rise {
                from { opacity: 0; transform: translateY(12px); }
                to { opacity: 1; transform: translateY(0); }
            }

            @keyframes drift {
                from { transform: translate3d(0, 0, 0); }
                to { transform: translate3d(18px, -12px, 0); }
            }

            @media (max-width: 900px) {
                .mini-grid {
                    grid-template-columns: 1fr;
                }
                .hero h1 {
                    font-size: 1.8rem;
                }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def hero() -> None:
        st.markdown(
            """
            <div class="hero">
                <h1>Hotel Cancellation Prediction Studio</h1>
                <p>
                    This app is prediction-only. Training, testing, cross-validation, SHAP reports,
                    confusion matrices, and segmentation are generated from the terminal and loaded here
                    as saved artifacts for a cleaner production workflow.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


class PredictionApp:
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
        self.artifacts_dir = artifacts_dir
        self.processor = HotelDataProcessor()

    @st.cache_data(show_spinner=False)
    def load_metadata(_self, artifacts_dir: Path) -> Dict[str, Any]:
        path = artifacts_dir / "reports" / "metadata.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @st.cache_data(show_spinner=False)
    def load_schema(_self, artifacts_dir: Path) -> Dict[str, Any]:
        path = artifacts_dir / "reports" / "prediction_schema.json"
        if not path.exists():
            return {"columns": []}
        return json.loads(path.read_text(encoding="utf-8"))

    @st.cache_data(show_spinner=False)
    def load_examples(_self, artifacts_dir: Path) -> pd.DataFrame:
        path = artifacts_dir / "reports" / "prediction_examples.csv"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    @st.cache_resource(show_spinner=False)
    def load_models(_self, artifacts_dir: Path) -> Dict[str, Any]:
        models: Dict[str, Any] = {}
        models_dir = artifacts_dir / "models"
        if not models_dir.exists():
            return models
        deployment_path = models_dir / "deployment_model.joblib"
        if deployment_path.exists():
            models["Deployment Model"] = joblib.load(deployment_path)
            return models
        for model_path in sorted(models_dir.glob("*.joblib")):
            if model_path.name == "best_model.joblib":
                continue
            models[model_path.stem.replace("_", " ").title()] = joblib.load(model_path)
        return models

    def run(self) -> None:
        st.set_page_config(page_title="Hotel Cancellation Prediction", layout="wide")
        PredictionStyle.apply()
        PredictionStyle.hero()

        metadata = self.load_metadata(self.artifacts_dir)
        schema = self.load_schema(self.artifacts_dir)
        examples = self.load_examples(self.artifacts_dir)
        models = self.load_models(self.artifacts_dir)

        if not models or not schema.get("columns"):
            st.error(
                "No trained artifacts were found. Run `python train_terminal.py` in the terminal first."
            )
            return

        left, right = st.columns([1.25, 0.75], gap="large")
        with left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Prediction Form")
            selected_model = st.selectbox("Trained model", list(models.keys()))
            input_frame = self.render_form(schema)

            if st.button("Predict Cancellation Risk", type="primary", use_container_width=True):
                self.render_prediction(models[selected_model], input_frame, selected_model)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Model Snapshot")
            self.render_snapshot(metadata, selected_model)
            if not examples.empty:
                with st.expander("Sample Training Rows Used For Form Defaults"):
                    st.dataframe(examples.head(10), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    def render_snapshot(self, metadata: Dict[str, Any], selected_model: str) -> None:
        trained_models = metadata.get("trained_models", [])
        st.markdown(
            f"""
            <div class="mini-grid">
                <div class="mini-card"><span>Best model</span><strong>{metadata.get('best_model', 'N/A')}</strong></div>
                <div class="mini-card"><span>Cloud model</span><strong>{metadata.get('deployment_model', metadata.get('best_model', 'N/A'))}</strong></div>
                <div class="mini-card"><span>Train / Test split</span><strong>70% / 30%</strong></div>
                <div class="mini-card"><span>Cross-validation</span><strong>{metadata.get('cross_validation_folds', 'N/A')}-fold</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Loaded models: {', '.join(trained_models) if trained_models else selected_model}")
        shap_notes = metadata.get("shap_explanations", [])
        if shap_notes:
            st.write("Risk drivers from the saved SHAP analysis")
            for item in shap_notes:
                st.markdown(f"- `{item['feature']}`: {item['explanation']}")

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

    def render_prediction(self, model: Any, raw_input: pd.DataFrame, model_name: str) -> None:
        model_input = self.processor.add_engineered_features(raw_input.copy())
        prediction = int(model.predict(model_input)[0])
        probabilities = _positive_probabilities(model, model_input)
        cancel_probability = float(probabilities[0]) if probabilities is not None else None

        st.divider()
        if prediction == 1:
            st.error(f"{model_name}: this booking is predicted to cancel.")
        else:
            st.success(f"{model_name}: this booking is predicted to stay active.")

        if cancel_probability is not None:
            stay_probability = 1 - cancel_probability
            metric_left, metric_right = st.columns(2)
            metric_left.metric("Cancellation probability", f"{cancel_probability * 100:.2f}%")
            metric_right.metric("Stay probability", f"{stay_probability * 100:.2f}%")

            risk_text = (
                "Risk is elevated, so this reservation deserves proactive retention handling."
                if cancel_probability >= 0.5
                else "Risk is lower, so this reservation currently looks stable."
            )
            st.info(risk_text)


if __name__ == "__main__":
    PredictionApp().run()
