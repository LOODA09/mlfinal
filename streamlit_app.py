from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from hotel_cancellation_oop import HotelDataProcessor, _positive_probabilities


ARTIFACTS_DIR = Path("artifacts")
DATA_PATH = Path("hotel_bookings.csv")


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
                padding: 34px 34px 30px;
                border-radius: 30px;
                margin-bottom: 22px;
                color: white;
                background: linear-gradient(135deg, #07111f 0%, #0f3d56 46%, #0f766e 100%);
                box-shadow: 0 34px 90px rgba(7, 17, 31, 0.20);
                animation: fadeLift .8s ease both;
            }

            .hero-shell::before,
            .hero-shell::after {
                content: "";
                position: absolute;
                border-radius: 999px;
                opacity: .55;
                filter: blur(10px);
            }

            .hero-shell::before {
                width: 280px;
                height: 280px;
                right: -70px;
                top: -85px;
                background: rgba(250, 204, 21, .18);
                animation: floatA 9s ease-in-out infinite alternate;
            }

            .hero-shell::after {
                width: 220px;
                height: 220px;
                left: -55px;
                bottom: -90px;
                background: rgba(255, 255, 255, .11);
                animation: floatB 12s ease-in-out infinite alternate;
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
            </style>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def hero(metadata: Dict[str, Any]) -> None:
        st.markdown(
            f"""
            <div class="hero-shell">
                <div class="hero-topline">Benchmark Dashboard • Prediction Console • SHAP Explainability</div>
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
            "Rows Used": f"{metadata.get('train_rows', 0) + metadata.get('test_rows', 0):,}",
        }
        html = "".join(
            f'<div class="metric-tile"><span>{label}</span><strong>{value}</strong></div>'
            for label, value in cards.items()
        )
        st.markdown(f'<div class="metrics-ribbon">{html}</div>', unsafe_allow_html=True)


class PredictionApp:
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
        self.artifacts_dir = artifacts_dir
        self.processor = HotelDataProcessor()

    @st.cache_data(show_spinner=False)
    def load_json(_self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    @st.cache_data(show_spinner=False)
    def load_csv(_self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    @st.cache_data(show_spinner=False)
    def load_raw_data(_self, path: Path) -> pd.DataFrame:
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

    def run(self) -> None:
        st.set_page_config(page_title="Hotel Cancellation Intelligence", layout="wide")
        DashboardStyle.apply()

        metadata = self.load_json(self.artifacts_dir / "reports" / "metadata.json", {})
        schema = self.load_json(self.artifacts_dir / "reports" / "prediction_schema.json", {"columns": []})
        examples = self.load_csv(self.artifacts_dir / "reports" / "prediction_examples.csv")
        holdout = self.load_csv(self.artifacts_dir / "reports" / "holdout_summary.csv")
        cv_results = self.load_csv(self.artifacts_dir / "reports" / "cross_validation_results.csv")
        guest_segments = self.load_csv(self.artifacts_dir / "reports" / "guest_segments.csv")
        raw_data = self.load_raw_data(DATA_PATH)
        models = self.load_models(self.artifacts_dir)

        if holdout.empty or not schema.get("columns"):
            st.error("Saved training artifacts are missing. Run terminal training and redeploy the app.")
            return

        DashboardStyle.hero(metadata)

        overview_tab, models_tab, explain_tab, predict_tab = st.tabs(
            ["Overview", "Model Comparison", "Explainability", "Prediction"]
        )

        with overview_tab:
            self.render_overview(holdout, cv_results, guest_segments, metadata, raw_data)

        with models_tab:
            self.render_model_comparison(holdout, cv_results)

        with explain_tab:
            self.render_explainability(metadata, raw_data)

        with predict_tab:
            self.render_prediction_console(models, schema, examples, metadata)

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

        left, right = st.columns([1.25, 0.75], gap="large")
        with left:
            st.plotly_chart(self.build_metric_radar(top_model), use_container_width=True)
        with right:
            cv_mean = cv_results[cv_results["fold"].astype(str) == "mean"].sort_values("f1", ascending=False)
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

    def render_model_comparison(self, holdout: pd.DataFrame, cv_results: pd.DataFrame) -> None:
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
            st.plotly_chart(self.build_holdout_bar(holdout, metric), use_container_width=True)
        with right:
            st.plotly_chart(self.build_accuracy_vs_time(holdout), use_container_width=True)

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
                "training_time_sec",
                "inference_ms_per_row",
                "complexity_score",
                "model_size_mb",
            ]
        ].rename(
            columns={
                "complexity_score": "complexity_proxy",
                "training_time_sec": "training_time_sec",
                "inference_ms_per_row": "inference_ms_per_row",
            }
        )
        numeric_columns = styled.select_dtypes(include=["number"]).columns
        st.dataframe(
            styled.style.format({column: "{:.4f}" for column in numeric_columns}),
            use_container_width=True,
        )
        st.caption("`complexity_proxy` is a heuristic size/structure measure derived from the trained estimator, not a universal complexity theorem.")

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

        model_options = holdout["model"].tolist()
        selected_confusion = st.selectbox("Confusion matrix model", model_options, index=0)
        confusion_name = selected_confusion.lower().replace(" ", "_")
        self.render_image_card(
            f"{selected_confusion} Confusion Matrix",
            "Saved confusion matrix from the holdout evaluation.",
            self.artifacts_dir / "plots" / f"{confusion_name}_confusion_matrix.png",
        )

    def render_explainability(self, metadata: Dict[str, Any], raw_data: pd.DataFrame) -> None:
        self.render_section_header(
            "Explainability",
            "SHAP plots explain which saved features most increased or decreased cancellation risk for the benchmarked model.",
        )

        clean = self.processor.clean_data(raw_data) if not raw_data.empty else pd.DataFrame()
        deposit_rates = (
            clean.groupby("deposit_type")["is_canceled"].mean().sort_values(ascending=False)
            if not clean.empty and {"deposit_type", "is_canceled"}.issubset(clean.columns)
            else pd.Series(dtype=float)
        )
        adr_summary = (
            clean.groupby("is_canceled")["adr"].mean()
            if not clean.empty and {"adr", "is_canceled"}.issubset(clean.columns)
            else pd.Series(dtype=float)
        )
        adr_corr = (
            clean[["adr", "is_canceled"]].corr().iloc[0, 1]
            if not clean.empty and {"adr", "is_canceled"}.issubset(clean.columns)
            else None
        )

        summary_left, summary_right = st.columns([1.1, 0.9], gap="large")
        with summary_left:
            self.render_image_card(
                "SHAP Summary Plot",
                "Global feature importance view. Features with wider SHAP spread have a stronger effect on cancellation decisions.",
                self.artifacts_dir / "plots" / "random_forest_shap_summary.png",
            )
        with summary_right:
            st.markdown("### SHAP Feature Notes")
            for item in metadata.get("shap_explanations", []):
                st.markdown(
                    f"""
                    <div class="insight-box">
                        <strong>{item['feature']}</strong>
                        <span>{item['explanation']} Mean absolute SHAP impact: {item['mean_abs_shap']:.4f}.</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if not deposit_rates.empty:
                st.markdown(
                    f"""
                    <div class="insight-box">
                        <strong>Observed deposit-type cancellation rates</strong>
                        <span>Non Refund: {deposit_rates.get('Non Refund', float('nan')) * 100:.2f}% canceled. No Deposit: {deposit_rates.get('No Deposit', float('nan')) * 100:.2f}% canceled. Refundable: {deposit_rates.get('Refundable', float('nan')) * 100:.2f}% canceled. These are empirical rates from the dataset, not direct SHAP probabilities.</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if adr_corr is not None and not adr_summary.empty:
                st.markdown(
                    f"""
                    <div class="insight-box">
                        <strong>ADR reality check</strong>
                        <span>ADR is not a dominant saved SHAP driver here. In the raw dataset, canceled bookings have mean ADR {adr_summary.get(1, float('nan')):.2f} versus {adr_summary.get(0, float('nan')):.2f} for non-canceled bookings, with a weak positive correlation of {adr_corr:.4f} to cancellation. So lower ADR does not appear to increase cancellation probability overall in this dataset.</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        dep_left, dep_right = st.columns(2, gap="large")
        with dep_left:
            self.render_image_card(
                "Country Effect",
                "Bookings tagged with this country indicator tended to push risk upward in the saved SHAP dependence analysis.",
                self.artifacts_dir / "plots" / "random_forest_shap_country_prt.png",
            )
            self.render_image_card(
                "No Deposit Effect",
                "This plot shows why no-deposit bookings tended to reduce risk relative to more restrictive deposit profiles.",
                self.artifacts_dir / "plots" / "random_forest_shap_deposit_type_no_deposit.png",
            )
        with dep_right:
            self.render_image_card(
                "Non-Refund Deposit Effect",
                "This dependence plot shows why non-refundable deposit type pushed many predictions toward higher cancellation risk.",
                self.artifacts_dir / "plots" / "random_forest_shap_deposit_type_non_refund.png",
            )

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
                self.render_prediction(models[model_name], input_frame, model_name)

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

    def render_image_card(self, title: str, copy: str, path: Path) -> None:
        st.markdown(f"### {title}")
        st.caption(copy)
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            st.warning(f"Missing artifact: {path.name}")

    def build_holdout_bar(self, holdout: pd.DataFrame, metric: str) -> go.Figure:
        chart = holdout.sort_values(metric, ascending=False).copy()
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

    def build_accuracy_vs_time(self, holdout: pd.DataFrame) -> go.Figure:
        fig = px.scatter(
            holdout,
            x="training_time_sec",
            y="accuracy",
            size="model_size_mb",
            color="model",
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
        frame = holdout.set_index("model")[metrics]
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


if __name__ == "__main__":
    PredictionApp().run()
