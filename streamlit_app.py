from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from hotel_cancellation_oop import (
    ANNModel,
    BaseHotelModel,
    MODEL_REGISTRY,
    ModelTester,
    ModelTrainer,
    NotebookEDAAnalyzer,
    RNNModel,
    SHAPAnalyzer,
    _positive_probabilities,
)


@dataclass
class DashboardConfig:
    data_path: str
    sample_size: int
    test_size: float
    folds: int
    remove_leakage: bool
    epochs: int
    model_names: List[str]


class StreamlitStyle:
    @staticmethod
    def apply() -> None:
        st.markdown(
            """
            <style>
            .stApp {
                background: #f5f7fb;
            }

            .main-header {
                position: relative;
                overflow: hidden;
                border-radius: 16px;
                padding: 28px 30px;
                margin-bottom: 18px;
                background: linear-gradient(135deg, #101828 0%, #184e77 48%, #00a896 100%);
                color: white;
                box-shadow: 0 22px 50px rgba(16, 24, 40, 0.18);
            }

            .main-header:after {
                content: "";
                position: absolute;
                inset: -80px;
                background:
                    radial-gradient(circle at 20% 30%, rgba(255,255,255,.30), transparent 22%),
                    radial-gradient(circle at 72% 45%, rgba(255,255,255,.18), transparent 20%),
                    radial-gradient(circle at 50% 80%, rgba(255,255,255,.12), transparent 18%);
                animation: floatLights 8s ease-in-out infinite alternate;
            }

            .main-header-content {
                position: relative;
                z-index: 2;
            }

            .main-header h1 {
                font-size: 2.15rem;
                line-height: 1.1;
                margin: 0 0 10px;
                letter-spacing: 0;
            }

            .main-header p {
                max-width: 800px;
                margin: 0;
                font-size: 1rem;
                line-height: 1.55;
                color: rgba(255,255,255,.88);
            }

            .moving-line {
                position: absolute;
                left: 0;
                right: 0;
                bottom: 0;
                height: 4px;
                background: linear-gradient(90deg, #ffbe0b, #fb5607, #ff006e, #8338ec, #3a86ff);
                background-size: 300% 100%;
                animation: colorRun 4s linear infinite;
                z-index: 3;
            }

            .soft-card {
                border: 1px solid rgba(16, 24, 40, 0.08);
                border-radius: 14px;
                background: rgba(255,255,255,.86);
                padding: 18px;
                box-shadow: 0 14px 34px rgba(16, 24, 40, 0.07);
                animation: slideUp .45s ease both;
            }

            .metric-row {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 12px;
                margin: 16px 0;
            }

            .metric-box {
                border-radius: 14px;
                padding: 16px;
                background: white;
                border: 1px solid rgba(16, 24, 40, 0.08);
                box-shadow: 0 10px 24px rgba(16, 24, 40, 0.06);
                animation: slideUp .5s ease both;
            }

            .metric-box span {
                display: block;
                color: #667085;
                font-size: .82rem;
                margin-bottom: 6px;
            }

            .metric-box strong {
                color: #101828;
                font-size: 1.35rem;
            }

            div.stButton > button {
                border-radius: 12px;
                min-height: 46px;
                font-weight: 700;
                transition: transform .16s ease, box-shadow .16s ease;
            }

            div.stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 12px 24px rgba(24, 78, 119, .18);
            }

            @keyframes floatLights {
                from { transform: translate3d(-20px, -10px, 0) rotate(0deg); }
                to { transform: translate3d(18px, 12px, 0) rotate(8deg); }
            }

            @keyframes colorRun {
                from { background-position: 0 0; }
                to { background-position: 300% 0; }
            }

            @keyframes slideUp {
                from { opacity: 0; transform: translateY(12px); }
                to { opacity: 1; transform: translateY(0); }
            }

            @media (max-width: 900px) {
                .metric-row {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }

            @media (max-width: 560px) {
                .metric-row {
                    grid-template-columns: 1fr;
                }
                .main-header h1 {
                    font-size: 1.55rem;
                }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def header() -> None:
        st.markdown(
            """
            <div class="main-header">
                <div class="main-header-content">
                    <h1>Hotel Cancellation Intelligence</h1>
                    <p>
                        A Streamlit dashboard built from the notebook workflow:
                        exploratory analysis, model training, holdout testing,
                        k-fold validation, and SHAP explanations.
                    </p>
                </div>
                <div class="moving-line"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def metric_row(values: Dict[str, str]) -> None:
        cards = "".join(
            f"""
            <div class="metric-box">
                <span>{label}</span>
                <strong>{value}</strong>
            </div>
            """
            for label, value in values.items()
        )
        st.markdown(f'<div class="metric-row">{cards}</div>', unsafe_allow_html=True)


class HotelCancellationDashboard:
    def __init__(self) -> None:
        self.trainer = ModelTrainer()
        self.tester = ModelTester()

    def run(self) -> None:
        st.set_page_config(page_title="Hotel Cancellation Dashboard", layout="wide")
        StreamlitStyle.apply()
        StreamlitStyle.header()

        config = self.sidebar()
        self.trainer = ModelTrainer(test_size=config.test_size)

        raw_data = self.load_raw_data(config.data_path, config.sample_size)
        x_data, y_data = self.load_features(
            config.data_path,
            config.sample_size,
            config.remove_leakage,
        )

        StreamlitStyle.metric_row(
            {
                "Rows": f"{len(x_data):,}",
                "Features": f"{x_data.shape[1]:,}",
                "Cancellation Rate": f"{y_data.mean() * 100:.1f}%",
                "Models": f"{len(config.model_names):,}",
            }
        )

        self.analysis_section(raw_data)
        self.model_section(config, x_data, y_data)

    def sidebar(self) -> DashboardConfig:
        with st.sidebar:
            st.header("Configuration")
            data_path = st.text_input("Dataset", "hotel_bookings.csv")
            sample_size = st.number_input(
                "Sample size",
                min_value=0,
                value=0,
                step=1000,
                help="Use 0 for the full dataset.",
            )
            remove_leakage = st.checkbox("Remove leakage features", value=True)
            test_size = st.slider("Test size", 0.1, 0.5, 0.3, 0.05)
            folds = st.slider("K-fold splits", 2, 10, 5)
            epochs = st.slider("ANN/RNN epochs", 1, 60, 8)
            model_names = st.multiselect(
                "Models",
                list(MODEL_REGISTRY.keys()),
                default=["Decision Tree"],
            )

        return DashboardConfig(
            data_path=data_path,
            sample_size=int(sample_size),
            test_size=float(test_size),
            folds=int(folds),
            remove_leakage=remove_leakage,
            epochs=int(epochs),
            model_names=model_names,
        )

    @staticmethod
    @st.cache_data(show_spinner=False)
    def load_raw_data(path: str, sample_size: int) -> pd.DataFrame:
        data = pd.read_csv(path)
        if sample_size and sample_size < len(data):
            return data.sample(sample_size, random_state=42)
        return data

    @staticmethod
    @st.cache_data(show_spinner=False)
    def load_features(path: str, sample_size: int, remove_leakage: bool):
        trainer = ModelTrainer()
        return trainer.prepare_data(
            data_path=path,
            sample_size=None if sample_size == 0 else sample_size,
            remove_leakage_features=remove_leakage,
        )

    def build_models(self, config: DashboardConfig) -> List[BaseHotelModel]:
        models: List[BaseHotelModel] = []
        for model_name in config.model_names:
            if model_name == "ANN":
                models.append(ANNModel(epochs=config.epochs))
            elif model_name == "RNN":
                models.append(RNNModel(epochs=config.epochs))
            else:
                models.append(MODEL_REGISTRY[model_name]())
        return models

    def analysis_section(self, raw_data: pd.DataFrame) -> None:
        analyzer = NotebookEDAAnalyzer(raw_data)

        st.subheader("Notebook Analysis")
        overview, demand, stay, correlation = st.tabs(
            ["Overview", "Demand", "Stay Length", "Correlation"]
        )

        with overview:
            left, right = st.columns([1, 1])
            with left:
                st.dataframe(analyzer.preview(), use_container_width=True)
            with right:
                st.dataframe(analyzer.null_summary(), use_container_width=True)
            c1, c2 = st.columns(2)
            c1.metric("Negative ADR Rows", len(analyzer.negative_adr_rows()))
            c2.metric("No-Guest Bookings", len(analyzer.empty_guest_bookings()))

        with demand:
            st.plotly_chart(analyzer.country_choropleth(), use_container_width=True)
            st.plotly_chart(analyzer.room_price_figure(), use_container_width=True)
            st.plotly_chart(analyzer.monthly_guest_figure(), use_container_width=True)

        with stay:
            st.plotly_chart(analyzer.stay_distribution_figure(), use_container_width=True)
            st.dataframe(analyzer.stay_distribution(), use_container_width=True)

        with correlation:
            st.dataframe(
                analyzer.target_correlation().to_frame("absolute_correlation"),
                use_container_width=True,
            )
            st.pyplot(analyzer.correlation_heatmap())

    def model_section(self, config: DashboardConfig, x_data: pd.DataFrame, y_data: pd.Series) -> None:
        st.subheader("Model Lab")
        train_col, fold_col = st.columns(2)

        if train_col.button("Train Selected Models", type="primary", use_container_width=True):
            self.train_selected_models(config, x_data, y_data)

        if fold_col.button("Run K-Fold Validation", use_container_width=True):
            self.run_cross_validation(config, x_data, y_data)

        self.render_holdout_results()
        self.render_kfold_results()
        self.prediction_section(x_data)

    def train_selected_models(
        self,
        config: DashboardConfig,
        x_data: pd.DataFrame,
        y_data: pd.Series,
    ) -> None:
        if not config.model_names:
            st.warning("Select at least one model.")
            return

        try:
            with st.spinner("Training models..."):
                x_train, x_test, y_train, y_test = self.trainer.split_data(x_data, y_data)
                holdout_specs = self.build_models(config)
                models = self.trainer.train_many(holdout_specs, x_train, y_train)
                summary, details = self.tester.test_many(models, x_test, y_test)
                final_models = self.trainer.train_many(self.build_models(config), x_data, y_data)
                full_fit_summary, _ = self.tester.test_many(final_models, x_data, y_data)

            st.session_state["trained_models"] = models
            st.session_state["final_prediction_models"] = final_models
            st.session_state["x_train"] = x_train
            st.session_state["x_test"] = x_test
            st.session_state["feature_template"] = x_data
            st.session_state["holdout_summary"] = summary
            st.session_state["full_fit_summary"] = full_fit_summary
            st.session_state["holdout_details"] = details
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    def run_cross_validation(
        self,
        config: DashboardConfig,
        x_data: pd.DataFrame,
        y_data: pd.Series,
    ) -> None:
        if not config.model_names:
            st.warning("Select at least one model.")
            return

        try:
            with st.spinner("Running k-fold validation..."):
                results = self.trainer.k_fold_cross_validate(
                    self.build_models(config),
                    x_data,
                    y_data,
                    n_splits=config.folds,
                )
            st.session_state["kfold_results"] = results
        except Exception as exc:
            st.error(f"K-fold validation failed: {exc}")

    def render_holdout_results(self) -> None:
        if "holdout_summary" not in st.session_state:
            return

        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.write("Holdout Evaluation")
        st.dataframe(st.session_state["holdout_summary"].style.format(precision=4), use_container_width=True)

        if "full_fit_summary" in st.session_state:
            st.write("Full-Data In-Sample Fit")
            st.caption("This is scored on the same rows used for training. It is not an honest unseen test.")
            st.dataframe(st.session_state["full_fit_summary"].style.format(precision=4), use_container_width=True)

        selected = st.selectbox("Inspect model", list(st.session_state["trained_models"].keys()))
        detail = st.session_state["holdout_details"][selected]

        left, right = st.columns([1, 2])
        with left:
            st.dataframe(
                pd.DataFrame(
                    detail["confusion_matrix"],
                    index=["Actual 0", "Actual 1"],
                    columns=["Predicted 0", "Predicted 1"],
                ),
                use_container_width=True,
            )
        with right:
            st.dataframe(pd.DataFrame(detail["classification_report"]).T, use_container_width=True)

        self.shap_section(selected)
        st.markdown("</div>", unsafe_allow_html=True)

    def shap_section(self, selected_model: str) -> None:
        st.write("SHAP Explanation")
        row_count = st.slider("Rows for SHAP", 5, 75, 20)
        if not st.button("Build SHAP Plot", use_container_width=True):
            return

        try:
            with st.spinner("Building SHAP plot..."):
                x_explain = st.session_state["x_test"].sample(
                    min(row_count, len(st.session_state["x_test"])),
                    random_state=42,
                )
                analyzer = SHAPAnalyzer()
                shap_values = analyzer.explain(
                    st.session_state["trained_models"][selected_model],
                    st.session_state["x_train"],
                    x_explain,
                )
                st.pyplot(analyzer.summary_plot(shap_values))
        except Exception as exc:
            st.error(f"SHAP failed: {exc}")

    def render_kfold_results(self) -> None:
        if "kfold_results" not in st.session_state:
            return

        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.write("K-Fold Results")
        results = st.session_state["kfold_results"]
        st.dataframe(results.style.format(precision=4), use_container_width=True)
        mean_results = results[results["fold"].astype(str) == "mean"]
        if not mean_results.empty:
            st.bar_chart(mean_results.set_index("model")["f1"])
        st.markdown("</div>", unsafe_allow_html=True)

    def prediction_section(self, x_data: pd.DataFrame) -> None:
        st.subheader("Manual Cancellation Prediction")
        if "final_prediction_models" not in st.session_state:
            st.info("Train at least one model first, then enter booking values here.")
            return

        model_name = st.selectbox(
            "Prediction model",
            list(st.session_state["final_prediction_models"].keys()),
            key="manual_prediction_model",
        )
        st.caption("Manual predictions use the final model trained on the full dataset.")
        input_row = self.manual_feature_form(x_data)

        if not st.button("Predict Cancellation", type="primary", use_container_width=True):
            return

        model = st.session_state["final_prediction_models"][model_name]
        prediction = int(model.predict(input_row)[0])
        probability = _positive_probabilities(model, input_row)
        cancel_probability = float(probability[0]) if probability is not None else None

        if prediction == 1:
            st.error("Prediction: Booking will be canceled")
        else:
            st.success("Prediction: Booking will not be canceled")

        if cancel_probability is not None:
            st.metric("Cancellation probability", f"{cancel_probability * 100:.2f}%")

    def manual_feature_form(self, x_data: pd.DataFrame) -> pd.DataFrame:
        values: Dict[str, Any] = {}
        categorical_columns = list(x_data.select_dtypes(include=["object", "category"]).columns)
        numeric_columns = [column for column in x_data.columns if column not in categorical_columns]

        st.write("Booking values")
        cat_columns = st.columns(3)
        for index, column in enumerate(categorical_columns):
            options = sorted(x_data[column].astype(str).fillna("Unknown").unique().tolist())
            default_value = (
                str(x_data[column].mode(dropna=True).iloc[0])
                if not x_data[column].mode(dropna=True).empty
                else options[0]
            )
            default_index = options.index(default_value) if default_value in options else 0
            values[column] = cat_columns[index % 3].selectbox(
                column,
                options,
                index=default_index,
                key=f"manual_{column}",
            )

        num_columns = st.columns(3)
        for index, column in enumerate(numeric_columns):
            series = pd.to_numeric(x_data[column], errors="coerce").fillna(0)
            min_value = float(series.min())
            max_value = float(series.max())
            default_value = float(series.median())
            step = 1.0 if pd.api.types.is_integer_dtype(series) else 0.1
            values[column] = num_columns[index % 3].number_input(
                column,
                min_value=min_value,
                max_value=max_value,
                value=default_value,
                step=step,
                key=f"manual_{column}",
            )

        return pd.DataFrame([values], columns=x_data.columns)


if __name__ == "__main__":
    HotelCancellationDashboard().run()
