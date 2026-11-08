
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Breast Cancer Classification")
st.write(
    "Machine Learning application using five classification algorithms "
    "on the Breast Cancer Wisconsin Diagnostic dataset."
)

st.divider()

# --------------------------------------------------
# 2. Model Configuration
# --------------------------------------------------

MODEL_DIR = "models"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl"
}


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# --------------------------------------------------
# 3. Sidebar - Model Selection
# --------------------------------------------------

st.sidebar.header("Model Configuration")

selected_model = st.sidebar.selectbox(
    "Select Classification Model",
    list(MODEL_FILES.keys())
)

model_path = os.path.join(
    MODEL_DIR,
    MODEL_FILES[selected_model]
)

# --------------------------------------------------
# 4. CSV Upload
# --------------------------------------------------

st.header("Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to start classification.")

    # Optional sample test data
    if os.path.exists("test_data.csv"):
        if st.button("Use Sample Test Dataset"):
            st.session_state["use_sample"] = True
    else:
        st.caption("You can also provide your own CSV file.")

    if st.session_state.get("use_sample", False):
        data = pd.read_csv("test_data.csv")
    else:
        st.stop()
else:
    data = pd.read_csv(uploaded_file)

# --------------------------------------------------
# 5. Display Uploaded Dataset
# --------------------------------------------------

st.subheader("Dataset Preview")

st.write(f"Total records: {data.shape[0]}")
st.write(f"Total columns: {data.shape[1]}")

st.dataframe(data.head(10), use_container_width=True)

# --------------------------------------------------
# 6. Data Preprocessing
# --------------------------------------------------

# Remove unnecessary columns
data = data.drop(
    columns=["id", "Unnamed: 32"],
    errors="ignore"
)

# Separate actual labels, if available
actual_labels = None

if "diagnosis" in data.columns:

    actual_labels = data["diagnosis"].copy()

    # Convert B/M labels to 0/1 if needed
    if actual_labels.dtype == "object":
        actual_labels = actual_labels.map({
            "B": 0,
            "M": 1
        })

    actual_labels = pd.to_numeric(
        actual_labels,
        errors="coerce"
    )

    if actual_labels.isna().any():
        st.error(
            "Diagnosis column contains unsupported or missing values. "
            "Use B/M or 0/1 labels."
        )
        st.stop()

    if not actual_labels.isin([0, 1]).all():
        st.error("Diagnosis must contain only 0/1 or B/M values.")
        st.stop()

    features = data.drop(columns=["diagnosis"])

else:
    features = data.copy()

# --------------------------------------------------
# 7. Load Selected Model
# --------------------------------------------------

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# Use feature names recorded during model training
if hasattr(model, "feature_names_in_"):
    expected_features = list(model.feature_names_in_)
else:
    st.error(
        "The selected model does not contain training feature names. "
        "Please check the saved model."
    )
    st.stop()

missing_features = [
    col for col in expected_features
    if col not in features.columns
]

if missing_features:
    st.error(
        "The uploaded dataset is missing required features: "
        + ", ".join(missing_features)
    )
    st.stop()

# Select features in the same order used during training
X = features[expected_features].copy()

# Check for invalid or missing feature values
X = X.apply(pd.to_numeric, errors="coerce")

if X.isna().any().any():
    st.error(
        "The uploaded dataset contains missing or non-numeric feature values. "
        "Please clean the dataset before prediction."
    )
    st.stop()

# --------------------------------------------------
# 8. Generate Predictions
# --------------------------------------------------

st.divider()

st.subheader("Classification Results")

if st.button("Run Classification", type="primary"):

    try:
        predictions = model.predict(X)

        probabilities = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[:, 1]

        results = pd.DataFrame({
            "Predicted Diagnosis": [
                "Malignant" if p == 1 else "Benign"
                for p in predictions
            ]
        })

        if probabilities is not None:
            results["Malignant Probability"] = probabilities

        # --------------------------------------------------
        # 9. Display Prediction Summary
        # --------------------------------------------------

        malignant_count = int(np.sum(predictions == 1))
        benign_count = int(np.sum(predictions == 0))

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Records", len(predictions))
        col2.metric("Benign Predictions", benign_count)
        col3.metric("Malignant Predictions", malignant_count)

        st.subheader("Predictions")

        display_results = pd.concat(
            [data.reset_index(drop=True),
             results.reset_index(drop=True)],
            axis=1
        )

        st.dataframe(
            display_results,
            use_container_width=True
        )

        # --------------------------------------------------
        # 10. Evaluation Metrics
        # --------------------------------------------------

        if actual_labels is not None:

            st.divider()
            st.subheader("Model Evaluation Metrics")

            accuracy = accuracy_score(
                actual_labels, predictions
            )

            precision = precision_score(
                actual_labels, predictions, zero_division=0
            )

            recall = recall_score(
                actual_labels, predictions, zero_division=0
            )

            f1 = f1_score(
                actual_labels, predictions, zero_division=0
            )

            mcc = matthews_corrcoef(
                actual_labels, predictions
            )

            auc = (
                roc_auc_score(actual_labels, probabilities)
                if probabilities is not None
                else None
            )

            metric_cols = st.columns(3)

            metric_cols[0].metric(
                "Accuracy", f"{accuracy:.4f}"
            )

            metric_cols[1].metric(
                "AUC Score",
                f"{auc:.4f}" if auc is not None else "N/A"
            )

            metric_cols[2].metric(
                "Precision", f"{precision:.4f}"
            )

            metric_cols = st.columns(3)

            metric_cols[0].metric(
                "Recall", f"{recall:.4f}"
            )

            metric_cols[1].metric(
                "F1 Score", f"{f1:.4f}"
            )

            metric_cols[2].metric(
                "MCC", f"{mcc:.4f}"
            )

            # --------------------------------------------------
            # 11. Confusion Matrix
            # --------------------------------------------------

            st.divider()
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                actual_labels,
                predictions,
                labels=[0, 1]
            )

            cm_df = pd.DataFrame(
                cm,
                index=["Actual Benign", "Actual Malignant"],
                columns=["Predicted Benign", "Predicted Malignant"]
            )

            st.dataframe(cm_df, use_container_width=True)

            # --------------------------------------------------
            # 12. Classification Report
            # --------------------------------------------------

            st.subheader("Classification Report")

            report = classification_report(
                actual_labels,
                predictions,
                labels=[0, 1],
                target_names=["Benign", "Malignant"],
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(report).transpose()

            st.dataframe(
                report_df.round(4),
                use_container_width=True
            )

        else:
            st.info(
                "Actual diagnosis labels were not provided. "
                "Predictions are available, but evaluation metrics "
                "cannot be calculated."
            )

        # --------------------------------------------------
        # 13. Download Predictions
        # --------------------------------------------------

        download_data = display_results.to_csv(index=False)

        st.download_button(
            label="Download Prediction Results",
            data=download_data,
            file_name="breast_cancer_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Classification failed: {e}")

st.divider()

st.caption(
    "Educational machine learning project. "
    "Predictions are not intended for clinical diagnosis."
)
