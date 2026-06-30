

import os
import re
import json

import joblib
import numpy as np
import pandas as pd

EPSILON = 1e-6

NUMERIC_COLUMNS = [
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Credit_History_Age",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
]


def _parse_credit_history_age(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return np.nan

    match = re.match(r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?", str(value))
    if not match:
        return np.nan

    years, months = int(match.group(1)), int(match.group(2))
    return years * 12 + months


def _count_loan_types(value):
    if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
        return 0

    return len([t for t in str(value).replace("and ", "").split(",") if t.strip()])


def _build_features(raw: dict) -> pd.DataFrame:

    data = dict(raw)

    data["Credit_History_Age"] = _parse_credit_history_age(data.get("Credit_History_Age"))
    data["Num_Loan_Types"] = _count_loan_types(data.get("Type_of_Loan"))
    data.pop("Type_of_Loan", None)

    df = pd.DataFrame([data])

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Debt_to_Income_Ratio"] = df["Outstanding_Debt"] / (df["Annual_Income"] + EPSILON)
    df["Loan_per_Bank_Account"] = df["Num_of_Loan"] / (df["Num_Bank_Accounts"] + EPSILON)
    df["EMI_to_Income_Ratio"] = df["Total_EMI_per_month"] / (df["Annual_Income"] + EPSILON)
    df["Investment_to_Income_Ratio"] = df["Amount_invested_monthly"] / (df["Annual_Income"] + EPSILON)
    df["Credit_per_Account"] = df["Num_Credit_Card"] / (df["Num_Bank_Accounts"] + EPSILON)
    df["Inquiry_per_Loan"] = df["Num_Credit_Inquiries"] / (df["Num_of_Loan"] + EPSILON)
    df["Total_Financial_Accounts"] = df["Num_Bank_Accounts"] + df["Num_Credit_Card"]

    return df


# ----------------------------------------------------------------------
# SageMaker required hooks
# ----------------------------------------------------------------------

def model_fn(model_dir):
    """Called once per worker to load every artifact bundled in model.tar.gz."""

    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.pkl"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))

    return {
        "model": model,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
    }


def input_fn(request_body, request_content_type="application/json"):
    """Parse the raw request body into a list of raw record dicts."""

    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    payload = json.loads(request_body)

    # Accept both {"instances": [{...}, ...]} and a single raw dict {...}
    if isinstance(payload, dict) and "instances" in payload:
        return payload["instances"]

    if isinstance(payload, dict):
        return [payload]

    if isinstance(payload, list):
        return payload

    raise ValueError("Request body must be a JSON object or a list of objects")


def predict_fn(input_data, model_artifacts):
    """Run prediction for every record in input_data."""

    model = model_artifacts["model"]
    preprocessor = model_artifacts["preprocessor"]
    label_encoder = model_artifacts["label_encoder"]

    results = []

    for raw in input_data:

        df = _build_features(raw)

        X = preprocessor.transform(df)
        if hasattr(X, "toarray"):
            X = X.toarray()

        pred_encoded = model.predict(X)[0]
        predicted_label = label_encoder.inverse_transform([pred_encoded])[0]

        probabilities = {}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            class_labels = label_encoder.inverse_transform(np.arange(len(proba)))
            probabilities = {str(c): float(p) for c, p in zip(class_labels, proba)}

        results.append({
            "prediction": str(predicted_label),
            "probabilities": probabilities,
        })

    return results


def output_fn(prediction, accept="application/json"):
    """Serialize predict_fn's output back to the caller."""

    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")

    return json.dumps({"predictions": prediction}), accept