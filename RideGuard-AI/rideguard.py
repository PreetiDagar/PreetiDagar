from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

TARGET_ORDER = ["Completed", "Cancelled", "Incomplete", "No Driver Found"]
NUMERIC_FEATURES = ["Hour"]
CATEGORICAL_FEATURES = ["Weekday", "Month", "Vehicle Type", "Pickup Location", "Drop Location"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = ["Booking Status", "Date", "Time", "Vehicle Type", "Pickup Location", "Drop Location"]


def _find_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    lookup = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    for candidate in candidates:
        for column in columns:
            if candidate.lower() in str(column).strip().lower():
                return column
    return None


def normalize_status(value: object) -> str | None:
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "unknown"}:
        return None
    if "no driver" in text or "driver not found" in text:
        return "No Driver Found"
    if "incomplete" in text:
        return "Incomplete"
    if "complete" in text or "success" in text:
        return "Completed"
    if "cancel" in text:
        return "Cancelled"
    return None


def prepare_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return leakage-safe feature matrix, normalized target, and cleaned source rows."""
    df = raw.copy()
    df.columns = df.columns.astype(str).str.strip()
    mapping = {
        "status": _find_column(df.columns, ["Booking Status", "Status"]),
        "date": _find_column(df.columns, ["Date", "Booking Date"]),
        "time": _find_column(df.columns, ["Time", "Booking Time"]),
        "vehicle": _find_column(df.columns, ["Vehicle Type", "Vehicle"]),
        "pickup": _find_column(df.columns, ["Pickup Location", "Pickup"]),
        "drop": _find_column(df.columns, ["Drop Location", "Destination"]),
    }
    missing = [key for key, value in mapping.items() if value is None]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
            + ". Required fields are Booking Status, Date, Time, Vehicle Type, Pickup Location and Drop Location."
        )

    date_values = pd.to_datetime(df[mapping["date"]], errors="coerce", dayfirst=True)
    time_values = pd.to_datetime(df[mapping["time"]].astype(str), errors="coerce", format="mixed")
    hour = time_values.dt.hour
    if hour.notna().sum() == 0:
        combined = pd.to_datetime(
            df[mapping["date"]].astype(str).str.strip() + " " + df[mapping["time"]].astype(str).str.strip(),
            errors="coerce",
            dayfirst=True,
        )
        hour = combined.dt.hour

    target = df[mapping["status"]].map(normalize_status)
    features = pd.DataFrame({
        "Hour": hour,
        "Weekday": date_values.dt.day_name(),
        "Month": date_values.dt.month_name(),
        "Vehicle Type": df[mapping["vehicle"]].astype("string").str.strip(),
        "Pickup Location": df[mapping["pickup"]].astype("string").str.strip(),
        "Drop Location": df[mapping["drop"]].astype("string").str.strip(),
    }, index=df.index)

    valid = target.notna()
    features = features.loc[valid].copy()
    target = target.loc[valid].astype(str)
    source = df.loc[valid].copy()
    for col in CATEGORICAL_FEATURES:
        features[col] = features[col].fillna("Unknown").replace("", "Unknown")
    return features, target, source


def build_pipeline(model_name: str, random_state: int = 42) -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=3)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    if model_name == "Logistic Regression":
        classifier = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        )
    elif model_name == "Random Forest":
        classifier = RandomForestClassifier(
            n_estimators=180,
            max_depth=18,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError("Choose Logistic Regression or Random Forest.")
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


@dataclass
class ModelResult:
    model: Pipeline
    model_name: str
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    report: dict
    matrix: np.ndarray
    labels: list[str]
    X_test: pd.DataFrame
    y_test: pd.Series
    predictions: np.ndarray


def train_evaluate(features: pd.DataFrame, target: pd.Series, model_name: str) -> ModelResult:
    counts = target.value_counts()
    keep = counts[counts >= 2].index
    mask = target.isin(keep)
    X, y = features.loc[mask], target.loc[mask]
    if y.nunique() < 2:
        raise ValueError("At least two outcome classes with two or more rows each are needed to train a model.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42, stratify=y
    )
    model = build_pipeline(model_name)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    labels = [label for label in TARGET_ORDER if label in set(y)]
    return ModelResult(
        model=model,
        model_name=model_name,
        accuracy=float(accuracy_score(y_test, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, predictions)),
        macro_f1=float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        report=classification_report(
            y_test, predictions, labels=labels, output_dict=True, zero_division=0
        ),
        matrix=confusion_matrix(y_test, predictions, labels=labels),
        labels=labels,
        X_test=X_test,
        y_test=y_test,
        predictions=predictions,
    )


def get_permutation_importance(result: ModelResult, max_rows: int = 600) -> pd.DataFrame:
    sample = result.X_test
    if len(sample) > max_rows:
        sample = sample.sample(max_rows, random_state=42)
    y = result.y_test.loc[sample.index]
    scores = permutation_importance(
        result.model, sample, y, n_repeats=3, random_state=42,
        scoring="balanced_accuracy", n_jobs=-1
    )
    return pd.DataFrame({
        "Feature": sample.columns,
        "Importance": scores.importances_mean,
        "Std": scores.importances_std,
    }).sort_values("Importance", ascending=False)


def make_demo_data(n_rows: int = 3200, seed: int = 42) -> pd.DataFrame:
    """Create explicitly synthetic demo data; never presented as real ride history."""
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 180, n_rows), unit="D"
    )
    hours = rng.integers(0, 24, n_rows)
    vehicles = np.array(["Auto", "Mini", "Sedan", "SUV", "Bike"])
    locations = np.array([
        "Connaught Place", "Dwarka", "Gurugram", "Noida", "Saket",
        "Rohini", "Karol Bagh", "IGI Airport", "Greater Noida", "Lajpat Nagar",
    ])
    vehicle = rng.choice(vehicles, n_rows, p=[.22, .30, .24, .14, .10])
    pickup = rng.choice(locations, n_rows)
    drop = rng.choice(locations, n_rows)
    rush = ((hours >= 8) & (hours <= 10)) | ((hours >= 17) & (hours <= 21))
    airport = (pickup == "IGI Airport") | (drop == "IGI Airport")
    no_driver_p = np.clip(.035 + .055 * rush + .04 * airport, .02, .22)
    cancel_p = np.clip(.10 + .07 * rush + .035 * (vehicle == "Bike"), .05, .35)
    incomplete_p = np.full(n_rows, .045)
    completed_p = 1 - no_driver_p - cancel_p - incomplete_p
    probs = np.column_stack([completed_p, cancel_p, incomplete_p, no_driver_p])
    probs = probs / probs.sum(axis=1, keepdims=True)
    labels = np.array(TARGET_ORDER)
    status = np.array([rng.choice(labels, p=p) for p in probs])
    minutes = rng.integers(0, 60, n_rows)
    return pd.DataFrame({
        "Date": dates.strftime("%d-%m-%Y"),
        "Time": [f"{h:02d}:{m:02d}:00" for h, m in zip(hours, minutes)],
        "Vehicle Type": vehicle,
        "Pickup Location": pickup,
        "Drop Location": drop,
        "Booking Status": status,
        "Booking Value": np.round(rng.uniform(60, 900, n_rows), 2),
        "Ride Distance": np.round(rng.uniform(1, 28, n_rows), 1),
    })
