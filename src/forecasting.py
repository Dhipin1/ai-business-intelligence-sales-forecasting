from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from src.features import make_time_features, make_lag_features

@dataclass
class ForecastMetrics:
    mae: float
    rmse: float
    mape: float

def _metrics(y_true, y_pred) -> ForecastMetrics:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(1e-9, y_true))) * 100.0)
    return ForecastMetrics(mae=mae, rmse=rmse, mape=mape)

# -------------------- Prophet --------------------
def train_prophet(df_daily: pd.DataFrame, test_days: int = 60):
    from prophet import Prophet
    df = df_daily.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds")

    train = df.iloc[:-test_days].copy() if len(df) > test_days else df.copy()
    test = df.iloc[-test_days:].copy() if len(df) > test_days else df.tail(0).copy()

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
    )
    m.fit(train[["ds", "y"]])

    if len(test) > 0:
        fc = m.predict(test[["ds"]])
        met = _metrics(test["y"], fc["yhat"])
    else:
        met = _metrics(train["y"], m.predict(train[["ds"]])["yhat"])

    return m, met

def save_prophet(model, path: Path):
    from prophet.serialize import model_to_json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model_to_json(model), encoding="utf-8")

def load_prophet(path: Path):
    from prophet.serialize import model_from_json
    return model_from_json(path.read_text(encoding="utf-8"))

# -------------------- XGBoost --------------------
def train_xgb(df_daily: pd.DataFrame, test_days: int = 60):
    from xgboost import XGBRegressor

    df = df_daily.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds")

    df = make_time_features(df)
    df = make_lag_features(df, lags=(1, 7, 14, 28))

    if len(df) > test_days:
        train = df.iloc[:-test_days].copy()
        test = df.iloc[-test_days:].copy()
    else:
        train = df.copy()
        test = df.tail(0).copy()

    features = [
        "dow", "week", "month", "year", "day", "is_weekend",
        "lag_1", "lag_7", "lag_14", "lag_28", "roll_7", "roll_14"
    ]

    X_train, y_train = train[features], train["y"]
    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)

    if len(test) > 0:
        yhat = model.predict(test[features])
        met = _metrics(test["y"], yhat)
    else:
        yhat = model.predict(X_train)
        met = _metrics(y_train, yhat)

    return model, met, features

def save_xgb(model, features, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features}, path)

def load_xgb(path: Path):
    return joblib.load(path)