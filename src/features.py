from __future__ import annotations
import pandas as pd

def make_time_features(df_daily: pd.DataFrame) -> pd.DataFrame:
    out = df_daily.copy()
    out["ds"] = pd.to_datetime(out["ds"])
    out["dow"] = out["ds"].dt.dayofweek
    out["week"] = out["ds"].dt.isocalendar().week.astype(int)
    out["month"] = out["ds"].dt.month
    out["year"] = out["ds"].dt.year
    out["day"] = out["ds"].dt.day
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    return out

def make_lag_features(df: pd.DataFrame, lags=(1, 7, 14, 28)) -> pd.DataFrame:
    """
    Uses rolling means of previous values (shifted), so it works for future prediction rows too.
    """
    out = df.sort_values("ds").copy()
    for lag in lags:
        out[f"lag_{lag}"] = out["y"].shift(lag)

    # rolling mean of history only (exclude current row)
    y_prev = out["y"].shift(1)
    out["roll_7"] = y_prev.rolling(7).mean()
    out["roll_14"] = y_prev.rolling(14).mean()

    out = out.dropna().reset_index(drop=True)
    return out