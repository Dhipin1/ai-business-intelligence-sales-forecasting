from __future__ import annotations
import pandas as pd

def rule_based_insights(kpi: dict, sales_daily: pd.DataFrame) -> list[str]:
    insights = []

    if kpi.get("total_gmv", 0) > 0:
        insights.append(f"Total GMV is {kpi['total_gmv']:.2f} (BRL) across the dataset period.")

    df = sales_daily.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds")

    if len(df) >= 60:
        last30 = df.tail(30)["y"].sum()
        prev30 = df.iloc[-60:-30]["y"].sum()
        if prev30 > 0:
            change = (last30 - prev30) / prev30 * 100.0
            direction = "up" if change >= 0 else "down"
            insights.append(f"GMV is {direction} {abs(change):.1f}% in the last 30 days vs the previous 30 days.")

    if len(df) >= 7:
        df["dow"] = df["ds"].dt.day_name()
        best = df.groupby("dow")["y"].mean().sort_values(ascending=False).head(1)
        insights.append(f"Highest average GMV day is {best.index[0]}.")

    return insights