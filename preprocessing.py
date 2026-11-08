"""Cleaning rules specific to the Telco Customer Churn dataset.
Used by both train_models.py and app.py, so an uploaded raw CSV is
treated exactly like the training data."""
import pandas as pd

TARGET = "Churn"
ID_COL = "customerID"


def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[ID_COL], errors="ignore")

    # Text columns: remove stray spaces (TotalCharges has " " for new customers)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # TotalCharges is stored as text; 11 rows are blank (tenure = 0).
    # Convert to numbers; blanks become NaN and are imputed in the pipeline.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # SeniorCitizen is 0/1 but is really a yes/no category like the others
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str).replace({"0": "No", "1": "Yes"})

    return df
