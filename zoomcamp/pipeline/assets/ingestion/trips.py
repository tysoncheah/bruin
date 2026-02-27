"""@bruin
name: ingestion.trips
type: python
image: python:3.11

materialization:
  type: table
  strategy: append

columns:
  - name: pickup_datetime
    type: timestamp
    description: "When the meter was engaged"
  - name: dropoff_datetime
    type: timestamp
    description: "When the meter was disengaged"
@bruin"""

import os
import json
from typing import List

import pandas as pd


BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year}-{month}.parquet"


def _generate_month_starts(start_date: str, end_date: str) -> List[pd.Timestamp]:
    """
    Generate a list of month-start timestamps between two ISO date strings (inclusive).
    """
    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()

    # "MS" = month start frequency
    return list(pd.date_range(start=start_ts, end=end_ts, freq="MS"))


def _normalize_columns(df: pd.DataFrame, taxi_type: str) -> pd.DataFrame:
    """
    Normalize raw NYC TLC schemas into a common set of columns expected downstream.
    This handles common variants between yellow/green/FHV style datasets.
    """
    # Handle pickup/dropoff timestamp column name differences (yellow vs green vs HVFHV).
    if "pickup_datetime" in df.columns:
        pickup_col = "pickup_datetime"
        dropoff_col = "dropoff_datetime"
    elif "tpep_pickup_datetime" in df.columns:
        pickup_col = "tpep_pickup_datetime"
        dropoff_col = "tpep_dropoff_datetime"
    elif "lpep_pickup_datetime" in df.columns:
        pickup_col = "lpep_pickup_datetime"
        dropoff_col = "lpep_dropoff_datetime"
    else:
        raise ValueError("Could not find pickup/dropoff datetime columns in source data.")

    rename_map = {
        pickup_col: "pickup_datetime",
        dropoff_col: "dropoff_datetime",
    }

    # Location IDs
    if "PULocationID" in df.columns:
        rename_map["PULocationID"] = "pickup_location_id"
    if "DOLocationID" in df.columns:
        rename_map["DOLocationID"] = "dropoff_location_id"

    df = df.rename(columns=rename_map)

    # Ensure required columns exist (create as null if missing so downstream SQL still works).
    required_columns = [
        "pickup_datetime",
        "dropoff_datetime",
        "pickup_location_id",
        "dropoff_location_id",
        "fare_amount",
        "payment_type",
        "taxi_type",
    ]

    for col in required_columns:
        if col not in df.columns:
            df[col] = pd.NA

    # Keep only the columns we care about, in a stable order.
    df = df[required_columns]

    # Attach taxi_type so staging/report layers can segment by fleet.
    df["taxi_type"] = taxi_type

    return df


def materialize():
    """
    Ingest NYC taxi trip data from public Parquet files on a month-by-month basis
    for the pipeline's requested time window (BRUIN_START_DATE to BRUIN_END_DATE).

    This function:
      - figures out which months fall inside the run's date window,
      - downloads the corresponding Parquet files for each configured taxi_type,
      - normalizes their schemas to a common layout,
      - filters rows to the exact [start_date, end_date) window,
      - and returns a single concatenated DataFrame.
    """
    start_date = os.environ["BRUIN_START_DATE"]
    end_date = os.environ["BRUIN_END_DATE"]
    taxi_types = json.loads(os.environ["BRUIN_VARS"]).get("taxi_types", ["yellow"])

    month_starts = _generate_month_starts(start_date, end_date)
    if not month_starts:
        # No months to process – return an empty DataFrame with the expected schema.
        return pd.DataFrame(
            columns=[
                "pickup_datetime",
                "dropoff_datetime",
                "pickup_location_id",
                "dropoff_location_id",
                "fare_amount",
                "payment_type",
                "taxi_type",
            ]
        )

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    dataframes = []

    for taxi_type in taxi_types:
        for month_start in month_starts:
            year = month_start.year
            month = month_start.month
            url = BASE_URL.format(
                taxi_type=taxi_type,
                year=year,
                month=f"{month:02d}",
            )

            try:
                df = pd.read_parquet(url)
            except Exception:
                # If a month/taxi_type combination does not exist, skip it.
                continue

            df = _normalize_columns(df, taxi_type=taxi_type)

            # Filter to the exact pipeline run window to avoid carrying extra history.
            mask = (df["pickup_datetime"] >= start_ts) & (df["pickup_datetime"] < end_ts)
            df = df.loc[mask]

            if not df.empty:
                dataframes.append(df)

    if not dataframes:
        return pd.DataFrame(
            columns=[
                "pickup_datetime",
                "dropoff_datetime",
                "pickup_location_id",
                "dropoff_location_id",
                "fare_amount",
                "payment_type",
                "taxi_type",
            ]
        )

    final_dataframe = pd.concat(dataframes, ignore_index=True)

    return final_dataframe