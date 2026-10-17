"""
Common utilities for MIMIC preprocessing scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple
import duckdb
import pandas as pd
import numpy as np


@dataclass
class MIMICPaths:
    db_path: str = "data/mimic/mimiciii.duckdb"
    vital_meta_path: str = "data/mimic/vital_metadata.csv"
    labs_meta_path: str = "data/mimic/labs_metadata.csv"


def connect_db(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path, read_only=True)


def load_metadata(paths: MIMICPaths) -> Tuple[pd.DataFrame, pd.DataFrame]:
    vital_meta = pd.read_csv(paths.vital_meta_path)
    labs_meta = pd.read_csv(paths.labs_meta_path)
    return vital_meta, labs_meta


def chunk_list(values: List[int], chunk_size: int) -> Iterable[List[int]]:
    for i in range(0, len(values), chunk_size):
        yield values[i:i + chunk_size]


def fetch_labevents_timeseries(
    conn: duckdb.DuckDBPyConnection,
    hadm_ids: List[int],
    itemids: List[int],
    value_col: str,
    min_val: float,
    max_val: float,
    batch_size: int = 500,
) -> pd.DataFrame:
    """
    Load labevents time series for given itemids and hadm_ids in batches.
    """
    if not hadm_ids or not itemids:
        return pd.DataFrame()

    frames = []
    for batch in chunk_list(hadm_ids, batch_size):
        query = f"""
        SELECT
            le.hadm_id::INTEGER AS hadm_id,
            le.charttime::TIMESTAMP AS charttime,
            EXTRACT(EPOCH FROM (le.charttime::TIMESTAMP - a.admittime::TIMESTAMP)) / 86400.0 AS time_days,
            CAST(le.valuenum AS REAL) AS {value_col}
        FROM labevents le
        INNER JOIN admissions a ON le.hadm_id = a.hadm_id
        WHERE
            le.itemid IN {tuple(itemids)}
            AND le.valuenum::REAL IS NOT NULL
            AND le.valuenum::REAL BETWEEN {min_val} AND {max_val}
            AND le.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.dischtime::TIMESTAMP
            AND le.hadm_id::INTEGER IN {tuple(batch)}
        """
        df = conn.execute(query).fetchdf()
        if len(df) > 0:
            df.columns = df.columns.str.lower()
            frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_vitals_labs(
    conn: duckdb.DuckDBPyConnection,
    hadm_ids: List[int],
    vital_items: List[int],
    lab_items: List[int],
    batch_size: int = 500,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not hadm_ids:
        return pd.DataFrame(), pd.DataFrame()

    if not vital_items:
        vital_items = []
    if not lab_items:
        lab_items = []

    vitals_frames = []
    labs_frames = []

    for batch in chunk_list(hadm_ids, batch_size):
        if vital_items:
            vitals_query = f"""
            SELECT chartevents.subject_id::INTEGER AS subject_id
                , chartevents.hadm_id::INTEGER AS hadm_id
                , chartevents.charttime::TIMESTAMP AS charttime
                , chartevents.itemid::INTEGER AS itemid
                , chartevents.valuenum::DOUBLE AS valuenum
                , admissions.admittime::TIMESTAMP AS admittime
            FROM chartevents
            INNER JOIN admissions
                ON chartevents.subject_id = admissions.subject_id
                AND chartevents.hadm_id = admissions.hadm_id
                AND chartevents.charttime::TIMESTAMP BETWEEN
                    (admissions.admittime::TIMESTAMP)
                    AND (admissions.dischtime::TIMESTAMP)
                AND itemid::INTEGER IN {tuple(vital_items)}
                AND chartevents.hadm_id::INTEGER IN {tuple(batch)}
            WHERE chartevents.error::INTEGER IS DISTINCT FROM 1
            """

            vdf = conn.execute(vitals_query).fetchdf()
            if len(vdf) > 0:
                vdf.columns = vdf.columns.str.lower()
                vitals_frames.append(vdf)

        if lab_items:
            labs_query = f"""
            SELECT labevents.subject_id::INTEGER AS subject_id
                , labevents.hadm_id::INTEGER AS hadm_id
                , labevents.charttime::TIMESTAMP AS charttime
                , labevents.itemid::INTEGER AS itemid
                , labevents.valuenum::DOUBLE AS valuenum
                , admissions.admittime::TIMESTAMP AS admittime
            FROM labevents
            INNER JOIN admissions
                ON labevents.subject_id = admissions.subject_id
                AND labevents.hadm_id = admissions.hadm_id
                AND labevents.charttime::TIMESTAMP BETWEEN
                    (admissions.admittime::TIMESTAMP)
                    AND (admissions.dischtime::TIMESTAMP)
                AND itemid::INTEGER IN {tuple(lab_items)}
                AND labevents.hadm_id::INTEGER IN {tuple(batch)}
            """

            ldf = conn.execute(labs_query).fetchdf()
            if len(ldf) > 0:
                ldf.columns = ldf.columns.str.lower()
                labs_frames.append(ldf)

    vitals_df = pd.concat(vitals_frames, ignore_index=True) if vitals_frames else pd.DataFrame()
    labs_df = pd.concat(labs_frames, ignore_index=True) if labs_frames else pd.DataFrame()
    return vitals_df, labs_df


def ensure_time_day(df: pd.DataFrame) -> pd.DataFrame:
    if 'time_days' in df.columns:
        df['time_day'] = np.floor(df['time_days']).astype(int)
    return df


def aggregate_vitals_labs(vitals_df: pd.DataFrame, labs_df: pd.DataFrame,
                          vital_meta: pd.DataFrame, labs_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate vitals and labs to daily min/max/mean.
    """
    vitals_df = vitals_df.merge(vital_meta, on='itemid', how='left')
    vitals_df = vitals_df[vitals_df['valuenum'].between(vitals_df['min'], vitals_df['max'], inclusive='both')]

    labs_df = labs_df.merge(labs_meta, on='itemid', how='left')
    labs_df = labs_df[labs_df['valuenum'].between(labs_df['min'], labs_df['max'], inclusive='both')]

    if 'units' in vitals_df.columns:
        vitals_df.loc[vitals_df['units'] == 'F', 'valuenum'] = (
            vitals_df.loc[vitals_df['units'] == 'F', 'valuenum'] - 32
        ) * 5.0/9.0
        vitals_df.loc[vitals_df['units'] == 'F', 'units'] = 'C'
        vitals_df.loc[vitals_df['feature name'] == 'TempF', 'feature name'] = 'TempC'

    vitals_labs = pd.concat([vitals_df, labs_df], ignore_index=True)

    vitals_labs_pivot = vitals_labs.pivot_table(
        index=['hadm_id', 'admittime', pd.Grouper(freq='1D', key='charttime')],
        columns='feature name',
        values='valuenum',
        aggfunc=['min', 'max', 'mean']
    )

    vitals_labs_pivot.columns = [f'{col[1]}_{col[0]}' for col in vitals_labs_pivot.columns]
    return vitals_labs_pivot


# Cohort criteria shared by all MIMIC tasks (as in the thesis Methods):
# adults aged 18-90 with an ICU stay of at least 2 days and at least
# MIN_TARGET_MEASUREMENTS measurements of the target clinical parameter.
MIN_ICU_LOS_DAYS = 2.0
MIN_TARGET_MEASUREMENTS = 5
ICU_LOS_SQL = (
    "EXISTS (SELECT 1 FROM icustays icu WHERE icu.hadm_id = a.hadm_id "
    f"AND icu.los::DOUBLE >= {MIN_ICU_LOS_DAYS})"
)

# Missing-data handling (as in the thesis Methods)
FFILL_LIMIT = 3              # consecutive fills for vitals and labs
MISSINGNESS_THRESHOLD = 0.7  # drop columns with more missing values than this
UNBOUNDED_FFILL_COLS = ("age", "gender")  # plus weight columns (low variance)


def ffill_within_stay(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Forward-fill within each stay (df must be sorted by id and time): static
    and low-variance parameters (age, gender, weight) without limit, all other
    columns for at most FFILL_LIMIT consecutive intervals.
    """
    # Positional, because the aggregated tables can hold duplicate column names
    # (e.g. glucose as both a vital sign and a lab).
    df = df.copy()
    groups = df[id_col].to_numpy()
    unbounded = [str(c) in UNBOUNDED_FFILL_COLS or str(c).lower().startswith("weight") for c in df.columns]
    capped_pos = [i for i, c in enumerate(df.columns) if c != id_col and not unbounded[i]]
    unbounded_pos = [i for i, c in enumerate(df.columns) if c != id_col and unbounded[i]]
    if capped_pos:
        filled = df.iloc[:, capped_pos].groupby(groups, sort=False).ffill(limit=FFILL_LIMIT)
        df.iloc[:, capped_pos] = filled.to_numpy()
    if unbounded_pos:
        filled = df.iloc[:, unbounded_pos].groupby(groups, sort=False).ffill()
        df.iloc[:, unbounded_pos] = filled.to_numpy()
    return df


def fetch_vasopressor_starts(
    conn: duckdb.DuckDBPyConnection,
    hadm_ids: List[int],
    mv_items: Tuple[int, ...],
    cv_items: Tuple[int, ...],
    out_col: str,
    seconds_per_unit: float,
) -> pd.DataFrame:
    """
    First vasopressor start per admission, in units since admission, from both
    MetaVision (inputevents_mv.starttime) and CareVue (inputevents_cv.charttime)
    records, so that patients from both MIMIC-III eras are covered.
    """
    query = f"""
    SELECT hadm_id, MIN(t) AS {out_col}
    FROM (
        SELECT a.hadm_id::INT AS hadm_id,
               EXTRACT(EPOCH FROM (ie.starttime::timestamp - a.admittime::timestamp)) / {seconds_per_unit} AS t
        FROM inputevents_mv ie
        INNER JOIN admissions a ON ie.hadm_id = a.hadm_id
        WHERE ie.itemid IN {tuple(mv_items)} AND a.hadm_id::INT IN {tuple(hadm_ids)}
        UNION ALL
        SELECT a.hadm_id::INT AS hadm_id,
               EXTRACT(EPOCH FROM (ie.charttime::timestamp - a.admittime::timestamp)) / {seconds_per_unit} AS t
        FROM inputevents_cv ie
        INNER JOIN admissions a ON ie.hadm_id = a.hadm_id
        WHERE ie.itemid IN {tuple(cv_items)} AND a.hadm_id::INT IN {tuple(hadm_ids)}
    )
    GROUP BY hadm_id
    """
    df = conn.execute(query).fetchdf()
    df.columns = df.columns.str.lower()
    return df


def drop_high_missing_columns(df: pd.DataFrame, threshold: float = 0.7, exclude_cols=None):
    """
    Drop columns with missing values above threshold after forward fill.
    """
    if exclude_cols is None:
        exclude_cols = ['hadm_id', 'time_day', 'target']

    missing = df.isna().mean()
    drop_cols = [c for c, frac in missing.items() if frac > threshold and c not in exclude_cols]
    cleaned = df.drop(columns=drop_cols)
    return cleaned, drop_cols
