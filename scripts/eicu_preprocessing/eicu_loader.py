"""
eICU Data Loader Module

Queries eICU-CRD v2.0 DuckDB and returns MIMIC-shaped dataframes for trajectory modeling.
Primary key: patientunitstayid (equivalent to MIMIC hadm_id for hospital admission)
Time reference: Minutes from unit admission (diagnosisoffset, labresultoffset, observationoffset)

Usage:
    from eicu_loader import EICULoader
    loader = EICULoader(db_path=os.path.join(os.environ.get("TRAJ_DATA_ROOT", "/home/gaga/data/physionet"), "eicu.duckdb"))
    aki_cohort = loader.load_aki_cohort()
    labs = loader.load_labs(['creatinine'])
"""

import os
import duckdb
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple


# Cohort and missing-data settings shared by all eICU tasks (as in the thesis
# Methods): at least MIN_TARGET_MEASUREMENTS measurements of the target clinical
# parameter; vitals and labs forward-filled for at most FFILL_LIMIT consecutive
# intervals (age, gender and weight without limit); columns with more than
# MISSINGNESS_THRESHOLD missing values dropped.
MIN_TARGET_MEASUREMENTS = 5
FFILL_LIMIT = 3
MISSINGNESS_THRESHOLD = 0.7
UNBOUNDED_FFILL_COLS = ("age", "gender")


def ffill_within_stay(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Forward-fill within each stay (df sorted by id and time); see FFILL_LIMIT."""
    df = df.copy()
    cols = [c for c in df.columns if c != id_col]
    unbounded = [c for c in cols if c in UNBOUNDED_FFILL_COLS or c.lower().startswith("weight")]
    capped = [c for c in cols if c not in unbounded]
    grouped = df.groupby(id_col, sort=False)
    if capped:
        df[capped] = grouped[capped].ffill(limit=FFILL_LIMIT)
    if unbounded:
        df[unbounded] = grouped[unbounded].ffill()
    return df


class EICULoader:
    """Query eICU-CRD v2.0 DuckDB and return MIMIC-shaped dataframes."""
    
    def __init__(self, db_path: str = os.path.join(os.environ.get("TRAJ_DATA_ROOT", "/home/gaga/data/physionet"), "eicu.duckdb")):
        """Initialize DuckDB connection."""
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)
        
        # Map common lab names and vitals for quick lookup
        self._lab_names = self._get_lab_names()
        self._vital_names = self._get_vital_names()
    
    def _get_lab_names(self) -> Dict[str, str]:
        """Fetch all unique lab names from database."""
        result = self.conn.execute("""
            SELECT DISTINCT labname FROM lab LIMIT 1000
        """).fetchall()
        return {name[0].lower(): name[0] for name in result}
    
    def _get_vital_names(self) -> Dict[str, str]:
        """Get vital signs column mapping (periodic table)."""
        return {
            'heart rate': 'heartrate',
            'systolic': 'systemicsystolic',
            'diastolic': 'systemicdiastolic',
            'mean bp': 'systemicmean',
            'respiratory rate': 'respiration',
            'o2 sat': 'sao2',
            'temperature': 'temperature',
        }
    
    def load_aki_cohort(self, 
                       age_min: int = 18, 
                       age_max: int = 90,
                       min_los_hours: float = 48) -> pd.DataFrame:
        """
        Load AKI cohort: general ICU patients aged 18-90 with ≥2 day LOS.
        
        Returns:
            DataFrame with columns: patientunitstayid, patienthealthsystemstayid, 
            gender, age, unitadmittime24, unitdischargetime24, hospital_expire_flag
        """
        # Note: LOS is computed from patient table's los column if available,
        # or estimated from admission/discharge offset in minutes
        query = f"""
        SELECT 
            p.patientunitstayid::INTEGER AS patientunitstayid,
            p.patienthealthsystemstayid::INTEGER AS patienthealthsystemstayid,
            p.gender,
            CASE 
                WHEN p.age LIKE '> %' THEN 90
                WHEN p.age LIKE '< %' THEN 18
                ELSE CAST(p.age AS INTEGER)
            END AS age,
            p.unitadmittime24 AS unitadmittime24,
            p.unitdischargetime24 AS unitdischargetime24,
            CAST(p.hospitaldischargestatus AS VARCHAR) AS hospitaldischargestatus,
            CASE WHEN p.hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END AS hospital_expire_flag
        FROM patient p
        WHERE 
            CASE 
                WHEN p.age LIKE '> %' THEN 90
                WHEN p.age LIKE '< %' THEN 18
                ELSE CAST(p.age AS INTEGER)
            END BETWEEN {age_min} AND {age_max}
            -- Use unitdischargeoffset (in minutes from admission) to filter LOS
            AND CAST(p.unitdischargeoffset AS INTEGER) / 60.0 >= {min_los_hours}
        ORDER BY p.patientunitstayid
        """
        return self.conn.execute(query).fetchdf()

    def load_general_cohort(self,
                            age_min: int = 18,
                            age_max: int = 90,
                            min_los_hours: float = 48) -> pd.DataFrame:
        """
        Load a general ICU cohort: adult patients aged 18-90 with ≥2 day LOS.

        Returns the same schema as load_aki_cohort().
        """
        return self.load_aki_cohort(age_min=age_min, age_max=age_max, min_los_hours=min_los_hours)
    
    def load_liver_cohort(self,
                         age_min: int = 18,
                         age_max: int = 90,
                         min_los_hours: float = 48) -> pd.DataFrame:
        """
        Load liver cohort: patients with chronic liver disease diagnoses.
        
        Diagnoses: ICD-9 codes 571.x (chronic liver disease and cirrhosis),
        572.2 (hepatic encephalopathy), 572.8 (other sequelae of chronic liver
        disease), V42.7 (liver transplant status).

        eICU stores dotted codes, often several per field ("571.5, K74.60"),
        so codes are matched at code boundaries rather than as MIMIC-style
        undotted prefixes (which matched only 571.x).
        """
        query = f"""
        WITH liver_patients AS (
            SELECT DISTINCT p.patientunitstayid
            FROM diagnosis d
            INNER JOIN patient p ON d.patientunitstayid = p.patientunitstayid
            WHERE
                regexp_matches(d.icd9code, '(^|, *)(571([.]|,|$)|572[.][28]|V42[.]7)')
        )
        SELECT 
            p.patientunitstayid::INTEGER AS patientunitstayid,
            p.patienthealthsystemstayid::INTEGER AS patienthealthsystemstayid,
            p.gender,
            CASE 
                WHEN p.age LIKE '> %' THEN 90
                WHEN p.age LIKE '< %' THEN 18
                ELSE CAST(p.age AS INTEGER)
            END AS age,
            p.unitadmittime24 AS unitadmittime24,
            p.unitdischargetime24 AS unitdischargetime24,
            CASE WHEN p.hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END AS hospital_expire_flag
        FROM patient p
        INNER JOIN liver_patients lp ON p.patientunitstayid = lp.patientunitstayid
        WHERE 
            CASE 
                WHEN p.age LIKE '> %' THEN 90
                WHEN p.age LIKE '< %' THEN 18
                ELSE CAST(p.age AS INTEGER)
            END BETWEEN {age_min} AND {age_max}
            AND CAST(p.unitdischargeoffset AS INTEGER) / 60.0 >= {min_los_hours}
        ORDER BY p.patientunitstayid
        """
        return self.conn.execute(query).fetchdf()
    
    def load_sepsis_cohort(self,
                          age_min: int = 18,
                          age_max: int = 90,
                          min_los_hours: float = 48) -> pd.DataFrame:
        """
        Load sepsis cohort: patients with sepsis diagnoses.
        
        Diagnoses: ICD-9 995.91 (sepsis), 995.92 (severe sepsis), 785.52
        (septic shock), ICD-10 R65.2x (severe sepsis / septic shock), or a
        diagnosis string mentioning sepsis or septic shock. Codes are matched
        at code boundaries because eICU fields hold several codes
        ("038.9, 785.52, R65.21"). Non-infectious SIRS (995.93/995.94,
        R65.1x) is not sepsis and is not included.
        """
        query = f"""
        WITH sepsis_patients AS (
            SELECT DISTINCT p.patientunitstayid
            FROM diagnosis d
            INNER JOIN patient p ON d.patientunitstayid = p.patientunitstayid
            WHERE
                regexp_matches(d.icd9code, '(^|, *)(995[.]9[12]|785[.]52|R65[.]2)') OR
                d.diagnosisstring LIKE '%sepsis%' OR
                d.diagnosisstring LIKE '%septic shock%'
        )
        SELECT 
            p.patientunitstayid::INTEGER AS patientunitstayid,
            p.patienthealthsystemstayid::INTEGER AS patienthealthsystemstayid,
            p.gender,
            CASE 
                WHEN p.age LIKE '> %' THEN 90
                WHEN p.age LIKE '< %' THEN 18
                ELSE CAST(p.age AS INTEGER)
            END AS age,
            p.unitadmittime24 AS unitadmittime24,
            p.unitdischargetime24 AS unitdischargetime24,
            CASE WHEN p.hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END AS hospital_expire_flag
        FROM patient p
        INNER JOIN sepsis_patients sp ON p.patientunitstayid = sp.patientunitstayid
        WHERE 
            CASE 
                WHEN p.age LIKE '> %' THEN 90
                WHEN p.age LIKE '< %' THEN 18
                ELSE CAST(p.age AS INTEGER)
            END BETWEEN {age_min} AND {age_max}
            AND CAST(p.unitdischargeoffset AS INTEGER) / 60.0 >= {min_los_hours}
        ORDER BY p.patientunitstayid
        """
        return self.conn.execute(query).fetchdf()
    
    def load_ventilator_cohort(self,
                              age_min: int = 18,
                              age_max: int = 90,
                              min_los_hours: float = 48) -> pd.DataFrame:
        """
        Load ventilator cohort: mechanically ventilated patients.
        
        Detected via respiratoryCharting using vent/intubation/extubation patterns.
        """
        query = f"""
        WITH ventilated_patients AS (
            SELECT DISTINCT patientunitstayid
            FROM respiratoryCharting
            WHERE 
                LOWER(respchartvaluelabel) LIKE '%vent%' OR 
                LOWER(respchartvaluelabel) LIKE '%intubat%' OR
                LOWER(respchartvaluelabel) LIKE '%extubat%' OR
                LOWER(respchartvalue) LIKE '%vent%' OR
                LOWER(respchartvalue) LIKE '%intubat%'
        )
        SELECT 
            p.patientunitstayid::INTEGER AS patientunitstayid,
            p.patienthealthsystemstayid::INTEGER AS patienthealthsystemstayid,
            p.gender,
            CASE 
                WHEN p.age LIKE '> %' THEN 90
                WHEN p.age LIKE '< %' THEN 18
                ELSE CAST(p.age AS INTEGER)
            END AS age,
            p.unitadmittime24 AS unitadmittime24,
            p.unitdischargetime24 AS unitdischargetime24,
            CASE WHEN p.hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END AS hospital_expire_flag
        FROM patient p
        INNER JOIN ventilated_patients vp ON p.patientunitstayid = vp.patientunitstayid
        WHERE 
            CASE 
                WHEN p.age LIKE '> %' THEN 90
                WHEN p.age LIKE '< %' THEN 18
                ELSE CAST(p.age AS INTEGER)
            END BETWEEN {age_min} AND {age_max}
            AND CAST(p.unitdischargeoffset AS INTEGER) / 60.0 >= {min_los_hours}
        ORDER BY p.patientunitstayid
        """
        return self.conn.execute(query).fetchdf()
    
    def load_labs(self, 
                 lab_names: List[str],
                 patient_unit_stay_ids: List[int]) -> pd.DataFrame:
        """
        Load lab values for specified labs and patients.
        
        Args:
            lab_names: List of lab names (e.g., ['creatinine', 'bilirubin', 'lactate'])
            patient_unit_stay_ids: List of patientunitstayid values
        
        Returns:
            DataFrame with columns: patientunitstayid, labresultoffset (minutes), 
            labname, labresult (numeric value)
        """
        # Normalize lab names to lowercase to match the DB query on LOWER(labname)
        lab_names_lower = [name.lower() for name in lab_names]
        lab_names_str = "', '".join(lab_names_lower)
        stay_ids_str = ",".join(map(str, patient_unit_stay_ids))
        
        query = f"""
        SELECT 
            l.patientunitstayid::INTEGER AS patientunitstayid,
            CAST(l.labresultoffset AS REAL) / 60 AS lab_time_hours,  -- Convert minutes to hours
            l.labname,
            CAST(l.labresult AS REAL) AS labresult
        FROM lab l
        WHERE 
            l.patientunitstayid::INTEGER IN ({stay_ids_str})
            AND LOWER(l.labname) IN ('{lab_names_str}')
            AND TRY_CAST(l.labresult AS REAL) IS NOT NULL
            AND CAST(l.labresult AS REAL) IS NOT NULL
        ORDER BY l.patientunitstayid, l.labresultoffset
        """
        return self.conn.execute(query).fetchdf()
    
    def load_vitals(self,
                   patient_unit_stay_ids: List[int]) -> pd.DataFrame:
        """
        Load vital signs from vitalPeriodic: HR, respiration, temperature, SpO2.
        Note: non-invasive BP is in vitalAperiodic (load separately if needed).

        Args:
            patient_unit_stay_ids: List of patientunitstayid values

        Returns:
            DataFrame with columns: patientunitstayid, vital_time_hours,
            heart_rate, temperature, respiratory_rate, o2_sat
        """
        stay_ids_str = ",".join(map(str, patient_unit_stay_ids))

        query = f"""
        SELECT 
            v.patientunitstayid::INTEGER AS patientunitstayid,
            CAST(v.observationoffset AS REAL) / 60 AS vital_time_hours,
            TRY_CAST(v.heartrate AS REAL) AS heart_rate,
            TRY_CAST(v.temperature AS REAL) AS temperature,
            TRY_CAST(v.respiration AS REAL) AS respiratory_rate,
            TRY_CAST(v.sao2 AS REAL) AS o2_sat
        FROM vitalPeriodic v
        WHERE v.patientunitstayid::INTEGER IN ({stay_ids_str})
        ORDER BY v.patientunitstayid, v.observationoffset
        """
        return self.conn.execute(query).fetchdf()
    
    def load_vitals_bp(self,
                      patient_unit_stay_ids: List[int]) -> pd.DataFrame:
        """
        Load non-invasive blood pressure from vitalAperiodic table.

        Args:
            patient_unit_stay_ids: List of patientunitstayid values

        Returns:
            DataFrame with columns: patientunitstayid, vital_time_hours,
            systolic, diastolic, mean_bp
        """
        stay_ids_str = ",".join(map(str, patient_unit_stay_ids))

        query = f"""
        SELECT 
            v.patientunitstayid::INTEGER AS patientunitstayid,
            CAST(v.observationoffset AS REAL) / 60 AS vital_time_hours,
            TRY_CAST(v.noninvasivesystolic AS REAL) AS systolic,
            TRY_CAST(v.noninvasivediastolic AS REAL) AS diastolic,
            TRY_CAST(v.noninvasivemean AS REAL) AS mean_bp
        FROM vitalAperiodic v
        WHERE v.patientunitstayid::INTEGER IN ({stay_ids_str})
        ORDER BY v.patientunitstayid, v.observationoffset
        """
        return self.conn.execute(query).fetchdf()
    
    def load_medications(self,
                        drug_names: List[str],
                        patient_unit_stay_ids: List[int],
                        match_mode: str = "exact") -> pd.DataFrame:
        """
        Load medication infusions (vasopressors, diuretics, etc.).
        
        Args:
            drug_names: List of drug names to search (e.g., ['norepinephrine', 'vasopressin'])
            patient_unit_stay_ids: List of patientunitstayid values
        
        Returns:
            DataFrame with columns: patientunitstayid, med_time_hours, drugname, infusionrate
        """
        if not drug_names or not patient_unit_stay_ids:
            return pd.DataFrame(columns=["patientunitstayid", "med_time_hours", "drugname", "infusionrate"])

        drug_names_lower = [name.lower() for name in drug_names]
        drug_names_str = "', '".join(drug_names_lower)
        stay_ids_str = ",".join(map(str, patient_unit_stay_ids))
        
        if match_mode == "contains":
            like_clauses = " OR ".join([f"LOWER(i.drugname) LIKE '%{name}%'" for name in drug_names_lower])
            drug_filter = f"({like_clauses})"
        else:
            drug_filter = f"LOWER(i.drugname) IN ('{drug_names_str}')"

        query = f"""
        SELECT 
            i.patientunitstayid::INTEGER AS patientunitstayid,
            CAST(i.infusionoffset AS REAL) / 60 AS med_time_hours,  -- Convert minutes to hours
            i.drugname,
            CAST(i.infusionrate AS REAL) AS infusionrate
        FROM infusionDrug i
        WHERE 
            i.patientunitstayid::INTEGER IN ({stay_ids_str})
            AND {drug_filter}
        ORDER BY i.patientunitstayid, i.infusionoffset
        """
        return self.conn.execute(query).fetchdf()
    
    def load_diagnoses(self,
                      patient_unit_stay_ids: List[int]) -> pd.DataFrame:
        """
        Load ICD-9 diagnoses for patients.
        
        Args:
            patient_unit_stay_ids: List of patientunitstayid values
        
        Returns:
            DataFrame with columns: patientunitstayid, icd9code, diagnosisstring
        """
        stay_ids_str = ",".join(map(str, patient_unit_stay_ids))
        
        query = f"""
        SELECT 
            d.patientunitstayid::INTEGER AS patientunitstayid,
            d.icd9code,
            d.diagnosisstring
        FROM diagnosis d
        WHERE 
            d.patientunitstayid::INTEGER IN ({stay_ids_str})
        ORDER BY d.patientunitstayid, d.icd9code
        """
        return self.conn.execute(query).fetchdf()
    
    def close(self):
        """Close DuckDB connection."""
        self.conn.close()


if __name__ == "__main__":
    # Test loader
    print("Testing eICU Loader...")
    loader = EICULoader()
    
    # Load all 4 cohorts
    print("=== Cohort Sizes ===")
    aki = loader.load_aki_cohort()
    print(f"AKI: {len(aki):,} patients")
    
    liver = loader.load_liver_cohort()
    print(f"Liver: {len(liver):,} patients")
    
    sepsis = loader.load_sepsis_cohort()
    print(f"Sepsis: {len(sepsis):,} patients")
    
    vent = loader.load_ventilator_cohort()
    print(f"Ventilator: {len(vent):,} patients")
    
    # Quick data availability check (single patient)
    print("\n=== Quick Data Check (1 AKI patient) ===")
    sample_ids = aki['patientunitstayid'].head(1).tolist()
    
    labs = loader.load_labs(['creatinine'], sample_ids)
    print(f"Labs (creatinine): {len(labs)} measurements")
    
    vitals = loader.load_vitals(sample_ids)
    print(f"Vitals: {len(vitals)} measurements")
    
    bp = loader.load_vitals_bp(sample_ids)
    print(f"BP: {len(bp)} measurements")
    
    print("\n✓ Loader verified - ready for preprocessing")
