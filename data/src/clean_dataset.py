from matplotlib.table import table
import numpy as np
import pandas as pd
from pathlib import Path

# Settings
IN_DIR = Path("./../data/preprocessed")
OUT_DIR = Path("./../data/processed")

YEAR = 2019 # Analysis year; WHO reporting.
WINDOW_START = 2013 # Window start year.
MAX_MISSING_COUNTRY = 0.2 # Exclude highly incomplete countries.

# Exclusions documented in initial EDA.
DROP_INDICATORS = ["mri_per1m", "caesarean_rate", "gamma_camera_per1m", "pet_per1m", "hpv_coverage", "avail_chemotherapy", "air_pollution_mortality","linear_accelerator_per1m", "telecobalt_per1m","death_registration_completeness","hepb3_coverage", "hib3_coverage", "pcv3_coverage", "rota_coverage"]

DATA_ERRORS = [
    ("POL", 2019, "ct_per1m"), # Correct EDA-identified data errors.
]

# Remove sparse and group causes.
GROUP_CAUSES = [
    "all_causes", "communicable_maternal_perinatal_and_nutritional_conditions",
    "infectious_and_parasitic_diseases", "stds_excluding_hiv", "childhood_cluster_diseases",
    "hepatitis", "parasitic_and_vector_diseases", "intestinal_nematode_infections",
    "respiratory_infectious", "neonatal_conditions", "nutritional_deficiencies",
    "noncommunicable_diseases", "malignant_neoplasms", "mouth_and_oropharynx_cancers",
    "liver_cancer", "melanoma_and_other_skin_cancers", "lymphomas_multiple_myeloma",
    "endocrine_blood_immune_disorders", "mental_and_substance_use_disorders",
    "drug_use_disorders", "childhood_behavioural_disorders", "neurological_conditions",
    "sense_organ_diseases", "cardiovascular_diseases", "stroke", "respiratory_diseases",
    "digestive_diseases", "cirrhosis_of_the_liver", "genitourinary_diseases", "kidney_diseases",
    "musculoskeletal_diseases", "congenital_anomalies", "oral_conditions",
    "injuries", "unintentional_injuries", "intentional_injuries",
]

ID_COLS = ["iso3", "country", "region", "income_group"] # Identifier columns.

def latest_indicator_values(data, indicators):
    """Get each country's latest indicators."""

    recent = data[data["year"].between(WINDOW_START, YEAR)].sort_values("year") # Sort recent years.
    latest = recent.groupby("iso3")[indicators].last() # Get latest country values.

    # Convert battle deaths to rates.
    # Missing values become zero.
    # Verify against source metadata.
    if "battle_deaths" in latest:
        latest["battle_deaths"] = latest["battle_deaths"].fillna(0) / latest["population"] * 100_000
        latest = latest.rename(columns={"battle_deaths": "battle_deaths_per100k"})

    return latest

def select_causes(data, cause_cols):
    """Select eligible causes for YEAR."""
    causes = data[data["year"] == YEAR].set_index("iso3")[cause_cols] # Select analysis-year causes.
    no_deaths = ((causes == 0).mean() >= 0.9) | (causes.max() < 0.5) # Exclude mostly-zero causes.
    groups = ["death_asdr_" + g for g in GROUP_CAUSES] # Build group names.
    keep = [c for c in cause_cols if not no_deaths[c] and c not in groups] # Keep eligible causes.
    print(f"Cause Selection: Dropped \n Dropped: \n No Deaths: {no_deaths.sum()} , Groups: {len(groups)}, \n Kept: {len(keep)}")
    return causes[keep]

def fill_missing(table, income_group):
    """Impute by group, then global median."""
    group_median = table.groupby(income_group).transform("median")
    return table.fillna(group_median).fillna(table.median())

def log_skewed(table):
    """Log-transform nonnegative skewed columns."""

    binary = [c for c in table if set(table[c].unique()) <= {0, 1}] # Find binary columns.
    skewed = [c for c in table if c not in binary and table[c].skew() > 1 and table[c].min() >= 0] # Exclude binary and negative columns.

    out = table.copy() 
    out[skewed] = np.log1p(out[skewed]) # Transform skewed columns.

    return out.copy(), skewed

def main():
    """Build cleaned datasets."""
    OUT_DIR.mkdir(parents=True, exist_ok=True) # Create output directory.

    # Load source data.
    data = pd.read_csv(IN_DIR / "global_health_dataset.csv")
    dictionary = pd.read_csv(IN_DIR / "dictionary.csv")

    # Validate country-year uniqueness.
    if data.duplicated(subset=["iso3", "year"]).any():
        raise ValueError(
            "Duplicate country-year records found in the input dataset."
        )

    # Apply known data corrections.
    for iso3, year, column in DATA_ERRORS:
        data.loc[(data["iso3"] == iso3) & (data["year"] == year), column] = np.nan

    is_cause = dictionary["group"] == "cause_of_death"
    indicators = [c for c in dictionary.loc[~is_cause, "column"] if c not in DROP_INDICATORS]
    cause_cols = dictionary.loc[is_cause, "column"].tolist()

    # Select causes and indicators.
    meta = data.drop_duplicates("iso3").set_index("iso3")[ID_COLS[1:]]
    latest = latest_indicator_values(data, indicators)
    indicators = latest.columns.tolist()  # Include corrected battle deaths.
    table = latest.join(select_causes(data, cause_cols)).copy()

    # Filter incomplete countries.
    share_missing = table[indicators].isna().mean(axis=1)
    dropped = share_missing[share_missing > MAX_MISSING_COUNTRY].index
    print(f"Countries dropped : {list(meta.loc[dropped, 'country'])}")# Print country names.

    table = table.drop(index=dropped) # Drop excluded rows.

    # Impute by income group.
    # Fall back to global median.
    missing_before = int(table.isna().sum().sum())

    table = fill_missing(
    table, meta.loc[table.index, "income_group"]
    )

    missing_after = int(table.isna().sum().sum())

    if missing_after > 0:
        raise ValueError(
            f"{missing_after} missing values remain after imputation."
        )

    print(f"Missing values filled: {missing_before - missing_after}")

    clean = meta.loc[table.index].join(table).reset_index()
    clean.to_csv(OUT_DIR / f"clean_{YEAR}.csv", index=False)

    # Save original and logged data.
    logged, skewed = log_skewed(table)
    meta.loc[table.index].join(logged).reset_index().to_csv(OUT_DIR / f"clean_{YEAR}_log.csv", index=False)

    print('Data Cleaning finished')

if __name__ == "__main__":
    main()