from matplotlib.table import table
import numpy as np
import pandas as pd
from pathlib import Path

# Config +++
IN_DIR = Path("./../data/preprocessed")
OUT_DIR = Path("./../data/processed")

YEAR = 2019 #Year used for analysis, 2019 due to WHO reporting
WINDOW_START = 2013 # start of period
MAX_MISSING_COUNTRY = 0.2 #Exclude countries with more than 20% missing indicator values

# Indicators to drop based on findings in "initial_eda.ipynb", which contains the explanations
DROP_INDICATORS = ["mri_per1m", "caesarean_rate", "gamma_camera_per1m", "pet_per1m", "hpv_coverage", "avail_chemotherapy", "air_pollution_mortality","linear_accelerator_per1m", "telecobalt_per1m","death_registration_completeness","hepb3_coverage", "hib3_coverage", "pcv3_coverage", "rota_coverage"]

DATA_ERRORS = [
    ("POL", 2019, "ct_per1m"), #Correct issue within data -> "initial_eda.ipynb"
]

# Remove redundant causes and causes with extremly low death rates
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

ID_COLS = ["iso3", "country", "region", "income_group"] #Identifiers
# Config ---


def latest_indicator_values(data, indicators):
    """Latest non-missing value of each indicator per country between WINDOW_START and YEAR.

    Parameters
    data: pandas.DataFrame, the merged country-year dataset
    indicators: list of str, indicator columns

    Returns
    pandas.DataFrame with one row per country
    """

    recent = data[data["year"].between(WINDOW_START, YEAR)].sort_values("year") # Recent years sorted by year
    latest = recent.groupby("iso3")[indicators].last() # Latest values per country

    # Convert battle deaths to a rate per 100,000 inhabitants.
    # Missing battle-death observations are currently treated as zero;
    # this assumption requires verification against the source metadata.
    if "battle_deaths" in latest:
        latest["battle_deaths"] = latest["battle_deaths"].fillna(0) / latest["population"] * 100_000
        latest = latest.rename(columns={"battle_deaths": "battle_deaths_per100k"})

    return latest


def select_causes(data, cause_cols):
    """Cause-of-death rates in YEAR, without group causes and causes with (almost) no deaths.

    A cause is dropped if it is 0 in at least 90% of countries or never above 0.5 per 100,000.

    Parameters
    data: pandas.DataFrame, main dataset
    cause_cols: list of str, cause of death columns

    Returns
    pandas.DataFrame, one row per country
    """
    causes = data[data["year"] == YEAR].set_index("iso3")[cause_cols] # Select causes for the specified year
    no_deaths = ((causes == 0).mean() >= 0.9) | (causes.max() < 0.5) # Causes with no deaths in majority of country
    groups = ["death_asdr_" + g for g in GROUP_CAUSES] #form groups
    keep = [c for c in cause_cols if not no_deaths[c] and c not in groups] #Keep
    print(f"Cause Selection: Dropped \n Dropped: \n No Deaths: {no_deaths.sum()} , Groups: {len(groups)}, \n Kept: {len(keep)}")
    return causes[keep]


def fill_missing(table, income_group):
    """Fill using median of income group, followed by overall median if still missing.

    Parameters
    table: pandas.DataFrame, build table from other functions
    income_group: pandas.Series, income group per country 

    Returns
    pandas.DataFrame, final table with filled values
    """
    group_median = table.groupby(income_group).transform("median")
    return table.fillna(group_median).fillna(table.median())


def log_skewed(table):
    """Use (1 + x) to reduce skewness of columns with skew > 1

    Parameters
    table: pandas.DataFrame, numeric

    Returns
    pandas.DataFrame
    """

    binary = [c for c in table if set(table[c].unique()) <= {0, 1}] # Binary columns
    skewed = [c for c in table if c not in binary and table[c].skew() > 1 and table[c].min() >= 0] #Exclude binary and neg. values
    
    out = table.copy() 
    out[skewed] = np.log1p(out[skewed]) #Log transformation of skewed columns

    return out.copy(), skewed


def main():
    """Main Function"""
    OUT_DIR.mkdir(parents=True, exist_ok=True) #Create output dir

    # Load data
    data = pd.read_csv(IN_DIR / "global_health_dataset.csv")
    dictionary = pd.read_csv(IN_DIR / "dictionary.csv")

    # Validate the uniqueness of country-year observations
    if data.duplicated(subset=["iso3", "year"]).any():
        raise ValueError(
            "Duplicate country-year records found in the input dataset."
        )

    #Correct known data errors identified during initial EDA
    for iso3, year, column in DATA_ERRORS:
        data.loc[(data["iso3"] == iso3) & (data["year"] == year), column] = np.nan


    is_cause = dictionary["group"] == "cause_of_death"
    indicators = [c for c in dictionary.loc[~is_cause, "column"] if c not in DROP_INDICATORS]
    cause_cols = dictionary.loc[is_cause, "column"].tolist()

    # Cause selection and indicator selection
    meta = data.drop_duplicates("iso3").set_index("iso3")[ID_COLS[1:]]
    latest = latest_indicator_values(data, indicators)
    indicators = latest.columns.tolist()  # Include the battle deaths correction
    table = latest.join(select_causes(data, cause_cols)).copy()

    # Countries dropped based on missing values
    share_missing = table[indicators].isna().mean(axis=1)
    dropped = share_missing[share_missing > MAX_MISSING_COUNTRY].index
    print(f"Countries dropped : {list(meta.loc[dropped, 'country'])}")#use list to avoid index printing

    table = table.drop(index=dropped) #Drop columns

    # Impute missing values using income-group medians,
    # followed by the global median when necessary
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
    
    # 
    clean = meta.loc[table.index].join(table).reset_index()
    clean.to_csv(OUT_DIR / f"clean_{YEAR}.csv", index=False)

    # Log Transform + Write to CSV
    logged, skewed = log_skewed(table)
    meta.loc[table.index].join(logged).reset_index().to_csv(OUT_DIR / f"clean_{YEAR}_log.csv", index=False)

    print('Data Cleaning finished')


if __name__ == "__main__":
    main()