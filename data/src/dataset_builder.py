import json
import re
from pathlib import Path
import pandas as pd

# Config +++
RAW_DIR = Path("./../data/raw")
OUT_DIR = Path("./../data/preprocessed")

start_year, last_year = 2000, 2023 #Starting Year and End Year of Data
ONLY_COUNTRIES_WITH_CAUSE_DATA = True

CAUSE_MEASURES = ["asdr"]
CAUSE_SEXES = ["both"]

GHE_FILES = {"asdr": "ghe2023_asdr_bycountry.xlsx", "crude": "ghe2023_crude_bycountry.xlsx"}

# Mapped Data
WORLDBANK = {
    "SH.XPD.CHEX.GD.ZS": ("health_exp_pct_gdp", "system_input"),
    "SH.XPD.CHEX.PP.CD": ("health_exp_pc_ppp", "system_input"),
    "SH.XPD.OOPC.CH.ZS": ("out_of_pocket_pct_che", "system_input"),
    "SH.XPD.GHED.CH.ZS": ("gov_health_exp_pct_che", "system_input"),
    "SH.XPD.EHEX.CH.ZS": ("external_health_exp_pct_che", "system_input"),
    "SH.IMM.IDPT": ("dtp3_coverage", "system_output"),
    "SH.IMM.MEAS": ("measles_coverage", "system_output"),
    "SH.STA.BRTC.ZS": ("skilled_birth_attendance", "system_output"),
    "SH.DYN.MORT": ("under5_mortality", "outcome"),
    "SP.DYN.IMRT.IN": ("infant_mortality", "outcome"),
    "SH.STA.MMRT": ("maternal_mortality_ratio", "outcome"),
    "SH.TBS.INCD": ("tb_incidence", "outcome"),
    "SH.DYN.AIDS.ZS": ("hiv_prevalence_15_49", "outcome"),
    "EN.ATM.PM25.MC.M3": ("pm25_exposure", "risk_factor"),
    "SH.H2O.BASW.ZS": ("basic_water_access", "risk_factor"),
    "SH.STA.BASS.ZS": ("basic_sanitation_access", "risk_factor"),
    "NY.GDP.PCAP.PP.KD": ("gdp_pc_ppp", "context"),
    "NY.GDP.MKTP.KD.ZG": ("gdp_growth", "context"),
    "SI.POV.DDAY": ("extreme_poverty", "context"),
    "SP.POP.TOTL": ("population", "context"),
    "SP.POP.65UP.TO.ZS": ("pop_65plus_pct", "context"),
    "SP.URB.TOTL.IN.ZS": ("urban_pct", "context"),
    "SP.DYN.TFRT.IN": ("fertility_rate", "context"),
    "VC.BTL.DETH": ("battle_deaths", "context"),
    "GOV_WGI_GE_SC": ("government_effectiveness", "context"),
    "SP.REG.DTHS.ZS": ("death_registration_completeness", "data_quality"),
} # Indicators Worldbank Dataset

#WHO
WHO_GHO = {
    "WHOSIS_000001": ("life_expectancy", "outcome"),
    "WHOSIS_000002": ("hale", "outcome"),
    "NCDMORT3070": ("ncd_mortality_30_70", "outcome"),
    "SDGMALARIA": ("malaria_incidence", "outcome"),
    "SDGHEPHBSAGPRV": ("hepb_prevalence", "outcome"),
    "SDGWSHBOD": ("wash_mortality", "outcome"),
    "AIR_42": ("air_pollution_mortality", "outcome"),
    "WHS6_102": ("beds_per10k", "system_input"),
    "HWF_0001": ("doctors_per10k", "system_input"),
    "HWF_0006": ("nurses_per10k", "system_input"),
    "HWF_0010": ("dentists_per10k", "system_input"),
    "HWF_0014": ("pharmacists_per10k", "system_input"),
    "MH_6": ("psychiatrists_per100k", "system_input"),
    "DEVICES08": ("mri_per1m", "system_input"),
    "DEVICES09": ("ct_per1m", "system_input"),
    "DEVICES10": ("pet_per1m", "system_input"),
    "DEVICES11": ("gamma_camera_per1m", "system_input"),
    "DEVICES12": ("linear_accelerator_per1m", "system_input"),
    "DEVICES21": ("telecobalt_per1m", "system_input"),
    "DEVICES22": ("radiotherapy_per1m", "system_input"),
    "NCD_CCS_dialysis": ("avail_dialysis", "treatment_availability"),
    "NCD_CCS_renaltrans": ("avail_kidney_transplant", "treatment_availability"),
    "NCD_CCS_chemo": ("avail_chemotherapy", "treatment_availability"),
    "NCD_CCS_coronary": ("avail_coronary_bypass_stent", "treatment_availability"),
    "NCD_CCS_thrombolytic": ("avail_thrombolysis", "treatment_availability"),
    "NCD_CCS_stroke": ("avail_stroke_care", "treatment_availability"),
    "NCD_CCS_Insulin": ("avail_insulin", "treatment_availability"),
    "NCD_CCS_Metformin": ("avail_metformin", "treatment_availability"),
    "NCD_CCS_Statins": ("avail_statins", "treatment_availability"),
    "NCD_CCS_ACE": ("avail_ace_inhibitors", "treatment_availability"),
    "NCD_CCS_Bronchodilators": ("avail_bronchodilators", "treatment_availability"),
    "NCD_CCS_DiabetesTest": ("avail_diabetes_testing", "treatment_availability"),
    "NCD_CCS_CervCy": ("avail_cervical_cytology", "treatment_availability"),
    "NCD_CCS_BreastCancer": ("avail_breast_screening", "treatment_availability"),
    "NCD_CCS_BowelCancer": ("avail_colon_screening", "treatment_availability"),
    "NCD_CCS_palliative_prim": ("avail_palliative_care", "treatment_availability"),
    "NCD_CCS_CancerRegNational": ("cancer_registry", "treatment_availability"),
    "UHC_INDEX_REPORTED": ("uhc_index", "system_output"),
    "UHC_SCI_RMNCH": ("uhc_sub_rmnch", "system_output"),
    "UHC_SCI_INFECT": ("uhc_sub_infectious", "system_output"),
    "UHC_SCI_NCD": ("uhc_sub_ncd", "system_output"),
    "UHC_SCI_CAPACITY": ("uhc_sub_capacity", "system_output"),
    "SDGIHR": ("ihr_capacity", "system_output"),
    "WHS4_117": ("hepb3_coverage", "system_output"),
    "WHS4_129": ("hib3_coverage", "system_output"),
    "PCV3": ("pcv3_coverage", "system_output"),
    "ROTAC": ("rota_coverage", "system_output"),
    "MCV2": ("mcv2_coverage", "system_output"),
    "SDGHPVRECEIVED": ("hpv_coverage", "system_output"),
    "WHS4_115": ("caesarean_rate", "system_output"),
    "NCD_HYP_PREVALENCE_A": ("hypertension_prevalence", "risk_factor"),
    "NCD_BMI_30A": ("obesity_prevalence", "risk_factor"),
    "NCD_GLUC_04": ("raised_glucose_prevalence", "risk_factor"),
    "NCD_BMI_MEAN": ("mean_bmi", "risk_factor"),
    "NCD_BMI_18A": ("underweight_prevalence", "risk_factor"),
    "NCD_PAA": ("insufficient_activity", "risk_factor"),
    "NUTSTUNTINGPREV": ("stunting_prevalence", "risk_factor"),
    "M_Est_tob_curr_std": ("tobacco_prevalence", "risk_factor"),
    "SA_0000001688": ("alcohol_per_capita", "risk_factor"),
}

#OWID
OWID = {"human-development-index": ("hdi", "context")}

# Breakdown values that mean "everyone" (used when WHO gives no plain total).
TOTAL_MARKERS = ["18-PLUS", "15-PLUS", "18PLUS", "15PLUS", "GHE000000", "TOTL", "TOTAL", "ALLAGES"]

# Cells like "A." or "1." in the WHO workbooks are list numbering, not cause names.
LIST_MARKER = re.compile(r"^[A-Za-z0-9]{1,4}\.$")

SEX_LABELS = {
    "both sexes": "both", "both": "both", "persons": "both", "total": "both", "btsx": "both",
    "sex_btsx": "both", "males": "male", "male": "male", "mle": "male", "sex_mle": "male",
    "females": "female", "female": "female", "fmle": "female", "sex_fmle": "female",
}

# Config ---

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def to_number(value):
    """Convert a raw value to a float.

    Parameters
    value: object

    Returns
    float or None.
    """

    # Non-existing values handler
    if value is None:
        return None

    #Check if value is numeric
    if isinstance(value, (int, float)):
        return None if pd.isna(value) else float(value) #NaN handler 
    
    #Normalized string
    text = str(value).strip().lower()

    # Conversion into binary for indicators
    if text in ("yes", "generally available", "available"):
        return 1.0
    if text in ("no", "generally not available", "not generally available", "not available", "none"):
        return 0.0

    match = re.search(r"-?\d+(\.\d+)?", text) #Regex Matching
    return float(match.group()) if match else None


def slug(text):
    """String to Slug conversion

    Parameters:
    text: str

    Returns:
    str

    """
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def read_json(folder, code):
    """Load JSON files for indicators

    Parameters
    folder : pathlib.Path
       
    code : str
        Indicator code.

    Returns
    list
    """
    # Sort by page number
    pages = sorted(folder.glob(f"{code}_p*.json"), key=lambda p: int(p.stem.rsplit("_p", 1)[1]))
    return [json.loads(p.read_text(encoding="utf-8")) for p in pages]


def read_worldbank_countries():
    """World Bank country list

    Returns
    pandas.DataFrame
        Columns iso3, country, region, income_group.
    """
    rows = []
    for page in read_json(RAW_DIR / "worldbank", "_countries"):
        for c in page[1] or []:
            region = (c.get("region") or {}).get("value", "").strip()
            if region and region != "Aggregates":
                rows.append({"iso3": c["id"], "country": c["name"], "region": region,"income_group": (c.get("incomeLevel") or {}).get("value")})
    
    return pd.DataFrame(rows, columns=["iso3", "country", "region", "income_group"])


def read_worldbank(countries):
    """Read World Bank indicators while checking for ISO3 country codes

    Parameters
    countries: Set of strings

    Returns
    pandas.DataFrame
        Columns iso3, year, column, value.
    """
    rows = []
    for code, (name, _) in WORLDBANK.items():
        pages = read_json(RAW_DIR / "worldbank", code)
        if not pages or "message" in str(pages[0])[:200]:
            print(f"  world bank: no usable file for {code} ({name})")
            continue
        for page in pages:
            for r in page[1] or []:
                if r.get("countryiso3code") in countries and r.get("value") is not None:
                    rows.append((r["countryiso3code"], int(r["date"]), name, float(r["value"])))
    return pd.DataFrame(rows, columns=["iso3", "year", "column", "value"]) 


def pick_total_rows(records):
    """Keep the 'total' records of one WHO indicator.

    Rules: both sexes (or no sex split); no other breakdown. If every record
    has a breakdown, use the one that means 'everyone' (adults 18+, all
    causes, ...) or, if there is only one breakdown value, that one.

    Parameters
    ----------
    records : list of dict
        Raw WHO GHO records.

    Returns
    -------
    list of dict
        The records to keep.
    """
    def other_dims(r):
        return tuple(str(r.get(f"Dim{i}")) for i in (1, 2, 3)
                     if r.get(f"Dim{i}Type") and r.get(f"Dim{i}Type") != "SEX" and r.get(f"Dim{i}") is not None)

    def is_both(r):
        for i in (1, 2, 3):
            if r.get(f"Dim{i}Type") == "SEX":
                return SEX_LABELS.get(str(r.get(f"Dim{i}")).lower()) == "both"
        return True

    records = [r for r in records if is_both(r)]
    plain = [r for r in records if not other_dims(r)]
    if plain:
        return plain
    breakdowns = {other_dims(r) for r in records}
    if len(breakdowns) == 1:
        return records
    return [r for r in records if any(m in " ".join(other_dims(r)).upper() for m in TOTAL_MARKERS)]


def read_who_gho(countries):
    """Read every WHO GHO indicator listed in WHO_GHO.

    Parameters
    ----------
    countries : set of str
        Valid ISO3 country codes.

    Returns
    -------
    pandas.DataFrame
        Columns iso3, year, column, value.
    """
    rows = []
    for code, (name, _) in WHO_GHO.items():
        pages = read_json(RAW_DIR / "who_gho", code)
        records = [r for page in pages for r in page.get("value", [])]
        records = [r for r in records if r.get("SpatialDimType") == "COUNTRY"
                   and r.get("SpatialDim") in countries and r.get("TimeDimType", "YEAR") == "YEAR"]
        kept = 0
        for r in pick_total_rows(records):
            value = r.get("NumericValue")
            value = float(value) if value is not None else to_number(r.get("Value"))
            if value is not None:
                rows.append((r["SpatialDim"], int(r["TimeDim"]), name, value))
                kept += 1
        if kept == 0:
            print(f"  WHO GHO: no usable data for {code} ({name})")
    return pd.DataFrame(rows, columns=["iso3", "year", "column", "value"])


def read_owid(countries):
    """Read every Our World in Data chart listed in OWID.

    Parameters
    ----------
    countries : set of str
        Valid ISO3 country codes.

    Returns
    -------
    pandas.DataFrame
        Columns iso3, year, column, value.
    """
    frames = []
    for chart, (name, _) in OWID.items():
        path = RAW_DIR / "owid" / f"{chart}.csv"
        if not path.exists():
            print(f"  OWID: missing {path.name}")
            continue
        df = pd.read_csv(path)
        value_col = [c for c in df.columns if c not in ("Entity", "Code", "Year")][0]
        df = df[df["Code"].isin(countries)]
        frames.append(pd.DataFrame({"iso3": df["Code"], "year": df["Year"], "column": name,
                                    "value": df[value_col]}))
    return pd.concat(frames) if frames else pd.DataFrame(columns=["iso3", "year", "column", "value"])


def read_ghe_workbook(path, measure, countries):
    """Read one WHO Global Health Estimates workbook (one sheet per year).

    Each sheet has one row per cause and sex and one column per country
    (a header row with ISO3 codes). The cause name is the right-most text
    cell left of the country columns.

    Parameters
    ----------
    path : pathlib.Path
        The .xlsx file.
    measure : str
        "asdr" or "crude" (stored in the output).
    countries : set of str
        Valid ISO3 country codes (used to find the header row).

    Returns
    -------
    pandas.DataFrame
        Columns iso3, year, measure, sex, cause, value.
    """
    out = []
    for sheet, raw in pd.read_excel(path, sheet_name=None, header=None).items():
        year_match = re.search(r"(19|20)\d{2}", str(sheet))
        if not year_match:
            continue  # e.g. the "Notes" sheet
        year = int(year_match.group())
        raw.columns = range(raw.shape[1])

        # header row = the first row that contains many country codes
        header = None
        for i in range(min(60, len(raw))):
            if raw.iloc[i].astype(str).str.strip().isin(countries).sum() >= 20:
                header = i
                break
        if header is None:
            print(f"  GHE: no country header in sheet {sheet!r}, skipped")
            continue
        head = raw.iloc[header].astype(str).str.strip()
        country_cols = [c for c in raw.columns if head[c] in countries]
        label_cols = list(range(min(country_cols)))
        body = raw.iloc[header + 1:]

        # sex column = the label column whose values are mostly sex labels
        sex_col = None
        for c in label_cols:
            values = body[c].dropna().astype(str).str.strip().str.lower()
            if len(values) and values.isin(SEX_LABELS).mean() > 0.8:
                sex_col = c

        for _, row in body.iterrows():
            # cause = right-most real text cell (skip list markers like "A." / "1.")
            cause = None
            for c in reversed(label_cols):
                v = row[c]
                if c != sex_col and isinstance(v, str) and v.strip() and not LIST_MARKER.match(v.strip()):
                    cause = v.strip()
                    break
            if cause is None or cause.lower().startswith("population"):
                continue
            sex = SEX_LABELS.get(str(row[sex_col]).strip().lower()) if sex_col is not None else "both"
            for c in country_cols:
                out.append((head[c], year, measure, sex, cause, to_number(row[c])))

    return pd.DataFrame(out, columns=["iso3", "year", "measure", "sex", "cause", "value"])


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    """Build global_health_dataset.csv (all indicators + causes, not cleaned) and dictionary.csv."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading country list ...")
    meta = read_worldbank_countries()
    countries = set(meta["iso3"])

    print("Reading WHO cause-of-death workbooks ...")
    causes = []
    for measure in CAUSE_MEASURES:
        causes.append(read_ghe_workbook(RAW_DIR / "who_ghe" / GHE_FILES[measure], measure, countries))
    causes = pd.concat(causes, ignore_index=True)
    causes = causes[causes["sex"].isin(CAUSE_SEXES)]
    print(f"  {causes['cause'].nunique()} causes, {causes['iso3'].nunique()} countries, "
          f"years {sorted(causes['year'].unique())}")

    print("Reading indicators ...")
    indicators = pd.concat([read_worldbank(countries), read_who_gho(countries), read_owid(countries)])

    # --- choose countries and years -----------------------------------------
    keep = sorted(causes["iso3"].unique()) if ONLY_COUNTRIES_WITH_CAUSE_DATA else sorted(countries)
    years = list(range(start_year, last_year + 1))
    grid = pd.MultiIndex.from_product([keep, years], names=["iso3", "year"]).to_frame(index=False)

    # --- indicators: long -> wide -------------------------------------------
    indicators = indicators[indicators["iso3"].isin(keep) & indicators["year"].between(start_year, last_year)]
    wide_ind = indicators.pivot_table(index=["iso3", "year"], columns="column", values="value",
                                      aggfunc="mean").reset_index()

    # --- causes: long -> wide (one column per cause / measure / sex) --------
    causes = causes.copy()
    causes["column"] = "death_" + causes["measure"] + "_" + causes["cause"].map(slug)
    causes.loc[causes["sex"] != "both", "column"] += "_" + causes["sex"]
    dupes = causes.duplicated(["iso3", "year", "column"]).sum()
    if dupes:
        print(f"  warning: {dupes} cause values share a column name and were averaged")
    wide_causes = causes.pivot_table(index=["iso3", "year"], columns="column", values="value",
                                     aggfunc="mean").reset_index()

    # --- merge everything -----------------------------------------------------
    data = grid.merge(meta, on="iso3", how="left")
    data = data.merge(wide_ind, on=["iso3", "year"], how="left")
    data = data.merge(wide_causes, on=["iso3", "year"], how="left")

    ind_cols = sorted(c for c in wide_ind.columns if c not in ("iso3", "year"))
    cause_cols = sorted(c for c in wide_causes.columns if c not in ("iso3", "year"))
    data = data[["iso3", "country", "region", "income_group", "year"] + ind_cols + cause_cols]
    data.to_csv(OUT_DIR / "global_health_dataset.csv", index=False)

    # --- data dictionary: one row per data column ------------------------------
    rows = []
    for source, table in (("World Bank", WORLDBANK), ("WHO GHO", WHO_GHO), ("OWID", OWID)):
        for code, (name, group) in table.items():
            if name in data.columns:
                rows.append({"column": name, "group": group, "source": source, "code": code})
    for column, cause in causes.drop_duplicates("column")[["column", "cause"]].values:
        rows.append({"column": column, "group": "cause_of_death", "source": "WHO GHE 2023", "code": cause})
    pd.DataFrame(rows).to_csv(OUT_DIR / "dictionary.csv", index=False)

    missing = [n for t in (WORLDBANK, WHO_GHO, OWID) for n, _ in t.values() if n not in data.columns]
    print(f"\nDone: {len(data)} rows ({len(keep)} countries x {len(years)} years), "
          f"{len(ind_cols)} indicators + {len(cause_cols)} cause columns")
    print(f"Written to {OUT_DIR / 'global_health_dataset.csv'} and dictionary.csv")
    if missing:
        print("Indicators without data (not in the file):", ", ".join(missing))


if __name__ == "__main__":
    main()