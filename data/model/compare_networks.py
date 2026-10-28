import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import build_network as nb

# Config +++
DATA_FILE = Path("./../data/processed/clean_2019.csv")
DICTIONARY_FILE = Path("./../data/preprocessed/dictionary.csv")
OUT_DIR = Path("./../data/network")
N_BOOTSTRAP = 100 #Comparison of networks using resamplling
STABLE = 0.5   # a link is stable if it appears in at least half of the resampled networks

LEVEL2_GROUPS = [   # WHO second-level cause groups
    "infectious_and_parasitic_diseases", "respiratory_infectious", "neonatal_conditions",
    "nutritional_deficiencies", "malignant_neoplasms", "endocrine_blood_immune_disorders",
    "mental_and_substance_use_disorders", "neurological_conditions", "cardiovascular_diseases",
    "respiratory_diseases", "digestive_diseases", "genitourinary_diseases", "musculoskeletal_diseases",
    "congenital_anomalies", "unintentional_injuries", "intentional_injuries",
]
OWN_GROUP = ["maternal_conditions", "other_neoplasms", "diabetes_mellitus", "skin_diseases",
             "sudden_infant_death_syndrome"]   # causes without a second-level group

KNOWN_PAIRS = [   # cause pairs with a known shared cause (risk factor, infection, vaccine, ...)
    ("alcohol_use_disorders", "cirrhosis_due_to_alcohol_use"),
    ("alcohol_use_disorders", "liver_cancer_secondary_to_alcohol_use"),
    ("cirrhosis_due_to_alcohol_use", "liver_cancer_secondary_to_alcohol_use"),
    ("cirrhosis_due_to_hepatitis_b", "liver_cancer_secondary_to_hepatitis_b"),
    ("cirrhosis_due_to_hepatitis_c", "liver_cancer_secondary_to_hepatitis_c"),
    ("acute_hepatitis_b", "cirrhosis_due_to_hepatitis_b"),
    ("diabetes_mellitus", "chronic_kidney_disease_due_to_diabetes"),
    ("ischaemic_heart_disease", "ischaemic_stroke"),
    ("hypertensive_heart_disease", "haemorrhagic_stroke"),
    ("trachea_bronchus_lung_cancers", "chronic_obstructive_pulmonary_disease"),
    ("trachea_bronchus_lung_cancers", "larynx_cancer"),
    ("larynx_cancer", "lip_and_oral_cavity"),
    ("hiv_aids", "tuberculosis"),
    ("hiv_aids", "cervix_uteri_cancer"),
    ("measles", "whooping_cough"),
    ("diphtheria", "tetanus"),
    ("diarrhoeal_diseases", "protein_energy_malnutrition"),
    ("preterm_birth_complications", "neonatal_sepsis_and_infections"),
    ("maternal_conditions", "birth_asphyxia_and_birth_trauma"),
    ("opioid_use_disorders", "cocaine_use_disorders"),
    ("thalassaemias", "other_haemoglobinopathies_and_haemolytic_anaemias"),
    ("breast_cancer", "ovary_cancer"),
]
# Config ---


def who_groups(dictionary):
    """WHO second-level group of every cause (e.g. tuberculosis -> infectious and parasitic diseases).

    Parameters
    dictionary: pandas.DataFrame, dictionary.csv (causes in WHO order)

    Returns
    dict, cause column -> group name
    """
    groups, current = {}, None
    for col in dictionary.loc[dictionary["group"] == "cause_of_death", "column"]:
        name = col.replace("death_asdr_", "")
        if name in LEVEL2_GROUPS or name in ("communicable_maternal_perinatal_and_nutritional_conditions",
                                             "noncommunicable_diseases", "injuries"):
            current = name
        groups[col] = name if name in OWN_GROUP else current
    return groups


def links(data, causes, method, alpha=None):
    """Link strength matrix and the set of links above the threshold.

    Parameters
    data: pandas.DataFrame, cleaned dataset (or a resample)
    causes: list of str
    method: "glasso" or "correlation"
    alpha: float or None, graphical lasso penalty

    Returns
    pandas.DataFrame, link strength matrix
    set of tuple, links (cause a, cause b) with strength >= threshold
    float or None, alpha used
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") #Ignore warning from glasso
        matrix, alpha = nb.cause_links(data, causes, method, alpha)
    values = matrix.values
    rows, cols = np.where(np.triu(values >= nb.THRESHOLD[method], k=1)) # upper triangle of matrix, excluding diagonal
    return matrix, {(causes[i], causes[j]) for i, j in zip(rows, cols)}, alpha


def one_resample(data, causes, method, alpha, i):
    """Links of one resampled network.

    Returns
    set of tuple, or empty set if the model could not be fitted
    """
    rows = np.random.default_rng(nb.SEED + i).integers(0, len(data), len(data))
    try:
        return links(data.iloc[rows].reset_index(drop=True), causes, method, alpha)[1]
    except FloatingPointError:
        return set()


def link_stability(data, causes, method, alpha):
    """Share of resampled networks in which every link appears.

    Returns
    dict, link -> share (0-1)
    """
    runs = Parallel(n_jobs=-1)(delayed(one_resample)(data, causes, method, alpha, i) for i in range(N_BOOTSTRAP))
    counts = {}
    for run in runs:
        for link in run:
            counts[link] = counts.get(link, 0) + 1
    return {link: n / len(runs) for link, n in counts.items()}


def link_table(matrix, edges, stab, groups, other_edges):
    """One row per link with strength, stability, same WHO group, known pair, in the other network.

    Returns
    pandas.DataFrame
    """
    known = {frozenset(("death_asdr_" + a, "death_asdr_" + b)) for a, b in KNOWN_PAIRS}
    rows = [{"a": nb.label(a), "b": nb.label(b), "strength": matrix.loc[a, b],
             "stability": stab.get((a, b), 0.0), "same_group": groups[a] == groups[b],
             "known": frozenset((a, b)) in known, "in_other": (a, b) in other_edges,
             "group_a": groups[a], "group_b": groups[b]} for a, b in edges]
    return pd.DataFrame(rows).sort_values("strength", ascending=False)


def summary(table, name):
    """Print the quality numbers of one network."""
    stable = table["stability"] >= STABLE
    print(f"\n{name}: {len(table)} links")
    print(f"  stable links (in >= {STABLE:.0%} of resamples): {stable.mean():.0%}")
    print(f"  same WHO group: all links {table['same_group'].mean():.0%} | stable {table.loc[stable, 'same_group'].mean():.0%}"
          f" | unstable {table.loc[~stable, 'same_group'].mean():.0%}")
    print(f"  known pairs found: {table['known'].sum()} / {len(KNOWN_PAIRS)}")


def main():
    data = pd.read_csv(DATA_FILE)
    causes = [c for c in data.columns if c.startswith("death_asdr_")]
    groups = who_groups(pd.read_csv(DICTIONARY_FILE))
    n_pairs = len(causes) * (len(causes) - 1) / 2
    same_pairs = sum(groups[a] == groups[b] for i, a in enumerate(causes) for b in causes[i + 1:])
    print(f"baseline: {same_pairs / n_pairs:.0%} of all possible pairs are in the same WHO group")

    corr, corr_edges, _ = links(data, causes, "correlation")
    glasso, glasso_edges, alpha = links(data, causes, "glasso")
    corr_stab = link_stability(data, causes, "correlation", None)
    glasso_stab = link_stability(data, causes, "glasso", alpha)

    corr_table = link_table(corr, corr_edges, corr_stab, groups, glasso_edges)
    glasso_table = link_table(glasso, glasso_edges, glasso_stab, groups, corr_edges)
    summary(corr_table, "CORRELATION (co-occurrence, |rho| >= 0.35)")
    summary(glasso_table, f"GLASSO (direct links, >= 0.05, alpha {alpha:.3f})")
    print(f"\nshared links: {len(corr_edges & glasso_edges)} | only correlation: {len(corr_edges - glasso_edges)}"
          f" | only glasso: {len(glasso_edges - corr_edges)}")

    print("\nglasso links by strength:")
    for low, high in [(0.05, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 1)]:
        part = glasso_table[glasso_table["strength"].between(low, high, inclusive="left")]
        print(f"  {low:.2f}-{high:.2f}: {len(part):3d} links | stable {np.mean(part['stability'] >= STABLE):.0%}"
              f" | same group {part['same_group'].mean():.0%} | known {part['known'].sum()}")

    print("\nknown pairs:")
    for a, b in KNOWN_PAIRS:
        a, b = "death_asdr_" + a, "death_asdr_" + b
        a, b = sorted((a, b), key=causes.index)
        print(f"  {nb.label(a)[:30]:30s} - {nb.label(b)[:30]:30s} corr {corr.loc[a, b]:+.2f} {'Y' if (a, b) in corr_edges else '.'}"
              f" | glasso {glasso.loc[a, b]:+.2f} {'Y' if (a, b) in glasso_edges else '.'}")

    corr_table.to_csv(OUT_DIR / "links_correlation.csv", index=False)
    glasso_table.to_csv(OUT_DIR / "links_glasso.csv", index=False)


if __name__ == "__main__":
    main()