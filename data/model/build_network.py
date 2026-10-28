import json
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from sklearn.metrics import adjusted_rand_score

# Config +++
DATA_FILE = Path("./../data/processed/clean_2019.csv")
DICTIONARY_FILE = Path("./../data/preprocessed/dictionary.csv")
OUT_DIR = Path("./../data/network")

CONTROL = "gdp_pc_ppp"  # wealth effect removed from every cause before correlating

# How causes are linked (both networks are built and exported, the viewer switches between them):
#   "glasso"      = direct links: partial correlation between two causes given ALL other causes
#                   (graphical lasso). Removes links that only exist through a shared pattern.
#   "correlation" = co-occurrence: plain correlation between two causes (GDP removed only)
METHODS = ["glasso", "correlation"]
THRESHOLD = {"glasso": 0.05, "correlation": 0.35}  # minimum link strength for a cause-cause edge
                                                   # (0.35 = significant after Bonferroni, n = 182)
SEED = 42               # makes the community detection reproducible
N_BOOTSTRAP = 200       # number of resampled networks for the stability check (runs on all CPU cores)

# Health-system layer: indicators as a second type of node
INDICATOR_THRESHOLD = 0.35  # minimum |correlation| for an indicator-cause edge (both signs are kept)
NOT_IN_NETWORK = [          # indicators that are not used as nodes (see extended EDA)
    "population", "gdp_growth", "infant_mortality", "hale", "maternal_mortality_ratio",
]

# Links we expect from medical knowledge: (indicator, cause, expected sign)
# -1 = more of the indicator, fewer deaths   +1 = more of the indicator, more deaths
EXPECTED_LINKS = [
    ("dtp3_coverage", "death_asdr_diphtheria", -1),
    ("dtp3_coverage", "death_asdr_tetanus", -1),
    ("dtp3_coverage", "death_asdr_whooping_cough", -1),
    ("measles_coverage", "death_asdr_measles", -1),
    ("skilled_birth_attendance", "death_asdr_maternal_conditions", -1),
    ("skilled_birth_attendance", "death_asdr_birth_asphyxia_and_birth_trauma", -1),
    ("skilled_birth_attendance", "death_asdr_neonatal_sepsis_and_infections", -1),
    ("basic_water_access", "death_asdr_diarrhoeal_diseases", -1),
    ("basic_sanitation_access", "death_asdr_diarrhoeal_diseases", -1),
    ("avail_dialysis", "death_asdr_chronic_kidney_disease_due_to_diabetes", -1),
    ("avail_dialysis", "death_asdr_other_chronic_kidney_disease", -1),
    ("avail_insulin", "death_asdr_diabetes_mellitus", -1),
    ("avail_thrombolysis", "death_asdr_ischaemic_stroke", -1),
    ("avail_coronary_bypass_stent", "death_asdr_ischaemic_heart_disease", -1),
    ("avail_cervical_cytology", "death_asdr_cervix_uteri_cancer", -1),
    ("avail_bronchodilators", "death_asdr_asthma", -1),
    ("tb_incidence", "death_asdr_tuberculosis", +1),
    ("hiv_prevalence_15_49", "death_asdr_hiv_aids", +1),
    ("hepb_prevalence", "death_asdr_liver_cancer_secondary_to_hepatitis_b", +1),
    ("hepb_prevalence", "death_asdr_cirrhosis_due_to_hepatitis_b", +1),
    ("alcohol_per_capita", "death_asdr_alcohol_use_disorders", +1),
    ("alcohol_per_capita", "death_asdr_cirrhosis_due_to_alcohol_use", +1),
    ("tobacco_prevalence", "death_asdr_trachea_bronchus_lung_cancers", +1),
    ("tobacco_prevalence", "death_asdr_chronic_obstructive_pulmonary_disease", +1),
    ("pm25_exposure", "death_asdr_chronic_obstructive_pulmonary_disease", +1),
    ("raised_glucose_prevalence", "death_asdr_diabetes_mellitus", +1),
    ("obesity_prevalence", "death_asdr_diabetes_mellitus", +1),
    ("stunting_prevalence", "death_asdr_protein_energy_malnutrition", +1),
    ("battle_deaths_per100k", "death_asdr_collective_violence_and_legal_intervention", +1),
]

BROAD_GROUPS = {        # the three WHO cause groups (used as node category)
    "death_asdr_communicable_maternal_perinatal_and_nutritional_conditions": "communicable",
    "death_asdr_noncommunicable_diseases": "noncommunicable",
    "death_asdr_injuries": "injuries",
}
# Config ---


def label(col):
    """Readable label for a cause column.

    Parameters
    col: str, e.g. "death_asdr_ischaemic_heart_disease"

    Returns
    str, e.g. "ischaemic heart disease"
    """
    return col.replace("death_asdr_", "").replace("_", " ")


def cause_categories(dictionary):
    """Broad WHO group of every cause (communicable / noncommunicable / injuries).

    The dictionary lists the causes in WHO order: each broad group is followed by all its causes,
    so every cause belongs to the last broad group above it.

    Parameters
    dictionary: pandas.DataFrame, dictionary.csv

    Returns
    dict, cause column -> category
    """
    categories, current = {}, None
    for col in dictionary.loc[dictionary["group"] == "cause_of_death", "column"]:
        current = BROAD_GROUPS.get(col, current)
        categories[col] = current
    return categories


def remove_control(table, control):
    """Remove the effect of a control variable (e.g. GDP) from every column, using ranks.

    Every column is ranked, the part explained by the rank of the control variable is removed
    with a linear regression, and the residuals are returned. Correlating the residuals gives the
    partial Spearman correlation.

    Parameters
    table: pandas.DataFrame, columns to adjust (one row per country)
    control: pandas.Series, control variable (same rows)

    Returns
    pandas.DataFrame, residual ranks
    """
    control_rank = control.rank()
    residuals = {}
    for col in table.columns:
        col_rank = table[col].rank()
        slope, intercept = np.polyfit(control_rank, col_rank, 1)
        residuals[col] = col_rank - (slope * control_rank + intercept)
    return pd.DataFrame(residuals)


def direct_links(residuals, alpha=None):
    """Direct links between all columns: partial correlations from the graphical lasso.

    The graphical lasso estimates the inverse correlation matrix with a penalty (alpha) that sets weak
    links to exactly 0. The partial correlation of two columns is their correlation after accounting
    for all other columns. If alpha is None it is chosen by 5-fold cross-validation.

    Parameters
    residuals: pandas.DataFrame, residual ranks (output of remove_control)
    alpha: float or None, penalty strength (None = choose by cross-validation)

    Returns
    pandas.DataFrame, partial correlation matrix
    float, the alpha that was used
    """
    z = ((residuals - residuals.mean()) / residuals.std().replace(0, 1)).fillna(0)   # standardise
    if alpha is None:
        with warnings.catch_warnings():
            # some penalties tried during cross-validation diverge; they are discarded, the chosen alpha is valid
            warnings.simplefilter("ignore", RuntimeWarning)
            model = GraphicalLassoCV(cv=5, max_iter=500).fit(z.values)
    else:
        model = GraphicalLasso(alpha=alpha, max_iter=500).fit(z.values)
    precision = model.precision_
    scale = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    pcorr = -precision / scale
    np.fill_diagonal(pcorr, 1.0)
    return pd.DataFrame(pcorr, index=residuals.columns, columns=residuals.columns), (alpha or model.alpha_)


def cause_links(data, causes, method, alpha=None):
    """Link strength between all causes.

    Parameters
    data: pandas.DataFrame, cleaned dataset (or a resampled version of it)
    causes: list of str, cause columns
    method: str, "glasso" or "correlation"
    alpha: float or None, graphical lasso penalty (None = cross-validation; ignored for "correlation")

    Returns
    pandas.DataFrame, link strength matrix
    float or None, the alpha that was used
    """
    residuals = remove_control(data[causes], data[CONTROL])
    if method == "correlation":
        return residuals.corr(), None
    return direct_links(residuals, alpha)


def build_network(corr, threshold):
    """Network with one node per column and an edge for every link strength >= threshold.

    Only positive links become edges (diseases that are high in the same countries).

    Parameters
    corr: pandas.DataFrame, correlation matrix
    threshold: float, minimum correlation for an edge

    Returns
    networkx.Graph, edge attribute "weight" = correlation
    """
    G = nx.Graph()
    G.add_nodes_from(corr.columns)
    values = corr.values
    rows, cols = np.where(np.triu(values >= threshold, k=1))  # pairs above the threshold (each pair once)
    for i, j in zip(rows, cols):
        G.add_edge(corr.columns[i], corr.columns[j], weight=round(float(values[i, j]), 4))
    return G


def find_communities(G, seed):
    """Detect communities with the Louvain method.

    Parameters
    G: networkx.Graph
    seed: int, random seed (Louvain is randomised)

    Returns
    dict, node -> community number (1 = largest community, 0 = isolated node)
    list of sets, the communities (needed for the modularity)
    """
    communities = nx.community.louvain_communities(G, weight="weight", seed=seed)
    communities = sorted(communities, key=len, reverse=True)  # community 1 = largest
    partition = {}
    for number, members in enumerate(communities, start=1):
        for node in members:
            partition[node] = number if len(members) > 1 else 0
    return partition, communities


def add_communities(G):
    """Detect communities and store them as node attribute "community".

    Parameters
    G: networkx.Graph

    Returns
    float, modularity of the partition (0 = no structure, higher = clearer communities)
    """
    partition, communities = find_communities(G, SEED)
    nx.set_node_attributes(G, partition, "community")
    return nx.community.modularity(G, communities, weight="weight")


def bootstrap_partition(data, causes, method, alpha, i):
    """Communities of one resampled network (one bootstrap run).

    Parameters
    data: pandas.DataFrame, cleaned dataset
    causes: list of str, cause columns
    method: str, "glasso" or "correlation"
    alpha: float or None, graphical lasso penalty of the original network (kept fixed for speed)
    i: int, number of the run (used as random seed)

    Returns
    numpy.ndarray, community number of every cause, or None if the model could not be fitted
    """
    rows = np.random.default_rng(SEED + i).integers(0, len(data), len(data))   # resample countries
    sample = data.iloc[rows].reset_index(drop=True)
    try:
        with warnings.catch_warnings():
            # some resampled datasets converge slowly or contain a constant column; the estimate is still usable
            warnings.simplefilter("ignore")
            corr, _ = cause_links(sample, causes, method, alpha)
    except FloatingPointError:   # rare: the resampled data is too ill-conditioned for the graphical lasso
        return None
    partition, _ = find_communities(build_network(corr, THRESHOLD[method]), seed=i)
    return np.array([partition[c] for c in causes])


def stability(data, causes, G, method, alpha):
    """Bootstrap stability of the communities.

    The countries are resampled with replacement N_BOOTSTRAP times. For every sample the network and
    its communities are rebuilt, and it is counted how often every pair of causes ends up in the same
    community ("co-assignment"). A cause is stable if it stays together with the members of its
    original community in most samples. The runs are spread over all CPU cores.

    Parameters
    data: pandas.DataFrame, cleaned dataset (one row per country)
    causes: list of str, cause columns (network nodes)
    G: networkx.Graph, original network with node attribute "community"
    method: str, "glasso" or "correlation"
    alpha: float or None, graphical lasso penalty of the original network

    Returns
    pandas.Series, stability per cause (0-1, share of samples it stays with its original community)
    pandas.DataFrame, co-assignment matrix (share of samples two causes are in the same community)
    list of float, adjusted Rand index of every sample vs. the original communities (1 = identical)
    """
    original = np.array([G.nodes[c]["community"] for c in causes])
    runs = Parallel(n_jobs=-1)(delayed(bootstrap_partition)(data, causes, method, alpha, i) for i in range(N_BOOTSTRAP))
    runs = [labels for labels in runs if labels is not None]

    together = np.zeros((len(causes), len(causes)))
    agreement = []
    for labels in runs:
        together += labels[:, None] == labels[None, :]                    # 1 where two causes share a community
        agreement.append(adjusted_rand_score(original, labels))
    together /= len(runs)
    scores = {}
    for i, cause in enumerate(causes):
        members = (original == original[i]) & (np.arange(len(causes)) != i)   # same original community, not itself
        scores[cause] = together[i, members].mean() if original[i] > 0 and members.any() else np.nan
    return pd.Series(scores), pd.DataFrame(together, index=causes, columns=causes), agreement


def add_node_info(G, data, categories):
    """Add label, category, median death rate and centrality measures to every node.

    Parameters
    G: networkx.Graph
    data: pandas.DataFrame, cleaned dataset
    categories: dict, cause column -> broad WHO group

    Returns
    None (the graph is changed in place)
    """
    betweenness = nx.betweenness_centrality(G)
    for node in G.nodes:
        G.nodes[node]["type"] = "cause"
        G.nodes[node]["label"] = label(node)
        G.nodes[node]["category"] = categories.get(node, "unknown")
        G.nodes[node]["median_rate"] = round(float(data[node].median()), 3)  # deaths per 100,000
        G.nodes[node]["degree"] = G.degree(node)                              # number of links
        G.nodes[node]["strength"] = round(float(G.degree(node, weight="weight")), 3)  # sum of link weights
        G.nodes[node]["betweenness"] = round(betweenness[node], 4)            # bridge between communities


def add_indicators(G, corr, indicators, groups):
    """Add the indicators as a second type of node, linked to the causes they correlate with.

    Indicator-cause edges are added for |correlation| >= INDICATOR_THRESHOLD. Both signs are kept:
    a negative link means "more of the indicator, fewer deaths from this cause". Every indicator is
    assigned to the disease community it is most strongly linked to (0 = no links).

    Parameters
    G: networkx.Graph, disease network with node attribute "community"
    corr: pandas.DataFrame, correlation matrix of causes and indicators (GDP removed)
    indicators: list of str, indicator columns
    groups: dict, indicator -> group (system_input, risk_factor, ...)

    Returns
    None (the graph is changed in place)
    """
    causes = [n for n in G.nodes]
    for ind in indicators:
        G.add_node(ind, type="indicator", label=ind.replace("_", " "), category=groups.get(ind, "unknown"))
        strength_per_community = {}
        for cause in causes:
            rho = corr.loc[ind, cause]
            if abs(rho) >= INDICATOR_THRESHOLD:
                G.add_edge(ind, cause, weight=round(abs(float(rho)), 4), rho=round(float(rho), 4), sign=int(np.sign(rho)))
                community = G.nodes[cause]["community"]
                strength_per_community[community] = strength_per_community.get(community, 0) + abs(rho)
        G.nodes[ind]["community"] = max(strength_per_community, key=strength_per_community.get) if strength_per_community else 0
        G.nodes[ind]["degree"] = G.degree(ind)


def check_expected_links(corr):
    """Compare the network with links expected from medical knowledge.

    A link is "found" if the correlation has the expected sign and |correlation| >= INDICATOR_THRESHOLD.

    Parameters
    corr: pandas.DataFrame, correlation matrix of causes and indicators (GDP removed)

    Returns
    pandas.DataFrame, one row per expected link
    """
    rows = []
    for ind, cause, sign in EXPECTED_LINKS:
        rho = corr.loc[ind, cause]
        rows.append({
            "indicator": ind, "cause": label(cause), "expected": "-" if sign < 0 else "+",
            "rho": round(float(rho), 3),
            "found": bool(np.sign(rho) == sign and abs(rho) >= INDICATOR_THRESHOLD),
            "right_direction": bool(np.sign(rho) == sign),
        })
    return pd.DataFrame(rows)


def export(G, method):
    """Save the network as nodes_<method>.csv, edges_<method>.csv, network_<method>.json (for the HTML viewer)
    and network_<method>.gexf (Gephi).

    Parameters
    G: networkx.Graph
    method: str, "glasso" or "correlation" (added to the file names)

    Returns
    None
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nodes = pd.DataFrame([{"id": n, **attrs} for n, attrs in G.nodes(data=True)]).fillna(0)
    edges = pd.DataFrame([{"source": a, "target": b, "weight": d["weight"], "rho": d.get("rho", d["weight"]),
                           "type": "indicator-cause" if "sign" in d else "cause-cause"}
                          for a, b, d in G.edges(data=True)])
    nodes.sort_values(["community", "degree"], ascending=[True, False]).to_csv(OUT_DIR / f"nodes_{method}.csv", index=False)
    edges.to_csv(OUT_DIR / f"edges_{method}.csv", index=False)

    # {"nodes": [...], "links": [...]} is the format D3.js force layouts read directly
    network = {
        "meta": {"method": method, "control": CONTROL, "threshold": THRESHOLD[method],
                 "indicator_threshold": INDICATOR_THRESHOLD,
                 "nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
        "nodes": nodes.to_dict(orient="records"),
        "links": edges.to_dict(orient="records"),
    }
    with open(OUT_DIR / f"network_{method}.json", "w", encoding="utf-8") as f:
        json.dump(network, f, indent=1)

    G_gexf = G.copy()
    for n in G_gexf.nodes:  # Gephi needs every attribute on every node
        for key, default in (("median_rate", 0.0), ("strength", 0.0), ("betweenness", 0.0), ("stability", 0.0)):
            G_gexf.nodes[n].setdefault(key, default)
    for a, b in G_gexf.edges:
        G_gexf.edges[a, b].setdefault("rho", G_gexf.edges[a, b]["weight"])
        G_gexf.edges[a, b].setdefault("sign", 1)
    nx.write_gexf(G_gexf, OUT_DIR / f"network_{method}.gexf")


def export_coassignment(together, method):
    """Save the co-assignment matrix (for a heatmap in the report).

    Parameters
    together: pandas.DataFrame, output of stability()
    method: str, "glasso" or "correlation" (added to the file name)

    Returns
    None
    """
    together.rename(index=label, columns=label).round(3).to_csv(OUT_DIR / f"coassignment_{method}.csv")


def build(data, dictionary, method):
    """Build one disease network (causes + health-system layer), export it and print a summary.

    Parameters
    data: pandas.DataFrame, cleaned dataset
    dictionary: pandas.DataFrame, dictionary.csv
    method: str, "glasso" or "correlation"

    Returns
    None
    """
    causes = [c for c in data.columns if c.startswith("death_asdr_")]

    # 1) link strength between causes after removing the wealth effect
    corr, alpha = cause_links(data, causes, method)

    # 2) network: causes = nodes, strong positive links = edges
    G = build_network(corr, THRESHOLD[method])

    # 3) communities and node information
    modularity = add_communities(G)
    add_node_info(G, data, cause_categories(dictionary))

    # 4) stability: how often does each cause stay in its community when the countries are resampled?
    print(f"\n===== Method: {method}" + (f" (graphical lasso, alpha = {alpha:.3f})" if alpha else "") + " =====")
    print(f"Stability check: {N_BOOTSTRAP} resampled networks on all CPU cores ...")
    scores, together, agreement = stability(data, causes, G, method, alpha)
    for node in G.nodes:
        G.nodes[node]["stability"] = 0.0 if np.isnan(scores[node]) else round(float(scores[node]), 3)  # isolated causes: 0

    # 5) health-system layer: indicators linked to causes (GDP removed as well, same for both methods)
    indicators = [c for c in data.columns if c not in ["iso3", "country", "region", "income_group", CONTROL]
                  and c not in causes and c not in NOT_IN_NETWORK]
    corr_all = remove_control(data[causes + indicators], data[CONTROL]).corr()
    groups = dictionary.set_index("column")["group"].to_dict()
    groups["battle_deaths_per100k"] = "context"
    disease_edges = G.number_of_edges()
    add_indicators(G, corr_all, indicators, groups)

    # 6) export
    export(G, method)
    export_coassignment(together, method)

    # summary
    cause_nodes = [n for n in G if G.nodes[n]["type"] == "cause"]
    sizes = pd.Series({n: G.nodes[n]["community"] for n in cause_nodes}).value_counts().sort_index()
    print(f"Disease network: {len(cause_nodes)} causes, {disease_edges} edges")
    print(f"Communities (Louvain): {len(sizes[sizes.index > 0])}, modularity: {modularity:.3f}")
    print("Community sizes:", sizes.to_dict(), "(0 = isolated causes)")
    for number in sizes.index[sizes.index > 0]:
        members = sorted((n for n in cause_nodes if G.nodes[n]["community"] == number), key=G.degree, reverse=True)
        print(f"  {number}: {', '.join(label(m) for m in members[:8])}{' ...' if len(members) > 8 else ''}")
    community_stability = scores.groupby(pd.Series({n: G.nodes[n]["community"] for n in cause_nodes})).mean().drop(0, errors="ignore")
    print(f"Agreement with the original communities (mean adjusted Rand index over {len(agreement)} runs): {np.mean(agreement):.2f}")
    print("Stability per community (share of samples members stay together):", community_stability.round(2).to_dict())
    linked = [i for i in indicators if G.degree(i) > 0]
    print(f"Health-system layer: {len(indicators)} indicators, {len(linked)} with links, "
          f"{G.number_of_edges() - disease_edges} indicator-cause edges")


def main():
    """Build both disease networks from clean_2019.csv, export them and check the expected indicator links."""
    data = pd.read_csv(DATA_FILE)
    dictionary = pd.read_csv(DICTIONARY_FILE)
    for method in METHODS:
        build(data, dictionary, method)

    # expected health-system links (the indicator layer is the same in both networks)
    causes = [c for c in data.columns if c.startswith("death_asdr_")]
    indicators = [c for c in data.columns if c not in ["iso3", "country", "region", "income_group", CONTROL]
                  and c not in causes and c not in NOT_IN_NETWORK]
    expected = check_expected_links(remove_control(data[causes + indicators], data[CONTROL]).corr())
    expected.to_csv(OUT_DIR / "expected_links.csv", index=False)
    print(f"\nExpected links found: {expected['found'].sum()} of {len(expected)} "
          f"(right direction: {expected['right_direction'].sum()} of {len(expected)})")
    print(expected.to_string(index=False))
    print(f"\nWritten to {OUT_DIR}: nodes_*.csv, edges_*.csv, network_*.json, network_*.gexf, coassignment_*.csv, "
          f"expected_links.csv  (* = {', '.join(METHODS)})")


if __name__ == "__main__":
    main()