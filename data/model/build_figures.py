import warnings
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import build_network as nb

# Settings
DATA_FILE = Path("./../data/processed/clean_2019.csv")
NETWORK_DIR = Path("./../data/network")
OUT_DIR = Path("./../figures")

STRONG = 0.5   # Strong-link threshold for Figure 1.
EXAMPLE_PAIRS = [   # Example pairs for Figure 2.
    ("hiv_aids", "drowning"),
    ("alcohol_use_disorders", "cirrhosis_due_to_alcohol_use"),
    ("measles", "whooping_cough"),
    ("trachea_bronchus_lung_cancers", "chronic_obstructive_pulmonary_disease"),
]
COMMUNITY_NAMES = {
    1: "Cancers",
    2: "Low-mortality causes & acute hepatitis",
    3: "Mixed: dementia, chronic & congenital",
    4: "Maternal, neonatal & tropical",
    5: "Mixed: infections & digestive",
    6: "Malnutrition, diarrhoea & diabetes",
    7: "Cardiovascular & injuries",
    8: "Alcohol & liver",
    9: "Vaccine-preventable, TB & hepatitis B",
    10: "HIV, cervical cancer & violence",
    11: "Drug use & head-and-neck cancers",
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # Match viewer colors.
GREY, INK, MUTED = "#b8b8b8", "#222222", "#6b6b6b"
CATEGORY_COLORS = {"communicable": COLORS[1], "noncommunicable": COLORS[0], "injuries": COLORS[2]}
ONLY_HERE = "#e34948"   # Compare network-only links.
PULL = 15      # Strengthen within-community layout.
WIDTH = 7.0    # Full text width, inches.
DPI = 300

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "axes.edgecolor": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True,
})

def pair_values(matrix):
    """Return unique upper-triangle values."""
    return matrix.values[np.triu_indices(len(matrix), k=1)]

def save(fig, name):
    """Save PNG and PDF versions."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

def fig_wealth(raw, adjusted):
    """Plot raw and adjusted correlations."""
    fig, ax = plt.subplots(figsize=(WIDTH / 2, 2.3))
    bins = np.linspace(-1, 1, 41)
    for values, color, name in [(pair_values(raw), COLORS[1], "raw"), (pair_values(adjusted), COLORS[0], "GDP removed")]:
        strong = np.mean(np.abs(values) >= STRONG)
        ax.hist(values, bins=bins, histtype="step", linewidth=1.6, color=color,
                label=f"{name}: {strong:.0%} of pairs |ρ| ≥ {STRONG}")
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)   # Reserve legend space.
    ax.set_xlabel("Spearman ρ between two causes of death")
    ax.set_ylabel("cause pairs")
    ax.set_title("Most co-occurrence is wealth", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    save(fig, "fig1_wealth")

def fig_direct(raw, adjusted, direct):
    """Plot example link strengths."""
    fig, ax = plt.subplots(figsize=(WIDTH / 2, 2.3))
    names = [f"{nb.label(a)} –\n{nb.label(b)}" for a, b in EXAMPLE_PAIRS]
    y = np.arange(len(EXAMPLE_PAIRS))
    steps = [(raw, COLORS[1], "raw"), (adjusted, COLORS[0], "GDP removed"), (direct, COLORS[2], "direct link")]
    for k, (matrix, color, name) in enumerate(steps):
        values = [matrix.loc[f"death_asdr_{a}", f"death_asdr_{b}"] for a, b in EXAMPLE_PAIRS]
        ax.barh(y + (k - 1) * 0.26, values, height=0.24, color=color, label=name)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(y, names, fontsize=6.5)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("link strength")
    ax.set_title("Direct links remove shared patterns", loc="left")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    save(fig, "fig2_direct_links")

def node_style(community):
    """Return colors for a node."""
    if community == 0:
        return GREY, GREY
    color = COLORS[(community - 1) % len(COLORS)]
    return (color, "white") if community <= len(COLORS) else ("white", color)

def community_layout(G, community):
    """Position communities with stronger internal links."""
    for a, b in G.edges:
        same = community[a] == community[b] > 0
        G.edges[a, b]["pull"] = G.edges[a, b]["weight"] * (PULL if same else 1)
    return nx.spring_layout(G, weight="pull", seed=nb.SEED, k=0.3, iterations=500)

def cause_graph(nodes, edges):
    """Build the cause-only graph."""
    causes = nodes[nodes["type"] == "cause"].set_index("id")
    G = nx.Graph()
    G.add_nodes_from(causes.index)
    links = edges[edges["type"] == "cause-cause"]
    G.add_weighted_edges_from(links[["source", "target", "weight"]].itertuples(index=False))
    return G, causes

def fig_network(nodes, edges):
    """Plot causes by community."""
    G, causes = cause_graph(nodes, edges)
    pos = community_layout(G, causes["community"])

    fig, ax = plt.subplots(figsize=(WIDTH, 4.6))
    nx.draw_networkx_edges(G, pos, ax=ax, width=[0.3 + 3 * G.edges[e]["weight"] for e in G.edges],
                           edge_color="#cfcfcf")
    for community, group in causes.groupby("community"):
        face, edge = node_style(community)
        ax.scatter([pos[n][0] for n in group.index], [pos[n][1] for n in group.index],
                   s=12 + 6 * group["degree"], c=face, edgecolors=edge, linewidths=1.2, zorder=3,
                   label=f"{community}  {COMMUNITY_NAMES[community]} ({len(group)})" if community else f"isolated ({len(group)})")
    for community, group in causes[causes["community"] > 0].groupby("community"):   # Community label at center.
        x, y = np.mean([pos[n] for n in group.index], axis=0)
        ax.text(x, y, str(community), fontsize=9, fontweight="bold", color=INK, ha="center", va="center",
                zorder=5, path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    ax.set_axis_off()
    ax.legend(frameon=False, fontsize=6.5, loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=0.8)
    save(fig, "fig3_network")

def fig_compare(adjusted, direct, nodes, edges):
    """Compare co-occurrence and direct links."""
    G, causes = cause_graph(nodes, edges)
    pos = community_layout(G, causes["community"])
    nets = [("Co-occurrence (GDP removed, |ρ| ≥ 0.35)", nb.build_network(adjusted, nb.THRESHOLD["correlation"])),
            ("Direct links (graphical lasso, ≥ 0.05)", nb.build_network(direct, nb.THRESHOLD["glasso"]))]
    sets = [{frozenset(e) for e in net.edges} for _, net in nets]

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.6))
    for ax, (title, net), own, other in zip(axes, nets, sets, sets[::-1]):
        shared = [tuple(e) for e in own & other]
        only = [tuple(e) for e in own - other]
        nx.draw_networkx_edges(net, pos, edgelist=shared, ax=ax, width=0.4, edge_color="#c8c8c8")
        nx.draw_networkx_edges(net, pos, edgelist=only, ax=ax, width=0.4, edge_color=ONLY_HERE, alpha=0.45)
        for category, color in CATEGORY_COLORS.items():
            members = causes.index[causes["category"] == category]
            ax.scatter([pos[n][0] for n in members], [pos[n][1] for n in members], s=14, c=color,
                       edgecolors="white", linewidths=0.6, zorder=3, label=category)
        ax.set_title(f"{title}\n{len(own)} links, {len(only)} only here", loc="left", fontsize=8)
        ax.set_axis_off()
    handles = axes[0].get_legend_handles_labels()[0] + [
        plt.Line2D([], [], color="#c8c8c8", lw=1.5), plt.Line2D([], [], color=ONLY_HERE, lw=1.5)]
    labels = list(CATEGORY_COLORS) + ["link in both networks", "link only in this network"]
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.04))
    save(fig, "fig6_compare_networks")

def fig_stability(nodes):
    """Plot mean community stability."""
    causes = nodes[(nodes["type"] == "cause") & (nodes["community"] > 0)]
    table = causes.groupby("community")["stability"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(WIDTH / 2, 2.6))
    ax.barh([COMMUNITY_NAMES[c] for c in table.index], table.values, height=0.6, color=COLORS[0])
    for i, value in enumerate(table.values):
        ax.text(value - 0.015, i, f"{value:.2f}", va="center", ha="right", fontsize=6.5, color="white")
    ax.set_xlim(0, 1)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of bootstrap runs a cause stays with its community")
    ax.set_title("Community stability (200 resamples)", loc="left")
    save(fig, "fig4_stability")

def fig_expected(expected):
    """Plot expected indicator links."""
    table = expected.assign(name=expected["indicator"].str.replace("_", " ") + " → " + expected["cause"])
    table = table.sort_values("rho")
    fig, ax = plt.subplots(figsize=(WIDTH / 2, 4.2))
    y = np.arange(len(table))
    for right, color, name in [(True, COLORS[0], "expected direction"), (False, COLORS[1], "opposite direction")]:
        part = table["right_direction"] == right
        ax.scatter(table["rho"][part], y[part.values], s=18, color=color, label=name, zorder=3)
    for x in (-nb.INDICATOR_THRESHOLD, nb.INDICATOR_THRESHOLD):
        ax.axvline(x, color=MUTED, linewidth=0.8, linestyle="--")
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(y, table["name"], fontsize=5.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Spearman ρ, GDP removed  (dashed = network threshold)")
    ax.set_title("Health-system links we expected", loc="left")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    save(fig, "fig5_expected_links")

def main():
    # Load analysis data.
    data = pd.read_csv(DATA_FILE)
    causes = [c for c in data.columns if c.startswith("death_asdr_")]

    # Build correlation matrices.
    raw = data[causes].corr(method="spearman")
    adjusted = nb.remove_control(data[causes], data[nb.CONTROL]).corr()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        direct, _ = nb.direct_links(nb.remove_control(data[causes], data[nb.CONTROL]))
    nodes = pd.read_csv(NETWORK_DIR / "nodes_glasso.csv")
    edges = pd.read_csv(NETWORK_DIR / "edges_glasso.csv")
    expected = pd.read_csv(NETWORK_DIR / "expected_links.csv")

    # Render all figures.
    fig_wealth(raw, adjusted)
    fig_direct(raw, adjusted, direct)
    fig_network(nodes, edges)
    fig_stability(nodes)
    fig_expected(expected)
    fig_compare(adjusted, direct, nodes, edges)

if __name__ == "__main__":
    main()