# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:27:44 2026

@author: yavuzb2
"""

# Input file paths.
MAF_FILE = "Merged_Genie_Tcga_VAFgreater12point5Percent__new.txt"
META_FILE = "TCGA_GENIE_Vol6_VAF_0125_Subtype_Tissue_Primary_Metastasis_Info_ccBioPortal_API.txt"
PICKLE_FILE = "GeneMutation_tumor_dictionary.p"
KEGG_FILE = "KEGG_2021_Human.txt"

# Output folder.
OUTDIR = "stats_out"

# Minimum mutation count in at least one group for later tests.
MIN_MUT_ANY_GROUP = 1

import os

# Create the output folder if it does not exist.
os.makedirs(OUTDIR, exist_ok=True)

print("OUTDIR:", OUTDIR)
print("MIN_MUT_ANY_GROUP:", MIN_MUT_ANY_GROUP)

# Imports and helper functions.

import pickle
import numpy as np
import pandas as pd

from itertools import combinations
from collections import defaultdict
from scipy.stats import chi2_contingency, fisher_exact, kruskal, mannwhitneyu


def bh_fdr(pvals):
    """Benjamini–Hochberg FDR """
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)

    mask = np.isfinite(p)
    p0 = p[mask]
    if p0.size == 0:
        return q

    order = np.argsort(p0)
    ranks = np.arange(1, p0.size + 1)
    q0 = p0[order] * (p0.size / ranks)
    q0 = np.minimum.accumulate(q0[::-1])[::-1]
    q0 = np.clip(q0, 0, 1)

    out = np.empty_like(p0)
    out[order] = q0
    q[mask] = out
    return q


def odds_ratio_ci(a, b, c, d):
    """
    OR and Wald 95% CI on log(OR)
    2x2 table:
        group1: a (mut), b (wt)
        group2: c (mut), d (wt)
    """
    a, b, c, d = float(a), float(b), float(c), float(d)
    if min(a, b, c, d) == 0:
        a += 0.5; b += 0.5; c += 0.5; d += 0.5

    or_ = (a * d) / (b * c)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z = 1.959963984540054
    lo = np.exp(np.log(or_) - z * se)
    hi = np.exp(np.log(or_) + z * se)
    return or_, lo, hi


def load_groups(meta_path):
    """
    meta file format (tab-delimited, no header), columns:
      0: Tumor_Sample_Barcode
      1: label (e.g., MEL)
      2: tissue (e.g., BRAIN)
      3: Primary/Metastasis
    """
    df = pd.read_csv(meta_path, sep="\t", header=None, dtype=str)

    primary_mel = set(df[(df[1] == "MEL") & (df[3] == "Primary")][0].tolist())
    met_mel = set(df[(df[1] == "MEL") & (df[3] == "Metastasis")][0].tolist())
    brain_met = set(df[(df[2] == "BRAIN") & (df[3] == "Metastasis")][0].tolist())

    groups = {"PrimaryMel": primary_mel, "MetMel": met_mel, "BrainMet": brain_met}
    return groups


def convert_mutation_to_gene_tumor_dict(mutation_dict):
    """mutation_dict: {'TP53_R175': set(tumors)} -> {'TP53': set(tumors)}"""
    gene_tumor = defaultdict(set)
    for mut, tumors in mutation_dict.items():
        gene = str(mut).split("_")[0]
        gene_tumor[gene].update(set(tumors))
    return dict(gene_tumor)


def compute_feature_stats(feature_to_tumors, groups, min_mutated_any_group=1):
    """
    For each feature (gene / alteration / pathway), test mutation frequency differences across 3 groups:
      - overall 3x2 chi-square (Group x mutated yes/no)
      - pairwise Fisher exact + OR(95% CI)
      - BH-FDR for overall + for each pairwise family
    """
    group_names = list(groups.keys())
    totals = {g: len(groups[g]) for g in group_names}

    rows = []
    for feat, tumors_all in feature_to_tumors.items():
        tumors_all = set(tumors_all)

        mut_counts = {g: len(tumors_all & groups[g]) for g in group_names}
        if max(mut_counts.values()) < min_mutated_any_group:
            continue

        wt_counts = {g: totals[g] - mut_counts[g] for g in group_names}

        # Overall chi-square test.
        table = np.array([[mut_counts[g], wt_counts[g]] for g in group_names], dtype=float)
        if table.sum() == 0 or table[:, 0].sum() == 0 or table[:, 1].sum() == 0:
            chi2 = np.nan
            dof = np.nan
            p_overall = 1.0
        else:
            chi2, p_overall, dof, _ = chi2_contingency(table, correction=False)

        row = {
            "feature": feat,
            "chi2_overall": chi2,
            "dof_overall": dof,
            "p_overall": p_overall,
        }

        # Mutation counts and proportions for each group.
        for g in group_names:
            row[f"{g}_mut"] = mut_counts[g]
            row[f"{g}_total"] = totals[g]
            row[f"{g}_prop"] = (mut_counts[g] / totals[g]) if totals[g] > 0 else np.nan

        # Pairwise Fisher tests and odds ratios.
        for g1, g2 in combinations(group_names, 2):
            a = mut_counts[g1]; b = wt_counts[g1]
            c = mut_counts[g2]; d = wt_counts[g2]

            try:
                _, p_pair = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            except Exception:
                p_pair = np.nan

            or_, lo, hi = odds_ratio_ci(a, b, c, d)

            row[f"p_fisher_{g1}_vs_{g2}"] = p_pair
            row[f"or_{g1}_vs_{g2}"] = or_
            row[f"or95lo_{g1}_vs_{g2}"] = lo
            row[f"or95hi_{g1}_vs_{g2}"] = hi

        rows.append(row)

    res = pd.DataFrame(rows)
    if res.empty:
        return res

    # Adjust the overall p-values.
    res["q_overall_BH"] = bh_fdr(res["p_overall"].values)

    # Adjust p-values separately for each pairwise comparison.
    fisher_cols = [c for c in res.columns if c.startswith("p_fisher_")]
    for c in fisher_cols:
        res[c.replace("p_", "q_BH_")] = bh_fdr(res[c].values)

    return res.sort_values("p_overall", kind="mergesort").reset_index(drop=True)


def kegg2021_pathway_gene_dictionary(file_name):
    """
    Enrichr-like format:
      PathwayName<TAB>GENE1<TAB>GENE2...
    Keep only pathway names that include both 'signaling' and 'pathway' (case-insensitive).
    """
    out = {}
    with open(file_name, "r") as f:
        for line in f:
            low = line.lower()
            if ("signaling" in low) and ("pathway" in low):
                parts = line.rstrip("\n").split("\t")
                pathway = parts[0]
                genes = set(g for g in parts[1:] if g)
                out[pathway] = genes
    return out


def safe_mannwhitneyu(x, y):
    """Compatible with SciPy versions with/without method=."""
    try:
        return mannwhitneyu(x, y, alternative="two-sided", method="auto")
    except TypeError:
        return mannwhitneyu(x, y, alternative="two-sided")


def show_subset(df, features=("TP53", "BRAF", "NRAS", "IDH1")):
    if df.empty:
        print("Empty results.")
        return
    sub = df[df["feature"].isin(features)].copy()
    if sub.empty:
        print("None of", features, "found in results.")
        return
    cols = ["feature", "PrimaryMel_mut", "PrimaryMel_total", "PrimaryMel_prop",
            "MetMel_mut", "MetMel_total", "MetMel_prop",
            "BrainMet_mut", "BrainMet_total", "BrainMet_prop",
            "p_overall", "q_overall_BH"]
    print(sub[cols].to_string(index=False))
    
    
# Load the sample groups from the metadata file.
groups = load_groups(META_FILE)

# Print the number of sample IDs in each group.
print("Group sizes:")
for g, s in groups.items():
    print(f"  {g}: {len(s)}")

# Load the sample groups.
groups = load_groups(META_FILE)

# Load the mutation dictionary
with open(PICKLE_FILE, "rb") as f:
    mutation_dict = pickle.load(f)

# Store the sample IDs for each mutation as a set.
mutation_dict = {k: set(v) for k, v in mutation_dict.items()}

# Combine mutation entries by gene.
gene_dict = convert_mutation_to_gene_tumor_dict(mutation_dict)

# Compare gene mutation frequencies across the groups.
gene_stats = compute_feature_stats(
    feature_to_tumors=gene_dict,
    groups=groups,
    min_mutated_any_group=MIN_MUT_ANY_GROUP
)

print("Gene-level results shape:", gene_stats.shape)
print("\nTop 10 by overall p-value:")



show_subset(gene_stats, features=("TP53", "BRAF", "NRAS", "IDH1", "PTEN", "NF1"))

# Save the results as a tab-separated file.
outpath = os.path.join(OUTDIR, "GeneFrequency_3GroupStats.tsv")
gene_stats.to_csv(outpath, sep="\t", index=False)
print("\nSaved:", outpath)

# Load the sample groups.
groups = load_groups(META_FILE)

# Load the mutation dictionary.
with open(PICKLE_FILE, "rb") as f:
    mutation_dict = pickle.load(f)

# Store the sample IDs for each mutation as a set.
mutation_dict = {k: set(v) for k, v in mutation_dict.items()}

# Compare frequencies using the original mutation dictionary keys.
alteration_stats = compute_feature_stats(
    feature_to_tumors=mutation_dict,
    groups=groups,
    min_mutated_any_group=MIN_MUT_ANY_GROUP
)

print("Alteration-level results shape:", alteration_stats.shape)
print("\nTop 10 by overall p-value:")


# Select up to 10 entries each for TP53 and BRAF.
tp53_hits = alteration_stats[alteration_stats["feature"].astype(str).str.startswith("TP53_")].head(10)
braf_hits = alteration_stats[alteration_stats["feature"].astype(str).str.startswith("BRAF_")].head(10)



# Save the results as a tab-separated file.
outpath = os.path.join(OUTDIR, "AlterationFrequency_3GroupStats.tsv")
alteration_stats.to_csv(outpath, sep="\t", index=False)
print("\nSaved:", outpath)


# Load the sample groups.
groups = load_groups(META_FILE)

# Load the mutation dictionary.
with open(PICKLE_FILE, "rb") as f:
    mutation_dict = pickle.load(f)
mutation_dict = {k: set(v) for k, v in mutation_dict.items()}

# Combine mutation entries by gene.
gene_dict = convert_mutation_to_gene_tumor_dict(mutation_dict)

# Load pathway names and their genes.
pathway_gene = kegg2021_pathway_gene_dictionary(KEGG_FILE)

print("Loaded KEGG pathways (signaling + pathway):", len(pathway_gene))

# Merge Ras into MAPK when both pathway entries exist.
ras_name = "Ras signaling pathway"
mapk_name = "MAPK signaling pathway"
if ras_name in pathway_gene and mapk_name in pathway_gene:
    pathway_gene[mapk_name] = pathway_gene[mapk_name].union(pathway_gene[ras_name])
    pathway_gene.pop(ras_name, None)
    print("Merged Ras signaling pathway into MAPK signaling pathway and removed Ras.")

# Collect samples with a mutation in any gene in each pathway.
pathway_to_tumors = {}
for pathway, genes in pathway_gene.items():
    tumors = set()
    for g in genes:
        tumors |= gene_dict.get(g, set())
    pathway_to_tumors[pathway] = tumors

# Compare pathway mutation frequencies across the groups.
pathway_stats = compute_feature_stats(
    feature_to_tumors=pathway_to_tumors,
    groups=groups,
    min_mutated_any_group=MIN_MUT_ANY_GROUP
)

print("Pathway-level results shape:", pathway_stats.shape)
print("\nTop 10 by overall p-value:")


# Show results for selected pathways.
check_pathways = [
    "Neurotrophin signaling pathway",
    "MAPK signaling pathway",
    "PI3K-Akt signaling pathway",
    "p53 signaling pathway",
    "Wnt signaling pathway",
    "ErbB signaling pathway",
]
sub = pathway_stats[pathway_stats["feature"].isin(check_pathways)][
    ["feature","PrimaryMel_prop","MetMel_prop","BrainMet_prop","p_overall","q_overall_BH"]
].copy()
print("\nSelected pathways (if present):")


# Save the results as a tab-separated file.
outpath = os.path.join(OUTDIR, "PathwayFrequency_3GroupStats.tsv")
pathway_stats.to_csv(outpath, sep="\t", index=False)
print("\nSaved:", outpath)


# Load the sample groups.
groups = load_groups(META_FILE)

# Read the mutation table.
df_maf = pd.read_csv(MAF_FILE, sep="\t", low_memory=False)
print("MAF shape:", df_maf.shape)
print("MAF columns (first 25):", list(df_maf.columns)[:25])

# Check that the sample ID and gene columns exist.
required = {"Tumor_Sample_Barcode", "Hugo_Symbol"}
if not required.issubset(df_maf.columns):
    raise ValueError(f"MAF missing required columns {required}. Found: {set(df_maf.columns)}")

# Remove rows missing a sample ID or gene name.
df_maf = df_maf.dropna(subset=["Tumor_Sample_Barcode", "Hugo_Symbol"]).copy()
df_maf["Tumor_Sample_Barcode"] = df_maf["Tumor_Sample_Barcode"].astype(str)
df_maf["Hugo_Symbol"] = df_maf["Hugo_Symbol"].astype(str)

# Count unique mutated genes per sample.
per_tumor_burden = df_maf.groupby("Tumor_Sample_Barcode")["Hugo_Symbol"].nunique()

# Collect counts by group and assign zero to samples missing from the counts.
burden = {
    g: per_tumor_burden.reindex(list(tumors)).fillna(0).astype(int).values
    for g, tumors in groups.items()
}

print("\nBurden summary (unique mutated genes per tumor):")
for g in ["PrimaryMel", "MetMel", "BrainMet"]:
    arr = burden[g]
    print(f"{g}: n={len(arr)}  mean={arr.mean():.3f}  median={np.median(arr):.3f}  min={arr.min()}  max={arr.max()}")

# Compare the three groups using the Kruskal–Wallis test.
H_stat, p_kw = kruskal(burden["PrimaryMel"], burden["MetMel"], burden["BrainMet"])
overall_df = pd.DataFrame([{"H_kruskal": H_stat, "p_kruskal": p_kw}])



# Compare each pair of groups using the Mann–Whitney U test.
pairs = [("PrimaryMel", "MetMel"), ("PrimaryMel", "BrainMet"), ("MetMel", "BrainMet")]
pair_rows = []
for g1, g2 in pairs:
    U, p = safe_mannwhitneyu(burden[g1], burden[g2])
    pair_rows.append({"pair": f"{g1}_vs_{g2}", "U": U, "p": p})

# Adjust the three pairwise p-values.
pair_df = pd.DataFrame(pair_rows)
pair_df["q_BH"] = bh_fdr(pair_df["p"].values)



# Save the overall and pairwise test results.
out1 = os.path.join(OUTDIR, "Burden_KruskalWallis.tsv")
out2 = os.path.join(OUTDIR, "Burden_PairwiseMannWhitney.tsv")
overall_df.to_csv(out1, sep="\t", index=False)
pair_df.to_csv(out2, sep="\t", index=False)
print("\nSaved:", out1)
print("Saved:", out2)

import os
import numpy as np
import pandas as pd

# Folder containing the saved results.
OUTDIR = "stats_out"

GENE_STATS_TSV = os.path.join(OUTDIR, "GeneFrequency_3GroupStats.tsv")
PATHWAY_STATS_TSV = os.path.join(OUTDIR, "PathwayFrequency_3GroupStats.tsv")
BURDEN_KW_TSV = os.path.join(OUTDIR, "Burden_KruskalWallis.tsv")
BURDEN_PAIR_TSV = os.path.join(OUTDIR, "Burden_PairwiseMannWhitney.tsv")

# Thresholds for later result filters.
Q_OVERALL = 0.05
Q_PAIRWISE = 0.05

# Check that the gene results file exists.
if not os.path.exists(GENE_STATS_TSV):
    raise FileNotFoundError(f"Missing: {GENE_STATS_TSV} (run the stats code first to generate it).")

# Load the saved gene results.
gene_df = pd.read_csv(GENE_STATS_TSV, sep="\t")

print("gene_df shape:", gene_df.shape)
print("gene_df columns (first 40):")
print(list(gene_df.columns)[:40])

# Show the first five rows in file order.
print("\nTop 5 rows (by file order):")
######3##
# Convert proportions to percentages.
def pct(x):
    return 100.0 * x


def invert_or_ci(or_, lo, hi):
    """Invert OR and CI: OR' = 1/OR, CI' = (1/hi, 1/lo)."""
    if pd.isna(or_) or pd.isna(lo) or pd.isna(hi) or float(or_) == 0:
        return (np.nan, np.nan, np.nan)
    return (1.0/float(or_), 1.0/float(hi), 1.0/float(lo))


def add_summary_cols(df):
    df = df.copy()

    # Add mutation percentages for each group.
    for g in ["PrimaryMel", "MetMel", "BrainMet"]:
        if f"{g}_prop" not in df.columns:
            raise ValueError(f"Missing {g}_prop in TSV. Found columns: {df.columns.tolist()}")
        df[f"{g}_pct"] = pct(df[f"{g}_prop"].astype(float))

    # Label the group with the highest mutation proportion.
    props = df[["PrimaryMel_prop", "MetMel_prop", "BrainMet_prop"]].astype(float)
    df["dominant_group"] = props.idxmax(axis=1).str.replace("_prop", "", regex=False)

    # Calculate differences in percentage points.
    df["delta_Brain_vs_Primary_pct"] = df["BrainMet_pct"] - df["PrimaryMel_pct"]
    df["delta_Brain_vs_Met_pct"] = df["BrainMet_pct"] - df["MetMel_pct"]

    # Reverse the stored odds ratios so BrainMet is the first group.
    needed_pb = ["or_PrimaryMel_vs_BrainMet", "or95lo_PrimaryMel_vs_BrainMet", "or95hi_PrimaryMel_vs_BrainMet"]
    needed_mb = ["or_MetMel_vs_BrainMet", "or95lo_MetMel_vs_BrainMet", "or95hi_MetMel_vs_BrainMet"]

    if all(c in df.columns for c in needed_pb):
        inv = df.apply(
            lambda r: invert_or_ci(r["or_PrimaryMel_vs_BrainMet"], r["or95lo_PrimaryMel_vs_BrainMet"], r["or95hi_PrimaryMel_vs_BrainMet"]),
            axis=1, result_type="expand"
        )
        df[["OR_Brain_vs_Primary", "OR_Brain_vs_Primary_lo", "OR_Brain_vs_Primary_hi"]] = inv
    else:
        df[["OR_Brain_vs_Primary", "OR_Brain_vs_Primary_lo", "OR_Brain_vs_Primary_hi"]] = np.nan

    if all(c in df.columns for c in needed_mb):
        inv2 = df.apply(
            lambda r: invert_or_ci(r["or_MetMel_vs_BrainMet"], r["or95lo_MetMel_vs_BrainMet"], r["or95hi_MetMel_vs_BrainMet"]),
            axis=1, result_type="expand"
        )
        df[["OR_Brain_vs_Met", "OR_Brain_vs_Met_lo", "OR_Brain_vs_Met_hi"]] = inv2
    else:
        df[["OR_Brain_vs_Met", "OR_Brain_vs_Met_lo", "OR_Brain_vs_Met_hi"]] = np.nan

    return df


# Add summary columns to a copy of the gene results.
gene_df2 = add_summary_cols(gene_df)

print("Added summary columns. Example columns now include:")
print([c for c in ["PrimaryMel_pct","MetMel_pct","BrainMet_pct","dominant_group","OR_Brain_vs_Primary"] if c in gene_df2.columns])

#####
# Check that the overall adjusted p-value column exists.
if "q_overall_BH" not in gene_df2.columns:
    raise ValueError("Missing q_overall_BH in gene TSV. Did you generate the TSV with the provided stats code?")

# Filter genes and sort by the overall adjusted p-value.
sig_genes = gene_df2[gene_df2["q_overall_BH"] < Q_OVERALL].copy()
sig_genes = sig_genes.sort_values("q_overall_BH", kind="mergesort").reset_index(drop=True)

print(f"Significant genes by overall test: {sig_genes.shape[0]} (q_overall_BH < {Q_OVERALL})")

# Select available columns for display.
cols = [
    "feature",
    "PrimaryMel_mut","PrimaryMel_total","PrimaryMel_pct",
    "MetMel_mut","MetMel_total","MetMel_pct",
    "BrainMet_mut","BrainMet_total","BrainMet_pct",
    "dominant_group",
    "p_overall","q_overall_BH",
    "q_BH_fisher_PrimaryMel_vs_BrainMet",
    "q_BH_fisher_MetMel_vs_BrainMet",
    "OR_Brain_vs_Primary","OR_Brain_vs_Met"
]
cols = [c for c in cols if c in sig_genes.columns]



# Select genes whose highest mutation proportion is in BrainMet.
brain_sig = sig_genes[sig_genes["dominant_group"] == "BrainMet"].copy()


# Select genes whose highest mutation proportion is in either melanoma group.
primary_sig = sig_genes[sig_genes["dominant_group"] == "PrimaryMel"].copy()
met_sig = sig_genes[sig_genes["dominant_group"] == "MetMel"].copy()
print(f"\nPrimary-enriched significant genes: {primary_sig.shape[0]}")
print(f"Metastatic-melanoma–enriched significant genes: {met_sig.shape[0]}")

##
# Check that the pathway results file exists.
if not os.path.exists(PATHWAY_STATS_TSV):
    raise FileNotFoundError(f"Missing: {PATHWAY_STATS_TSV} (run the stats code first to generate it).")

# Load the saved pathway results and add summary columns.
pathway_df = pd.read_csv(PATHWAY_STATS_TSV, sep="\t")
pathway_df2 = add_summary_cols(pathway_df)

print("pathway_df shape:", pathway_df2.shape)
print("pathway_df columns (first 40):")
print(list(pathway_df2.columns)[:40])

# Check that the overall adjusted p-value column exists.
if "q_overall_BH" not in pathway_df2.columns:
    raise ValueError("Missing q_overall_BH in pathway TSV. Did you generate the TSV with the provided stats code?")

# Filter pathways and sort by the overall adjusted p-value.
sig_paths = pathway_df2[pathway_df2["q_overall_BH"] < Q_OVERALL].copy()
sig_paths = sig_paths.sort_values("q_overall_BH", kind="mergesort").reset_index(drop=True)

print(f"Significant pathways by overall test: {sig_paths.shape[0]} (q_overall_BH < {Q_OVERALL})")

# Select available columns for display.
cols_p = [
    "feature",
    "PrimaryMel_mut","PrimaryMel_total","PrimaryMel_pct",
    "MetMel_mut","MetMel_total","MetMel_pct",
    "BrainMet_mut","BrainMet_total","BrainMet_pct",
    "dominant_group",
    "p_overall","q_overall_BH",
    "q_BH_fisher_PrimaryMel_vs_BrainMet",
    "q_BH_fisher_MetMel_vs_BrainMet",
    "OR_Brain_vs_Primary","OR_Brain_vs_Met"
]
cols_p = [c for c in cols_p if c in sig_paths.columns]

print("\nTop significant pathways (by q_overall_BH):")


# Select pathways whose highest mutation proportion is in BrainMet.
brain_sig_paths = sig_paths[sig_paths["dominant_group"] == "BrainMet"].copy()

####################
#Bubble plot-Fig 1b

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Input table and plot settings.
INFILE = Path("stats_out/GeneFrequency_3GroupStats.tsv")
Q_CUTOFF = 0.10
TOP_N = None
SIZE_SCALE = 2600
MIN_SIZE = 18
X_SPACING = 0.72

# Cap color values at the 95th percentile.
CLIP_COLOR_AT_PERCENTILE = 95

OUT_PNG = INFILE.parent / f"bubble_sig_genes_q_lt_{Q_CUTOFF}_dpi600.png"
OUT_SVG = INFILE.parent / f"bubble_sig_genes_q_lt_{Q_CUTOFF}.svg"
OUT_TSV = INFILE.parent / f"significant_genes_q_lt_{Q_CUTOFF}.tsv"

# Set the font choices and keep SVG text editable.
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 12,
    "svg.fonttype": "none",
})

# Read the saved gene statistics.
df = pd.read_csv(INFILE, sep="\t")

GENE_COL = "feature"

# Adjusted p-value columns.
QCOL_OVERALL = "q_overall_BH"
QCOL_PM_MM = "q_BH_fisher_PrimaryMel_vs_MetMel"
QCOL_PM_BM = "q_BH_fisher_PrimaryMel_vs_BrainMet"
QCOL_MM_BM = "q_BH_fisher_MetMel_vs_BrainMet"
q_cols_all = [QCOL_OVERALL, QCOL_PM_MM, QCOL_PM_BM, QCOL_MM_BM]

# Group labels, mutation counts, and pairwise comparisons.
GROUPS = [
    {
        "key": "PrimaryMel",
        "label": "Primary\nmelanoma",
        "n_col": "PrimaryMel_mut",
        "N_col": "PrimaryMel_total",
        "q_cols_involving_group": [QCOL_PM_MM, QCOL_PM_BM],
    },
    {
        "key": "MetMel",
        "label": "Metastatic\nmelanoma",
        "n_col": "MetMel_mut",
        "N_col": "MetMel_total",
        "q_cols_involving_group": [QCOL_PM_MM, QCOL_MM_BM],
    },
    {
        "key": "BrainMet",
        "label": "Metastatic\nbrain",
        "n_col": "BrainMet_mut",
        "N_col": "BrainMet_total",
        "q_cols_involving_group": [QCOL_PM_BM, QCOL_MM_BM],
    },
]

# Check that all required columns exist.
needed_cols = [GENE_COL] + q_cols_all
for g in GROUPS:
    needed_cols += [g["n_col"], g["N_col"]]
missing = [c for c in needed_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}\nColumns found: {list(df.columns)}")

# Convert the q-values and counts to numeric values.
for c in q_cols_all + [g["n_col"] for g in GROUPS] + [g["N_col"] for g in GROUPS]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Keep genes with any adjusted p-value below the cutoff.
df["q_min_any"] = df[q_cols_all].min(axis=1, skipna=True)
sig = df[df["q_min_any"] < Q_CUTOFF].copy()

if sig.empty:
    raise ValueError(f"No genes found with any of {q_cols_all} < {Q_CUTOFF}")

# Sort by the smallest adjusted p-value.
sig = sig.sort_values("q_min_any", ascending=True)

# Apply the gene limit when TOP_N is set.
if TOP_N is not None and len(sig) > TOP_N:
    sig = sig.iloc[:TOP_N].copy()

# Save the selected genes.
sig.to_csv(OUT_TSV, sep="\t", index=False)
print(f"Saved filtered table: {OUT_TSV}")
print(f"Genes plotted: {len(sig)}")

# Create one plotting row per gene and group.
rows = []
for g in GROUPS:
    # Use the smaller pairwise q-value involving this group.
    q_group = sig[g["q_cols_involving_group"]].min(axis=1, skipna=True).fillna(1.0)

    n = sig[g["n_col"]]
    N = sig[g["N_col"]]
    frac = (n / N).replace([np.inf, -np.inf], np.nan)

    rows.append(pd.DataFrame({
        "gene": sig[GENE_COL].astype(str),
        "group_key": g["key"],
        "group_label": g["label"],
        "n": n,
        "N": N,
        "frac": frac,
        "q_group": q_group,
        "q_min_any": sig["q_min_any"],
    }))

plot_df = pd.concat(rows, ignore_index=True).dropna(subset=["frac", "q_group"])

# Place genes vertically and cohorts horizontally.
gene_order = sig[GENE_COL].astype(str).tolist()
group_order = [g["key"] for g in GROUPS]
group_label_map = {g["key"]: g["label"] for g in GROUPS}

y_map = {gene: i for i, gene in enumerate(gene_order)}
x_map = {key: i for i, key in enumerate(group_order)}

plot_df["y"] = plot_df["gene"].map(y_map)
plot_df["x"] = plot_df["group_key"].map(x_map).astype(float) * X_SPACING

# Set bubble areas from mutation fractions, with a minimum size.
plot_df["size"] = (plot_df["frac"].clip(lower=0) * SIZE_SCALE) + MIN_SIZE

# Set color values from the pairwise q-value summary.
plot_df["color_val"] = -np.log10(plot_df["q_group"].clip(lower=1e-300, upper=1.0))

# Set the color range and cap values when a percentile is set.
vmin = 0.0
if CLIP_COLOR_AT_PERCENTILE is None:
    vmax = float(np.nanmax(plot_df["color_val"]))
else:
    vmax = float(np.nanpercentile(plot_df["color_val"], CLIP_COLOR_AT_PERCENTILE))
    plot_df["color_val"] = plot_df["color_val"].clip(upper=vmax)

norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
cmap = plt.get_cmap("viridis")

# Scale figure height by the number of genes.
fig_h = max(6, 0.28 * len(gene_order) + 1.8)
fig, ax = plt.subplots(figsize=(6.6, fig_h))

sc = ax.scatter(
    plot_df["x"],
    plot_df["y"],
    s=plot_df["size"],
    c=plot_df["color_val"],
    cmap=cmap,
    norm=norm,
    alpha=0.90,
)

# Add cohort labels.
xticks = [x_map[k] * X_SPACING for k in group_order]
xlabels = [group_label_map[k] for k in group_order]
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, rotation=0)

# Add italic gene labels.
ax.set_yticks(range(len(gene_order)))
ax.set_yticklabels(gene_order)
for lab in ax.get_yticklabels():
    lab.set_fontstyle("italic")

ax.set_xlabel("Cancer group")
ax.set_ylabel("Gene (feature)")
ax.set_title(f"Significant genes (any BH q < {Q_CUTOFF})\n"
             f"Bubble size = fraction mutated (n/N); Color = -log10(q)")

# Set horizontal limits around the cohort positions.
ax.set_xlim(min(xticks) - 0.35, max(xticks) + 0.35)

# Add the colorbar.
cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
cbar.set_label(r"$-\log_{10}(q)$ (group-specific)")

# Add a size legend below the plot.
legend_fracs = [0.05, 0.10, 0.25, 0.50]
handles = [
    ax.scatter([], [], s=f * SIZE_SCALE + MIN_SIZE,
               facecolors="none", edgecolors="black", linewidths=0.8)
    for f in legend_fracs
]
labels = [f"{int(f * 100)}%" for f in legend_fracs]

leg = ax.legend(
    handles, labels,
    title="Fraction mutated",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.11),
    ncol=len(legend_fracs),
    frameon=False,
    handletextpad=0.8,
    columnspacing=1.6,
)

# Leave space below the axes for the legend.
fig.subplots_adjust(bottom=0.18)

# Save the PNG and SVG, then display the figure.
fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
fig.savefig(OUT_SVG, bbox_inches="tight")
plt.show()

print(f"Saved PNG (dpi=600): {OUT_PNG}")
print(f"Saved SVG: {OUT_SVG}")


#################

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

# Figure size and margins.
FIGSIZE = (7.4, 4.6)
LEFT = 0.29
RIGHT = 0.98
TOP = 0.90
BOTTOM = 0.14

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 12,
    "svg.fonttype": "none",   # keep text editable in SVG
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

BAR_COLOR = "#4C647A"
DOT_COLOR = "#1A1A1A"
BG_DOT_COLOR = "#D9D9D9"
GRID_COLOR = "#EAEAEA"


def plot_upset(counts, title, outfile):
    sets = ["Primary melanoma", "Metastatic melanoma", "Metastatic brain"]

    # Membership patterns in a fixed order.
    # Tuple order: (Primary melanoma, Metastatic melanoma, Metastatic brain)
    patterns = [
        (1, 0, 0),  # Primary only
        (0, 1, 0),  # Metastatic melanoma only
        (0, 0, 1),  # Metastatic brain only
        (1, 1, 0),  # Primary and metastatic melanoma only
        (1, 0, 1),  # Primary and metastatic brain only
        (0, 1, 1),  # Metastatic melanoma and brain only
        (1, 1, 1),  # All three
    ]

    # Read the entered intersection counts.
    values = [counts[p] for p in patterns]
    x = np.arange(len(patterns))
    row_y = np.array([2, 1, 0])  # top to bottom order for the 3 sets

    # Sum the intersection counts for each cohort.
    set_sizes = [
        sum(counts[p] for p in patterns if p[0]),
        sum(counts[p] for p in patterns if p[1]),
        sum(counts[p] for p in patterns if p[2]),
    ]

    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        width_ratios=[2.35, 4.65],
        height_ratios=[3.0, 1.55],
        wspace=0.05,
        hspace=0.05,
    )

    ax_empty = fig.add_subplot(gs[0, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_sets = fig.add_subplot(gs[1, 0])
    ax_matrix = fig.add_subplot(gs[1, 1], sharex=ax_top)

    ax_empty.axis("off")

    # Draw the intersection count bars.
    ymax = max(values) * 1.18
    ax_top.bar(x, values, width=0.72, color=BAR_COLOR, edgecolor=BAR_COLOR)
    for xi, yi in zip(x, values):
        ax_top.text(xi, yi + ymax * 0.02, f"{yi}", ha="center", va="bottom")

    ax_top.set_xlim(-0.6, len(x) - 0.4)
    ax_top.set_ylim(0, ymax)
    ax_top.set_ylabel("Intersection size")
    ax_top.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax_top.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax_top.set_axisbelow(True)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    # Draw the membership dots and connecting lines.
    for i, pattern in enumerate(patterns):
        ax_matrix.scatter([i] * len(row_y), row_y, s=80, color=BG_DOT_COLOR, zorder=1)
        present = row_y[np.array(pattern, dtype=bool)]
        if len(present) > 1:
            ax_matrix.plot([i, i], [present.min(), present.max()], color=DOT_COLOR, linewidth=1.8, zorder=2)
        ax_matrix.scatter([i] * len(present), present, s=90, color=DOT_COLOR, zorder=3)

    for y in row_y:
        ax_matrix.axhline(y, color="#F2F2F2", linewidth=0.8, zorder=0)

    ax_matrix.set_ylim(-0.5, 2.5)
    ax_matrix.set_yticks(row_y)
    ax_matrix.set_yticklabels(["", "", ""])
    ax_matrix.set_xticks(x)
    ax_matrix.set_xticklabels([""] * len(x))
    ax_matrix.tick_params(axis="both", length=0)
    for side in ["top", "right", "left", "bottom"]:
        ax_matrix.spines[side].set_visible(False)

    # Draw the total size of each cohort set.
    ax_sets.barh(row_y, set_sizes, height=0.56, color=BAR_COLOR, edgecolor=BAR_COLOR)
    ax_sets.set_yticks(row_y)
    ax_sets.set_yticklabels(sets)
    ax_sets.set_xlabel("Set size")
    ax_sets.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax_sets.set_axisbelow(True)
    ax_sets.set_ylim(-0.5, 2.5)
    ax_sets.set_xlim(0, max(set_sizes) * 1.18)

    for y, s in zip(row_y, set_sizes):
        ax_sets.text(s + max(set_sizes) * 0.02, y, f"{s}", va="center", ha="left")

    ax_sets.spines["top"].set_visible(False)
    ax_sets.spines["right"].set_visible(False)
    ax_sets.spines["left"].set_visible(False)
    ax_sets.tick_params(axis="y", length=0)

    # Save the SVG and close the figure.
    fig.subplots_adjust(left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM)
    fig.savefig(outfile, format="svg", dpi=600)
    plt.close(fig)


# Fixed intersection counts for mutated proteins.
counts = {
    (1, 0, 0): 28,
    (0, 1, 0): 102,
    (0, 0, 1): 38,
    (1, 1, 0): 70,
    (1, 0, 1): 13,
    (0, 1, 1): 50,
    (1, 1, 1): 120,
}

plot_upset(counts, "Mutated proteins", "mutated_proteins_upset.svg")

###
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

# Figure size and margins.
FIGSIZE = (7.4, 4.6)
LEFT = 0.29
RIGHT = 0.98
TOP = 0.90
BOTTOM = 0.14

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 12,
    "svg.fonttype": "none",   # keep text editable in SVG
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

BAR_COLOR = "#4C647A"
DOT_COLOR = "#1A1A1A"
BG_DOT_COLOR = "#D9D9D9"
GRID_COLOR = "#EAEAEA"


def plot_upset(counts, title, outfile):
    sets = ["Primary melanoma", "Metastatic melanoma", "Metastatic brain"]

    # Membership patterns in a fixed order.
    # Tuple order: (Primary melanoma, Metastatic melanoma, Metastatic brain)
    patterns = [
        (1, 0, 0),  # Primary only
        (0, 1, 0),  # Metastatic melanoma only
        (0, 0, 1),  # Metastatic brain only
        (1, 1, 0),  # Primary and metastatic melanoma only
        (1, 0, 1),  # Primary and metastatic brain only
        (0, 1, 1),  # Metastatic melanoma and brain only
        (1, 1, 1),  # All three
    ]

    # Read the entered intersection counts.
    values = [counts[p] for p in patterns]
    x = np.arange(len(patterns))
    row_y = np.array([2, 1, 0])  # top to bottom order for the 3 sets

    # Sum the intersection counts for each cohort.
    set_sizes = [
        sum(counts[p] for p in patterns if p[0]),
        sum(counts[p] for p in patterns if p[1]),
        sum(counts[p] for p in patterns if p[2]),
    ]

    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        width_ratios=[2.35, 4.65],
        height_ratios=[3.0, 1.55],
        wspace=0.05,
        hspace=0.05,
    )

    ax_empty = fig.add_subplot(gs[0, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_sets = fig.add_subplot(gs[1, 0])
    ax_matrix = fig.add_subplot(gs[1, 1], sharex=ax_top)

    ax_empty.axis("off")

    # Draw the intersection count bars.
    ymax = max(values) * 1.18
    ax_top.bar(x, values, width=0.72, color=BAR_COLOR, edgecolor=BAR_COLOR)
    for xi, yi in zip(x, values):
        ax_top.text(xi, yi + ymax * 0.02, f"{yi}", ha="center", va="bottom")

    ax_top.set_xlim(-0.6, len(x) - 0.4)
    ax_top.set_ylim(0, ymax)
    ax_top.set_ylabel("Intersection size")
    ax_top.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax_top.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax_top.set_axisbelow(True)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    # Draw the membership dots and connecting lines.
    for i, pattern in enumerate(patterns):
        ax_matrix.scatter([i] * len(row_y), row_y, s=80, color=BG_DOT_COLOR, zorder=1)
        present = row_y[np.array(pattern, dtype=bool)]
        if len(present) > 1:
            ax_matrix.plot([i, i], [present.min(), present.max()], color=DOT_COLOR, linewidth=1.8, zorder=2)
        ax_matrix.scatter([i] * len(present), present, s=90, color=DOT_COLOR, zorder=3)

    for y in row_y:
        ax_matrix.axhline(y, color="#F2F2F2", linewidth=0.8, zorder=0)

    ax_matrix.set_ylim(-0.5, 2.5)
    ax_matrix.set_yticks(row_y)
    ax_matrix.set_yticklabels(["", "", ""])
    ax_matrix.set_xticks(x)
    ax_matrix.set_xticklabels([""] * len(x))
    ax_matrix.tick_params(axis="both", length=0)
    for side in ["top", "right", "left", "bottom"]:
        ax_matrix.spines[side].set_visible(False)

    # Draw the total size of each cohort set.
    ax_sets.barh(row_y, set_sizes, height=0.56, color=BAR_COLOR, edgecolor=BAR_COLOR)
    ax_sets.set_yticks(row_y)
    ax_sets.set_yticklabels(sets)
    ax_sets.set_xlabel("Set size")
    ax_sets.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax_sets.set_axisbelow(True)
    ax_sets.set_ylim(-0.5, 2.5)
    ax_sets.set_xlim(0, max(set_sizes) * 1.18)

    for y, s in zip(row_y, set_sizes):
        ax_sets.text(s + max(set_sizes) * 0.02, y, f"{s}", va="center", ha="left")

    ax_sets.spines["top"].set_visible(False)
    ax_sets.spines["right"].set_visible(False)
    ax_sets.spines["left"].set_visible(False)
    ax_sets.tick_params(axis="y", length=0)

    # Save the SVG and close the figure.
    fig.subplots_adjust(left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM)
    fig.savefig(outfile, format="svg", dpi=600)
    plt.close(fig)


# Fixed intersection counts for mutated proteins.
counts = {
    (1, 0, 0): 28,
    (0, 1, 0): 102,
    (0, 0, 1): 38,
    (1, 1, 0): 70,
    (1, 0, 1): 13,
    (0, 1, 1): 50,
    (1, 1, 1): 120,
}

plot_upset(counts, "Mutated proteins", "mutated_proteins_upset.svg")


##############
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

# Figure size and margins.
FIGSIZE = (7.4, 4.6)
LEFT = 0.29
RIGHT = 0.98
TOP = 0.90
BOTTOM = 0.14

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 12,
    "svg.fonttype": "none",   # keep text editable in SVG
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

BAR_COLOR = "#4C647A"
DOT_COLOR = "#1A1A1A"
BG_DOT_COLOR = "#D9D9D9"
GRID_COLOR = "#EAEAEA"


def plot_upset(counts, title, outfile):
    sets = ["Primary melanoma", "Metastatic melanoma", "Metastatic brain"]

    # Membership patterns in a fixed order.
    # Tuple order: (Primary melanoma, Metastatic melanoma, Metastatic brain)
    patterns = [
        (1, 0, 0),  # Primary only
        (0, 1, 0),  # Metastatic melanoma only
        (0, 0, 1),  # Metastatic brain only
        (1, 1, 0),  # Primary and metastatic melanoma only
        (1, 0, 1),  # Primary and metastatic brain only
        (0, 1, 1),  # Metastatic melanoma and brain only
        (1, 1, 1),  # All three
    ]

    # Read the entered intersection counts.
    values = [counts[p] for p in patterns]
    x = np.arange(len(patterns))
    row_y = np.array([2, 1, 0])  # top to bottom order for the 3 sets

    # Sum the intersection counts for each cohort.
    set_sizes = [
        sum(counts[p] for p in patterns if p[0]),
        sum(counts[p] for p in patterns if p[1]),
        sum(counts[p] for p in patterns if p[2]),
    ]

    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        width_ratios=[2.35, 4.65],
        height_ratios=[3.0, 1.55],
        wspace=0.05,
        hspace=0.05,
    )

    ax_empty = fig.add_subplot(gs[0, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_sets = fig.add_subplot(gs[1, 0])
    ax_matrix = fig.add_subplot(gs[1, 1], sharex=ax_top)

    ax_empty.axis("off")

    # Draw the intersection count bars.
    ymax = max(values) * 1.18
    ax_top.bar(x, values, width=0.72, color=BAR_COLOR, edgecolor=BAR_COLOR)
    for xi, yi in zip(x, values):
        ax_top.text(xi, yi + ymax * 0.02, f"{yi}", ha="center", va="bottom")

    ax_top.set_xlim(-0.6, len(x) - 0.4)
    ax_top.set_ylim(0, ymax)
    ax_top.set_ylabel("Intersection size")
    ax_top.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax_top.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax_top.set_axisbelow(True)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    # Draw the membership dots and connecting lines.
    for i, pattern in enumerate(patterns):
        ax_matrix.scatter([i] * len(row_y), row_y, s=80, color=BG_DOT_COLOR, zorder=1)
        present = row_y[np.array(pattern, dtype=bool)]
        if len(present) > 1:
            ax_matrix.plot([i, i], [present.min(), present.max()], color=DOT_COLOR, linewidth=1.8, zorder=2)
        ax_matrix.scatter([i] * len(present), present, s=90, color=DOT_COLOR, zorder=3)

    for y in row_y:
        ax_matrix.axhline(y, color="#F2F2F2", linewidth=0.8, zorder=0)

    ax_matrix.set_ylim(-0.5, 2.5)
    ax_matrix.set_yticks(row_y)
    ax_matrix.set_yticklabels(["", "", ""])
    ax_matrix.set_xticks(x)
    ax_matrix.set_xticklabels([""] * len(x))
    ax_matrix.tick_params(axis="both", length=0)
    for side in ["top", "right", "left", "bottom"]:
        ax_matrix.spines[side].set_visible(False)

    # Draw the total size of each cohort set.
    ax_sets.barh(row_y, set_sizes, height=0.56, color=BAR_COLOR, edgecolor=BAR_COLOR)
    ax_sets.set_yticks(row_y)
    ax_sets.set_yticklabels(sets)
    ax_sets.set_xlabel("Set size")
    ax_sets.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax_sets.set_axisbelow(True)
    ax_sets.set_ylim(-0.5, 2.5)
    ax_sets.set_xlim(0, max(set_sizes) * 1.18)

    for y, s in zip(row_y, set_sizes):
        ax_sets.text(s + max(set_sizes) * 0.02, y, f"{s}", va="center", ha="left")

    ax_sets.spines["top"].set_visible(False)
    ax_sets.spines["right"].set_visible(False)
    ax_sets.spines["left"].set_visible(False)
    ax_sets.tick_params(axis="y", length=0)

    # Save the SVG and close the figure.
    fig.subplots_adjust(left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM)
    fig.savefig(outfile, format="svg", dpi=600)
    plt.close(fig)


# Fixed intersection counts for mutated residues.
counts = {
    (1, 0, 0): 468,
    (0, 1, 0): 1257,
    (0, 0, 1): 521,
    (1, 1, 0): 129,
    (1, 0, 1): 20,
    (0, 1, 1): 45,
    (1, 1, 1): 27,
}

plot_upset(counts, "Mutated residues", "mutated_residues_upset.svg")

#pathway bubble
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Input table and filtering thresholds.
INFILE = "stats_out/PathwayFrequency_3GroupStats.tsv"

Q_THRESH = 0.10
MIN_PROP_ANY = 0.25

# Figure dimensions and bubble scaling.
FIG_WIDTH_IN = 8.0

HEIGHT_PER_ROW_IN = 0.24
MIN_FIG_HEIGHT_IN = 6.5

MAX_DIAM_FRACTION = 0.82

# Output folder and file names.
OUTDIR = "plots"
os.makedirs(OUTDIR, exist_ok=True)
OUTBASE = os.path.join(
    OUTDIR,
    f"PathwayBubblePlot_q{str(Q_THRESH).replace('.', 'p')}_minProp{int(MIN_PROP_ANY*100)}_focus-BrainMet"
)

# Cohort labels and table columns.
GROUPS = [
    {"label": "Primary melanoma",    "mut": "PrimaryMel_mut", "total": "PrimaryMel_total", "prop": "PrimaryMel_prop"},
    {"label": "Metastatic melanoma", "mut": "MetMel_mut",     "total": "MetMel_total",     "prop": "MetMel_prop"},
    {"label": "Metastatic brain",    "mut": "BrainMet_mut",   "total": "BrainMet_total",   "prop": "BrainMet_prop"},
]

# Horizontal positions of the cohorts.
XPOS = {
    "Primary melanoma": 0.00,
    "Metastatic melanoma": 0.75,
    "Metastatic brain": 1.50,
}

# Adjusted p-value columns.
Q_OVERALL = "q_overall_BH"
Q_PM_MM = "q_BH_fisher_PrimaryMel_vs_MetMel"
Q_PM_BM = "q_BH_fisher_PrimaryMel_vs_BrainMet"
Q_MM_BM = "q_BH_fisher_MetMel_vs_BrainMet"
Q_COLS = [Q_OVERALL, Q_PM_MM, Q_PM_BM, Q_MM_BM]

# Set font sizes and keep SVG text editable.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 12,

    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 12,

    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# Check that required columns exist.
def require_columns(df: pd.DataFrame, cols: list[str], context: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        prefix = f"{context}: " if context else ""
        raise ValueError(f"{prefix}Missing required column(s): {missing}")


# Read the saved pathway statistics and check the columns.
df = pd.read_csv(INFILE, sep="\t")
df.columns = df.columns.str.strip()

prop_cols = [g["prop"] for g in GROUPS]
mut_cols = [g["mut"] for g in GROUPS]
tot_cols = [g["total"] for g in GROUPS]
require_columns(df, ["feature"] + prop_cols + mut_cols + tot_cols + Q_COLS, context="Input TSV")

# Replace missing or nonnumeric q-values with 1.0 and limit values to 0–1.
for c in Q_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(1.0).clip(0.0, 1.0)

# Replace missing or nonnumeric proportions with zero and limit values to 0–1.
for c in prop_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(0.0, 1.0)

# Keep pathways passing at least one adjusted p-value threshold.
sig_mask = df[Q_COLS].lt(Q_THRESH).any(axis=1)
df_sig = df.loc[sig_mask].copy()

# Require a mutation proportion of at least MIN_PROP_ANY in one cohort.
df_sig["max_prop_any"] = df_sig[prop_cols].max(axis=1)
df_sig = df_sig.loc[df_sig["max_prop_any"] >= MIN_PROP_ANY].copy()

if df_sig.empty:
    raise RuntimeError(
        f"No pathways remained after filtering:\n"
        f"  - any BH q < {Q_THRESH}\n"
        f"  - max(prop across cohorts) >= {MIN_PROP_ANY}\n"
        f"Try relaxing thresholds."
    )

# Summarize pairwise q-values for each cohort.
df_sig["q_group_primary"] = df_sig[[Q_PM_MM, Q_PM_BM]].min(axis=1)
df_sig["q_group_met"]     = df_sig[[Q_PM_MM, Q_MM_BM]].min(axis=1)
df_sig["q_group_brain"]   = df_sig[[Q_PM_BM, Q_MM_BM]].min(axis=1)
df_sig["min_q_any"] = df_sig[[Q_OVERALL, "q_group_primary", "q_group_met", "q_group_brain"]].min(axis=1)

# Sort first by decreasing BrainMet mutation proportion.
df_sig = df_sig.sort_values(
    by=["BrainMet_prop", "q_group_brain", "min_q_any", "feature"],
    ascending=[False, True, True, True]
).copy()

feature_order = df_sig["feature"].drop_duplicates().tolist()

# Save the filtered and sorted pathways.
df_sig.to_csv(f"{OUTBASE}_filtered_sorted.tsv", sep="\t", index=False)


# Create one plotting row per pathway and cohort.
def make_long(group_label: str, mut_col: str, total_col: str, prop_col: str, q_group_col: str) -> pd.DataFrame:
    out = df_sig[["feature", mut_col, total_col, prop_col, q_group_col]].copy()
    out["group"] = group_label
    out.rename(columns={mut_col: "mut", total_col: "total", prop_col: "prop", q_group_col: "q_group"}, inplace=True)
    return out


long = pd.concat([
    make_long("Primary melanoma",    "PrimaryMel_mut", "PrimaryMel_total", "PrimaryMel_prop", "q_group_primary"),
    make_long("Metastatic melanoma", "MetMel_mut",     "MetMel_total",     "MetMel_prop",     "q_group_met"),
    make_long("Metastatic brain",    "BrainMet_mut",   "BrainMet_total",   "BrainMet_prop",   "q_group_brain"),
], ignore_index=True)

long["prop"] = pd.to_numeric(long["prop"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
long["q_group"] = pd.to_numeric(long["q_group"], errors="coerce").fillna(1.0).clip(0.0, 1.0)

# Apply the pathway order.
long["feature"] = pd.Categorical(long["feature"], categories=feature_order, ordered=True)
long = long.sort_values(["feature", "group"]).copy()

# Assign bubble positions.
long["x"] = long["group"].map(XPOS)
long["y"] = long["feature"].cat.codes.astype(float)

# Convert the pairwise q-value summary to a color value.
EPS = 1e-300
long["neglog10_q"] = -np.log10(np.clip(long["q_group"].values, EPS, 1.0))

# Create the main plot and a side panel for the legends.
n_features = len(feature_order)
fig_height = max(MIN_FIG_HEIGHT_IN, HEIGHT_PER_ROW_IN * n_features + 1.8)

fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_height))
gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.0, 0.18], wspace=0.03)

ax = fig.add_subplot(gs[0, 0])
ax_side = fig.add_subplot(gs[0, 1])
ax_side.axis("off")

# Add labels and ticks.
ax.set_title(f"Significant pathways (any BH q < {Q_THRESH})", pad=8)
ax.set_xlabel("Cancer group")
ax.set_ylabel("Pathway (feature)")

ax.set_xticks([XPOS[g["label"]] for g in GROUPS])
ax.set_xticklabels([g["label"] for g in GROUPS], rotation=20, ha="right")

ax.set_yticks(np.arange(n_features))
ax.set_yticklabels(feature_order)

# Place the largest BrainMet proportions at the top.
xmin = min(XPOS.values()) - 0.35
xmax = max(XPOS.values()) + 0.35
ax.set_xlim(xmin, xmax)
ax.set_ylim(n_features - 0.5, -0.5)

# Add horizontal grid lines.
ax.grid(axis="y", linewidth=0.3, alpha=0.25)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Update the axis transforms before measuring spacing.
fig.canvas.draw()

# Measure vertical row spacing in points.
p0 = ax.transData.transform((0, 0))
p1 = ax.transData.transform((0, 1))
dy_px = abs(p1[1] - p0[1])
dy_pt = dy_px * 72.0 / fig.dpi

# Measure the smallest horizontal cohort spacing in points.
xvals = sorted([XPOS[g["label"]] for g in GROUPS])
min_dx_data = min(np.diff(xvals))
q0 = ax.transData.transform((0, 0))
q1 = ax.transData.transform((min_dx_data, 0))
dx_px = abs(q1[0] - q0[0])
dx_pt = dx_px * 72.0 / fig.dpi

# Scale bubble areas using the available spacing.
gap_pt = min(dy_pt, dx_pt)
max_diam_pt = MAX_DIAM_FRACTION * gap_pt
max_area_pt2 = np.pi * (max_diam_pt / 2.0) ** 2

max_prop = float(long["prop"].max()) if len(long) else 1.0
SIZE_SCALE = max_area_pt2 / max(max_prop, 1e-12)

long["size"] = long["prop"] * SIZE_SCALE

# Draw the bubbles.
norm = Normalize(vmin=0, vmax=float(long["neglog10_q"].max()) if len(long) else 1.0)
sc = ax.scatter(
    long["x"], long["y"],
    s=long["size"],
    c=long["neglog10_q"],
    cmap="viridis",
    norm=norm,
    linewidths=0.30,
    edgecolors="black",
    alpha=0.95
)

# Add the colorbar to the side panel.
cax = ax_side.inset_axes([0.05, 0.08, 0.30, 0.84])
cbar = fig.colorbar(sc, cax=cax)
cbar.set_label(r"$-\log_{10}(q)$ (group-specific)")

# Add a size legend using fractions within the plotted range.
legend_props = [0.25, 0.50, 0.75]
legend_props = [p for p in legend_props if p <= max_prop + 1e-9]
if not legend_props:
    legend_props = [max_prop] if max_prop > 0 else [0.25]

handles = [
    ax_side.scatter([], [], s=p * SIZE_SCALE, facecolors="none", edgecolors="black", linewidths=0.8)
    for p in legend_props
]
labels = [f"{int(round(p * 100))}%" for p in legend_props]

ax_side.legend(
    handles, labels,
    title="Fraction altered",
    loc="upper left",
    bbox_to_anchor=(0.42, 1.00),
    frameon=False,
    borderaxespad=0.0,
    labelspacing=0.9,
    handletextpad=1.0
)

# Save the PNG, SVG, and close the figure.
fig.savefig(f"{OUTBASE}.png", dpi=600, bbox_inches="tight")
fig.savefig(f"{OUTBASE}.svg", bbox_inches="tight")
plt.close(fig)

print(f"Saved:\n  {OUTBASE}.png\n  {OUTBASE}.svg\n  {OUTBASE}_filtered_sorted.tsv")