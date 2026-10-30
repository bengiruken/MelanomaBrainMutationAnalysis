# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 14:46:40 2025

@author: yavuzb2
"""

# Load the mutation table from a tab-separated file.
import pandas as pd
MAF_file = 'Merged_Genie_Tcga_VAFgreater12point5Percent__new.txt'
df_mutant_residues = pd.read_csv(MAF_file, sep ='\t')


#======
import pandas as pd
import pickle

# Load the metadata with numbered columns and no header row.
filename = 'TCGA_GENIE_Vol6_VAF_0125_Subtype_Tissue_Primary_Metastasis_Info_ccBioPortal_API.txt'

df = pd.read_csv(filename, sep = '\t', header = None)

# Select rows labeled BRAIN in column 2.
df_brain = df[df[2] == 'BRAIN']
df_brain

# Keep the original variable name used in later cells.
df_gbm = df_brain
#throughout the codes df_gbm = df_brain

# Select rows labeled SKIN in column 2.
df_skin = df[df[2] == 'SKIN']
df_skin

# Select rows labeled MEL in column 1.
df_mel = df[df[1] == 'MEL']


# Collect sample IDs for primary brain samples.
brain_primary = set(df_brain[(df_brain[3] == 'Primary') ][0].to_list())

# Collect sample IDs for metastatic brain samples.
brain_metastasis = set(df_brain[(df_brain[3] == 'Metastasis') ][0].to_list())

# Collect sample IDs for primary melanoma samples.
melanoma_primary = set(df_mel[(df_mel[3] == 'Primary') & (df_mel[1] == 'MEL')][0].to_list())

# Collect sample IDs for metastatic melanoma samples.
melanoma_metastasis = set(df_mel[(df_mel[3] == 'Metastasis') & (df_mel[1] == 'MEL')][0].to_list())

# Select mutation records for primary melanoma samples.
df_mutant_res_primary_melanoma = df_mutant_residues[df_mutant_residues['Tumor_Sample_Barcode'].isin(melanoma_primary )]

# Select mutation records for metastatic melanoma samples.
df_mutant_res_metastatic_melanoma = df_mutant_residues[df_mutant_residues['Tumor_Sample_Barcode'].isin(melanoma_metastasis )]

# Select mutation records for metastatic brain samples.
df_mutant_res_metastatic_brain = df_mutant_residues[df_mutant_residues['Tumor_Sample_Barcode'].isin(brain_metastasis )]

##################

# draw WT and mutant residue bar plot

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_gene_position_mutations(df, gene, position, tag = 'primary melanoma', save_path=None):
    """
    Plot WT→Mutant residue changes at a specific gene and position.

    Parameters:
        df (pd.DataFrame): mutation dataframe
        gene (str): gene name (e.g., "BRAF")
        position (int): residue position (e.g., 600)
        save_path (str, optional): file path to save PNG
                                   (default: "<gene>_WT_mutant_residue_<position>_res.png")
    """
    # ensure ResidueNumber is numeric
    df = df.copy()
    df['ResidueNumber'] = pd.to_numeric(df['ResidueNumber'], errors="coerce")

    # filter for gene & position
    sub = df[(df['Hugo_Symbol'] == gene) & (df['ResidueNumber'] == position)].copy()

    if sub.empty:
        print(f"No mutations found for {gene} at position {position}")
        return

    # create a label for WT→Mut
    sub['Mutation_Label'] = sub['WildTypeResidue'].astype(str) + str(position) + sub['MutantResidue'].astype(str)

    # count occurrences
    counts = sub['Mutation_Label'].value_counts()

    # define pastel colors
    colors = sns.color_palette("pastel", n_colors=len(counts))

    # plot
    plt.figure(figsize=(8,5))
    counts.plot(kind='bar', color=colors, edgecolor="black", width=0.6)
    plt.title(f"{tag} - {gene} mutations at position {position}", fontsize=14)
    plt.xlabel("Mutation", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # default save path if not provided
    if save_path is None:
        save_path = f"{tag}_{gene}_WT_mutant_residue_{position}_res.png"

    # save figure
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    
    #KRAS G12  
# 
plot_gene_position_mutations(df_mutant_res_primary_melanoma, 'KRAS', 12, tag = 'primary melanoma')

# 
plot_gene_position_mutations(df_mutant_res_metastatic_melanoma, 'KRAS', 12, tag = 'metastatic melanoma')

#
plot_gene_position_mutations(df_mutant_res_metastatic_brain, 'KRAS', 12, tag = 'metastatic brain')

####
# Plot NRAS position 61 in primary melanoma.
plot_gene_position_mutations(df_mutant_res_primary_melanoma, 'NRAS', 61, tag = 'primary melanoma')

# Plot NRAS position 61 in metastatic mel
plot_gene_position_mutations(df_mutant_res_metastatic_melanoma, 'NRAS', 61, tag = 'metastatic melanoma')

# Plot 
plot_gene_position_mutations(df_mutant_res_metastatic_brain, 'NRAS', 61, tag = 'metastatic brain')

# Plot NRAS position 61 in primary melanoma.
plot_gene_position_mutations(df_mutant_res_primary_melanoma, 'NRAS', 61, tag = 'primary melanoma')
plot_gene_position_mutations(df_mutant_res_metastatic_melanoma, 'NRAS', 61, tag = 'metastatic melanoma')
plot_gene_position_mutations(df_mutant_res_metastatic_brain, 'NRAS', 61, tag = 'metastatic brain')


# Upload the mutations
import pickle

mutation_pickle_file = "GeneMutation_tumor_dictionary.p"

with open(mutation_pickle_file, 'rb') as file:
    gene_mutation_tumor_dict = pickle.load(file)
    
# Keep matching samples for each mutation in the brain metastasis group.
brain_metastasis_mutation_tumor_dict = {}
for mut in gene_mutation_tumor_dict.keys():
    intersecting = gene_mutation_tumor_dict[mut].intersection(brain_metastasis)
    if len(intersecting) > 0:
        brain_metastasis_mutation_tumor_dict[mut] = intersecting

#  matching samples for each mutation in the primary melanoma group
melanoma_primary_mutation_tumor_dict = {}
for mut in gene_mutation_tumor_dict.keys():
    intersecting = gene_mutation_tumor_dict[mut].intersection(melanoma_primary)
    if len(intersecting) > 0:
        melanoma_primary_mutation_tumor_dict[mut] = intersecting

#  matching samples for each mutation in the metastatic melanoma group
melanoma_metastasis_mutation_tumor_dict = {}
for mut in gene_mutation_tumor_dict.keys():
    intersecting = gene_mutation_tumor_dict[mut].intersection(melanoma_metastasis)
    if len(intersecting) > 0:
        melanoma_metastasis_mutation_tumor_dict[mut] = intersecting    
    

# correct typo in var. name
brain_metastatis_mutation_tumor_dict =brain_metastasis_mutation_tumor_dict


# Write OncoPrint input for one gene.
def write_oncoprint_input_for_gene(mutation_dict, gene_name, output_filename):
    with open(output_filename, 'w') as outfile:
        for mut_key, tumor_set in mutation_dict.items():
            if mut_key.split('_')[0] == gene_name and len(tumor_set) >= 3:
                    for tumor in tumor_set:
                        # The final two fields are fixed labels.
                        outfile.write(f"{tumor}\t{mut_key}\tHOMDEL\tCNA\n")


    
    
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

# Primary melanoma.
write_oncoprint_input_for_gene(melanoma_primary_mutation_tumor_dict, 'TP53', 'PrimaryMelanoma_TP53_Oncoprint.txt')
write_oncoprint_input_for_gene(melanoma_primary_mutation_tumor_dict, 'BRAF', 'PrimaryMelanoma_BRAF_Oncoprint.txt')

# Metastatic melanoma.
write_oncoprint_input_for_gene(melanoma_metastasis_mutation_tumor_dict, 'TP53', 'MetastaticMelanoma_TP53_Oncoprint.txt')
write_oncoprint_input_for_gene(melanoma_metastasis_mutation_tumor_dict, 'BRAF', 'MetastaticMelanoma_BRAF_Oncoprint.txt')

# Brain metastasis.
write_oncoprint_input_for_gene(brain_metastasis_mutation_tumor_dict, 'TP53', 'BrainMetastasis_TP53_Oncoprint.txt')
write_oncoprint_input_for_gene(brain_metastasis_mutation_tumor_dict, 'BRAF', 'BrainMetastasis_BRAF_Oncoprint.txt')

###3
# List unique gene names for primary melanoma.
melanoma_primary_mutated_gene_list = sorted({key.split('_')[0] for key in melanoma_primary_mutation_tumor_dict})

# List unique gene names for metastatic melanoma.
melanoma_metastasis_mutated_gene_list = sorted({key.split('_')[0] for key in melanoma_metastasis_mutation_tumor_dict})

# List unique gene names for brain metastasis.
brain_metastasis_mutated_gene_list = sorted({key.split('_')[0] for key in brain_metastatis_mutation_tumor_dict})

#
import pandas as pd
import upsetplot
from upsetplot import from_memberships, UpSet
import matplotlib.pyplot as plt

# Create gene sets for the three sample groups.
melanoma_primary_genes = set(melanoma_primary_mutated_gene_list)
melanoma_metastasis_genes = set(melanoma_metastasis_mutated_gene_list)
brain_metastasis_genes = set(brain_metastasis_mutated_gene_list)

# Record which groups contain each gene.
memberships = []
all_genes = melanoma_primary_genes | melanoma_metastasis_genes | brain_metastasis_genes

for gene in all_genes:
    sets = []
    if gene in melanoma_primary_genes:
        sets.append('Primary melanoma')
    if gene in melanoma_metastasis_genes:
        sets.append('Metastatic melanoma')
    if gene in brain_metastasis_genes:
        sets.append('Metastatic brain')
    memberships.append(sets)

# Prepare the membership data for UpSetPlot.
data = from_memberships(memberships)

# Draw the overlap plot.
plt.figure(figsize=(10, 6))
upset = UpSet(data, subset_size='count', show_counts=True, sort_by='degree')
upset.plot()
plt.suptitle("UpSet Plot of Mutated Genes in Melanoma and Brain Metastasis", fontsize=14)
plt.tight_layout()

# Save the figure as SVG.
plt.savefig("mutated_genes_upset_plot.svg", format="svg")
plt.show()

####Venn diagram

import matplotlib.pyplot as plt
from matplotlib_venn import venn3

# Create gene sets for the three sample groups.
melanoma_primary_genes = set(melanoma_primary_mutated_gene_list)
melanoma_metastasis_genes = set(melanoma_metastasis_mutated_gene_list)
brain_metastasis_genes = set(brain_metastasis_mutated_gene_list)

# Draw the Venn diagram.
plt.figure(figsize=(8, 8))
venn3(
    [melanoma_primary_genes, melanoma_metastasis_genes, brain_metastasis_genes],
    set_labels=("Primary melanoma", "Metastatic melanoma", "Metastatic brain")
)

plt.title("Venn Diagram of Mutated Genes in Melanoma and Brain Metastasis", fontsize=14)
plt.tight_layout()

# Save the figure as SVG.
plt.savefig("mutated_genes_venn_plot.svg", format="svg")
plt.show()