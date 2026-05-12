import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

def get_key_genes(cohort=''):
    # Load bulk RNA-seq expression data
    expression_rnaseq = pd.read_csv(f'../Datasets/RNA-seq Expression/{cohort}_cohort.csv')
    sample_ids = expression_rnaseq.iloc[:, 0]
    expression_rnaseq = expression_rnaseq.iloc[:, 1:]
    expression_rnaseq.index = sample_ids
    expression_rnaseq = pd.DataFrame(expression_rnaseq, index=sample_ids, columns=expression_rnaseq.columns)

    # Load single-cell RNA expression data
    expression_cell = pd.read_csv("../Datasets/Single-Cell Expression/{cohort}_matrix.csv")
    gene_ids = expression_cell.iloc[:, 0]
    expression_cell = expression_cell.iloc[:, 1:]
    expression_cell.index = gene_ids

    # Align bulk RNA-seq with single-cell genes
    expression_rnaseq = expression_rnaseq.loc[gene_ids]

    # Compute correlation matrix between bulk and single-cell data
    combined_data = pd.concat([expression_rnaseq, expression_cell], axis=1)
    correlation_matrix = combined_data.corr().iloc[:len(expression_rnaseq.columns), len(expression_rnaseq.columns):]

    # Initialize dataframes for storing results
    cor_scores = pd.DataFrame()
    key_genes = pd.DataFrame()
    key_genes_fdr = pd.DataFrame()

    n_samples = correlation_matrix.shape[0]
    n_genes = expression_cell.shape[0]

    # Compute correlations and p-values
    for i in range(n_samples):
        scores = correlation_matrix.iloc[i, :]
        p_values = []
        correlations = []

        for j in range(n_genes):
            exp_values = expression_cell.iloc[j, :].values
            correlation, p_value = spearmanr(exp_values, scores)
            p_values.append(p_value)
            correlations.append(correlation)

        # Adjust p-values with the Benjamini-Hochberg method
        fdr_values = multipletests(p_values, method='fdr_bh')[1]

        cor_scores = pd.concat([cor_scores, pd.DataFrame(correlations)], axis=1)
        key_genes = pd.concat([key_genes, pd.DataFrame(p_values)], axis=1)
        key_genes_fdr = pd.concat([key_genes_fdr, pd.DataFrame(fdr_values)], axis=1)

    # Calculate ranking scores
    rank_score_p = pd.DataFrame()
    rank_score_fdr = pd.DataFrame()

    for i in range(cor_scores.shape[1]):
        sig_values = []
        sig_values_fdr = []

        for j in range(cor_scores.shape[0]):
            if cor_scores.iloc[j, i] > 0:
                sig_p = 1 - 2 * key_genes.iloc[j, i]
                sig_fdr = 1 - 2 * key_genes_fdr.iloc[j, i]
            else:
                sig_p = 2 * key_genes.iloc[j, i] - 1
                sig_fdr = 2 * key_genes_fdr.iloc[j, i] - 1

            sig_values.append(sig_p)
            sig_values_fdr.append(sig_fdr)

        rank_score_p = pd.concat([rank_score_p, pd.DataFrame(sig_values)], axis=1)
        rank_score_fdr = pd.concat([rank_score_fdr, pd.DataFrame(sig_values_fdr)], axis=1)

    # Extract key genes for each sample
    key_genes_dict = {}
    for i in range(n_samples):
        significant_genes = gene_ids[
            (key_genes_fdr.iloc[:, i] <= 0.05) & (abs(rank_score_fdr.iloc[:, i]) >= 0.995)
        ]
        key_genes_dict[correlation_matrix.index[i]] = significant_genes

    # Save key genes as a numpy file
    np.save(f'../Datasets/KeyGenes/{cohort}_keyGene.npy', np.array(key_genes_dict, dtype=object))
