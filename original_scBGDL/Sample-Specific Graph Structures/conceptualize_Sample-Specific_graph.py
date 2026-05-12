import pandas as pd
import numpy as np
import argparse
import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix
from sklearn.preprocessing import QuantileTransformer

# Parse arguments for standardized file paths and parameters
def get_args(cohort):
    parser = argparse.ArgumentParser(description='Graph-based deep learning')
    parser.add_argument('--workspace', default=f'../Datasets/Survival_data/{cohort}_cohort.csv')
    parser.add_argument('--RNAseqMatrix', default=f'../Datasets/RNA-seq Expression/{cohort}_cohort.csv')
    parser.add_argument('--ScMatrix', default=f'../Datasets/Single-Cell Expression/{cohort}_matrix.csv')
    parser.add_argument('--TopGene', default=f'../Datasets/KeyGenes/{cohort}_KeyGene.npy')
    return parser.parse_args()

# Construct graphs for samples based on RNAseq and single-cell expression data
def graph_construction(sample_ids, cohort, args):
    top_genes = np.load(args.TopGene, allow_pickle=True).item()

    # Load single-cell expression data
    sc_matrix = pd.read_csv(args.ScMatrix, index_col=0)

    # Load RNAseq expression data
    rnaseq_matrix = pd.read_csv(args.RNAseqMatrix, index_col=0)
    rnaseq_matrix = pd.DataFrame(
        QuantileTransformer(output_distribution='uniform', n_quantiles=100, random_state=0)
        .fit_transform(rnaseq_matrix),
        index=rnaseq_matrix.index,
        columns=rnaseq_matrix.columns
    )
    rnaseq_matrix = rnaseq_matrix.loc[sc_matrix.index]

    slides_patch, slides_patch_feature, slides_edge = [], [], []

    # Iterate through each sample to construct graph nodes and edges
    for count, sample_id in enumerate(sample_ids, start=1):
        print(f'Processing sample {count}: {sample_id}')

        # Select key genes for the sample
        gene_names = top_genes[sample_id]

        # Process single-cell data
        sc_features = sc_matrix.loc[gene_names].values
        sc_patches = list(sc_matrix.loc[gene_names].index)
        sc_adj_matrix = compute_adjacency_matrix(sc_features, sc_patches)

        # Process RNAseq data
        rnaseq_features = rnaseq_matrix.loc[gene_names].values
        rnaseq_patches = list(rnaseq_matrix.loc[gene_names].index)
        rnaseq_adj_matrix = compute_adjacency_matrix(rnaseq_features, rnaseq_patches)

        # Combine adjacency matrices
        combined_adj_matrix = (sc_adj_matrix + rnaseq_adj_matrix)
        combined_adj_matrix[combined_adj_matrix != 0] = 1
        combined_adj_matrix = csr_matrix(combined_adj_matrix)

        slides_patch.append(sc_patches)
        slides_patch_feature.append(sc_features)
        slides_edge.append(combined_adj_matrix)

    # Save graph components to files
    np.save(f'../Datasets/graphs/{cohort}_patches_name.npy', np.array(slides_patch, dtype=object))
    np.save(f'../Datasets/graphs/{cohort}_nodes.npy', np.array(slides_patch_feature, dtype=object))
    np.save(f'../Datasets/graphs/{cohort}_edges.npy', np.array(slides_edge))

# Helper function to compute adjacency matrix using UMAP
def compute_adjacency_matrix(features, patches):
    obs = pd.DataFrame({'patches': patches})
    var = pd.DataFrame(index=range(features.shape[1]))
    adata = ad.AnnData(features, obs=obs, var=var)
    sc.pp.neighbors(adata, n_neighbors=9, method='umap', use_rep='X')
    adjacency_matrix = adata.obsp['distances'].toarray()
    adjacency_matrix[adjacency_matrix != 0] = 1
    return adjacency_matrix
