# amines/aldehydes clustering and success-rate analysis
# Save this file as clustering_amines_aldehydes.py or open in a notebook.
# Requirements: rdkit, pandas, numpy, scikit-learn, umap-learn, matplotlib
#
# This script is adapted to the CSV you provided (semicolon-delimited with header:
# AMINE;ALDEHYDE;SUCCESS). It will:
#  - detect delimiter (semicolon/comma)
#  - create per-reagent datasets for AMINE and ALDEHYDE
#  - cluster each reagent set independently and compute per-SMILES and per-cluster success rates
#  - save CSV outputs and UMAP PNGs

import math
import argparse
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

# RDKit imports - make sure RDKit is installed in your environment
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.ML.Cluster import Butina
except Exception as e:
    print('ERROR: RDKit imports failed. Install RDKit in your Python environment.
', e)
    raise

from sklearn.cluster import AgglomerativeClustering
import umap
import matplotlib.pyplot as plt


# ----------------------- helpers -----------------------

def mol_from_smiles(smi):
    if pd.isna(smi):
        return None
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def morgan_fp_bits(mol, radius=2, nBits=2048):
    # Returns RDKit ExplicitBitVect
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)


def tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def compute_distance_matrix(fps):
    n = len(fps)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            sim = tanimoto_similarity(fps[i], fps[j])
            dist = 1.0 - sim
            d[i, j] = dist
            d[j, i] = dist
    return d


def butina_cluster_from_fps(fps, cutoff=0.4):
    n = len(fps)
    if n == 0:
        return []
    dists = []
    for i in range(n-1):
        for j in range(i+1, n):
            dists.append(1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    return clusters


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + z**2 / n
    centre = phat + z**2 / (2*n)
    adj = z * math.sqrt((phat*(1-phat) + z**2/(4*n)) / n)
    return ((centre - adj) / denom, (centre + adj) / denom)


# ----------------------- core pipeline -----------------------

def cluster_and_summarize(df, smiles_col, label_col, which_name='amines',
                          fp_radius=2, fp_bits=2048,
                          min_count=5, smoothing_alpha=1.0,
                          butina_cutoff=0.4,
                          use_butina=True,
                          umap_n_components=2,
                          umap_neighbors=15,
                          out_prefix='results'):
    """
    df: pandas DataFrame with at least smiles_col and label_col
    smiles_col: column name containing SMILES (strings)
    label_col: column name containing binary success indicator (0/1)
    which_name: string used for labeling outputs (e.g. 'amines' or 'aldehydes')
    Returns summary dataframe and cluster assignments
    """

    df = df.copy()
    # Parse molecules and fingerprints
    df['mol'] = df[smiles_col].apply(mol_from_smiles)
    invalid = df[df['mol'].isnull()]
    if len(invalid) > 0:
        print(f"Warning: {len(invalid)} rows with invalid SMILES in column '{smiles_col}' will be skipped.
First invalid examples:")
        print(invalid[[smiles_col, label_col]].head(5).to_string(index=False))

    df = df[df['mol'].notnull()].reset_index(drop=True)
    if df.empty:
        raise ValueError(f'No valid molecules parsed from SMILES in column: {smiles_col}')

    df['fp'] = df['mol'].apply(lambda m: morgan_fp_bits(m, radius=fp_radius, nBits=fp_bits))

    # Group by unique SMILES (some SMILES may repeat) -> compute per-SMILES counts
    grouped = df.groupby(smiles_col)

    entries = []
    for name, g in grouped:
        trials = len(g)
        successes = int(g[label_col].sum())
        mol = g['mol'].iloc[0]
        fp = g['fp'].iloc[0]
        entries.append({'smiles': name, 'n': trials, 'k': successes, 'mol': mol, 'fp': fp})

    entries_df = pd.DataFrame(entries)
    entries_df = entries_df.sort_values('n', ascending=False).reset_index(drop=True)

    # Clustering
    fps = list(entries_df['fp'])
    if use_butina:
        clusters = butina_cluster_from_fps(fps, cutoff=butina_cutoff)
        labels = -1 * np.ones(len(fps), dtype=int)
        for cid, cl in enumerate(clusters):
            for idx in cl:
                labels[idx] = cid
        missing = np.where(labels == -1)[0]
        next_label = len(clusters)
        for i in missing:
            labels[i] = next_label
            next_label += 1
    else:
        D = compute_distance_matrix(fps)
        n_clusters = max(2, int(len(fps) / 20))
        agg = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
        labels = agg.fit_predict(D)

    entries_df['cluster'] = labels

    # compute cluster stats
    cluster_rows = []
    for cid in sorted(entries_df['cluster'].unique()):
        members = entries_df[entries_df['cluster'] == cid]
        total_trials = int(members['n'].sum())
        total_success = int(members['k'].sum())
        raw_rate = total_success / total_trials if total_trials > 0 else np.nan
        smoothed_rate = (total_success + smoothing_alpha) / (total_trials + 2*smoothing_alpha)
        lower, upper = wilson_ci(total_success, total_trials)
        distinct = len(members)
        repr_row = members.sort_values(['n','k'], ascending=[False, False]).iloc[0]
        cluster_rows.append({
            'cluster_id': cid,
            'n_smiles': distinct,
            'total_trials': total_trials,
            'total_success': total_success,
            'raw_rate': raw_rate,
            'smoothed_rate': smoothed_rate,
            'wilson_lower': lower,
            'wilson_upper': upper,
            'repr_smiles': repr_row['smiles'],
            'repr_n': int(repr_row['n']),
            'repr_k': int(repr_row['k'])
        })

    cluster_df = pd.DataFrame(cluster_rows).sort_values('smoothed_rate', ascending=False).reset_index(drop=True)
    cluster_df['reliability'] = cluster_df['total_trials'].apply(lambda x: 'low' if x < min_count else 'ok')

    entries_df['raw_rate'] = entries_df.apply(lambda r: r['k']/r['n'] if r['n']>0 else np.nan, axis=1)
    entries_df['smoothed_rate'] = entries_df.apply(lambda r: (r['k'] + smoothing_alpha) / (r['n'] + 2*smoothing_alpha), axis=1)
    entries_df[['wilson_lower','wilson_upper']] = entries_df.apply(lambda r: pd.Series(wilson_ci(r['k'], r['n'])), axis=1)
    entries_df['reliability'] = entries_df['n'].apply(lambda x: 'low' if x < min_count else 'ok')

    # UMAP embedding for visualization
    try:
        arr = np.zeros((len(fps), fp_bits), dtype=np.uint8)
        for i, fp in enumerate(fps):
            onbits = list(fp.GetOnBits())
            arr[i, onbits] = 1
        reducer = umap.UMAP(n_components=umap_n_components, n_neighbors=umap_neighbors, random_state=42)
        emb = reducer.fit_transform(arr)
        entries_df['umap_x'] = emb[:,0]
        entries_df['umap_y'] = emb[:,1]
    except Exception as e:
        print('UMAP failed:', e)
        entries_df['umap_x'] = np.nan
        entries_df['umap_y'] = np.nan

    # Save outputs
    entries_df.to_csv(f"{out_prefix}_{which_name}_per_smiles.csv", index=False)
    cluster_df.to_csv(f"{out_prefix}_{which_name}_cluster_summary.csv", index=False)

    members_out = []
    for cid in sorted(entries_df['cluster'].unique()):
        members = entries_df[entries_df['cluster']==cid][['smiles','n','k','raw_rate','smoothed_rate','wilson_lower','wilson_upper','reliability']]
        members = members.copy()
        members['cluster_id'] = cid
        members_out.append(members)
    members_out_df = pd.concat(members_out, ignore_index=True)
    members_out_df.to_csv(f"{out_prefix}_{which_name}_members.csv", index=False)

    # Plot UMAP colored by smoothed_rate and sized by n
    try:
        plt.figure(figsize=(8,6))
        sc = plt.scatter(entries_df['umap_x'], entries_df['umap_y'],
                         c=entries_df['smoothed_rate'], cmap='viridis', s=np.clip(entries_df['n']*4, 8, 300), alpha=0.9)
        plt.colorbar(sc, label='smoothed success rate')
        plt.title(f'UMAP of {which_name} (points sized by n, colored by smoothed success rate)')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_{which_name}_umap.png", dpi=200)
        plt.close()
    except Exception as e:
        print('UMAP plot failed:', e)

    return entries_df, cluster_df, members_out_df


# ----------------------- CLI -----------------------

def main():
    parser = argparse.ArgumentParser(description='Cluster SMILES (amines or aldehydes) and compute success-rate summaries')
    parser.add_argument('--input', required=True, help='CSV input file with columns for AMINE, ALDEHYDE, SUCCESS')
    parser.add_argument('--amine_col', default='AMINE', help='Column name for amine SMILES (default: AMINE)')
    parser.add_argument('--aldehyde_col', default='ALDEHYDE', help='Column name for aldehyde SMILES (default: ALDEHYDE)')
    parser.add_argument('--label_col', default='SUCCESS', help='Column name for binary success label (0/1) (default: SUCCESS)')
    parser.add_argument('--out_prefix', default='results', help='Prefix for output files')
    parser.add_argument('--min_count', type=int, default=3, help='Minimum total trials per cluster to be considered reliable (default:3)')
    parser.add_argument('--butina_cutoff', type=float, default=0.35, help='Tanimoto cutoff for Butina clustering (default:0.35)')
    parser.add_argument('--smoothing_alpha', type=float, default=1.0, help='Laplace smoothing alpha')
    args = parser.parse_args()

    # Read CSV with delimiter detection (supports ; or ,)
    try:
        df = pd.read_csv(args.input, sep=None, engine='python')
    except Exception as e:
        print('Auto-detection failed, trying semicolon...')
        df = pd.read_csv(args.input, sep=';')

    print('Loaded', len(df), 'rows from', args.input)

    # Normalize column names to uppercase for robustness
    df.columns = [c.strip() for c in df.columns]

    required = [args.amine_col, args.aldehyde_col, args.label_col]
    for col in required:
        if col not in df.columns:
            print(f"ERROR: required column '{col}' not found in input. Available columns: {df.columns.tolist()}")
            sys.exit(1)

    # Ensure label column is numeric 0/1
    df[args.label_col] = pd.to_numeric(df[args.label_col], errors='coerce')
    if df[args.label_col].isnull().any():
        print('Warning: some SUCCESS values could not be converted to numeric and will be treated as NaN -> dropped')
    df = df.dropna(subset=[args.label_col])
    df[args.label_col] = df[args.label_col].astype(int)

    # Prepare two datasets: amines and aldehydes
    amine_df = df[[args.amine_col, args.label_col]].rename(columns={args.amine_col: 'smiles', args.label_col: 'success'})
    aldehyde_df = df[[args.aldehyde_col, args.label_col]].rename(columns={args.aldehyde_col: 'smiles', args.label_col: 'success'})

    # Add a prep step to remove empty SMILES
    amine_df = amine_df[amine_df['smiles'].notnull() & (amine_df['smiles'].astype(str).str.strip() != '')].reset_index(drop=True)
    aldehyde_df = aldehyde_df[aldehyde_df['smiles'].notnull() & (aldehyde_df['smiles'].astype(str).str.strip() != '')].reset_index(drop=True)

    # Run clustering and summarization for amines
    print('
Processing amines...')
    try:
        a_entries, a_clusters, a_members = cluster_and_summarize(amine_df, 'smiles', 'success', which_name='amines',
                                                                  min_count=args.min_count, butina_cutoff=args.butina_cutoff,
                                                                  smoothing_alpha=args.smoothing_alpha, out_prefix=args.out_prefix)
        print('Amines: per-smiles ->', f"{args.out_prefix}_amines_per_smiles.csv")
        print('Amines: cluster summary ->', f"{args.out_prefix}_amines_cluster_summary.csv")
    except Exception as e:
        print('Amines processing failed:', e)

    # Run clustering and summarization for aldehydes
    print('
Processing aldehydes...')
    try:
        d_entries, d_clusters, d_members = cluster_and_summarize(aldehyde_df, 'smiles', 'success', which_name='aldehydes',
                                                                  min_count=args.min_count, butina_cutoff=args.butina_cutoff,
                                                                  smoothing_alpha=args.smoothing_alpha, out_prefix=args.out_prefix)
        print('Aldehydes: per-smiles ->', f"{args.out_prefix}_aldehydes_per_smiles.csv")
        print('Aldehydes: cluster summary ->', f"{args.out_prefix}_aldehydes_cluster_summary.csv")
    except Exception as e:
        print('Aldehydes processing failed:', e)

    print('
Done. Check CSV outputs and UMAP PNGs in the current directory.')


if __name__ == '__main__':
    main()
