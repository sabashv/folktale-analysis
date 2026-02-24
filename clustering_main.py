import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from collections import defaultdict
import logging
import os
from datetime import datetime
import atu_classification
import thompson_comparison
from itertools import product


log_dir = "clustering_tuning_logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.info(f"Script started. Single log file: {log_file}")r

def run_tuning(
    n_components=200,
    distance_threshold=0.6,
    linkage="complete",
    min_stories_per_motif=2,
    test_size=0.2,
    min_stories_for_stratify=5,
    seed=42,
    embeddings_path='embeddings.npy',
    df_path='full_metadata_df.pkl',
    atu_to_tmi_path="ATU_to_TMI.csv",
    sim_thr=0.4
):
    
    logger.info(f"Starting run with params: n_components={n_components}, "
                f"threshold={distance_threshold}, linkage={linkage}, "
                f"min_motif_stories={min_stories_per_motif}")

    sim_thr =  1 - distance_threshold

    df = pd.read_pickle(df_path)
    logger.info(f"Loaded {len(df)} stories")

    all_embeddings = np.load(embeddings_path)
    all_units = np.array([u for units in df['normalized_units'] for u in units])
    all_story_ids = np.array([sid for sid, units in zip(df['story_id'], df['normalized_units']) for _ in units])
    logger.info(f"Embeddings: {all_embeddings.shape}, Units: {len(all_units)}")

    # 1. atu classification
    atu_accuracy = atu_classification.run_atu_classification(df, all_embeddings, n_components, distance_threshold, linkage, min_stories_per_motif, test_size, min_stories_for_stratify, seed)

    
    svd = TruncatedSVD(n_components = n_components, random_state=seed)
    reduced = svd.fit_transform(all_embeddings)

    clustering = AgglomerativeClustering( n_clusters = None, distance_threshold = distance_threshold, metric="cosine", linkage = linkage)
    labels = clustering.fit_predict(reduced)

    # 2. silhouette score
    sil_score = silhouette_score(reduced, labels)

    clusters = defaultdict(list)
    for i, cid in enumerate(labels):
        clusters[cid].append(i)

    motif_clusters = [cid for cid, idxs in clusters.items() if len(set(all_story_ids[idxs])) >= min_stories_per_motif]
    num_all_clusters = len(clusters)
    num_motif_clusters = len(motif_clusters)

    # Create motif centroids and map to stories 
    cluster_to_centroid = {}
    for cid in motif_clusters:
        idxs = clusters[cid]
        cluster_embs = reduced[idxs]
        centroid = cluster_embs.mean(axis=0)
        cluster_to_centroid[cid] = centroid

    # only motif clusters
    unit_to_cluster = dict(zip(all_units, labels))
    unit_to_centroid = {u: cluster_to_centroid.get(unit_to_cluster.get(u)) 
                        for u in all_units if unit_to_cluster.get(u) in cluster_to_centroid}

    def get_motif_vectors(units):
        vectors = []
        seen = set()
        for u in units:
            cent = unit_to_centroid.get(u)
            if cent is not None and id(cent) not in seen: 
                seen.add(id(cent))
                vectors.append(cent)
        return vectors

    df["canonical_units"] = df["normalized_units"].apply(get_motif_vectors)
    logger.info("Added 'canonical_units' column to df (motif centroid vectors per story)")

    stories_with_motif = set()
    for cid in motif_clusters:
        stories_with_motif.update(all_story_ids[clusters[cid]])

    num_stories_without = len(set(df["story_id"]) - stories_with_motif)

    compression_ratio = len(all_units) / num_motif_clusters if num_motif_clusters > 0 else 0.0

    # 3. cluster entropy (lower better)
    entropy = compute_cluster_entropy(reduced, labels)

    # 4. clusters distance (mena distance)
    between_dist = between_cluster_distances(reduced, labels, motif_clusters)

    # 5. thompson recall
    th_recall = thompson_comparison.compare_with_thompson(df, pd.read_csv(atu_to_tmi_path), svd, sim_thr, text_col="thompson_motif_text", embeddings_path='tmi_raw_embeddings.npy')

    metrics = {
        "silhouette_score": sil_score,
        "num_all_clusters": num_all_clusters,
        "num_motif_clusters": num_motif_clusters,
        "compression_ratio": compression_ratio,
        "num_stories_without_motif": num_stories_without,
        "motif_cluster_entropy": entropy,
        "between_motif_distance": between_dist,
        "thompson_recall": th_recall,
        "atu_prediction_accuracy": atu_accuracy
    }

    logger.info("Final metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    return metrics

def compute_cluster_entropy(embeddings, labels):
    global_mean = embeddings.mean(axis=0)
    global_var = np.mean(np.sum((embeddings - global_mean) ** 2, axis=1))

    entropies = []
    for cid in np.unique(labels):
        mask = labels == cid
        if mask.sum() <= 1: continue
        cm = embeddings[mask].mean(axis=0)
        cv = np.mean(np.sum((embeddings[mask] - cm) ** 2, axis=1))
        entropies.append(cv / global_var)

    return np.mean(entropies) if entropies else 0.0

def between_cluster_distances(embeddings, labels, motif_clusters):
    centroids = [embeddings[labels == cid].mean(axis=0) for cid in motif_clusters if (labels == cid).sum() > 0]
    if len(centroids) < 2: return 0.0
    centroids = np.vstack(centroids)
    dist_matrix = cosine_distances(centroids)
    distances = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    return distances.mean() if distances.size > 0 else 0.0


if __name__ == "__main__":

    param_grid = {
    'n_components':      [100, 150, 200, 250],
    'distance_threshold': [0.55, 0.6, 0.65, 0.7],
    'linkage':           ['complete', 'average'],
    'min_stories_per_motif': [2, 3] 
    }

    results_list = []

    for params in product(*param_grid.values()):
        kwargs = dict(zip(param_grid.keys(), params))
        logger.info(f"Starting run with params: {kwargs}")

        try:
            metrics = run_tuning(**kwargs)
            results_list.append({**kwargs, **metrics})
            logger.info(f"Experiment finished successfully")
        except Exception as e:
            logger.error(f"Run failed: {e}")
            results_list.append({**kwargs, 'error': str(e)})

    df_results = pd.DataFrame(results_list)
    df_results.to_csv("tuning_results.csv", index=False)
    logger.info(f"Saved {len(df_results)} experiments to tuning_results.csv")