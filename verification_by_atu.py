import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report
import logging
from datetime import datetime
import os

##params:
seed = 42
test_size = 0.2
motif_cluster_min_story_count = 2
min_stories_for_stratify = 5
dim_reduction = 100
dist_threshold = 0.6
cluster_linkage = "complete"

log_dir = "./verification_logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"verification_by_atu_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logger = logging.getLogger("ATUVerification")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info("Starting ATU verification")
logger.info(f"Log file: {log_filename}")
logger.info(f"Parameters:\n -Test size: {test_size} \n - Motif cluster min story count: {motif_cluster_min_story_count} \n - Min stories for stratify: {min_stories_for_stratify} \n - Reduced dimnetionality: {dim_reduction} \n - Distance Threshold: {dist_threshold} \n - Clustering linkage: {cluster_linkage} \n - seed : {seed}")


np.random.seed(seed)  

def prepare_data(df, column_name='normalized_units'):
    df_units = df.explode(column_name).reset_index(drop=True)
    df_units = df_units.rename(columns={column_name: "unit"})
    df_units["embedding_index"] = np.arange(len(df_units))
    return df_units

def load_or_compute_embeddings(embeddings_path='embeddings.npy'):
    #todo: add the model to compute embeddings if they don't exist
    if os.path.exists(embeddings_path):
        logger.info(f"Loading pre-computed embeddings from {embeddings_path}")
        return np.load(embeddings_path)
    else:
        logger.info("embeddings not available (no pre-computed file found)")
        raise ValueError("embeddings not available")
 

def split_data(df, df_units, min_stories_for_stratify = 5, test_size = 0.2, seed=42):

    atu_counts = df.groupby("atu_id", dropna=True)["story_id"].nunique()
    valid_atus_split = atu_counts[atu_counts >= min_stories_for_stratify].index #for stratified train/test split

    df_non_nan_multi_split = df[df["atu_id"].notna() & df["atu_id"].isin(valid_atus_split)].copy()
    #df_non_nan_multi_cluster = df[df["atu_id"].notna() & df["atu_id"].isin(valid_atus_cluster)].copy()
    df_non_nan_single = df[df["atu_id"].notna() & ~df["atu_id"].isin(valid_atus_split)].copy()
    df_nan = df[df["atu_id"].isna()].copy()

    unique_stories_split = df_non_nan_multi_split["story_id"].unique()
    atu_per_story = df_non_nan_multi_split.groupby("story_id")["atu_id"].first()

    if len(unique_stories_split) == 0:
        logger.warning("No stories available for stratified split. Using all as train.")
        train_stories_non_nan = np.array([])
        test_stories = np.array([])
    else:
        try:
            train_stories_non_nan, test_stories = train_test_split(
                unique_stories_split,
                test_size=test_size,
                random_state=seed,
                stratify=atu_per_story.loc[unique_stories_split]
            )
        except ValueError as e:
            logger.warning(f"Stratification failed ({e}). Falling back to random split.")
            train_stories_non_nan, test_stories = train_test_split(
                unique_stories_split,
                test_size=test_size,
                random_state=seed
            )

    nan_stories = df_nan["story_id"].unique()
    single_stories = df_non_nan_single["story_id"].unique()

    logger.info(f"Yashpeh (NaN ATU) stories: {len(nan_stories)}")
    logger.info(f"Single-story ATU stories: {len(single_stories)}")
    logger.info(f"Multi-story ATU stories available for split: {len(unique_stories_split)}")

    train_stories = np.concatenate([
        train_stories_non_nan,
        single_stories,
        nan_stories
    ])

    train_mask = df_units["story_id"].isin(train_stories)
    test_mask = df_units["story_id"].isin(test_stories)

    logger.info(f"Train stories: {len(np.unique(train_stories))}")
    logger.info(f"Test stories:  {len(np.unique(test_stories)) if len(test_stories) > 0 else 0}")

    return df_units[train_mask], df_units[test_mask]

def perform_clustering(embeddings, n_components=100, distance_threshold=0.6, cluster_linkage = "complete", seed = 42):
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    reduced = svd.fit_transform(embeddings)
    logger.info(f"Reduced embeddings shape: {reduced.shape}")
    logger.info(f"SVD explained variance ratio (sum): {svd.explained_variance_ratio_.sum():.4f}")

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage = cluster_linkage
    )
    labels = clustering.fit_predict(reduced)
    logger.info(f"Clustering completed – {len(np.unique(labels))} clusters found")
    return reduced, labels, svd

def filter_valid_clusters(train_df, motif_cluster_min_story_count = 2):
    valid_clusters = train_df.groupby("cluster_label")["story_id"].nunique()
    valid_clusters = valid_clusters[valid_clusters >= motif_cluster_min_story_count].index.tolist()
    return train_df[train_df["cluster_label"].isin(valid_clusters)].copy(), valid_clusters

def compute_centroids(reduced, train_df, valid_clusters):
    centroids = {}
    for cid in valid_clusters:
        rows = train_df[train_df["cluster_label"] == cid]["row_id"].values
        centroids[cid] = reduced[rows].mean(axis=0)
    return centroids

def assign_test_clusters(test_reduced, centroids):
    centroid_matrix = np.vstack(list(centroids.values()))
    centroid_ids = list(centroids.keys())
    sim_matrix = cosine_similarity(test_reduced, centroid_matrix)
    return [centroid_ids[i] for i in sim_matrix.argmax(axis=1)]

def aggregate_story_clusters(df_part):
    return (
        df_part.groupby("story_id")["cluster_label"]
        .apply(lambda x: list(set(x)))
        .reset_index()
    )

def compute_atu_prototypes(train_story_clusters, centroids, df):
    train_story_clusters = train_story_clusters.merge(df[["story_id", "atu_id"]], on="story_id")
    prototypes = {}
    for atu, group in train_story_clusters.groupby("atu_id"):
        cluster_ids = [c for clusters in group["cluster_label"] for c in clusters]
        vectors = [centroids[c] for c in set(cluster_ids) if c in centroids]
        if vectors:
            prototypes[atu] = np.mean(vectors, axis=0)
    return prototypes

def predict_atu(test_story_clusters, prototypes, centroids, df):
    test_story_clusters = test_story_clusters.merge(df[["story_id", "atu_id"]], on="story_id")
    predictions = []
    true_labels = []
    for _, row in test_story_clusters.iterrows():
        clusters = row["cluster_label"]
        story_vectors = [centroids[c] for c in clusters if c in centroids]
        if not story_vectors:
            continue
        story_vector = np.mean(story_vectors, axis=0)
        best_atu, best_sim = None, -1
        for atu, proto in prototypes.items():
            sim = cosine_similarity([story_vector], [proto])[0][0]
            if sim > best_sim:
                best_sim = sim
                best_atu = atu
        predictions.append(best_atu)
        true_labels.append(row["atu_id"])
    return predictions, true_labels


# Main execution
df = pd.read_pickle('full_metadata_df.pkl')  
df_units = prepare_data(df)
all_embeddings = load_or_compute_embeddings(embeddings_path='embeddings.npy') 
train_units, test_units = split_data(df, df_units, min_stories_for_stratify, test_size, seed)
train_embeddings = all_embeddings[train_units["embedding_index"].values]
test_embeddings = all_embeddings[test_units["embedding_index"].values]

train_reduced, train_cluster_labels, svd = perform_clustering(train_embeddings, dim_reduction, dist_threshold, cluster_linkage, seed)
train_df = train_units.copy()
train_df["cluster_label"] = train_cluster_labels
train_df["row_id"] = np.arange(len(train_reduced))

train_df, valid_clusters = filter_valid_clusters(train_df, motif_cluster_min_story_count)
centroids = compute_centroids(train_reduced, train_df, valid_clusters)

test_reduced = svd.transform(test_embeddings)

assigned_clusters = assign_test_clusters(test_reduced, centroids)
test_units["cluster_label"] = assigned_clusters

train_story_clusters = aggregate_story_clusters(train_df)
test_story_clusters = aggregate_story_clusters(test_units)

prototypes = compute_atu_prototypes(train_story_clusters, centroids, df)

logger.info("Start prediction...")
predictions, true_labels = predict_atu(test_story_clusters, prototypes, centroids, df)

accuracy = accuracy_score(true_labels, predictions)

logger.info(f"Cross-Story Generalization Accuracy: {round(accuracy, 4)}")
logger.info(classification_report(true_labels, predictions, zero_division=0))

logger.info(f"True labels: {true_labels}")
logger.info(f"Predicted labels: {predictions}")

logger.info("finished successfully")