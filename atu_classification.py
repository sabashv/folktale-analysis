import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report
import logging

def run_atu_classification(df, all_embeddings, n_components, distance_threshold, linkage, min_stories_per_motif, test_size, min_stories_for_stratify, seed):
    logger = logging.getLogger(__name__)
    np.random.seed(seed)

    df_units = df.explode('normalized_units').reset_index(drop=True)
    df_units = df_units.rename(columns={'normalized_units': 'unit'})
    df_units["embedding_index"] = np.arange(len(df_units))

    atu_counts = df.groupby("atu_id", dropna=True)["story_id"].nunique()
    valid_atus_split = atu_counts[atu_counts >= min_stories_for_stratify].index

    df_multi = df[df["atu_id"].notna() & df["atu_id"].isin(valid_atus_split)].copy()
    unique_stories = df_multi["story_id"].unique()
    atu_per_story = df_multi.groupby("story_id")["atu_id"].first()

    try:
        train_stories_non_nan, test_stories = train_test_split(
            unique_stories, test_size=test_size, random_state=seed, stratify=atu_per_story.loc[unique_stories]
        )
    except ValueError:
        logger.warning("Stratification not possible, random split")
        train_stories_non_nan, test_stories = train_test_split(
            unique_stories, test_size=test_size, random_state=seed
        )

    train_stories = np.concatenate([
        train_stories_non_nan,
        df[df["atu_id"].isna()]["story_id"].unique(),
        df[df["atu_id"].notna() & ~df["atu_id"].isin(valid_atus_split)]["story_id"].unique()
    ])

    train_mask = df_units["story_id"].isin(train_stories)
    test_mask = df_units["story_id"].isin(test_stories)

    train_emb = all_embeddings[df_units[train_mask]["embedding_index"]]
    test_emb = all_embeddings[df_units[test_mask]["embedding_index"]]

    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    train_reduced = svd.fit_transform(train_emb)

    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold, metric="cosine", linkage=linkage
    )
    train_labels = clustering.fit_predict(train_reduced)

    train_df = df_units[train_mask].copy()
    train_df["cluster_label"] = train_labels
    train_df["row_id"] = np.arange(len(train_reduced))

    valid_clusters = train_df.groupby("cluster_label")["story_id"].nunique()
    valid_clusters = valid_clusters[valid_clusters >= min_stories_per_motif].index.tolist()

    centroids = {}
    for cid in valid_clusters:
        rows = train_df[train_df["cluster_label"] == cid]["row_id"].values
        centroids[cid] = train_reduced[rows].mean(axis=0)

    test_reduced = svd.transform(test_emb)
    sim_matrix = cosine_similarity(test_reduced, np.vstack(list(centroids.values())) if centroids else np.empty((0, n_components)))
    assigned = [list(centroids.keys())[i] if centroids else -1 for i in sim_matrix.argmax(1)] if centroids else [-1] * len(test_reduced)

    test_df = df_units[test_mask].copy()
    test_df["cluster_label"] = assigned

    train_story_cl = train_df.groupby("story_id")["cluster_label"].apply(set).apply(list).reset_index()
    test_story_cl = test_df.groupby("story_id")["cluster_label"].apply(set).apply(list).reset_index()

    train_story_cl = train_story_cl.merge(df[["story_id", "atu_id"]], on="story_id")
    prototypes = {}
    for atu, g in train_story_cl.groupby("atu_id"):
        cids = set(c for cl in g["cluster_label"] for c in cl)
        vecs = [centroids[c] for c in cids if c in centroids]
        if vecs:
            prototypes[atu] = np.mean(vecs, axis=0)

    preds, trues = [], []
    test_story_cl = test_story_cl.merge(df[["story_id", "atu_id"]], on="story_id")
    for _, row in test_story_cl.iterrows():
        vecs = [centroids[c] for c in row["cluster_label"] if c in centroids]
        if not vecs: continue
        vec = np.mean(vecs, axis=0)
        best_atu = max(prototypes, key=lambda atu: cosine_similarity([vec], [prototypes[atu]])[0][0] if prototypes else None)
        if best_atu is None: continue
        preds.append(best_atu)
        trues.append(row["atu_id"])

    accuracy = accuracy_score(trues, preds) if trues else 0.0
    logger.info(f"ATU accuracy: {accuracy:.4f}")
    logger.info(classification_report(trues, preds, zero_division=0))
    #logger.info(f"True labels: {trues}")
    #logger.info(f"Predicted labels: {preds}")

    return accuracy