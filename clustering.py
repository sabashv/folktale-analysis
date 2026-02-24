import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import logging
from datetime import datetime
import os

##clustering params:
dim_reduction = 200
dist_threshold = 0.65
cluster_linkage = "complete"
seed = 42

log_dir = "./clustering_logs"
os.makedirs(log_dir, exist_ok=True)

log_filename = os.path.join(
    log_dir,
    f"clustering_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)


logger = logging.getLogger("ClusteringRun")
logger.setLevel(logging.INFO)


formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)



logger.info("Starting clustering")
logger.info(f"Log file: {log_filename}")
logger.info(f"Clustering Parameters:\n - Reduced dimnetionality: {dim_reduction} \n - Distance Threshold: {dist_threshold} \n - Clustering linkage: {cluster_linkage} \n - seed : {seed}")

logger.info("Loading embeddings...")
embeddings = np.load('embeddings.npy')
logger.info(f"Embeddings loaded – shape: {embeddings.shape}")
logger.info(f"Memory usage (approx): {embeddings.nbytes / 1e9:.2f} GB")


logger.info("Starting TruncatedSVD for dimensionality reduction...")
svd = TruncatedSVD(n_components = dim_reduction, random_state = seed)
reduced = svd.fit_transform(embeddings)
logger.info(f"Reduced embeddings shape: {reduced.shape}")
logger.info(f"SVD explained variance ratio (sum): {svd.explained_variance_ratio_.sum():.4f}")

# Clustering
logger.info("Starting AgglomerativeClustering...")
clustering = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=dist_threshold,
    metric="cosine",
    linkage=cluster_linkage,
)

labels = clustering.fit_predict(reduced)
logger.info(f"Clustering completed – {len(np.unique(labels))} clusters found")
logger.info(f"Labels shape: {labels.shape}")
logger.info(f"Memory usage after clustering (approx): {reduced.nbytes / 1e9:.2f} GB")

# Evaluation
logger.info("Calculating silhouette score...")
sil_score = silhouette_score(reduced, labels)
logger.info(f"Silhouette score: {sil_score:.4f}")

logger.info(f"Min cluster size: {np.bincount(labels).min()}")
logger.info(f"Max cluster size: {np.bincount(labels).max()}")
logger.info(f"Number of noise/singleton points (label -1 or small): {(labels == -1).sum() if -1 in labels else 0}")


logger.info("Saving cluster labels...")
np.save('cluster_labels.npy', labels)

logger.info("Clustering finished.")
logger.info(f"All output saved. Log file: {log_filename}")