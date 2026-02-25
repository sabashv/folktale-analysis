## Dataset: 3430 folktales, 1503 of them are annotated with ATU type (180 different ATUs)

## Clustering evaluation:
1. External Verification:
- **ATU Prediction**  
  We hold out 20% of the labeled stories (stratified by ATU type, only ATUs that have ≥5 stories) as test data. We perform clustering only on the training data, filter motif clusters (≥2 stories per cluster), compute ATU prototypes as the mean of motif centroids per ATU in train, then assign test units to the closest train motif centroid. For each story in test set we aggregate per-story motif vectors, and predict ATU by finding the nearest prototype. 

- **TMI Comparison**  
  The ATU_tom_TMI dataset 895 ATU types to their associated thompson motifs. On average each ATU type is associated with 2.5 thompson motifs. From 180 different atu types in our folktale dataset, 160 of them appear in ATU_tom_TMI dataset.
  We use our full dataset(3430 stories) to cluster units and find motifs, then for each ATU type available We compare the discovered motif centroids against gold-standard Thompson Motif Index (TMI) descriptions. We report recall and for similarity threshold we use 1-distance threshold used in the clustering algorithm.

- **TMI Comparison**  
  The `ATU_to_TMI` dataset maps 895 ATU types to their associated Thompson motifs, with an average of 2.5 motifs per ATU type.  
  Of the 180 distinct ATU types present in our folktale dataset , 160 of them appear in `ATU_to_TMI`.  

  We perform clustering on the full dataset (3430 stories) to discover motif centroids.  
  For each ATU type present in our data, we compare the aggregated motif vectors (motif vector = mean of all units that belong to one cluster) against the corresponding Thompson Motif text embeddings .  
  We report recall and precision at a similarity threshold of `1 - distance_threshold` (where distance_threshold is the clustering threshold).

2. Clustering Quality:
- **silhouette_score** : 
- **entropy inside clusters** : the average variance of points around their cluster mean, normalized by the global variance (the lower the better)
- **distance between cluster** : average distance between centroids of different clusters (The higher the better)
- **compression_ratio** : number of units over number of motif clusters
- **num_stories_without_motif**
- **num_all_clusters** 
- **num_motif_clusters**

