import numpy as np
import pandas as pd
import torch
from sentence_transformers import util
import logging
import os

def compare_with_thompson(df, atu_to_tmi, reducer=None, sim_thr=0.4, text_col="thompson_motif_text", embeddings_path='tmi_raw_embeddings.npy'):
    logger = logging.getLogger(__name__)

    if not os.path.exists(embeddings_path):
        raise FileNotFoundError("Pre-computed TMI embeddings file missing")
    ref_emb_full = np.load(embeddings_path)
    logger.info(f"Loaded pre-computed TMI embeddings: {ref_emb_full.shape}")

    ref_grouped = atu_to_tmi.groupby("atu_id").agg({text_col: list}).rename(columns={text_col: "ref_texts"})


    atu_offsets = {}
    offset = 0
    for atu, row in ref_grouped.iterrows():
        texts = row['ref_texts']
        if texts:
            atu_offsets[atu] = (offset, offset + len(texts))
            offset += len(texts)

    if reducer is not None:
        ref_emb_reduced = reducer.transform(ref_emb_full)
    else:
        ref_emb_reduced = ref_emb_full

    ref_embeddings = {}
    for atu, (start, end) in atu_offsets.items():
        ref_embeddings[atu] = ref_emb_reduced[start:end]


    df_agg = (
        df.groupby("atu_id")["canonical_units"]
        .apply(lambda x: [v for subl in x for v in subl])
        .reset_index(name="aggregated_canonical")
    )

    comparison_df = df_agg.merge(ref_grouped.reset_index(), on="atu_id", how="inner")

    results = []
    for _, row in comparison_df.iterrows():
        atu = row["atu_id"]
        canon_vecs = row["aggregated_canonical"]

        if not canon_vecs:
            results.append({
                "atu_id": atu,
                "n_stories": 0,
                "n_your_motifs": 0,
                "n_ref_motifs": len(row["ref_texts"]),
                "recall": 0.0,
                "precision": 0.0
            })
            continue

        y_emb = torch.tensor(canon_vecs, dtype=torch.float32)
        r_emb = torch.tensor(ref_embeddings.get(atu, []), dtype=torch.float32)

        if r_emb.numel() == 0:
            continue

        y_emb = torch.nn.functional.normalize(y_emb, dim=1)
        r_emb = torch.nn.functional.normalize(r_emb, dim=1)

        sim = util.cos_sim(y_emb, r_emb)

        recall = (sim.max(dim=0).values >= sim_thr).float().mean().item()
        precision = (sim.max(dim=1).values >= sim_thr).float().mean().item()

        results.append({
            "atu_id": atu,
            "n_stories": len(df[df["atu_id"] == atu]),
            "n_your_motifs": len(canon_vecs),
            "n_ref_motifs": len(row["ref_texts"]),
            "recall": round(recall, 3),
            "precision": round(precision, 3)
        })

    results_df = pd.DataFrame(results)
    mean_recall = results_df['recall'].mean() if not results_df.empty else 0.0
    logger.info(f"Comparison completed. Mean recall: {mean_recall:.3f}")

    return mean_recall