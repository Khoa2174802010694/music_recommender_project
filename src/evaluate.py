"""Embeat Task 2.3 evaluation: audio, hybrid, MMR, random and popularity baselines."""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.recommender import EmbeatRecommender

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = PROJECT_ROOT / 'artifacts' / 'embeat_recommender.joblib'
OUTPUT_DIR = PROJECT_ROOT / 'reports'
TOP_N = 10
N_QUERIES = 30
ALPHA = 0.7
MMR_LAMBDA = 0.7
RANDOM_STATE = 42
MODEL_CONFIGS = [
    {"name": "Audio-only", "mode": "audio"},
    {"name": "Hybrid 70/30", "mode": "hybrid"},
    {"name": "Hybrid + MMR", "mode": "mmr"},
    {"name": "Random Baseline", "mode": "random"},
    {"name": "Popularity Baseline", "mode": "popularity"},
]


def load_model():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(f'Model artifact not found: {ARTIFACT_PATH}')
    model = EmbeatRecommender.load_artifact(str(ARTIFACT_PATH))
    print('Loaded:', len(model.df), 'tracks; audio:', model.feature_matrix.shape,
          '; genres:', model.genre_matrix.shape)
    return model


def extract_genres(result):
    return {genre.strip().lower()
            for value in result['artist_genres'].fillna('')
            for genre in str(value).split(',') if genre.strip()}


def eligible_indices(model, query_id):
    """Exclude the query and same artist, preferring ID with name fallback."""
    query_idx = model.id_to_idx[query_id]
    mask = np.ones(len(model.df), dtype=bool)
    mask[query_idx] = False

    query_row = model.df.iloc[query_idx]
    matched_artist_metadata = False

    if "artist_id" in model.df.columns:
        query_artist_id = query_row["artist_id"]
        if pd.notna(query_artist_id):
            mask &= ~(
                model.df["artist_id"].notna()
                & model.df["artist_id"].eq(query_artist_id)
            ).to_numpy()
            matched_artist_metadata = True

    # Name is a fallback when ID is absent for the query, and also catches
    # rows whose ID is missing but whose artist name matches the query.
    if "artist_name" in model.df.columns:
        query_artist_name = query_row["artist_name"]
        if pd.notna(query_artist_name):
            mask &= ~(
                model.df["artist_name"].notna()
                & model.df["artist_name"].astype(str).str.casefold().eq(
                    str(query_artist_name).casefold()
                )
            ).to_numpy()
            matched_artist_metadata = True

    if not matched_artist_metadata:
        # No artist metadata for this query: cannot infer same-artist rows.
        return np.flatnonzero(mask)

    return np.flatnonzero(mask)


def baseline_result(model, query_id, method, rng):
    candidates = eligible_indices(model, query_id)
    if method == 'Random Baseline':
        chosen = rng.choice(candidates, size=min(TOP_N, len(candidates)), replace=False)
    else:
        popularity = pd.to_numeric(model.df.iloc[candidates]['popularity'], errors='coerce').fillna(-np.inf).to_numpy()
        # Stable ranking, with index order as tie-breaker.
        chosen = candidates[np.argsort(-popularity, kind='stable')[:TOP_N]]
    return model.df.iloc[chosen][['track_id', 'track_name', 'artist_name', 'artist_genres', 'popularity']].reset_index(drop=True).copy()


def measure_list(model, query_id, result):
    indices = [model.id_to_idx[tid] for tid in result['track_id']]
    query_idx = model.id_to_idx[query_id]
    if not indices:
        return {'mean_audio_cosine': np.nan, 'mean_acoustic_distance': np.nan,
                'mean_hybrid_cosine': np.nan, 'audio_diversity': np.nan,
                'hybrid_diversity': np.nan}
    query_audio = cosine_similarity(model.feature_matrix[query_idx:query_idx+1],
                                    model.feature_matrix[indices]).ravel()
    query_genre = cosine_similarity(model.genre_matrix[query_idx], model.genre_matrix[indices]).ravel()
    audio_pair = cosine_similarity(model.feature_matrix[indices])
    genre_pair = cosine_similarity(model.genre_matrix[indices])
    if len(indices) > 1:
        upper = np.triu_indices(len(indices), k=1)
        audio_div = float(1 - audio_pair[upper].mean())
        hybrid_div = float(1 - (ALPHA * audio_pair + (1 - ALPHA) * genre_pair)[upper].mean())
    else:
        audio_div = hybrid_div = 0.0
    return {'mean_audio_cosine': float(query_audio.mean()),
            'mean_acoustic_distance': float((1 - query_audio).mean()),
            'mean_hybrid_cosine': float((ALPHA * query_audio + (1 - ALPHA) * query_genre).mean()),
            'audio_diversity': audio_div, 'hybrid_diversity': hybrid_div}



def evaluate_model(model, query_ids, model_name, mode=None):
    # Giữ tương thích với cách gọi cũ.
    # Nếu chưa truyền mode, suy ra từ model_name.
    if mode is None:
        legacy_modes = {
            "Audio-only": "audio",
            "Hybrid 70/30": "hybrid",
            "Hybrid + MMR": "mmr",
            "Random Baseline": "random",
            "Popularity Baseline": "popularity",
        }

        if model_name not in legacy_modes:
            raise ValueError(f"Unknown model: {model_name}")

        mode = legacy_modes[model_name]

    valid_modes = {"audio", "hybrid", "mmr", "random", "popularity"}
    if mode not in valid_modes:
        raise ValueError(f"Unknown evaluation mode: {mode}")

    records, per_query, latencies = [], [], []
    rng = np.random.default_rng(RANDOM_STATE + 100)

    for query_id in query_ids:
        start = time.perf_counter()

        if mode == "audio":
            result = model.recommend_by_track(
                query_id, top_n=TOP_N, exclude_same_artist=True
            )

        elif mode == "hybrid":
            result = model.recommend_hybrid(
                query_id, top_n=TOP_N,
                alpha=ALPHA, exclude_same_artist=True
            )

        elif mode == "mmr":
            result = model.recommend_hybrid_mmr(
                query_id,
                top_n=TOP_N,
                alpha=ALPHA,
                mmr_lambda=MMR_LAMBDA,
                exclude_same_artist=True
            )

        elif mode in ("random", "popularity"):
            baseline_name = (
                "Random Baseline" if mode == "random"
                else "Popularity Baseline"
            )
            result = baseline_result(
                model, query_id, baseline_name, rng
            )

        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

        if result.empty:
            raise ValueError(f"No recommendations: {mode}, query {query_id}")

        if query_id in set(result["track_id"]):
            raise AssertionError(f"Query track returned: {query_id}")

        measures = measure_list(model, query_id, result)

        per_query.append({
            "query_id": query_id,
            "unique_artists": result["artist_name"].nunique(),
            "unique_genres": len(extract_genres(result)),
            **measures
        })

        for rank, (_, row) in enumerate(result.iterrows(), start=1):
            record = {
                "model": model_name,
                "query_id": query_id,
                "rank": rank,
                "track_id": row["track_id"],
                "track_name": row["track_name"],
                "artist_name": row["artist_name"],
                "artist_genres": row["artist_genres"],
                "popularity": (
                    float(row["popularity"])
                    if pd.notna(row["popularity"]) else None
                ),
                **measures
            }

            for column in [
                "similarity", "audio_similarity",
                "genre_similarity", "hybrid_similarity", "mmr_score"
            ]:
                if column in row.index and pd.notna(row[column]):
                    record[column] = float(row[column])

            records.append(record)

    details = pd.DataFrame(records)
    query_metrics = pd.DataFrame(per_query)

    summary = {
        "model": model_name,
        "mode": mode,
        "total_queries": len(query_ids),
        "total_recommendations": len(details),
        "unique_recommended_tracks": int(details["track_id"].nunique()),
        "avg_unique_artists": float(query_metrics["unique_artists"].mean()),
        "avg_unique_genres": float(query_metrics["unique_genres"].mean()),
        "avg_popularity": float(details["popularity"].mean()),
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies))
    }

    for metric in [
        "mean_audio_cosine", "mean_acoustic_distance",
        "mean_hybrid_cosine", "audio_diversity", "hybrid_diversity"
    ]:
        summary[metric] = float(query_metrics[metric].mean())

    summary["avg_similarity"] = summary["mean_audio_cosine"]
    summary["intra_list_diversity"] = summary["audio_diversity"]

    if "mmr_score" in details:
        summary["avg_mmr_score"] = float(details["mmr_score"].mean())

    return summary, details


def save_dashboard(summary_df):
    """
    Generate evaluation charts and save them
    under reports/figures/.
    """

    figure_dir = OUTPUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    chart_metrics = [
        (
            "mean_audio_cosine",
            "Mean Audio Cosine Similarity",
            "Similarity"
        ),
        (
            "mean_acoustic_distance",
            "Mean Acoustic Distance",
            "Distance"
        ),
        (
            "audio_diversity",
            "Audio Intra-list Diversity",
            "Diversity"
        ),
        (
            "avg_unique_artists",
            "Average Unique Artists",
            "Number of Artists"
        ),
        (
            "avg_unique_genres",
            "Average Unique Genres",
            "Number of Genres"
        ),
        (
            "coverage_pct",
            "Catalog Coverage",
            "Coverage (%)"
        ),
        (
            "mean_latency_ms",
            "Mean Recommendation Latency",
            "Latency (ms)"
        )
    ]

    fig, axes = plt.subplots(
        4, 2,
        figsize=(17, 19)
    )

    for ax, (metric, title, ylabel) in zip(
        axes.flat,
        chart_metrics
    ):
        bars = ax.bar(
            summary_df["model"],
            summary_df[metric]
        )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(
            axis="x",
            rotation=25
        )
        ax.grid(
            axis="y",
            alpha=0.25
        )

        for bar in bars:
            value = bar.get_height()

            ax.annotate(
                f"{value:.3f}",
                (
                    bar.get_x() + bar.get_width() / 2,
                    value
                ),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8
            )

    axes.flat[-1].axis("off")

    fig.suptitle(
        "Embeat Recommendation Model Evaluation",
        fontsize=17
    )

    fig.tight_layout()

    output_path = (
        figure_dir / "evaluation_dashboard.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Dashboard saved:", output_path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model()
    if model.genre_matrix is None:
        raise ValueError('Genre matrix is required for this evaluation')
    has_artist_metadata = (
        ("artist_id" in model.df.columns and model.df["artist_id"].notna().any())
        or ("artist_name" in model.df.columns and model.df["artist_name"].notna().any())
    )
    if not has_artist_metadata:
        raise ValueError(
            "Main artifact has neither usable artist_id nor artist_name metadata."
        )
    rng = np.random.default_rng(RANDOM_STATE)
    query_ids = rng.choice(model.df['track_id'].to_numpy(),
                           size=min(N_QUERIES, len(model.df)), replace=False).tolist()
    summaries, details_list = [], []
    for config in MODEL_CONFIGS:
        name = config["name"]
        mode = config["mode"]

        print(f"Evaluating: {name} (mode={mode})")

        summary, details = evaluate_model(
            model=model,
            query_ids=query_ids,
            model_name=name,
            mode=mode
        )

        summary["catalog_size"] = len(model.df)
        summary["coverage_pct"] = (
            100 * summary["unique_recommended_tracks"] / len(model.df)
        )

        summaries.append(summary)
        details_list.append(details)
    summary_df = pd.DataFrame(summaries)
    details_df = pd.concat(details_list, ignore_index=True)
    metrics = {
        'evaluation_config': {
            'n_queries': len(query_ids),
            'top_n': TOP_N,
            'alpha': ALPHA,
            'mmr_lambda': MMR_LAMBDA,
            'candidate_pool': None,
            'candidate_pool_policy': '3 * top_n when not explicitly provided',
            'random_state': RANDOM_STATE
        },
               'catalog_size': len(model.df), 'query_track_ids': query_ids, 'results': summaries,
               'evaluation_notes': [
                   'No ground-truth relevance labels; Precision@K and Recall@K are not computed.',
                   'All five methods use the same query tracks and exclude query and same artist.',
                   'Mean acoustic distance = mean(1 - audio cosine(query, recommendation)); not Euclidean distance.',
                   'avg_similarity = mean_audio_cosine for comparable cross-method evaluation.',
                   'intra_list_diversity = audio_diversity for comparable cross-method evaluation.',
                   'mean_hybrid_cosine and hybrid_diversity are supplementary metrics.',
                   'Latency measures recommendation generation only, not evaluation metric calculation.',
                   'Random baseline is deterministic given RANDOM_STATE; timing varies by run.'
               ]}
    with (OUTPUT_DIR / 'metrics.json').open('w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    summary_df.to_csv(OUTPUT_DIR / 'evaluation_summary.csv', index=False)
    details_df.to_csv(OUTPUT_DIR / 'evaluation_details.csv', index=False)
    save_dashboard(summary_df)
    print(summary_df[['model', 'mean_audio_cosine', 'mean_acoustic_distance',
                      'audio_diversity', 'coverage_pct', 'mean_latency_ms']].to_string(index=False))
    print('EVALUATION COMPLETED SUCCESSFULLY')


if __name__ == '__main__':
    main()
