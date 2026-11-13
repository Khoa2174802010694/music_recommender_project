
import argparse
import json

from pathlib import Path

import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

from scipy.sparse import csr_matrix

from src.recommender import EmbeatRecommender


DEFAULT_MAX_TRACKS = 300_000


def select_compact_catalog(df, max_tracks=DEFAULT_MAX_TRACKS):
    """Keep the most popular tracks for the deployable compact artifact."""
    if max_tracks is None:
        return df.copy()

    if not isinstance(max_tracks, int) or max_tracks <= 0:
        raise ValueError("max_tracks must be a positive integer.")

    compact_df = df.copy()

    if "popularity" in compact_df.columns:
        compact_df["_popularity_sort"] = pd.to_numeric(
            compact_df["popularity"], errors="coerce"
        ).fillna(-1)
        compact_df = compact_df.sort_values(
            "_popularity_sort", ascending=False, kind="mergesort"
        ).drop(columns="_popularity_sort")
    else:
        print(
            "WARNING: popularity column is missing; "
            "keeping the first tracks instead of ranking by popularity."
        )

    return compact_df.head(max_tracks).reset_index(drop=True)


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "balanced_eda.csv"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "embeat_recommender.joblib"
)


# ==========================================
# 2. GENRE PREPROCESSING
# ==========================================

def build_genre_matrix(df):
    """
    Convert artist genres into sparse multi-hot encoding.
    """

    genre_list = (
        df["artist_genres"]
        .fillna("")
        .apply(
            lambda x: [
                genre.strip().lower()
                for genre in str(x).split(",")
                if genre.strip()
            ]
        )
    )

    mlb = MultiLabelBinarizer(
        sparse_output=True
    )

    genre_matrix = mlb.fit_transform(
        genre_list
    )

    genre_matrix = csr_matrix(
        genre_matrix
    )

    return genre_matrix, mlb


# ==========================================
# 3. TRAINING PIPELINE
# ==========================================

def train_model(input_path, output_path, max_tracks=DEFAULT_MAX_TRACKS):

    input_path = Path(input_path)
    output_path = Path(output_path)

    print("=" * 55)
    print("EMBEAT MODEL TRAINING")
    print("=" * 55)

    # 1. Load dataset
    print("\n[1/5] Loading dataset...")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    print("Original shape:", df.shape)

    # Keep a compact, popularity-ranked catalog for deployment.
    df = select_compact_catalog(df, max_tracks=max_tracks)
    print("Compact catalog shape:", df.shape)

    # 2. Train audio model
    print("\n[2/5] Training audio model...")

    model = EmbeatRecommender()

    model.fit(df)

    print("Cleaned tracks:", len(model.df))
    print(
        "Audio matrix:",
        model.feature_matrix.shape
    )

    # 3. Validate genre matrix created by model.fit()
    print("\n[3/5] Checking genre matrix...")

    if model.genre_matrix is None:
        raise ValueError(
            "Cannot train recommendation model: "
            "artist_genres is missing or unavailable."
        )

    print("Genre matrix:", model.genre_matrix.shape)
    print("Unique genres:", model.genre_matrix.shape[1])

    # 4. Save model artifact
    print("\n[4/5] Saving model artifact...")

    model.save_artifact(
        output_path
    )


    # 5. Generate training metadata
    print("\n[5/5] Generating training metadata...")

    metadata = {
        "model_name": "EmbeatRecommender",
        "training_method": "csv",
        "compact_max_tracks": max_tracks,
        "total_tracks": len(model.df),
        "audio_features": model.feature_matrix.shape[1],
        "total_genres": model.genre_matrix.shape[1],
        "audio_matrix_shape": list(model.feature_matrix.shape),
        "genre_matrix_shape": list(model.genre_matrix.shape),
        "artifact_path": output_path.name,
        "artifact_filename": output_path.name,
    }

    # Mỗi artifact có metadata riêng
    metadata_path = output_path.with_name(
        f"{output_path.stem}_metadata.json"
    )

    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("Metadata saved:", metadata_path)

    return model



# ==========================================
# 4. BATCH-BASED STREAMING TRAINING
# ==========================================

def train_model_from_stream(
    output_path=DEFAULT_OUTPUT,
    hf_token=None,
    sample_ratio=0.5,
    max_row_groups_per_file=None,
    max_total_tracks=None,
    min_popularity=0.0,
    fallback_to_local=True,
    max_tracks=DEFAULT_MAX_TRACKS,
):
    """
    Train the recommendation model from streamed batches.

    Incoming batches are accumulated first.
    The model is trained once after streaming completes.

    This is batch accumulation, not online learning.
    """

    from src.stream_embeat import stream_and_train_50pct

    output_path = Path(output_path)

    try:
        min_popularity = float(min_popularity)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_popularity must be between 0 and 1.") from exc
    if not 0.0 <= min_popularity <= 1.0:
        raise ValueError("min_popularity must be between 0 and 1.")

    print("=" * 55)
    print("EMBEAT STREAMING BATCH TRAINING")
    print("=" * 55)

    # 1. Initialize model
    model = EmbeatRecommender()

    # 2. Receive batches through callback
    print("\n[1/5] Receiving data batches...")

    stream_success = stream_and_train_50pct(
        hf_token=hf_token,
        callback_fn=model.fit_incremental,
        sample_ratio=sample_ratio,
        max_row_groups_per_file=max_row_groups_per_file,
        max_total_tracks=max_total_tracks,
        min_popularity=min_popularity,
        fallback_to_local=fallback_to_local,
    )

    if not stream_success:
        raise RuntimeError(
            "Streaming failed or returned no usable data. "
            "Model training was not finalized."
        )

    # 3. Finalize incremental training and apply compact selection once.
    print("\n[2/5] Finalizing incremental training...")

    model.finalize_incremental_training(max_tracks=max_tracks)
    print("Compact catalog tracks:", len(model.df))

    # 4. Validate genre matrix created during finalization
    print("\n[3/5] Checking genre matrix...")

    if model.genre_matrix is None:
        raise ValueError(
            "Genre matrix was not created during "
            "finalize_incremental_training()."
        )

    print("Genre matrix:", model.genre_matrix.shape)
    print("Unique genres:", model.genre_matrix.shape[1])

    # 5. Save model artifact
    print("\n[4/5] Saving model artifact...")

    model.save_artifact(output_path)

    # 6. Generate training metadata
    print("\n[5/5] Generating training metadata...")

    metadata = {
        "model_name": "EmbeatRecommender",
        "training_method": "batch_accumulation",
        "compact_max_tracks": max_tracks,
        "total_tracks": len(model.df),
        "audio_features": model.feature_matrix.shape[1],
        "total_genres": model.genre_matrix.shape[1],
        "audio_matrix_shape": list(model.feature_matrix.shape),
        "genre_matrix_shape": list(model.genre_matrix.shape),
        "artifact_path": output_path.name,
        "artifact_filename": output_path.name,
        "sample_ratio": sample_ratio,
        "max_total_tracks": max_total_tracks,
    }

    # Metadata riêng cho từng artifact
    metadata_path = output_path.with_name(
        f"{output_path.stem}_metadata.json"
    )

    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("Metadata saved:", metadata_path)

    return model


# ==========================================
# 5. COMMAND LINE ENTRY POINT
# ==========================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train Embeat Music Recommendation Model"
    )

    # Chọn phương thức huấn luyện
    parser.add_argument(
        "--mode",
        type=str,
        choices=["csv", "stream"],
        default="csv",
        help="Training mode: csv or stream"
    )

    # Đường dẫn dữ liệu và Artifact
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to input CSV dataset (CSV mode)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path to save model artifact"
    )

    parser.add_argument(
        "--max-tracks",
        type=int,
        default=DEFAULT_MAX_TRACKS,
        help="Maximum tracks in the compact deployment artifact"
    )

    # Cấu hình Streaming
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.5,
        help="Sampling ratio for streaming (1.0 = all data)"
    )

    parser.add_argument(
        "--max-row-groups-per-file",
        type=int,
        default=None,
        help="Maximum row groups to process per Parquet file"
    )

    parser.add_argument(
        "--max-total-tracks",
        type=int,
        default=None,
        help="Maximum number of tracks to receive"
    )

    parser.add_argument(
        "--min-popularity",
        type=float,
        default=0.0,
        help="Minimum popularity threshold"
    )

    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable fallback to local sample when streaming fails"
    )

    args = parser.parse_args()

    # Kiểm tra sampling ratio
    if not 0 < args.sample_ratio <= 1:
        parser.error("--sample-ratio must be greater than 0 and at most 1")

    if args.max_total_tracks is not None and args.max_total_tracks <= 0:
        parser.error("--max-total-tracks must be greater than 0")

    if args.max_tracks <= 0:
        parser.error("--max-tracks must be greater than 0")

    if not 0.0 <= args.min_popularity <= 1.0:
        parser.error("--min-popularity must be between 0 and 1")

    if (
        args.max_row_groups_per_file is not None
        and args.max_row_groups_per_file <= 0
    ):
        parser.error("--max-row-groups-per-file must be greater than 0")

    print("=" * 55)
    print("EMBEAT TRAINING CLI")
    print("=" * 55)
    print("Training mode:", args.mode)
    print("Output artifact:", args.output)

    if args.mode == "csv":

        print("\nStarting CSV training...")

        train_model(
            input_path=args.input,
            output_path=args.output,
            max_tracks=args.max_tracks
        )

    elif args.mode == "stream":

        print("\nStarting Streaming training...")

        train_model_from_stream(
            output_path=args.output,
            sample_ratio=args.sample_ratio,
            max_row_groups_per_file=args.max_row_groups_per_file,
            max_total_tracks=args.max_total_tracks,
            min_popularity=args.min_popularity,
            fallback_to_local=not args.no_fallback,
            max_tracks=args.max_tracks
        )

    print("\nCLI TRAINING COMPLETED")
