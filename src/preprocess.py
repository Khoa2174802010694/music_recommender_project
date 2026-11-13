
import pandas as pd
from sklearn.preprocessing import StandardScaler


AUDIO_FEATURES = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "loudness"
]

CORE_AUDIO_FEATURES = AUDIO_FEATURES


def clean_tracks(df):
    """
    Clean music metadata and audio features.
    """

    df = df.copy()

    # Remove duplicate tracks
    df = df.drop_duplicates(
        subset="track_id"
    )

    # Convert audio features to numeric
    for col in AUDIO_FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Remove missing audio features
    df = df.dropna(
        subset=AUDIO_FEATURES
    )

    # Remove invalid tempo
    df = df[df["tempo"] > 0]

    return df.reset_index(drop=True)


def fit_scaler(df):
    """
    Fit StandardScaler on audio features.
    """

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        df[AUDIO_FEATURES]
    )

    return scaler, scaled_features


def transform_features(df, scaler):
    """
    Transform audio features using fitted scaler.
    """

    return scaler.transform(
        df[AUDIO_FEATURES]
    )


def validate_and_clean_batch(
    df,
    deduplicate=True,
    min_popularity=0.0,
):
    """
    Validate and clean one incoming Embeat data batch.

    Popularity is normalized to the recommender's 0-1 scale.
    Source datasets using a 0-100 scale are converted automatically.
    min_popularity must therefore be expressed on a 0-1 scale.
    """
    try:
        min_popularity = float(min_popularity)
    except (TypeError, ValueError):
        raise ValueError("min_popularity must be a number between 0 and 1.")

    if not 0.0 <= min_popularity <= 1.0:
        raise ValueError("min_popularity must be between 0 and 1.")

    df = df.copy()

    # 1. Remove duplicated tracks within this batch
    if deduplicate and "track_id" in df.columns:
        df = df.drop_duplicates(subset="track_id")

    # 2. Validate required audio columns
    missing_features = [
        col for col in AUDIO_FEATURES
        if col not in df.columns
    ]
    if missing_features:
        raise ValueError(f"Missing audio features: {missing_features}")

    # 3. Convert audio features to numeric
    for col in AUDIO_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Remove rows with invalid audio features
    df = df.dropna(subset=AUDIO_FEATURES)

    # 5. Remove invalid tempo
    df = df[df["tempo"] > 0]

    # 6. Normalize popularity and apply the minimum threshold
    if "popularity" in df.columns:
        df["popularity"] = pd.to_numeric(
            df["popularity"], errors="coerce"
        )
        valid_popularity = df["popularity"].dropna()

        if not valid_popularity.empty:
            max_popularity = valid_popularity.max()
            if max_popularity > 1.0:
                if max_popularity > 100.0 or valid_popularity.min() < 0:
                    raise ValueError(
                        "Popularity must use either a 0-1 or 0-100 scale."
                    )
                df["popularity"] = df["popularity"] / 100.0
            elif valid_popularity.min() < 0:
                raise ValueError("Popularity cannot be negative.")

    elif min_popularity > 0:
        raise ValueError("Missing required column: popularity")

    if min_popularity > 0:
        df = df.dropna(subset=["popularity"])
        df = df[df["popularity"] >= min_popularity]

    return df.reset_index(drop=True)
