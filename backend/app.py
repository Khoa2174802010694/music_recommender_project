"""FastAPI Backend Server for Embeat Music Recommender System."""

from contextlib import asynccontextmanager
import logging
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd

from backend.schemas import (
    AudioFeatures,
    HealthResponse,
    RecommendationResponse,
    SearchResponse,
    TasteProfileRequest,
    TrackItem,
)
from src.preprocess import AUDIO_FEATURES
from src.recommender import EmbeatRecommender

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "embeat_recommender.joblib"
FALLBACK_CSV_PATHS = [
    PROJECT_ROOT / "data" / "processed" / "balanced_eda.csv",
    PROJECT_ROOT / "data" / "processed" / "embeat_sample_tracks.csv",
    PROJECT_ROOT / "data" / "processed" / "sample_tracks.csv",
]

AUDIO_FEATURE_FIELDS = (
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "loudness",
)

SCORE_COLUMNS = (
    "hybrid_similarity",
    "similarity",
    "mood_similarity",
)

# Global model state initialized once during startup lifespan
model: Optional[EmbeatRecommender] = None
model_mode: str = "unloaded"
loaded_artifact_path: Optional[str] = None
popularity_is_unit_scale: bool = True


def _detect_unit_popularity_scale(df: pd.DataFrame) -> bool:
    """Return True if dataset popularity column uses 0.0..1.0 unit scale instead of 0..100."""
    if df is None or "popularity" not in df.columns:
        return True
    pop_series = pd.to_numeric(df["popularity"], errors="coerce").dropna()
    if pop_series.empty:
        return True
    return float(pop_series.max()) <= 1.0


def _to_relative_path(path: Path) -> str:
    """Return path relative to PROJECT_ROOT using POSIX slashes, or basename if outside."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _load_model_once() -> EmbeatRecommender:
    """Load the EmbeatRecommender artifact or fallback to CSV dataset."""
    global model, model_mode, loaded_artifact_path, popularity_is_unit_scale

    if model is not None and model.df is not None:
        return model

    if ARTIFACT_PATH.exists():
        try:
            logger.info("Loading model artifact from %s", ARTIFACT_PATH)
            model = EmbeatRecommender.load_artifact(ARTIFACT_PATH)
            model_mode = "artifact_joblib"
            loaded_artifact_path = _to_relative_path(ARTIFACT_PATH)
            popularity_is_unit_scale = _detect_unit_popularity_scale(model.df)
            return model
        except Exception as exc:
            logger.warning(
                "Failed to load artifact %s (%s). Falling back to CSV...",
                ARTIFACT_PATH,
                exc,
            )

    for csv_path in FALLBACK_CSV_PATHS:
        if csv_path.exists():
            logger.info("Fitting fallback model from CSV %s", csv_path)
            df = pd.read_csv(csv_path)
            recommender = EmbeatRecommender()
            recommender.fit(df)
            model = recommender
            model_mode = "fallback_csv"
            loaded_artifact_path = _to_relative_path(csv_path)
            popularity_is_unit_scale = _detect_unit_popularity_scale(model.df)
            return model

    raise RuntimeError(
        "Could not load model: neither artifacts/embeat_recommender.joblib "
        "nor fallback CSV files in data/processed/ were found."
    )


def _get_model() -> EmbeatRecommender:
    """Retrieve the active recommender model, ensuring it is loaded."""
    if model is None or model.df is None:
        return _load_model_once()
    return model


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """FastAPI lifespan handler to load the recommendation model once at startup."""
    loaded = _load_model_once()
    app_instance.state.model = loaded
    app_instance.state.mode = model_mode
    app_instance.state.artifact_path = loaded_artifact_path
    yield


app = FastAPI(
    title="Embeat Music Recommender API",
    description="Content-Based & Hybrid Music Recommendation REST API powered by Embeat 45M Spotify Tracks.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _clean_scalar(value: Any) -> Any:
    """Convert NaN, np.nan, or +/-inf values to None and normalize numpy scalars."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (str, bytes, dict, list, tuple, set)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _normalize_popularity(raw_pop: Any) -> Optional[int]:
    """Convert popularity to integer (handling both 0..1 float scale and 0..100 int scale)."""
    cleaned_pop = _clean_scalar(raw_pop)
    if cleaned_pop is None:
        return None
    try:
        if isinstance(raw_pop, (int, np.integer)) and not isinstance(raw_pop, bool):
            return int(raw_pop)
        pop_float = float(cleaned_pop)
        if math.isnan(pop_float) or math.isinf(pop_float):
            return None
        if 0.0 < pop_float < 1.0 or (
            popularity_is_unit_scale
            and 0.0 <= pop_float <= 1.0
            and isinstance(raw_pop, (float, np.floating))
        ):
            return int(round(pop_float * 100))
        return int(round(pop_float))
    except (TypeError, ValueError):
        return None


def _row_to_track_item(row: dict, rank: int | None = None) -> TrackItem:
    """Convert a track dictionary row into a validated TrackItem schema."""
    cleaned: Dict[str, Any] = {k: _clean_scalar(v) for k, v in row.items()}

    track_id = str(cleaned.get("track_id") or "")
    track_name = str(cleaned.get("track_name") or "Unknown Track")

    artist_name = cleaned.get("artist_name")
    if artist_name is not None:
        artist_name = str(artist_name)

    artist_genres = cleaned.get("artist_genres")
    if artist_genres is not None:
        artist_genres = str(artist_genres)

    popularity = _normalize_popularity(row.get("popularity"))

    # Extract similarity score from priority score columns, rounded to 4 decimal places
    similarity_score: Optional[float] = None
    for col in SCORE_COLUMNS:
        if col in cleaned and cleaned[col] is not None:
            similarity_score = round(float(cleaned[col]), 4)
            break
    if similarity_score is None and cleaned.get("similarity_score") is not None:
        similarity_score = round(float(cleaned["similarity_score"]), 4)

    mmr_score: Optional[float] = None
    if cleaned.get("mmr_score") is not None:
        mmr_score = round(float(cleaned["mmr_score"]), 4)

    # Group the 9 acoustic features if present in row (rounded to 4 decimal places as standard float)
    audio_features: Optional[AudioFeatures] = None
    if all(feat in cleaned and cleaned[feat] is not None for feat in AUDIO_FEATURE_FIELDS):
        audio_features = AudioFeatures(
            **{feat: round(float(cleaned[feat]), 4) for feat in AUDIO_FEATURE_FIELDS}
        )
    elif isinstance(cleaned.get("audio_features"), AudioFeatures):
        af_dict = cleaned["audio_features"].model_dump()
        audio_features = AudioFeatures(
            **{feat: round(float(af_dict[feat]), 4) for feat in AUDIO_FEATURE_FIELDS}
        )
    elif isinstance(cleaned.get("audio_features"), dict):
        af_clean = {k: _clean_scalar(v) for k, v in cleaned["audio_features"].items()}
        if all(feat in af_clean and af_clean[feat] is not None for feat in AUDIO_FEATURE_FIELDS):
            audio_features = AudioFeatures(
                **{feat: round(float(af_clean[feat]), 4) for feat in AUDIO_FEATURE_FIELDS}
            )

    resolved_rank = rank if rank is not None else cleaned.get("rank")
    if resolved_rank is not None:
        resolved_rank = int(resolved_rank)

    return TrackItem(
        rank=resolved_rank,
        track_id=track_id,
        track_name=track_name,
        artist_name=artist_name,
        artist_genres=artist_genres,
        popularity=popularity,
        similarity_score=similarity_score,
        mmr_score=mmr_score,
        spotify_embed_url=f"https://open.spotify.com/embed/track/{track_id}",
        audio_features=audio_features,
    )


@app.get("/api/v1/health", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return server health and loaded model status."""
    recommender = _get_model()
    is_loaded = recommender is not None and recommender.df is not None
    total_tracks = int(len(recommender.df)) if is_loaded else 0
    return HealthResponse(
        status="ok" if is_loaded else "error",
        model_loaded=is_loaded,
        mode=model_mode,
        total_tracks=total_tracks,
        artifact_path=loaded_artifact_path,
    )


@app.get("/api/v1/search", response_model=SearchResponse)
def search_tracks(
    q: str = Query(..., description="Search query for track name or artist name"),
    limit: int = Query(15, ge=1, le=100, description="Maximum number of search results"),
) -> SearchResponse:
    """Search tracks by keyword and attach their 9 audio features."""
    start = time.perf_counter()
    recommender = _get_model()

    raw_results = recommender.search_tracks(q, limit=limit, as_dict=True)
    items: List[TrackItem] = []

    for idx, row in enumerate(raw_results, start=1):
        tid = row.get("track_id")
        merged_row = dict(row)
        has_all_audio = all(
            feat in merged_row and merged_row[feat] is not None
            for feat in AUDIO_FEATURE_FIELDS
        )
        if tid and not has_all_audio:
            try:
                details = recommender.get_track_details(tid)
                if details:
                    merged_row = {**details, **row}
            except ValueError:
                pass
        items.append(_row_to_track_item(merged_row, rank=idx))

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return SearchResponse(
        status="success",
        query=q,
        total=len(items),
        results=items,
        latency_ms=latency_ms,
    )


@app.get("/api/v1/tracks/{track_id}", response_model=TrackItem)
def get_track(track_id: str) -> TrackItem:
    """Retrieve metadata and 9 audio features for a single track by track_id."""
    recommender = _get_model()
    try:
        details = recommender.get_track_details(track_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not details:
        raise HTTPException(
            status_code=404, detail=f"Track ID not found: {track_id}"
        )

    return _row_to_track_item(details)


@app.get("/api/v1/recommend/track/{track_id}", response_model=RecommendationResponse)
def recommend_by_track(
    track_id: str,
    k: int = Query(10, ge=1, le=50, description="Number of tracks to recommend"),
    strategy: Literal["audio", "hybrid", "mmr"] = Query(
        "hybrid",
        description="Recommendation strategy: audio, hybrid, or mmr",
    ),
    alpha: float = Query(0.7, ge=0.0, le=1.0, description="Audio vs genre weight"),
    mmr_lambda: float = Query(
        0.7, ge=0.0, le=1.0, description="Relevance vs diversity weight for MMR"
    ),
    exclude_same_artist: bool = Query(
        True, description="Exclude tracks from the same artist as the seed track"
    ),
) -> RecommendationResponse:
    """Recommend top-K tracks similar to a single seed track."""
    start = time.perf_counter()
    recommender = _get_model()

    strategy_norm = strategy.strip().lower()

    try:
        seed_details = recommender.get_track_details(track_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    seed_item = _row_to_track_item(seed_details)

    def _invoke(exclude_artist_flag: bool) -> List[Dict[str, Any]]:
        if strategy_norm == "audio":
            return recommender.recommend_by_track(
                track_id,
                exclude_same_artist=exclude_artist_flag,
                top_k=k,
                as_dict=True,
            )
        if strategy_norm == "hybrid":
            return recommender.recommend_hybrid(
                track_id,
                alpha=alpha,
                exclude_same_artist=exclude_artist_flag,
                top_k=k,
                as_dict=True,
            )
        return recommender.recommend_hybrid_mmr(
            track_id,
            alpha=alpha,
            mmr_lambda=mmr_lambda,
            exclude_same_artist=exclude_artist_flag,
            top_k=k,
            as_dict=True,
        )

    try:
        raw_recs = _invoke(exclude_same_artist)
    except ValueError as exc:
        err_msg = str(exc)
        if exclude_same_artist and "artist_id" in err_msg:
            raw_recs = _invoke(False)
        elif "Track ID not found" in err_msg:
            raise HTTPException(status_code=404, detail=err_msg) from exc
        else:
            raise HTTPException(status_code=400, detail=err_msg) from exc

    recommendations = [
        _row_to_track_item(row, rank=idx)
        for idx, row in enumerate(raw_recs, start=1)
    ]
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return RecommendationResponse(
        status="success",
        mode=strategy_norm,
        seed_track=seed_item,
        recommendations=recommendations,
        latency_ms=latency_ms,
    )


@app.post("/api/v1/recommend/profile", response_model=RecommendationResponse)
def recommend_by_profile(payload: TasteProfileRequest) -> RecommendationResponse:
    """Recommend top-K tracks based on a multi-track user taste profile."""
    start = time.perf_counter()
    recommender = _get_model()

    unique_track_ids = list(dict.fromkeys(payload.track_ids))

    seed_items: List[TrackItem] = []
    for tid in unique_track_ids:
        try:
            details = recommender.get_track_details(tid)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        seed_items.append(_row_to_track_item(details))

    def _invoke(exclude_artist_flag: bool) -> List[Dict[str, Any]]:
        return recommender.recommend_by_taste_profile(
            unique_track_ids,
            alpha=payload.alpha,
            exclude_same_artist=exclude_artist_flag,
            top_k=payload.k,
            as_dict=True,
        )

    try:
        raw_recs = _invoke(payload.exclude_same_artist)
    except ValueError as exc:
        err_msg = str(exc)
        if payload.exclude_same_artist and "artist_id" in err_msg:
            raw_recs = _invoke(False)
        elif "Track ID not found" in err_msg:
            raise HTTPException(status_code=404, detail=err_msg) from exc
        else:
            raise HTTPException(status_code=400, detail=err_msg) from exc

    recommendations = [
        _row_to_track_item(row, rank=idx)
        for idx, row in enumerate(raw_recs, start=1)
    ]
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return RecommendationResponse(
        status="success",
        mode="taste_profile",
        seed_tracks=seed_items,
        recommendations=recommendations,
        latency_ms=latency_ms,
    )


@app.get("/api/v1/recommend/popular", response_model=RecommendationResponse)
def recommend_popular(
    k: int = Query(10, ge=1, le=50, description="Number of popular tracks to return"),
) -> RecommendationResponse:
    """Return top-K most popular tracks as cold-start baseline."""
    start = time.perf_counter()
    recommender = _get_model()

    raw_recs = recommender.recommend_by_popularity(top_k=k, as_dict=True)
    recommendations = [
        _row_to_track_item(row, rank=idx)
        for idx, row in enumerate(raw_recs, start=1)
    ]
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return RecommendationResponse(
        status="success",
        mode="popular",
        recommendations=recommendations,
        latency_ms=latency_ms,
    )


@app.get("/api/v1/recommend/mood/{mood_name}", response_model=RecommendationResponse)
def recommend_by_mood(
    mood_name: str,
    k: int = Query(10, ge=1, le=50, description="Number of mood tracks to return"),
) -> RecommendationResponse:
    """Recommend top-K tracks matching a target acoustic mood profile."""
    start = time.perf_counter()
    recommender = _get_model()

    try:
        raw_recs = recommender.recommend_by_mood(mood_name, top_k=k, as_dict=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    recommendations = [
        _row_to_track_item(row, rank=idx)
        for idx, row in enumerate(raw_recs, start=1)
    ]
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return RecommendationResponse(
        status="success",
        mode=f"mood_{mood_name.strip().lower()}",
        recommendations=recommendations,
        latency_ms=latency_ms,
    )
