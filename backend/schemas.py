"""Pydantic v2 schemas for Embeat Music Recommender REST API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class AudioFeatures(BaseModel):
    """Core 9 acoustic features of a track."""

    danceability: float
    energy: float
    valence: float
    tempo: float
    acousticness: float
    instrumentalness: float
    liveness: float
    speechiness: float
    loudness: float


class TrackItem(BaseModel):
    """Metadata, similarity score, and audio features of a single track."""

    rank: Optional[int] = None
    track_id: str
    track_name: str
    artist_name: Optional[str] = None
    artist_genres: Optional[str] = None
    popularity: Optional[int] = None
    similarity_score: Optional[float] = None
    mmr_score: Optional[float] = None
    spotify_embed_url: str
    audio_features: Optional[AudioFeatures] = None


class RecommendationResponse(BaseModel):
    """Standard JSON response for all recommendation endpoints."""

    status: str = "success"
    mode: str
    seed_track: Optional[TrackItem] = None
    seed_tracks: Optional[List[TrackItem]] = None
    recommendations: List[TrackItem]
    latency_ms: float


class SearchResponse(BaseModel):
    """Standard JSON response for track search endpoint."""

    status: str = "success"
    query: str
    total: int
    results: List[TrackItem]
    latency_ms: float


class TasteProfileRequest(BaseModel):
    """Request payload for multi-track taste profile recommendation."""

    track_ids: List[str] = Field(..., min_length=1, max_length=50)
    k: int = Field(10, ge=1, le=50)
    alpha: float = Field(0.7, ge=0.0, le=1.0)
    exclude_same_artist: bool = False


class HealthResponse(BaseModel):
    """Server health and loaded model status response."""

    status: str
    model_loaded: bool
    mode: str
    total_tracks: int
    artifact_path: Optional[str] = None
