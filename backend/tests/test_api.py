"""Integration tests for FastAPI Backend endpoints using TestClient."""

from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from backend.app import app


@pytest.fixture(scope="module")
def client():
    """Initialize TestClient with lifespan context to load the model once."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client: TestClient):
    """Verify /api/v1/health returns 200 with loaded model metadata and relative artifact_path."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["total_tracks"] > 0
    assert data["artifact_path"] is not None
    assert not Path(data["artifact_path"]).is_absolute()


def test_search_tracks(client: TestClient):
    """Verify /api/v1/search?q=a&limit=5 returns tracks with embed URL and 9 rounded audio features."""
    response = client.get("/api/v1/search", params={"q": "a", "limit": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total"] == len(data["results"])
    assert len(data["results"]) == 5

    for idx, item in enumerate(data["results"], start=1):
        assert item["rank"] == idx
        assert item["track_id"]
        assert item["spotify_embed_url"].startswith(
            "https://open.spotify.com/embed/track/"
        )
        assert item["audio_features"] is not None
        assert len(item["audio_features"]) == 9
        for val in item["audio_features"].values():
            assert isinstance(val, float)
            assert round(val, 4) == val


def test_get_track_details_valid_and_404(client: TestClient):
    """Verify /api/v1/tracks/{track_id} returns 200 for a valid ID and 404 with detail for unknown ID."""
    pop_res = client.get("/api/v1/recommend/popular", params={"k": 1})
    assert pop_res.status_code == 200
    valid_track_id = pop_res.json()["recommendations"][0]["track_id"]

    valid_res = client.get(f"/api/v1/tracks/{valid_track_id}")
    assert valid_res.status_code == 200
    track_data = valid_res.json()
    assert track_data["track_id"] == valid_track_id
    assert track_data["audio_features"] is not None

    not_found_res = client.get("/api/v1/tracks/non_existent_id_999")
    assert not_found_res.status_code == 404
    err_data = not_found_res.json()
    assert "detail" in err_data
    assert "Track ID not found" in err_data["detail"]
    assert "non_existent_id_999" in err_data["detail"]


def test_recommend_by_track_strategies(client: TestClient):
    """Verify /api/v1/recommend/track/{track_id} across audio, hybrid, and mmr strategies."""
    pop_res = client.get("/api/v1/recommend/popular", params={"k": 1})
    seed_id = pop_res.json()["recommendations"][0]["track_id"]

    for strategy in ("audio", "hybrid", "mmr"):
        response = client.get(
            f"/api/v1/recommend/track/{seed_id}",
            params={"k": 5, "strategy": strategy},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["mode"] == strategy
        assert data["seed_track"]["track_id"] == seed_id

        recs = data["recommendations"]
        assert len(recs) == 5
        rec_ids = {item["track_id"] for item in recs}
        assert seed_id not in rec_ids

        for idx, item in enumerate(recs, start=1):
            assert item["rank"] == idx
            assert item["similarity_score"] is not None
            assert 0.0 <= item["similarity_score"] <= 1.0
            assert item["audio_features"] is not None
            if strategy == "mmr":
                assert item["mmr_score"] is not None
            else:
                assert item["mmr_score"] is None


def test_recommend_mmr_similarity_score_non_negative(client: TestClient):
    """Verify strategy=mmr returns non-negative hybrid similarity_score even with high diversity weight."""
    pop_res = client.get("/api/v1/recommend/popular", params={"k": 1})
    seed_id = pop_res.json()["recommendations"][0]["track_id"]

    response = client.get(
        f"/api/v1/recommend/track/{seed_id}",
        params={"k": 10, "strategy": "mmr", "mmr_lambda": 0.3},
    )
    assert response.status_code == 200
    data = response.json()
    recs = data["recommendations"]
    assert len(recs) == 10
    for item in recs:
        assert item["similarity_score"] is not None
        assert 0.0 <= item["similarity_score"] <= 1.0
        assert item["mmr_score"] is not None


def test_recommend_by_track_404(client: TestClient):
    """Verify /api/v1/recommend/track/invalid_id returns 404 and checks detail content."""
    response = client.get("/api/v1/recommend/track/invalid_id")
    assert response.status_code == 404
    err_data = response.json()
    assert "detail" in err_data
    assert "Track ID not found" in err_data["detail"]
    assert "invalid_id" in err_data["detail"]


def test_recommend_taste_profile(client: TestClient):
    """Verify POST /api/v1/recommend/profile with valid track IDs and deduplication."""
    pop_res = client.get("/api/v1/recommend/popular", params={"k": 2})
    assert pop_res.status_code == 200
    track_ids = [item["track_id"] for item in pop_res.json()["recommendations"]]
    assert len(track_ids) == 2

    # Send with a duplicate ID to verify deduplication
    response = client.post(
        "/api/v1/recommend/profile",
        json={"track_ids": [track_ids[0], track_ids[1], track_ids[0]], "k": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["seed_tracks"]) == 2
    assert len(data["recommendations"]) == 5
    rec_ids = {item["track_id"] for item in data["recommendations"]}
    for tid in track_ids:
        assert tid not in rec_ids


def test_recommend_taste_profile_404(client: TestClient):
    """Verify POST /api/v1/recommend/profile with unknown track_id returns 404 and checks detail."""
    response = client.post(
        "/api/v1/recommend/profile",
        json={"track_ids": ["non_existent_track_999"], "k": 5},
    )
    assert response.status_code == 404
    err_data = response.json()
    assert "detail" in err_data
    assert "Track ID not found" in err_data["detail"]
    assert "non_existent_track_999" in err_data["detail"]


def test_recommend_popular(client: TestClient):
    """Verify /api/v1/recommend/popular?k=10 returns 200 and 10 tracks."""
    response = client.get("/api/v1/recommend/popular", params={"k": 10})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["recommendations"]) == 10


def test_recommend_mood_valid_and_400(client: TestClient):
    """Verify /api/v1/recommend/mood/happy?k=5 returns 200 and invalid_mood returns 400 with detail."""
    valid_res = client.get("/api/v1/recommend/mood/happy", params={"k": 5})
    assert valid_res.status_code == 200
    data = valid_res.json()
    assert data["status"] == "success"
    assert len(data["recommendations"]) == 5
    for item in data["recommendations"]:
        assert item["similarity_score"] is not None

    invalid_res = client.get("/api/v1/recommend/mood/invalid_mood", params={"k": 5})
    assert invalid_res.status_code == 400
    err_data = invalid_res.json()
    assert "detail" in err_data
    assert "Unsupported mood" in err_data["detail"]
    assert "invalid_mood" in err_data["detail"]


def test_validation_422_errors(client: TestClient):
    """Verify 422 Unprocessable Entity and detail structure for k out of range, invalid strategy, empty track_ids, and >50 track_ids."""
    pop_res = client.get("/api/v1/recommend/popular", params={"k": 1})
    seed_id = pop_res.json()["recommendations"][0]["track_id"]

    # 1. k out of bounds (k=0 and k=51)
    for invalid_k in (0, 51):
        res_k = client.get(
            f"/api/v1/recommend/track/{seed_id}", params={"k": invalid_k}
        )
        assert res_k.status_code == 422
        detail_k = res_k.json().get("detail")
        assert isinstance(detail_k, list) and len(detail_k) > 0
        assert any("k" in err.get("loc", []) for err in detail_k)

    # 2. Invalid strategy
    res_strategy = client.get(
        f"/api/v1/recommend/track/{seed_id}",
        params={"k": 5, "strategy": "invalid_strategy"},
    )
    assert res_strategy.status_code == 422
    detail_strategy = res_strategy.json().get("detail")
    assert isinstance(detail_strategy, list) and len(detail_strategy) > 0
    assert any("strategy" in err.get("loc", []) for err in detail_strategy)

    # 3. Empty track_ids list
    res_empty = client.post(
        "/api/v1/recommend/profile", json={"track_ids": [], "k": 5}
    )
    assert res_empty.status_code == 422
    detail_empty = res_empty.json().get("detail")
    assert isinstance(detail_empty, list) and len(detail_empty) > 0
    assert any("track_ids" in err.get("loc", []) for err in detail_empty)

    # 4. track_ids exceeding max_length=50
    res_too_many = client.post(
        "/api/v1/recommend/profile",
        json={"track_ids": [seed_id] * 51, "k": 5},
    )
    assert res_too_many.status_code == 422
    detail_many = res_too_many.json().get("detail")
    assert isinstance(detail_many, list) and len(detail_many) > 0
    assert any("track_ids" in err.get("loc", []) for err in detail_many)
