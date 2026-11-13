# ĐẶC TẢ GIAO ƯỚC RESTful API (BACKEND -> FRONTEND API CONTRACT)

> **Dự án:** Hệ gợi ý nhạc AI (Embeat 45M Spotify Tracks)  
> **Người soạn thảo:** Nhật Minh Phạm — Backend Engineer (Thành viên 3)  
> **Người tiếp nhận:** Lê Chí Thiện — Frontend / QA Engineer (Thành viên 4)  
> **Base URL mặc định:** `http://localhost:8000`  
> **Tài liệu tương tác (Swagger UI):** `http://localhost:8000/docs`  
> **ReDoc UI:** `http://localhost:8000/redoc`

---

## 1. Tổng Quan & Cấu Hình Kết Nối

- **CORS Policy:** Đã bật `CORSMiddleware` cho phép mọi Origin (`allow_origins=["*"]`), mọi HTTP Method (`allow_methods=["*"]`) và mọi Header (`allow_headers=["*"]`). Frontend có thể gọi trực tiếp từ `file://`, `http://localhost:3000`, hoặc `http://127.0.0.1:5500` (Live Server) mà không bị chặn CORS.
- **Định dạng dữ liệu Header:** `content-type: application/json`.
- **Xử lý giá trị đặc biệt & Làm tròn:**
  - Mọi giá trị `NaN`, `np.nan`, `±Infinity` từ tầng dữ liệu đã được làm sạch thành `null` chuẩn JSON.
  - Các chỉ số trong `audio_features`, `similarity_score` và `mmr_score` đều được ép về kiểu `float` chuẩn của Python và làm tròn **4 chữ số thập phân**.
- **Nhúng trình phát nhạc Spotify:** Mọi đối tượng bài hát (`TrackItem`) đều có sẵn trường `spotify_embed_url` theo mẫu:
  ```text
  https://open.spotify.com/embed/track/{track_id}
  ```

---

## 2. Cấu Trúc Schema Dữ Liệu Dùng Chung (`backend/schemas.py`)

### 2.1. `AudioFeatures` (9 Đặc trưng âm học phục vụ vẽ Radar Chart)
| Trường | Kiểu dữ liệu | Miền giá trị tham khảo | Mô tả |
| :--- | :---: | :---: | :--- |
| `danceability` | `float` | `0.0 – 1.0` | Độ bắt nhịp, phù hợp khiêu vũ |
| `energy` | `float` | `0.0 – 1.0` | Cường độ năng lượng, độ sôi động |
| `valence` | `float` | `0.0 – 1.0` | Sắc thái cảm xúc tích cực / vui tươi |
| `tempo` | `float` | `30.0 – 250.0` | Nhịp độ bài hát (BPM) |
| `acousticness` | `float` | `0.0 – 1.0` | Độ mộc mạc của nhạc cụ acoustic |
| `instrumentalness` | `float` | `0.0 – 1.0` | Mức độ thuần nhạc cụ không lời |
| `liveness` | `float` | `0.0 – 1.0` | Đặc trưng thu âm biểu diễn trực tiếp |
| `speechiness` | `float` | `0.0 – 1.0` | Mật độ lời nói / giọng đọc (Rap/Spoken word) |
| `loudness` | `float` | `-60.0 – 5.0` | Độ lớn âm lượng trung bình (dB) |

### 2.2. `TrackItem` (Thông tin 1 bài hát)
| Trường | Kiểu dữ liệu | Nullable? | Mô tả |
| :--- | :---: | :---: | :--- |
| `rank` | `int \| null` | Có | Thứ hạng trong danh sách tìm kiếm hoặc gợi ý (`1..K`), `null` đối với bài hát tra cứu đơn lẻ hoặc `seed_track` |
| `track_id` | `string` | Không | Spotify Track ID duy nhất |
| `track_name` | `string` | Không | Tên bài hát |
| `artist_name` | `string \| null` | Có | Tên nghệ sĩ trình bày |
| `artist_genres` | `string \| null` | Có | Chuỗi các thể loại phân cách bằng dấu phẩy |
| `popularity` | `int \| null` | Có | Độ phổ biến quy chuẩn (`0 – 100`) |
| `similarity_score` | `float \| null` | Có | Điểm tương đồng thực tế (`0.0 – 1.0`, làm tròn 4 chữ số thập phân): lấy từ `similarity` (chế độ `audio`), `hybrid_similarity` (chế độ `hybrid`, `mmr`, `taste_profile`), hoặc `mood_similarity` (chế độ `mood`). Bằng `null` ở `/search`, `/tracks/{id}`, `/recommend/popular` và trong `seed_track`/`seed_tracks` |
| `mmr_score` | `float \| null` | Có | Điểm mục tiêu Maximal Marginal Relevance (làm tròn 4 chữ số thập phân, có thể âm khi ưu tiên đa dạng hóa mạnh). Chỉ khác `null` khi `strategy="mmr"`, các chế độ khác đều là `null` |
| `spotify_embed_url` | `string` | Không | URL nhúng trực tiếp vào `<iframe src="...">` |
| `audio_features` | `AudioFeatures \| null` | Có | Đối tượng chứa đủ 9 chỉ số âm học đã làm tròn 4 chữ số thập phân |

---

## 3. Chi Tiết 7 RESTful API Endpoints (`backend/app.py`)

### 3.1. Kiểm tra trạng thái Server & Model (`Health Check`)
- **Method & URL:** `GET /api/v1/health` *(hoặc `GET /health`)*
- **Query Params:** Không có.
- **Lưu ý:** Trường `artifact_path` luôn trả về đường dẫn tương đối so với thư mục gốc dự án (không để lộ đường dẫn tuyệt đối của máy chủ).
- **Response `200 OK` (`HealthResponse`):**
```json
{
  "status": "ok",
  "model_loaded": true,
  "mode": "artifact_joblib",
  "total_tracks": 29985,
  "artifact_path": "artifacts/embeat_recommender.joblib"
}
```

---

### 3.2. Tìm kiếm bài hát theo từ khóa (`Search Tracks`)
- **Method & URL:** `GET /api/v1/search`
- **Query Params:**
  | Tên tham số | Kiểu | Bắt buộc? | Mặc định | Ràng buộc | Mô tả |
  | :--- | :---: | :---: | :---: | :---: | :--- |
  | `q` | `string` | **Có** | — | Chuỗi bất kỳ | Từ khóa tìm theo tên bài hát (`track_name`) hoặc tên nghệ sĩ (`artist_name`) |
  | `limit` | `int` | Không | `15` | `1 <= limit <= 100` | Số lượng kết quả tối đa trả về |
- **Ví dụ Request:** `GET /api/v1/search?q=In%20Da%20Club&limit=2`
- **Response `200 OK` (`SearchResponse`):** *(Mỗi bài hát trả về đã được đính kèm đầy đủ 9 `audio_features`)*
```json
{
  "status": "success",
  "query": "In Da Club",
  "total": 2,
  "results": [
    {
      "rank": 1,
      "track_id": "7iL6o9tox1zgHpKUfh9vuC",
      "track_name": "In Da Club",
      "artist_name": "50 Cent",
      "artist_genres": "hip hop, gangster rap, east coast hip hop, rap, queens hip hop, pop rap",
      "popularity": 82,
      "similarity_score": null,
      "mmr_score": null,
      "spotify_embed_url": "https://open.spotify.com/embed/track/7iL6o9tox1zgHpKUfh9vuC",
      "audio_features": {
        "danceability": 0.899,
        "energy": 0.713,
        "valence": 0.777,
        "tempo": 90.0,
        "acousticness": 0.255,
        "instrumentalness": 0.0,
        "liveness": 0.0708,
        "speechiness": 0.366,
        "loudness": -2.752
      }
    },
    {
      "rank": 2,
      "track_id": "4RY96Asd9IefaL3X4LOLZ8",
      "track_name": "In Da Club",
      "artist_name": "50 Cent",
      "artist_genres": "hip hop, gangster rap, east coast hip hop, rap, queens hip hop, pop rap",
      "popularity": 75,
      "similarity_score": null,
      "mmr_score": null,
      "spotify_embed_url": "https://open.spotify.com/embed/track/4RY96Asd9IefaL3X4LOLZ8",
      "audio_features": {
        "danceability": 0.904,
        "energy": 0.72,
        "valence": 0.805,
        "tempo": 90.0,
        "acousticness": 0.259,
        "instrumentalness": 0.0,
        "liveness": 0.0749,
        "speechiness": 0.348,
        "loudness": -2.776
      }
    }
  ],
  "latency_ms": 12.11
}
```

---

### 3.3. Lấy thông tin chi tiết 1 bài hát (`Get Track Details`)
- **Method & URL:** `GET /api/v1/tracks/{track_id}`
- **Path Params:** `track_id` (`string`, bắt buộc).
- **Ví dụ Request:** `GET /api/v1/tracks/7iL6o9tox1zgHpKUfh9vuC`
- **Response `200 OK` (`TrackItem`):**
```json
{
  "rank": null,
  "track_id": "7iL6o9tox1zgHpKUfh9vuC",
  "track_name": "In Da Club",
  "artist_name": "50 Cent",
  "artist_genres": "hip hop, gangster rap, east coast hip hop, rap, queens hip hop, pop rap",
  "popularity": 82,
  "similarity_score": null,
  "mmr_score": null,
  "spotify_embed_url": "https://open.spotify.com/embed/track/7iL6o9tox1zgHpKUfh9vuC",
  "audio_features": {
    "danceability": 0.899,
    "energy": 0.713,
    "valence": 0.777,
    "tempo": 90.0,
    "acousticness": 0.255,
    "instrumentalness": 0.0,
    "liveness": 0.0708,
    "speechiness": 0.366,
    "loudness": -2.752
  }
}
```
- **Response `404 Not Found`:** Khi `track_id` không tồn tại trong mô hình (`GET /api/v1/tracks/non_existent_id_999`):
```json
{
  "detail": "Track ID not found: non_existent_id_999"
}
```

---

### 3.4. Gợi ý bài hát tương đồng từ 1 bài hát mồi (`Single-Track Recommendation`)
- **Method & URL:** `GET /api/v1/recommend/track/{track_id}`
- **Path Params:** `track_id` (`string`, bắt buộc).
- **Query Params:**
  | Tên tham số | Kiểu | Mặc định | Ràng buộc | Mô tả |
  | :--- | :---: | :---: | :---: | :--- |
  | `k` | `int` | `10` | `1 <= k <= 50` | Số lượng bài hát gợi ý trả về |
  | `strategy` | `Literal` | `"hybrid"` | `"audio"` \| `"hybrid"` \| `"mmr"` | Chiến lược gợi ý: thuần âm học (`audio`), kết hợp âm học + thể loại (`hybrid`), hoặc đa dạng hóa bằng Maximal Marginal Relevance (`mmr`) |
  | `alpha` | `float` | `0.7` | `0.0 <= alpha <= 1.0` | Trọng số kết hợp giữa Audio (`alpha`) và Genre (`1 - alpha`) |
  | `mmr_lambda` | `float` | `0.7` | `0.0 <= mmr_lambda <= 1.0` | Trọng số cân bằng giữa độ liên quan và độ đa dạng khi `strategy="mmr"` |
  | `exclude_same_artist` | `bool` | `true` | `true` \| `false` | Loại bỏ các bài hát cùng nghệ sĩ với bài mồi (tự động fallback sang `false` nếu tập dữ liệu thiếu cột `artist_id`) |
- **Ví dụ Request 1 (`strategy=hybrid`):** `GET /api/v1/recommend/track/7iL6o9tox1zgHpKUfh9vuC?k=2&strategy=hybrid&alpha=0.7`
- **Response `200 OK` (`RecommendationResponse` - Hybrid):**
```json
{
  "status": "success",
  "mode": "hybrid",
  "seed_track": {
    "rank": null,
    "track_id": "7iL6o9tox1zgHpKUfh9vuC",
    "track_name": "In Da Club",
    "artist_name": "50 Cent",
    "artist_genres": "hip hop, gangster rap, east coast hip hop, rap, queens hip hop, pop rap",
    "popularity": 82,
    "similarity_score": null,
    "mmr_score": null,
    "spotify_embed_url": "https://open.spotify.com/embed/track/7iL6o9tox1zgHpKUfh9vuC",
    "audio_features": {
      "danceability": 0.899,
      "energy": 0.713,
      "valence": 0.777,
      "tempo": 90.0,
      "acousticness": 0.255,
      "instrumentalness": 0.0,
      "liveness": 0.0708,
      "speechiness": 0.366,
      "loudness": -2.752
    }
  },
  "seed_tracks": null,
  "recommendations": [
    {
      "rank": 1,
      "track_id": "63BcfK6YAzJYeISaTPr6IO",
      "track_name": "Big Poppa - 2007 Remaster",
      "artist_name": "The Notorious B.I.G.",
      "artist_genres": "east coast hip hop, hip hop, gangster rap, hardcore hip hop, rap",
      "popularity": 72,
      "similarity_score": 0.8826,
      "mmr_score": null,
      "spotify_embed_url": "https://open.spotify.com/embed/track/63BcfK6YAzJYeISaTPr6IO",
      "audio_features": {
        "danceability": 0.765,
        "energy": 0.671,
        "valence": 0.798,
        "tempo": 85.0,
        "acousticness": 0.344,
        "instrumentalness": 0.0,
        "liveness": 0.0506,
        "speechiness": 0.278,
        "loudness": -5.249
      }
    },
    {
      "rank": 2,
      "track_id": "2NBQmPrOEEjA8VbeWOQGxO",
      "track_name": "Drop It Like It's Hot",
      "artist_name": "Snoop Dogg",
      "artist_genres": "gangster rap, rap, west coast rap, pop rap, g funk, hip hop",
      "popularity": 78,
      "similarity_score": 0.8715,
      "mmr_score": null,
      "spotify_embed_url": "https://open.spotify.com/embed/track/2NBQmPrOEEjA8VbeWOQGxO",
      "audio_features": {
        "danceability": 0.888,
        "energy": 0.632,
        "valence": 0.662,
        "tempo": 92.0,
        "acousticness": 0.19,
        "instrumentalness": 0.0,
        "liveness": 0.101,
        "speechiness": 0.219,
        "loudness": -3.872
      }
    }
  ],
  "latency_ms": 5.71
}
```
- **Ví dụ Request 2 (`strategy=mmr`):** `GET /api/v1/recommend/track/7iL6o9tox1zgHpKUfh9vuC?k=2&strategy=mmr&alpha=0.7&mmr_lambda=0.7`
- **Response `200 OK` (`RecommendationResponse` - MMR, có cả `similarity_score` và `mmr_score`):**
```json
{
  "status": "success",
  "mode": "mmr",
  "seed_track": {
    "rank": null,
    "track_id": "7iL6o9tox1zgHpKUfh9vuC",
    "track_name": "In Da Club",
    "artist_name": "50 Cent",
    "artist_genres": "hip hop, gangster rap, east coast hip hop, rap, queens hip hop, pop rap",
    "popularity": 82,
    "similarity_score": null,
    "mmr_score": null,
    "spotify_embed_url": "https://open.spotify.com/embed/track/7iL6o9tox1zgHpKUfh9vuC",
    "audio_features": {
      "danceability": 0.899,
      "energy": 0.713,
      "valence": 0.777,
      "tempo": 90.0,
      "acousticness": 0.255,
      "instrumentalness": 0.0,
      "liveness": 0.0708,
      "speechiness": 0.366,
      "loudness": -2.752
    }
  },
  "seed_tracks": null,
  "recommendations": [
    {
      "rank": 1,
      "track_id": "63BcfK6YAzJYeISaTPr6IO",
      "track_name": "Big Poppa - 2007 Remaster",
      "artist_name": "The Notorious B.I.G.",
      "artist_genres": "east coast hip hop, hip hop, gangster rap, hardcore hip hop, rap",
      "popularity": 72,
      "similarity_score": 0.8826,
      "mmr_score": 0.6178,
      "spotify_embed_url": "https://open.spotify.com/embed/track/63BcfK6YAzJYeISaTPr6IO",
      "audio_features": {
        "danceability": 0.765,
        "energy": 0.671,
        "valence": 0.798,
        "tempo": 85.0,
        "acousticness": 0.344,
        "instrumentalness": 0.0,
        "liveness": 0.0506,
        "speechiness": 0.278,
        "loudness": -5.249
      }
    },
    {
      "rank": 2,
      "track_id": "2NBQmPrOEEjA8VbeWOQGxO",
      "track_name": "Drop It Like It's Hot",
      "artist_name": "Snoop Dogg",
      "artist_genres": "gangster rap, rap, west coast rap, pop rap, g funk, hip hop",
      "popularity": 78,
      "similarity_score": 0.8715,
      "mmr_score": 0.3731,
      "spotify_embed_url": "https://open.spotify.com/embed/track/2NBQmPrOEEjA8VbeWOQGxO",
      "audio_features": {
        "danceability": 0.888,
        "energy": 0.632,
        "valence": 0.662,
        "tempo": 92.0,
        "acousticness": 0.19,
        "instrumentalness": 0.0,
        "liveness": 0.101,
        "speechiness": 0.219,
        "loudness": -3.872
      }
    }
  ],
  "latency_ms": 5.52
}
```
- **Mã lỗi có thể gặp:**
  - `404 Not Found`: `track_id` không tồn tại.
  - `422 Unprocessable Entity`: `strategy` nằm ngoài `audio | hybrid | mmr`, hoặc `k`, `alpha`, `mmr_lambda` nằm ngoài miền hợp lệ.

---

### 3.5. Gợi ý theo Hồ sơ Gu âm nhạc nhiều bài (`Taste Profile Recommendation`)
- **Method & URL:** `POST /api/v1/recommend/profile`
- **Request Body (`TasteProfileRequest`):**
```json
{
  "track_ids": [
    "7fEoXCZTZFosUFvFQg1BmW",
    "7iL6o9tox1zgHpKUfh9vuC"
  ],
  "k": 2,
  "alpha": 0.7,
  "exclude_same_artist": false
}
```
  | Trường Body | Kiểu | Bắt buộc? | Mặc định | Ràng buộc |
  | :--- | :---: | :---: | :---: | :--- |
  | `track_ids` | `list[str]` | **Có** | — | Từ 1 đến 50 `track_id` (`min_length=1, max_length=50`). Nếu có ID trùng lặp, server tự động loại trùng giữ nguyên thứ tự xuất hiện đầu tiên |
  | `k` | `int` | Không | `10` | `1 <= k <= 50` |
  | `alpha` | `float` | Không | `0.7` | `0.0 <= alpha <= 1.0` |
  | `exclude_same_artist` | `bool` | Không | `false` | `true` \| `false` |
- **Response `200 OK` (`RecommendationResponse`):**
```json
{
  "status": "success",
  "mode": "taste_profile",
  "seed_track": null,
  "seed_tracks": [
    {
      "rank": null,
      "track_id": "7fEoXCZTZFosUFvFQg1BmW",
      "track_name": "Took Her To The O",
      "artist_name": "King Von",
      "artist_genres": "chicago rap",
      "popularity": 82,
      "similarity_score": null,
      "mmr_score": null,
      "spotify_embed_url": "https://open.spotify.com/embed/track/7fEoXCZTZFosUFvFQg1BmW",
      "audio_features": {
        "danceability": 0.82,
        "energy": 0.592,
        "valence": 0.4,
        "tempo": 160.0,
        "acousticness": 0.0149,
        "instrumentalness": 0.0,
        "liveness": 0.121,
        "speechiness": 0.29,
        "loudness": -7.002
      }
    },
    {
      "rank": null,
      "track_id": "7iL6o9tox1zgHpKUfh9vuC",
      "track_name": "In Da Club",
      "artist_name": "50 Cent",
      "artist_genres": "hip hop, gangster rap, east coast hip hop, rap, queens hip hop, pop rap",
      "popularity": 82,
      "similarity_score": null,
      "mmr_score": null,
      "spotify_embed_url": "https://open.spotify.com/embed/track/7iL6o9tox1zgHpKUfh9vuC",
      "audio_features": {
        "danceability": 0.899,
        "energy": 0.713,
        "valence": 0.777,
        "tempo": 90.0,
        "acousticness": 0.255,
        "instrumentalness": 0.0,
        "liveness": 0.0708,
        "speechiness": 0.366,
        "loudness": -2.752
      }
    }
  ],
  "recommendations": [
    {
      "rank": 1,
      "track_id": "4PmMVdIFpJTB9WQbmILf4p",
      "track_name": "How We Do",
      "artist_name": "50 Cent",
      "artist_genres": "hip hop, gangster rap, east coast hip hop, rap, queens hip hop, pop rap",
      "popularity": 76,
      "similarity_score": 0.8909,
      "mmr_score": null,
      "spotify_embed_url": "https://open.spotify.com/embed/track/4PmMVdIFpJTB9WQbmILf4p",
      "audio_features": {
        "danceability": 0.862,
        "energy": 0.648,
        "valence": 0.637,
        "tempo": 98.0,
        "acousticness": 0.0455,
        "instrumentalness": 0.0,
        "liveness": 0.0332,
        "speechiness": 0.251,
        "loudness": -7.401
      }
    },
    {
      "rank": 2,
      "track_id": "4RY96Asd9IefaL3X4LOLZ8",
      "track_name": "In Da Club",
      "artist_name": "50 Cent",
      "artist_genres": "hip hop, gangster rap, east coast hip hop, rap, queens hip hop, pop rap",
      "popularity": 75,
      "similarity_score": 0.8806,
      "mmr_score": null,
      "spotify_embed_url": "https://open.spotify.com/embed/track/4RY96Asd9IefaL3X4LOLZ8",
      "audio_features": {
        "danceability": 0.904,
        "energy": 0.72,
        "valence": 0.805,
        "tempo": 90.0,
        "acousticness": 0.259,
        "instrumentalness": 0.0,
        "liveness": 0.0749,
        "speechiness": 0.348,
        "loudness": -2.776
      }
    }
  ],
  "latency_ms": 4.79
}
```
- **Mã lỗi có thể gặp:**
  - `404 Not Found`: Bất kỳ `track_id` nào trong `track_ids` không tồn tại trong mô hình.
  - `422 Unprocessable Entity`: `track_ids` rỗng (`[]`), `track_ids` vượt quá 50 phần tử, hoặc `k`/`alpha` sai miền giá trị.

---

### 3.6. Gợi ý Top bài hát phổ biến nhất (`Popularity Baseline`)
- **Method & URL:** `GET /api/v1/recommend/popular`
- **Query Params:**
  - `k` (`int`, mặc định `10`, `1 <= k <= 50`): Số lượng bài hát phổ biến nhất cần lấy.
- **Ví dụ Request:** `GET /api/v1/recommend/popular?k=2`
- **Response `200 OK` (`RecommendationResponse`):**
```json
{
  "status": "success",
  "mode": "popular",
  "seed_track": null,
  "seed_tracks": null,
  "recommendations": [
    {
      "rank": 1,
      "track_id": "7qXptgpZGo8AFPhJCBVLFE",
      "track_name": "LINDA",
      "artist_name": "Tito Double P",
      "artist_genres": "corrido",
      "popularity": 82,
      "similarity_score": null,
      "mmr_score": null,
      "spotify_embed_url": "https://open.spotify.com/embed/track/7qXptgpZGo8AFPhJCBVLFE",
      "audio_features": {
        "danceability": 0.911,
        "energy": 0.761,
        "valence": 0.948,
        "tempo": 125.0,
        "acousticness": 0.135,
        "instrumentalness": 0.0,
        "liveness": 0.163,
        "speechiness": 0.0507,
        "loudness": -6.712
      }
    },
    {
      "rank": 2,
      "track_id": "7j74lucZ59vqN67Ipe2ZcY",
      "track_name": "Maneater",
      "artist_name": "Daryl Hall & John Oates",
      "artist_genres": "mellow gold, rock, heartland rock, classic rock, soft rock, yacht rock, singer-songwriter, album rock",
      "popularity": 82,
      "similarity_score": null,
      "mmr_score": null,
      "spotify_embed_url": "https://open.spotify.com/embed/track/7j74lucZ59vqN67Ipe2ZcY",
      "audio_features": {
        "danceability": 0.724,
        "energy": 0.498,
        "valence": 0.823,
        "tempo": 89.0,
        "acousticness": 0.0405,
        "instrumentalness": 0.0001,
        "liveness": 0.0761,
        "speechiness": 0.0423,
        "loudness": -13.083
      }
    }
  ],
  "latency_ms": 1.09
}
```

---

### 3.7. Gợi ý bài hát theo Tâm trạng (`Mood-Based Recommendation`)
- **Method & URL:** `GET /api/v1/recommend/mood/{mood_name}`
- **Path Params:**
  - `mood_name` (`string`, bắt buộc): Hỗ trợ 6 tâm trạng chuẩn (không phân biệt hoa thường):
    - `happy` (Vui tươi, sôi nổi)
    - `sad` (Trầm buồn, sâu lắng)
    - `calm` (Êm dịu, thư thái)
    - `chill` (Thư giãn, acoustic nhẹ nhàng)
    - `energetic` (Bùng nổ năng lượng, vận động/tập luyện)
    - `focus` (Tập trung học tập/làm việc, giàu nhạc cụ không lời)
- **Query Params:**
  - `k` (`int`, mặc định `10`, `1 <= k <= 50`): Số lượng bài hát gợi ý.
- **Ví dụ Request:** `GET /api/v1/recommend/mood/happy?k=2`
- **Response `200 OK` (`RecommendationResponse`):**
```json
{
  "status": "success",
  "mode": "mood_happy",
  "seed_track": null,
  "seed_tracks": null,
  "recommendations": [
    {
      "rank": 1,
      "track_id": "1yWoGDkJZNnjibNjpbS8LM",
      "track_name": "Asibolanga",
      "artist_name": "Mzbel",
      "artist_genres": "hiplife, azonto",
      "popularity": 7,
      "similarity_score": 0.9927,
      "mmr_score": null,
      "spotify_embed_url": "https://open.spotify.com/embed/track/1yWoGDkJZNnjibNjpbS8LM",
      "audio_features": {
        "danceability": 0.914,
        "energy": 0.894,
        "valence": 0.956,
        "tempo": 125.0,
        "acousticness": 0.031,
        "instrumentalness": 0.0004,
        "liveness": 0.0924,
        "speechiness": 0.0634,
        "loudness": -3.235
      }
    },
    {
      "rank": 2,
      "track_id": "1xnht7Yhl694jLR2nZKzKZ",
      "track_name": "Kalos Sas Vrika - Mixed",
      "artist_name": "Giorgos Mazonakis",
      "artist_genres": "laiko, greek pop",
      "popularity": 7,
      "similarity_score": 0.9908,
      "mmr_score": null,
      "spotify_embed_url": "https://open.spotify.com/embed/track/1xnht7Yhl694jLR2nZKzKZ",
      "audio_features": {
        "danceability": 0.844,
        "energy": 0.895,
        "valence": 0.911,
        "tempo": 128.0,
        "acousticness": 0.144,
        "instrumentalness": 0.0,
        "liveness": 0.135,
        "speechiness": 0.0426,
        "loudness": -4.436
      }
    }
  ],
  "latency_ms": 3.19
}
```
- **Response `400 Bad Request`:** Khi `mood_name` không thuộc danh sách hỗ trợ (`GET /api/v1/recommend/mood/invalid_mood?k=5`):
```json
{
  "detail": "Unsupported mood: 'invalid_mood'. Supported moods: happy, sad, calm, chill, energetic, focus"
}
```

---

## 4. Bảng Tổng Hợp Mã Trạng Thái HTTP & Cấu Trúc Body Lỗi

| HTTP Code | Kiểu của trường `detail` | Ý nghĩa & Khi nào xảy ra | Cách xử lý khuyến nghị ở Frontend |
| :---: | :---: | :--- | :--- |
| **`200 OK`** | — | Truy vấn hợp lệ | Render danh sách bài hát, Spotify Embed và Radar Chart |
| **`400 Bad Request`** | `string` | `mood_name` không thuộc 6 tâm trạng hỗ trợ | Hiển thị trực tiếp chuỗi `response.detail` lên Toast cảnh báo |
| **`404 Not Found`** | `string` | `track_id` không tồn tại trong dữ liệu hoặc URL không khớp route | Hiển thị trạng thái Empty/Not Found kèm chuỗi `response.detail` |
| **`422 Unprocessable Entity`** | `array[object]` | Lỗi kiểm định schema Pydantic v2: thiếu `q`, `strategy` không hợp lệ, `track_ids` rỗng hoặc `> 50` phần tử, `k`/`limit`/`alpha`/`mmr_lambda` ngoài khoảng cho phép | Duyệt mảng `response.detail`, đọc `loc` và `msg` để báo lỗi trường đầu vào |

### Ví dụ thực tế khác biệt giữa Body lỗi `400`/`404` và `422`:

- **Lỗi `400 Bad Request` / `404 Not Found` (`detail` là `string`):**
```json
{
  "detail": "Track ID not found: non_existent_id_999"
}
```

- **Lỗi `422 Unprocessable Entity` (`detail` là `array[object]` — ví dụ `GET /api/v1/recommend/track/7iL6o9tox1zgHpKUfh9vuC?k=0&strategy=invalid_strategy`):**
```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": [
        "query",
        "k"
      ],
      "msg": "Input should be greater than or equal to 1",
      "input": "0",
      "ctx": {
        "ge": 1
      }
    },
    {
      "type": "literal_error",
      "loc": [
        "query",
        "strategy"
      ],
      "msg": "Input should be 'audio', 'hybrid' or 'mmr'",
      "input": "invalid_strategy",
      "ctx": {
        "expected": "'audio', 'hybrid' or 'mmr'"
      }
    }
  ]
}
```
