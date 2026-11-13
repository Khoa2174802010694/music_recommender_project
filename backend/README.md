# Embeat Music Recommender — Backend API Server

Tầng Backend RESTful API được xây dựng bằng **FastAPI** và **Pydantic v2**, đóng vai trò cầu nối giữa mô hình học máy [`EmbeatRecommender`](../src/recommender.py) và giao diện Web Frontend.

---

## 1. Cấu Trúc Thư Mục `backend/`

```text
backend/
├── __init__.py          # Khởi tạo package backend
├── app.py               # FastAPI application, lifespan loader & 7 REST endpoints
├── schemas.py           # Pydantic v2 models (AudioFeatures, TrackItem, Responses...)
├── README.md            # Tài liệu hướng dẫn cài đặt và vận hành Backend
└── tests/
    ├── __init__.py
    └── test_api.py      # Bộ kiểm thử tự động 11 kịch bản API bằng pytest & TestClient
```

---

## 2. Cài Đặt Môi Trường

Đảm bảo bạn đang đứng tại thư mục gốc của dự án (`music_recommender_project/`) và sử dụng Python 3.10+:

```bash
pip install -r requirements.txt
```

Các thư viện cốt lõi phục vụ Backend bao gồm:
- `fastapi>=0.104.0`
- `uvicorn[standard]>=0.24.0`
- `pydantic>=2.4.0`
- `pytest>=7.4.0` & `httpx>=0.25.0`

---

## 3. Cơ Chế Nạp Model Thông Minh (`lifespan`)

Khi khởi động, hàm `lifespan` trong `backend/app.py` chỉ nạp mô hình **1 lần duy nhất** vào bộ nhớ RAM theo thứ tự ưu tiên:

1. **Chế độ Chính (`mode = "artifact_joblib"`):**
   - Tự động kiểm tra và nạp file `artifacts/embeat_recommender.joblib` thông qua `EmbeatRecommender.load_artifact(...)`.
   - Bao gồm đầy đủ `StandardScaler`, ma trận đặc trưng âm học (`feature_matrix`), ma trận thể loại thưa (`genre_matrix` CSR) và bảng ánh xạ tra cứu $O(1)$ (`id_to_idx`).
2. **Chế độ Dự phòng (`mode = "fallback_csv"`):**
   - Nếu máy chưa có file `.joblib`, server tự động fallback đọc `data/processed/balanced_eda.csv` (hoặc `data/processed/embeat_sample_tracks.csv` / `sample_tracks.csv`) và gọi `EmbeatRecommender().fit(df)` ngay khi khởi động để đảm bảo API luôn sẵn sàng hoạt động.
3. **Tự động thích ứng cột `artist_id`:**
   - Khi gọi gợi ý với `exclude_same_artist=True`, nếu tập dữ liệu fallback không có cột `artist_id`, Backend sẽ tự động bắt ngoại lệ `ValueError` và thực thi lại với `exclude_same_artist=False`.

---

## 4. Khởi Chạy Server FastAPI

Chạy lệnh sau từ thư mục gốc của dự án:

```bash
uvicorn backend.app:app --reload --port 8000
```

Sau khi server khởi chạy thành công:
- **Base API URL:** `http://localhost:8000`
- **Swagger UI (Tài liệu & Test API trực tiếp):** `http://localhost:8000/docs`
- **ReDoc UI:** `http://localhost:8000/redoc`
- **Health Check nhanh:** `http://localhost:8000/api/v1/health`

---

## 5. Danh Sách Endpoints Chính

| Method | Endpoint | Mô tả ngắn |
| :---: | :--- | :--- |
| `GET` | `/api/v1/health` (hoặc `/health`) | Kiểm tra trạng thái server, chế độ model và tổng số bài hát |
| `GET` | `/api/v1/search?q=...&limit=15` | Tìm kiếm bài hát theo tên bài/nghệ sĩ kèm đủ 9 đặc trưng âm học |
| `GET` | `/api/v1/tracks/{track_id}` | Lấy chi tiết metadata và 9 chỉ số âm học của 1 bài hát |
| `GET` | `/api/v1/recommend/track/{track_id}` | Gợi ý Top-K bài tương đồng theo chiến lược `audio`, `hybrid`, hoặc `mmr` |
| `POST` | `/api/v1/recommend/profile` | Gợi ý Top-K bài hát dựa trên danh sách nhiều bài yêu thích (`track_ids`) |
| `GET` | `/api/v1/recommend/popular?k=10` | Gợi ý Top-K bài hát có độ phổ biến cao nhất (Cold-start baseline) |
| `GET` | `/api/v1/recommend/mood/{mood_name}?k=10` | Gợi ý bài hát theo tâm trạng (`happy`, `sad`, `calm`, `chill`, `energetic`, `focus`) |

Chi tiết đầy đủ về Request/Response JSON được mô tả tại [`docs/api_contract.md`](../docs/api_contract.md).

---

## 6. Chạy Kiểm Thử Tự Động (`pytest`)

Để kiểm tra toàn bộ 11 kịch bản tích hợp API (Health với `artifact_path` tương đối, Search & làm tròn 4 chữ số thập phân `audio_features`, Track Details & 404, Audio/Hybrid/MMR Strategies kèm `mmr_score` & `similarity_score` không âm, Taste Profile kèm loại trùng & 404, Popularity, Mood Filtering & 400, cùng kiểm định lỗi 422), chạy lệnh:

```bash
pytest backend/tests/test_api.py -v
```
