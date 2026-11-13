# Kế Hoạch Debug & Phân Chia Nhiệm Vụ (TV2 & TV3)

> Tài liệu tổng hợp toàn bộ các bug cần khắc phục, tối ưu hoá và phân chia trách nhiệm cụ thể giữa **Thành viên 2 (Minh Khoa - ML Engineer)** và **Thành viên 3 (Nhật Minh Phạm - Backend / API Engineer)**, liên kết trực tiếp với notebook [04_evaluate_model.ipynb](file:///d:/music_recommender_projectv2/notebooks/04_evaluate_model.ipynb) và pipeline triển khai của hệ thống.

---

## 2. Thành viên 2 — Minh Khoa (ML Engineer | Task 2.1–2.3)

*Trọng tâm giao việc ngay: Gồm 7 bug mức P0/P1 và 3 đầu việc P2.*

### Danh sách Bug P0 / P1
- [x] **[BUG-01] (P0 — Chủ trì Task 2.2) — ĐÃ BỔ SUNG CƠ CHẾ, CHƯA KIỂM THỬ Ở QUY MÔ 100K–300K:**
  - **Vị trí:** [src/train.py](file:///d:/music_recommender_projectv2/src/train.py)
  - **Mô tả & giải pháp:** Đã bổ sung cơ chế chọn tối đa `max_tracks` theo `popularity` khi finalize incremental training để tạo catalog compact. Tuy nhiên, lần chạy hiện tại chỉ có `29.985` tracks; chưa có dữ liệu đủ lớn để kiểm thử chọn `100k–300k` tracks, dung lượng artifact dưới `100 MB` hoặc xác nhận hiệu quả trên catalog mục tiêu. Vì vậy, mới xác nhận cơ chế đã được bổ sung, chưa xác nhận mục tiêu quy mô và dung lượng.
- [x] **[BUG-03] (P0 — Chủ trì Task 2.1) — ĐÃ KIỂM THỬ:**
  - **Vị trí:** [src/recommender.py](file:///d:/music_recommender_projectv2/src/recommender.py)
  - **Mô tả & giải pháp:** Đã dùng `StandardScaler.partial_fit()` theo batch và ghi batch ra Parquet tạm, không tích lũy `self._pending_batches` hoặc dùng `pd.concat`. Notebook 03 chạy thành công với 3 batches và finalize 29.985 tracks. **Giới hạn:** sau finalize, DataFrame và các ma trận đặc trưng vẫn được nạp vào RAM để phục vụ backend hiện tại; chưa phải pipeline out-of-core hoàn toàn.
- [x] **[BUG-06] (P1 — Phối hợp Task 2.2) — ĐÃ KIỂM TRA:**
  - **Vị trí:** [src/train.py](file:///d:/music_recommender_projectv2/src/train.py)
  - **Mô tả & giải pháp:** Đã xử lý chuẩn popularity theo thang `0–1` và `0–100` trong preprocessing. Kiểm thử với ngưỡng `0.5`: từ 5.000 dòng còn 1.673 tracks, tất cả đều đạt ngưỡng. 
- [x] **[BUG-07] (P1 — Chủ trì Task 2.1) — ĐÃ KIỂM THỬ:**
  - **Vị trí:** [src/recommender.py](file:///d:/music_recommender_projectv2/src/recommender.py)
  - **Mô tả & giải pháp:** Đã bổ sung `_normalize_similarity()` để chuẩn hóa điểm similarity đầu ra về `[0.0, 1.0]`. Đã kiểm tra các luồng audio, hybrid, MMR và mood, kể cả top-k lớn.
- [x] **[BUG-08] (P1 — Chủ trì Task 2.1) — ĐÃ TỐI ƯU:**
  - **Vị trí:** [src/recommender.py](file:///d:/music_recommender_projectv2/src/recommender.py)
  - **Mô tả & giải pháp:** Đã tối ưu các thao tác tìm kiếm và lấy Top-K theo hướng giảm chi phí trên catalog:
    - Precompute mảng index sắp xếp theo `popularity` khi `fit()` / `load_artifact()`.
    - Thay `np.argsort` thành `np.argpartition` (`O(N)`) trong `_rank_scores()` cho việc lấy Top-K.
    - Tối ưu `search_tracks()`: tránh tạo nhiều bản sao Series `.fillna("").str.contains()` lặp lại và sắp xếp kết quả theo `popularity`.
- [x] **[BUG-09] (P1 — Chủ trì Task 2.1) — ĐÃ KIỂM THỬ CÁC TRƯỜNG HỢP THIẾU:**
  - **Vị trí:** [src/recommender.py](file:///d:/music_recommender_projectv2/src/recommender.py)
  - **Mô tả & giải pháp:** Đã bổ sung xử lý artist/genre thiếu và fallback theo `artist_name` khi không có `artist_id` khả dụng; tránh xem các giá trị thiếu là cùng nghệ sĩ; xử lý an toàn khi genre matrix không khả dụng. Colab test `MISSING ARTIST/GENRE TEST: PASS`. 
- [x] **[BUG-11] (P1 — Chủ trì Task 2.2 / 2.3) — ĐÃ KIỂM THỬ:**
  - **Vị trí:** [notebooks/04_evaluate_model.ipynb](file:///d:/music_recommender_projectv2/notebooks/04_evaluate_model.ipynb) và [notebooks/03_train_model.ipynb](file:///d:/music_recommender_projectv2/notebooks/03_train_model.ipynb)
  - **Mô tả & giải pháp:** 
    - Notebook 04 đã bỏ phụ thuộc vào đường dẫn Colab cố định, tự dò project root và lưu/đọc dashboard tại `reports/figures/evaluation_dashboard.png`.
    - Notebook 03 đã đồng bộ tên metadata theo `<artifact_stem>_metadata.json`; kiểm tra metadata và tính portable đều PASS.
    - Sửa [notebooks/04_evaluate_model.ipynb](file:///d:/music_recommender_projectv2/notebooks/04_evaluate_model.ipynb) bỏ hardcode `/content/drive/MyDrive/music_recommender_project` và trỏ đúng ảnh dashboard ở `reports/figures/evaluation_dashboard.png`.
    - Đồng bộ tên file metadata trong [notebooks/03_train_model.ipynb](file:///d:/music_recommender_projectv2/notebooks/03_train_model.ipynb) (`training_metadata.json` vs `*_metadata.json` do [src/train.py](file:///d:/music_recommender_projectv2/src/train.py) sinh ra) và tạo lại metadata không chứa đường dẫn Google Colab.

> **Ghi chú trạng thái:** Các checkbox của TV2 được cập nhật theo code và kết quả notebook hiện có. BUG-01 chỉ xác nhận đã bổ sung cơ chế compact; chưa xác nhận mục tiêu 100k–300k tracks và artifact dưới 100 MB do dataset thử nghiệm hiện tại chỉ có 29.985 tracks. Các đầu việc Backend/API của TV3 bên dưới giữ nguyên trạng thái để TV3 tự xác nhận.

### 3 Đầu Việc P2 Thuộc TV2
- [x] **P2-1 — ĐÃ KIỂM THỬ:** `candidate_pool=None` tự tính theo `top_n * 3`; đã kiểm tra các mức top-k khác nhau, không còn mặc định cứng 50.
- [x] **P2-2 — ĐÃ KIỂM TRA:** Đã bỏ lần gọi `build_genre_matrix(model.df)` trùng trong [src/train.py](file:///d:/music_recommender_projectv2/src/train.py); `model.fit()` tự xây dựng genre matrix và xử lý trường hợp thiếu `artist_genres`.
- [x] **P2-3 & P2-5 — ĐÃ KIỂM TRA VÀ CHẠY EVALUATION:** Đã bổ sung kiểm tra DataFrame rỗng; đồng bộ output `search_tracks()` giữa hai chế độ; xử lý vector genre bằng ndarray thay cho `np.matrix`; notebook 04 đã chạy lại và tạo `metrics.json`, `evaluation_summary.csv`, `evaluation_details.csv` cùng dashboard.

---

## 3. Thành viên 3 — Nhật Minh Phạm (Backend / API Engineer | Task 3.1–3.4)

### Danh sách Nhiệm Vụ & Phối Hợp Sửa Lỗi
- [ ] **[BUG-01] (P0 — Chủ trì Task 3.3):**
  - **Vị trí:** [backend/app.py](file:///d:/music_recommender_projectv2/backend/app.py)
  - **Mô tả & giải pháp:** Cập nhật nạp `DATA_SCALE_MODE` (từ file cấu hình `.env`), ưu tiên nạp Compact artifact thay vì hardcode duy nhất file `3.816 GB`.
  - Không chạy `_load_model_once()` đồng bộ trực tiếp trên event loop của `async def lifespan()` (chuyển sang threadpool / worker task) và ghi log / xử lý lỗi rõ ràng thay vì nuốt ngoại lệ `MemoryError`.
- [ ] **[BUG-06] (P1 — Chủ trì Task 3.3):**
  - **Vị trí:** [backend/app.py](file:///d:/music_recommender_projectv2/backend/app.py)
  - **Mô tả & giải pháp:** Sửa hàm `_normalize_popularity()`: xử lý đúng trường hợp giá trị nguyên `0`/`1` khi `popularity_is_unit_scale=True`, không nhân nhầm `100` với số thập phân `(0.0, 1.0)` khi `popularity_is_unit_scale=False`, clamp giá trị trong khoảng `[0, 100]` và loại bỏ phụ thuộc global state dễ gây lệch khi swap model trong test.
- [ ] **[BUG-07] (P1 — Phối hợp Task 3.1 / 3.3 / 3.4):**
  - **Vị trí:** [docs/api_contract.md](file:///d:/music_recommender_projectv2/docs/api_contract.md), [backend/app.py](file:///d:/music_recommender_projectv2/backend/app.py), [backend/tests/test_api.py](file:///d:/music_recommender_projectv2/backend/tests/test_api.py)
  - **Mô tả & giải pháp:** Đồng bộ tài liệu và assertions kiểm thử theo quy chuẩn `similarity_score` luôn nằm trong `[0.0, 1.0]` thống nhất cùng TV2.
- [ ] **[BUG-09] (P1 — Phối hợp Task 3.3):**
  - **Vị trí:** [backend/app.py](file:///d:/music_recommender_projectv2/backend/app.py)
  - **Mô tả & giải pháp:** Gỡ bỏ workaround tắt `exclude_same_artist=False` tại API backend sau khi TV2 đã hỗ trợ fallback tự lọc theo `artist_name`.
- [ ] **P2-1 (Phối hợp Task 3.3):**
  - **Vị trí:** [backend/app.py](file:///d:/music_recommender_projectv2/backend/app.py)
  - **Mô tả & giải pháp:** Truyền `candidate_pool` đủ lớn (ví dụ `max(k * 3, 150)`) khi gọi `recommend_hybrid_mmr()` để thuật toán MMR không bị mất tác dụng lọc đa dạng hóa khi người dùng yêu cầu `k=50`.
- [ ] **P2-6 (Chủ trì Task 3.4):**
  - **Vị trí:** [backend/tests/test_api.py](file:///d:/music_recommender_projectv2/backend/tests/test_api.py)
  - **Mô tả & giải pháp:** 
    - Tách test suite khỏi sự phụ thuộc vào file `artifacts/embeat_recommender.joblib` `3.816 GB` trên đĩa (dùng fixture CSV mẫu hoặc model nhỏ tạm thời để chạy CI mượt mà).
    - Bổ sung các test cases còn thiếu: kiểm tra chế độ `fallback_csv`, các giá trị biên của `_normalize_popularity` (`0`, `1`, `0.0`, `1.0`, `NaN`), giá trị `±Inf` trong các đặc trưng âm học, endpoint `/health`, `exclude_same_artist=False`, tìm kiếm với chuỗi rỗng / không ra kết quả, và kiểm thử toàn diện 5 mood còn lại (`sad`, `calm`, `chill`, `energetic`, `focus`).

