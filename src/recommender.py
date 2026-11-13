import numpy as np
import pandas as pd
import joblib
import tempfile

from pathlib import Path
from scipy.sparse import csr_matrix

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

from src.preprocess import (
    AUDIO_FEATURES,
    clean_tracks,
    fit_scaler
)


class EmbeatRecommender:
    """
    Embeat Music Recommendation Model.

    Supported methods:
    - Audio-only Recommendation
    - Hybrid Recommendation
    - Hybrid + MMR Recommendation
    - Taste Profile Recommendation
    - Popularity-based Recommendation
    """

    def __init__(self):
        self.df = None
        self.scaler = None
        self.feature_matrix = None
        self.id_to_idx = {}
        self.genre_matrix = None

        # Cached indexes for fast popularity ranking and track search.
        self._popularity_scores = np.array([], dtype=float)
        self._popularity_order = np.array([], dtype=int)
        self._popularity_rank = np.array([], dtype=int)
        self._track_name_search = np.array([], dtype=str)
        self._artist_name_search = np.array([], dtype=str)

        # Incremental-training state: batches are spooled to disk, not kept in RAM.
        self._incremental_tmp = None
        self._incremental_path = None
        self._incremental_writer = None
        self._incremental_schema = None
        self._incremental_scaler = None
        self._incremental_seen_ids = set()
        self._incremental_batch_count = 0
        self._incremental_track_count = 0

    # ==========================================
    # 1. MODEL TRAINING
    # ==========================================

    def _build_search_and_popularity_indexes(self):
        """
        Precompute arrays used by search and popularity ranking.

        This avoids rebuilding pandas Series/string conversions and sorting
        the complete catalog on every request.
        """
        if self.df is None:
            self._popularity_scores = np.array([], dtype=float)
            self._popularity_order = np.array([], dtype=int)
            self._popularity_rank = np.array([], dtype=int)
            self._track_name_search = np.array([], dtype=str)
            self._artist_name_search = np.array([], dtype=str)
            return

        row_count = len(self.df)

        if "popularity" in self.df.columns:
            popularity = pd.to_numeric(
                self.df["popularity"], errors="coerce"
            ).to_numpy(dtype=float)
            popularity[~np.isfinite(popularity)] = -np.inf
        else:
            popularity = np.full(row_count, -np.inf, dtype=float)

        self._popularity_scores = popularity
        self._popularity_order = np.argsort(
            -popularity, kind="stable"
        ).astype(int)

        ranks = np.empty(row_count, dtype=int)
        ranks[self._popularity_order] = np.arange(row_count)
        self._popularity_rank = ranks

        if "track_name" in self.df.columns:
            self._track_name_search = (
                self.df["track_name"].fillna("").astype(str)
                .str.casefold().to_numpy(dtype=str)
            )
        else:
            self._track_name_search = np.full(row_count, "", dtype=str)

        if "artist_name" in self.df.columns:
            self._artist_name_search = (
                self.df["artist_name"].fillna("").astype(str)
                .str.casefold().to_numpy(dtype=str)
            )
        else:
            self._artist_name_search = np.full(row_count, "", dtype=str)

    def _reset_incremental_state(self):
        """Close and remove temporary files used by incremental training."""
        writer = getattr(self, "_incremental_writer", None)
        if writer is not None:
            writer.close()
        tmp = getattr(self, "_incremental_tmp", None)
        if tmp is not None:
            tmp.cleanup()
        self._incremental_tmp = None
        self._incremental_path = None
        self._incremental_writer = None
        self._incremental_schema = None
        self._incremental_scaler = None
        self._incremental_seen_ids = set()
        self._incremental_batch_count = 0
        self._incremental_track_count = 0

    def fit(self, df):
        """
        Clean data and train the audio feature scaler.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Cannot fit recommender: input DataFrame is empty.")

        required_columns = ["track_id", *AUDIO_FEATURES]
        missing_columns = [
            column for column in required_columns
            if column not in df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Cannot fit recommender. Missing required columns: {missing_columns}"
            )

        self._reset_incremental_state()
        self.df = clean_tracks(df)

        if self.df.empty:
            raise ValueError(
                "Cannot fit recommender: no valid tracks remain after clean_tracks()."
            )

        self.scaler, self.feature_matrix = fit_scaler(
            self.df
        )

        self.id_to_idx = {
            track_id: idx
            for idx, track_id in enumerate(
                self.df["track_id"]
            )
        }

        # Build genre matrix automatically if artist_genres is present,
        # otherwise reset it to None.
        if "artist_genres" in self.df.columns:
            genre_list = (
                self.df["artist_genres"]
                .fillna("")
                .apply(
                    lambda x: [
                        genre.strip().lower()
                        for genre in str(x).split(",")
                        if genre.strip()
                    ]
                )
            )
            mlb = MultiLabelBinarizer(sparse_output=True)
            self.genre_matrix = csr_matrix(mlb.fit_transform(genre_list))
        else:
            # Genre metadata is optional. An empty CSR matrix preserves the
            # artifact contract; hybrid methods fall back to audio similarity.
            self.genre_matrix = csr_matrix((len(self.df), 0), dtype=np.float32)

        self._build_search_and_popularity_indexes()
        return self


    def fit_incremental(
        self,
        df: pd.DataFrame,
        is_first_batch: bool = False
    ) -> None:
        """
        Update StandardScaler batch-by-batch and spool cleaned rows to disk.
        No DataFrame batch list is retained in memory.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if not isinstance(is_first_batch, bool):
            raise TypeError("is_first_batch must be a boolean.")
        if is_first_batch:
            self._reset_incremental_state()
        if df.empty:
            return None

        required_columns = ["track_id", *AUDIO_FEATURES]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        batch = clean_tracks(df)
        if not batch.empty:
            batch = batch.loc[
                ~batch["track_id"].isin(self._incremental_seen_ids)
            ].copy()
            self._incremental_seen_ids.update(
                batch["track_id"].dropna().tolist()
            )
        if batch.empty:
            return None

        if self._incremental_scaler is None:
            from sklearn.preprocessing import StandardScaler
            self._incremental_scaler = StandardScaler()
        self._incremental_scaler.partial_fit(batch[AUDIO_FEATURES])

        if self._incremental_tmp is None:
            self._incremental_tmp = tempfile.TemporaryDirectory(
                prefix="embeat_incremental_"
            )
            self._incremental_path = (
                Path(self._incremental_tmp.name) / "tracks.parquet"
            )

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "Incremental training requires pyarrow. "
                "Install it with: pip install pyarrow"
            ) from exc

        table = pa.Table.from_pandas(batch, preserve_index=False)
        if self._incremental_writer is None:
            self._incremental_schema = table.schema
            self._incremental_writer = pq.ParquetWriter(
                self._incremental_path,
                self._incremental_schema,
                compression="snappy"
            )
        else:
            table = table.cast(self._incremental_schema)
        self._incremental_writer.write_table(table)

        self._incremental_batch_count += 1
        self._incremental_track_count += len(batch)
        return None

    def finalize_incremental_training(self, max_tracks=None):
        """
        Finalize the incrementally fitted scaler and construct model matrices.
        Metadata is loaded from a single temporary Parquet file, without
        retaining pending DataFrames or using pd.concat().
        """
        if (
            self._incremental_writer is None
            or self._incremental_scaler is None
            or self._incremental_track_count == 0
        ):
            raise ValueError(
                "No pending batches. Call fit_incremental() first."
            )

        self._incremental_writer.close()
        self._incremental_writer = None

        print("=== FINALIZING INCREMENTAL TRAINING ===")
        print("Received batches:", self._incremental_batch_count)
        print("Total unique tracks:", self._incremental_track_count)

        # Read the spooled rows, then apply the same compact-catalog policy
        # as CSV training. The scaler remains fitted on every accepted batch;
        # only the deployed catalog is compacted.
        self.df = pd.read_parquet(self._incremental_path).reset_index(drop=True)

        if max_tracks is not None:
            if (
                not isinstance(max_tracks, (int, np.integer))
                or isinstance(max_tracks, bool)
                or max_tracks <= 0
            ):
                raise ValueError("max_tracks must be a positive integer or None.")

            if "popularity" in self.df.columns:
                popularity = pd.to_numeric(
                    self.df["popularity"], errors="coerce"
                ).fillna(-1)
                self.df = (
                    self.df.assign(_compact_popularity=popularity)
                    .sort_values(
                        "_compact_popularity",
                        ascending=False,
                        kind="mergesort"
                    )
                    .drop(columns="_compact_popularity")
                    .head(int(max_tracks))
                    .reset_index(drop=True)
                )
            else:
                self.df = self.df.head(int(max_tracks)).reset_index(drop=True)

        self.scaler = self._incremental_scaler
        self.feature_matrix = self.scaler.transform(self.df[AUDIO_FEATURES])
        self.id_to_idx = {
            track_id: idx
            for idx, track_id in enumerate(self.df["track_id"])
        }

        if "artist_genres" in self.df.columns:
            genre_list = (
                self.df["artist_genres"]
                .fillna("")
                .apply(
                    lambda value: [
                        genre.strip().lower()
                        for genre in str(value).split(",")
                        if genre.strip()
                    ]
                )
            )
            mlb = MultiLabelBinarizer(sparse_output=True)
            self.genre_matrix = csr_matrix(mlb.fit_transform(genre_list))
        else:
            self.genre_matrix = csr_matrix((len(self.df), 0), dtype=np.float32)

        print("Cleaned tracks:", len(self.df))
        print("Audio matrix:", self.feature_matrix.shape)
        if self.genre_matrix is not None:
            print("Genre matrix:", self.genre_matrix.shape)
        self._build_search_and_popularity_indexes()

        print("INCREMENTAL TRAINING COMPLETED")

        self._reset_incremental_state()
        return self

    def set_genre_matrix(self, genre_matrix, track_ids=None):
        """
        Attach preprocessed genre matrix to the model.

        Parameters
        ----------
        genre_matrix : array-like or sparse matrix
            Preprocessed genre matrix.

        track_ids : array-like, optional
            Track IDs corresponding to each row of genre_matrix.
            Used to validate row ordering.
        """

        if self.df is None:
            raise ValueError("Model has not been fitted.")

        if genre_matrix.shape[0] != len(self.df):
            raise ValueError(
                "Genre matrix and track data have different row counts."
            )

        if track_ids is not None:
            track_ids = list(track_ids)
            model_track_ids = self.df["track_id"].tolist()

            if track_ids != model_track_ids:
                raise ValueError(
                    "Genre matrix Track ID order does not match model data."
                )

        self.genre_matrix = csr_matrix(genre_matrix)

        return self

    # ==========================================
    # 2. INTERNAL HELPERS
    # ==========================================

    def _validate_model(self, require_genre=False, require_artist=False):

        if self.df is None:
            raise ValueError("Model has not been fitted.")

        if self.feature_matrix is None:
            raise ValueError("Audio feature matrix is missing.")

        if require_genre and self.genre_matrix is None:
            raise ValueError("Genre matrix has not been set.")


    def _get_artist_column(self):
        """Prefer artist_id; fall back to artist_name when available."""
        if self.df is None:
            return None
        if "artist_id" in self.df.columns and self.df["artist_id"].notna().any():
            return "artist_id"
        if "artist_name" in self.df.columns and self.df["artist_name"].notna().any():
            return "artist_name"
        return None

    def _get_same_artist_indices(self, query_idx):
        """
        Return indices for the query artist without treating missing values
        as one shared artist. If artist metadata is unavailable, return [].
        """
        artist_column = self._get_artist_column()
        if artist_column is None:
            return []

        query_artist = self.df.iloc[query_idx][artist_column]
        if pd.isna(query_artist):
            return []

        values = self.df[artist_column]
        return np.flatnonzero(values.notna().to_numpy() & values.eq(query_artist).to_numpy()).tolist()

    def _get_seed_artist_indices(self, seed_indices):
        """Return rows belonging to seed artists using ID/name fallback."""
        artist_column = self._get_artist_column()
        if artist_column is None:
            return []

        seed_artists = (
            self.df.iloc[seed_indices][artist_column]
            .dropna()
            .unique()
        )
        if len(seed_artists) == 0:
            return []

        values = self.df[artist_column]
        return np.flatnonzero(
            values.notna().to_numpy()
            & values.isin(seed_artists).to_numpy()
        ).tolist()

    def _genre_similarity(self, query_matrix, catalog_matrix):
        """Compute genre cosine similarity, or zeros when genres are absent."""
        if catalog_matrix is None or catalog_matrix.shape[1] == 0:
            return np.zeros(len(self.df), dtype=float)
        return cosine_similarity(query_matrix, catalog_matrix).flatten()

    def _get_track_index(self, track_id):

        if track_id not in self.id_to_idx:
            raise ValueError(
                f"Track ID not found: {track_id}"
            )

        return self.id_to_idx[track_id]

    
    @staticmethod
    def _normalize_similarity(scores):
        """
        Map cosine similarity from [-1, 1] to [0, 1].
        Clipping protects the public score contract from floating-point drift.
        """
        values = np.asarray(scores, dtype=float)
        return np.clip((values + 1.0) / 2.0, 0.0, 1.0)

    def _get_result(self, indices, scores=None):
        """
        Build recommendation results with track metadata
        and normalized similarity scores in [0, 1].
        """

        metadata_columns = [
            "track_id",
            "track_name",
            "artist_id",
            "artist_name",
            "album_id",
            "album_name",
            "isrc",
            "artist_genres",
            "popularity",
            "explicit",
            "release_year",
            "duration",
            *AUDIO_FEATURES,
        ]

        # Chỉ lấy những cột thực sự tồn tại
        available_columns = [
            col for col in metadata_columns
            if col in self.df.columns
        ]

        result = self.df.iloc[indices][
            available_columns
        ].copy()

        # Bổ sung các điểm similarity
        if scores is not None:
            for column, values in scores.items():
                selected_scores = np.asarray(values, dtype=float)[indices]

                # Normalize cosine-based similarity fields for the API/report.
                # Ranking still uses the original raw cosine values.
                if "similarity" in column.lower():
                    selected_scores = self._normalize_similarity(
                        selected_scores
                    )

                result[column] = selected_scores

        return result.reset_index(drop=True)


    def _rank_scores(self, scores, top_n, excluded_indices=None):
        """
        Return the highest-scoring valid indices without sorting the full
        catalog when only a small top_n is requested.
        """
        scores = np.asarray(scores, dtype=float).reshape(-1)

        if not isinstance(top_n, (int, np.integer)) or isinstance(top_n, bool):
            raise ValueError("top_n must be a positive integer.")
        if top_n < 1:
            raise ValueError("top_n must be a positive integer.")

        valid_mask = np.isfinite(scores)

        if excluded_indices is not None:
            excluded = np.asarray(list(excluded_indices), dtype=int)
            excluded = excluded[(excluded >= 0) & (excluded < len(scores))]
            valid_mask[excluded] = False

        valid_indices = np.flatnonzero(valid_mask)
        if valid_indices.size == 0:
            return []

        k = min(int(top_n), valid_indices.size)
        valid_scores = scores[valid_indices]

        if k == valid_indices.size:
            selected_positions = np.arange(valid_indices.size)
        else:
            selected_positions = np.argpartition(
                valid_scores, valid_scores.size - k
            )[-k:]

        selected_indices = valid_indices[selected_positions]
        order = np.argsort(scores[selected_indices], kind="stable")[::-1]

        return selected_indices[order].tolist()
    # ==========================================
    # 3. TRACK SEARCH AND DETAILS
    # ==========================================

    def search_tracks(
        self,
        query,
        limit=20,
        *,
        as_dict=False
    ):
        """
        Search track/artist names using cached lowercase arrays.
        Results are ordered by popularity without rebuilding pandas Series.
        """
        self._validate_model()

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")
        if (
            not isinstance(limit, (int, np.integer))
            or isinstance(limit, bool)
            or limit < 1
        ):
            raise ValueError("Limit must be a positive integer.")

        query = query.strip().casefold()
        if not query:
            return [] if as_dict else pd.DataFrame()

        # Rebuild lazily for older artifacts created before cache fields existed.
        if (
            len(getattr(self, "_track_name_search", [])) != len(self.df)
            or len(getattr(self, "_artist_name_search", [])) != len(self.df)
            or len(getattr(self, "_popularity_rank", [])) != len(self.df)
        ):
            self._build_search_and_popularity_indexes()

        track_matches = np.char.find(self._track_name_search, query) >= 0
        artist_matches = np.char.find(self._artist_name_search, query) >= 0
        matched_indices = np.flatnonzero(track_matches | artist_matches)

        if matched_indices.size == 0:
            return [] if as_dict else pd.DataFrame()

        # Stable popularity ordering, using the precomputed rank index.
        order = np.argsort(
            self._popularity_rank[matched_indices], kind="stable"
        )
        selected_indices = matched_indices[order[:int(limit)]]

        result_columns = [
            "track_id",
            "track_name",
            "artist_name",
            "artist_genres",
            "popularity",
            *AUDIO_FEATURES
        ]
        result_columns = [
            column for column in result_columns
            if column in self.df.columns
        ]

        result = self.df.iloc[selected_indices][result_columns].reset_index(drop=True)

        if as_dict:
            return result.to_dict(orient="records")

        return result

    def get_track_details(self, track_id):
        """
        Retrieve metadata and audio features of a track.
        """

        self._validate_model()

        idx = self._get_track_index(track_id)

        return self.df.iloc[idx].to_dict()

    # ==========================================
    # 4. AUDIO-ONLY RECOMMENDATION
    # ==========================================

    
    def recommend_by_track(
        self,
        track_id,
        top_n=10,
        exclude_same_artist=False,
        *,
        top_k=None,
        as_dict=False
    ):
        """
        Recommend tracks using audio feature similarity.

        Supports:
        - top_n: Legacy parameter for notebooks.
        - top_k: Standard parameter for Backend.
        - as_dict: Return list of dictionaries when True.
        """

        self._validate_model(
          require_artist=exclude_same_artist
      )

        # Support both top_n and top_k
        if top_k is not None:
            top_n = top_k

        if not isinstance(top_n, (int, np.integer)) or isinstance(top_n, bool) or top_n < 1:
            raise ValueError("top_n/top_k must be a positive integer.")

        query_idx = self._get_track_index(track_id)

        # Calculate audio similarity
        similarities = cosine_similarity(
            self.feature_matrix[query_idx:query_idx + 1],
            self.feature_matrix
        ).flatten()

        # Exclude original track
        excluded = [query_idx]

        # Optionally exclude tracks from the same artist
        if exclude_same_artist:
            excluded.extend(self._get_same_artist_indices(query_idx))

        # Rank recommendations
        top_indices = self._rank_scores(
            similarities,
            top_n,
            excluded
        )

        # Build result
        result = self._get_result(
            top_indices,
            {"similarity": similarities}
        )

        # Support Backend response format
        if as_dict:
            return result.to_dict(orient="records")

        return result


    # ==========================================
    # 5. HYBRID RECOMMENDATION
    # ==========================================

    
    def recommend_hybrid(
        self,
        track_id,
        top_n=10,
        alpha=0.7,
        exclude_same_artist=True,
        *,
        top_k=None,
        as_dict=False
    ):
        """
        Combine audio and genre similarities.

        alpha:
            0.7 = 70% Audio + 30% Genre

        Supports:
            top_n: Legacy parameter.
            top_k: Backend parameter.
            as_dict: Return list of dictionaries.
        """

        self._validate_model(
          require_genre=True,
          require_artist=exclude_same_artist
      )

        # Support both top_n and top_k
        if top_k is not None:
            top_n = top_k

        if not isinstance(top_n, (int, np.integer)) or isinstance(top_n, bool) or top_n < 1:
            raise ValueError("top_n/top_k must be a positive integer.")

        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1.")

        query_idx = self._get_track_index(track_id)

        # Audio similarity
        audio_similarity = cosine_similarity(
            self.feature_matrix[query_idx:query_idx + 1],
            self.feature_matrix
        ).flatten()

        # Genre similarity
        genre_similarity = self._genre_similarity(
            self.genre_matrix[query_idx],
            self.genre_matrix
        )

        # Hybrid similarity
        hybrid_similarity = (
            alpha * audio_similarity
            + (1 - alpha) * genre_similarity
        )

        # Exclude original track and optionally same artist
        excluded = [query_idx]

        if exclude_same_artist:
            excluded.extend(self._get_same_artist_indices(query_idx))

        # Rank recommendations
        top_indices = self._rank_scores(
            hybrid_similarity,
            top_n,
            excluded
        )

        # Build result
        result = self._get_result(
            top_indices,
            {
                "audio_similarity": audio_similarity,
                "genre_similarity": genre_similarity,
                "hybrid_similarity": hybrid_similarity
            }
        )

        if as_dict:
            return result.to_dict(orient="records")

        return result


    # ==========================================
    # 6. HYBRID + MMR
    # ==========================================


    def recommend_hybrid_mmr(
        self,
        track_id,
        top_n=10,
        alpha=0.7,
        mmr_lambda=0.7,
        candidate_pool=None,
        exclude_same_artist=True,
        *,
        top_k=None,
        as_dict=False
    ):
        """
        Hybrid recommendation with Maximal Marginal Relevance.

        MMR balances relevance and diversity.

        Parameters:
            top_n: Legacy parameter.
            top_k: Backend parameter.
            alpha: Audio/Genre weighting.
            mmr_lambda: Relevance/Diversity weighting.
            candidate_pool: Number of initial candidates.
            exclude_same_artist: Exclude query artist.
            as_dict: Return list of dictionaries.
        """

        self._validate_model(
          require_genre=True,
          require_artist=exclude_same_artist
      )

        # Support both top_n and top_k
        if top_k is not None:
            top_n = top_k

        if not isinstance(top_n, (int, np.integer)) or isinstance(top_n, bool) or top_n < 1:
            raise ValueError("top_n/top_k must be a positive integer.")

        if candidate_pool is None:
            candidate_pool = top_n * 3

        if (
            not isinstance(candidate_pool, (int, np.integer))
            or isinstance(candidate_pool, bool)
            or candidate_pool < 1
        ):
            raise ValueError("candidate_pool must be a positive integer.")

        # Đảm bảo số ứng viên không nhỏ hơn số kết quả cần trả về.
        candidate_pool = max(candidate_pool, top_n)

        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1.")

        if not 0 <= mmr_lambda <= 1:
            raise ValueError("MMR lambda must be between 0 and 1.")

        query_idx = self._get_track_index(track_id)

        # 1. Audio similarity
        audio_similarity = cosine_similarity(
            self.feature_matrix[query_idx:query_idx + 1],
            self.feature_matrix
        ).flatten()

        # 2. Genre similarity
        genre_similarity = self._genre_similarity(
            self.genre_matrix[query_idx],
            self.genre_matrix
        )

        # 3. Hybrid similarity
        hybrid_similarity = (
            alpha * audio_similarity
            + (1 - alpha) * genre_similarity
        )

        # 4. Exclude original track and optionally same artist
        excluded = [query_idx]

        if exclude_same_artist:
            excluded.extend(self._get_same_artist_indices(query_idx))

        # 5. Select candidate pool
        candidates = self._rank_scores(
            hybrid_similarity,
            candidate_pool,
            excluded
        )

        if not candidates:
            return [] if as_dict else pd.DataFrame()

        # 6. Calculate pairwise similarity
        audio_pairwise = cosine_similarity(
            self.feature_matrix[candidates]
        )

        if self.genre_matrix is None or self.genre_matrix.shape[1] == 0:
            # No genre features: MMR diversity is computed from audio only.
            genre_pairwise = np.zeros(
                (len(candidates), len(candidates)),
                dtype=float
            )
        else:
            genre_pairwise = cosine_similarity(
                self.genre_matrix[candidates]
            )

        pairwise_similarity = (
            alpha * audio_pairwise
            + (1 - alpha) * genre_pairwise
        )

        # 7. Greedy MMR selection
        selected = []
        selected_scores = []

        remaining = list(range(len(candidates)))

        while remaining and len(selected) < top_n:

            best_position = None
            best_score = -np.inf

            for pos in remaining:

                relevance = hybrid_similarity[
                    candidates[pos]
                ]

                if selected:
                    redundancy = max(
                        pairwise_similarity[pos, previous]
                        for previous in selected
                    )
                else:
                    redundancy = 0

                score = (
                    mmr_lambda * relevance
                    - (1 - mmr_lambda) * redundancy
                )

                if score > best_score:
                    best_score = score
                    best_position = pos

            selected.append(best_position)
            selected_scores.append(best_score)

            remaining.remove(best_position)

        # 8. Build result
        selected_indices = [
            candidates[pos] for pos in selected
        ]

        result = self._get_result(
            selected_indices,
            {
                "audio_similarity": audio_similarity,
                "genre_similarity": genre_similarity,
                "hybrid_similarity": hybrid_similarity
            }
        )

        result["mmr_score"] = selected_scores

        if as_dict:
            return result.to_dict(orient="records")

        return result



    # ==========================================
    # 7. TASTE PROFILE RECOMMENDATION
    # ==========================================

    def recommend_by_taste_profile(
        self,
        track_ids,
        top_n=10,
        alpha=0.7,
        exclude_same_artist=False,
        *,
        top_k=None,
        as_dict=False
    ):
        """
        Recommend tracks based on multiple
        favorite tracks supplied by the user.

        Supports:
            top_n: Legacy Notebook parameter.
            top_k: Backend API parameter.
            as_dict: Return list of dictionaries.
        """

        self._validate_model(
          require_genre=True,
          require_artist=exclude_same_artist
      )

        # Support both top_n and top_k
        if top_k is not None:
            top_n = top_k

        if (
            not isinstance(top_n, (int, np.integer))
            or isinstance(top_n, bool)
            or top_n < 1
        ):
            raise ValueError(
                "top_n/top_k must be a positive integer."
            )

        if isinstance(track_ids, str):
            track_ids = [track_ids]

        if track_ids is None or len(track_ids) == 0:
            raise ValueError(
                "At least one track ID is required."
            )

        if not 0 <= alpha <= 1:
            raise ValueError(
                "Alpha must be between 0 and 1."
            )

        seed_indices = list(dict.fromkeys(
            self._get_track_index(track_id)
            for track_id in track_ids
        ))

        # Average audio preference
        audio_profile = np.asarray(
            self.feature_matrix[seed_indices].mean(axis=0)
        ).reshape(1, -1)

        audio_similarity = cosine_similarity(
            audio_profile,
            self.feature_matrix
        ).flatten()

        # Average genre preference
        genre_mean = self.genre_matrix[seed_indices].mean(axis=0)
        genre_profile = csr_matrix(
            np.asarray(genre_mean, dtype=np.float32).reshape(1, -1)
        )

        genre_similarity = self._genre_similarity(
            genre_profile,
            self.genre_matrix
        )

        # Hybrid similarity
        hybrid_similarity = (
            alpha * audio_similarity
            + (1 - alpha) * genre_similarity
        )

        # Exclude original tracks
        excluded = list(seed_indices)

        # Exclude tracks from seed artists
        if exclude_same_artist:
            excluded.extend(self._get_seed_artist_indices(seed_indices))

        # Rank recommendations
        top_indices = self._rank_scores(
            hybrid_similarity,
            top_n,
            excluded
        )

        result = self._get_result(
            top_indices,
            {
                "audio_similarity": audio_similarity,
                "genre_similarity": genre_similarity,
                "hybrid_similarity": hybrid_similarity
            }
        )

        if as_dict:
            return result.to_dict(orient="records")

        return result


    # ==========================================
    # 8. MOOD-BASED RECOMMENDATION
    # ==========================================

    def recommend_by_mood(
        self,
        mood,
        top_n=10,
        *,
        top_k=None,
        as_dict=False
    ):
        """
        Recommend tracks based on a predefined audio profile
        representing the requested mood.

        Supported moods:
            happy, sad, calm, chill, energetic, focus

        Supports:
            top_n: Legacy Notebook parameter.
            top_k: Backend parameter.
            as_dict: Return list of dictionaries.
        """

        self._validate_model()

        # Support both top_n and top_k
        if top_k is not None:
            top_n = top_k

        if (
            not isinstance(top_n, (int, np.integer))
            or isinstance(top_n, bool)
            or top_n < 1
        ):
            raise ValueError(
                "top_n/top_k must be a positive integer."
            )

        if not isinstance(mood, str):
            raise TypeError("Mood must be a string.")

        mood = mood.strip().lower()

        # Representative audio profiles.
        # Values follow the original Spotify audio feature scales.
        mood_profiles = {
            "happy": {
                "danceability": 0.80,
                "energy": 0.80,
                "valence": 0.85,
                "tempo": 120.0,
                "acousticness": 0.15,
                "instrumentalness": 0.05,
                "speechiness": 0.08,
                "liveness": 0.15,
                "loudness": -5.0
            },

            "sad": {
                "danceability": 0.40,
                "energy": 0.25,
                "valence": 0.20,
                "tempo": 75.0,
                "acousticness": 0.65,
                "instrumentalness": 0.10,
                "speechiness": 0.05,
                "liveness": 0.15,
                "loudness": -18.0
            },

            "calm": {
                "danceability": 0.35,
                "energy": 0.25,
                "valence": 0.45,
                "tempo": 85.0,
                "acousticness": 0.80,
                "instrumentalness": 0.35,
                "speechiness": 0.04,
                "liveness": 0.12,
                "loudness": -18.0
            },

            "chill": {
                "danceability": 0.35,
                "energy": 0.25,
                "valence": 0.45,
                "tempo": 85.0,
                "acousticness": 0.80,
                "instrumentalness": 0.35,
                "speechiness": 0.04,
                "liveness": 0.12,
                "loudness": -18.0
            },

            "energetic": {
                "danceability": 0.75,
                "energy": 0.90,
                "valence": 0.65,
                "tempo": 140.0,
                "acousticness": 0.10,
                "instrumentalness": 0.05,
                "speechiness": 0.08,
                "liveness": 0.20,
                "loudness": -5.0
            },

            "focus": {
                "danceability": 0.30,
                "energy": 0.30,
                "valence": 0.25,
                "tempo": 95.0,
                "acousticness": 0.55,
                "instrumentalness": 0.85,
                "speechiness": 0.03,
                "liveness": 0.12,
                "loudness": -17.0
            }
        }

        if mood not in mood_profiles:
            supported_moods = ", ".join(mood_profiles.keys())

            raise ValueError(
                f"Unsupported mood: '{mood}'. "
                f"Supported moods: {supported_moods}"
            )

        # Convert the target profile into the same
        # standardized feature space as the trained model.
        target_df = pd.DataFrame(
            [mood_profiles[mood]],
            columns=AUDIO_FEATURES
        )

        target_vector = self.scaler.transform(target_df)

        # Calculate cosine similarity against all tracks.
        similarities = cosine_similarity(
            target_vector,
            self.feature_matrix
        ).flatten()

        # Rank tracks by mood similarity.
        top_indices = self._rank_scores(
            similarities,
            top_n
        )

        result = self._get_result(
            top_indices,
            {"mood_similarity": similarities}
        )

        if as_dict:
            return result.to_dict(orient="records")

        return result

    
    # ==========================================
    # 8. POPULARITY-BASED BASELINE
    # ==========================================

    def recommend_by_popularity(
        self,
        top_n=10,
        exclude_track_ids=None,
        *,
        top_k=None,
        as_dict=False
    ):
        """
        Popularity-based baseline for comparison.

        Supports:
            top_n: Legacy Notebook parameter.
            top_k: Backend API parameter.
            as_dict: Return list of dictionaries.
        """

        self._validate_model()

        # Support both top_n and top_k
        if top_k is not None:
            top_n = top_k

        if (
            not isinstance(top_n, (int, np.integer))
            or isinstance(top_n, bool)
            or top_n < 1
        ):
            raise ValueError(
                "top_n/top_k must be a positive integer."
            )

        if isinstance(exclude_track_ids, str):
            exclude_track_ids = [exclude_track_ids]

        excluded_indices = [
            self.id_to_idx[tid]
            for tid in (exclude_track_ids or [])
            if tid in self.id_to_idx
        ]

        if (
            len(getattr(self, "_popularity_order", [])) != len(self.df)
            or len(getattr(self, "_popularity_scores", [])) != len(self.df)
        ):
            self._build_search_and_popularity_indexes()

        # Reuse the precomputed descending popularity order.
        excluded_set = set(excluded_indices)
        top_indices = [
            int(idx)
            for idx in self._popularity_order
            if int(idx) not in excluded_set
        ][:int(top_n)]

        # Sử dụng chung helper để tạo recommendation response
        result = self._get_result(top_indices)

        if as_dict:
            return result.to_dict(orient="records")

        return result

    
    # ==========================================
    # 9. MODEL SERIALIZATION
    # ==========================================

    def save_artifact(self, path):
        """
        Save the complete trained model.
        """

        self._validate_model(require_genre=True)

        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        artifact = {
            "version": 1,
            "model": self
        }

        joblib.dump(
            artifact,
            path,
            compress=3
        )

        print(f"Model saved successfully: {path}")

        return path

    @classmethod
    def load_artifact(cls, path):
        """
        Load a previously saved model.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Artifact not found: {path}"
            )

        artifact = joblib.load(path)

        if artifact.get("version") != 1:
            raise ValueError(
                "Unsupported model artifact version."
            )

        model = artifact.get("model")

        if not isinstance(model, cls):
            raise TypeError(
                "Invalid model artifact."
            )

        model._validate_model(require_genre=True)
        model._build_search_and_popularity_indexes()

        return model



