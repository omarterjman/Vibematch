import pandas as pd
import numpy as np

from typing import List, Optional, Tuple, Dict

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


class RecommendationSystem:
    def __init__(
        self,
        data_path: str,
        mongo_uri: str = "mongodb://localhost:27017/",
        db_name: str = "vibematch",
        listeners_collection: str = "listeners",
    ) -> None:
        self.df: pd.DataFrame = self._load_and_preprocess(data_path)
        if self.df is None or self.df.empty:
            raise FileNotFoundError(f"Could not load or preprocess data from {data_path}")

        self.feature_cols: List[str] = [
            "acousticness",
            "danceability",
            "energy",
            "instrumentalness",
            "liveness",
            "loudness",
            "speechiness",
            "tempo",
            "valence",
        ]
        self.feature_cols = [c for c in self.feature_cols if c in self.df.columns]
        if not self.feature_cols:
            raise ValueError("No expected audio feature columns found in dataset.")

        self.df_features: pd.DataFrame = self.df[self.feature_cols].fillna(0.0)
        self.scaler = MinMaxScaler()
        self.df_features_scaled: np.ndarray = self.scaler.fit_transform(self.df_features)
        self.cbf_similarity_matrix: np.ndarray = cosine_similarity(self.df_features_scaled)

        self.client: Optional[MongoClient] = None
        self.db = None
        self.listeners_col = None
        self._connect_mongo(mongo_uri, db_name, listeners_collection)

        self.user_item_matrix: Optional[pd.DataFrame] = None

    def _connect_mongo(self, uri: str, db_name: str, collection_name: str) -> None:
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=2000)
            _ = self.client.server_info()
            self.db = self.client[db_name]
            self.listeners_col = self.db[collection_name]
        except Exception:
            self.client = None
            self.db = None
            self.listeners_col = None

    def _load_and_preprocess(self, data_path: str) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(data_path)

            if "track_id" not in df.columns:
                raise ValueError("Expected column 'track_id' not found in dataset.")

            if "track_name" in df.columns:
                df.rename(columns={"track_name": "Track"}, inplace=True)
            if "track_artist" in df.columns:
                df.rename(columns={"track_artist": "Artist"}, inplace=True)

            if "Track" not in df.columns:
                df["Track"] = "Unknown Track"
            if "Artist" not in df.columns:
                df["Artist"] = "Unknown Artist"

            df["Track"] = df["Track"].fillna("Unknown Track")
            df["Artist"] = df["Artist"].fillna("Unknown Artist")

            if "playlist_genre" not in df.columns:
                df["playlist_genre"] = "Unknown"

            df["playlist_genre"] = df["playlist_genre"].fillna("Unknown")

            df.drop_duplicates(subset=["track_id"], inplace=True)
            df.set_index("track_id", inplace=True)

            return df
        except Exception as exc:
            print(f"Error loading or preprocessing data: {exc}")
            return None

    def add_like_for_listener(self, name: str, track_id: str) -> Optional[str]:
        if self.listeners_col is None:
            return "MongoDB is not available. Please check your connection."

        if track_id not in self.df.index:
            return "Selected track is not part of the catalog."

        try:
            self.listeners_col.update_one(
                {"name": name},
                {"$addToSet": {"liked_tracks": track_id}},
                upsert=True,
            )
            return None
        except PyMongoError as exc:
            return f"Failed to update MongoDB: {exc}"

    def remove_like_for_listener(self, name: str, track_id: str) -> Optional[str]:
        """
        Remove a single track from a listener's liked_tracks array in MongoDB.
        """
        if self.listeners_col is None:
            return "MongoDB is not available. Please check your connection."

        try:
            result = self.listeners_col.update_one(
                {"name": name},
                {"$pull": {"liked_tracks": track_id}},
            )
            if result.matched_count == 0:
                return f"No listener named '{name}' was found in MongoDB."
            # If track_id wasn't there, $pull is simply a no-op (no error).
            return None
        except PyMongoError as exc:
            return f"Failed to update MongoDB: {exc}"

    def get_liked_tracks_for_listener(
        self, name: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Return a table of this listener's liked songs."""
        if self.listeners_col is None:
            return None, "MongoDB is not available. Please check your connection."

        if not name.strip():
            return None, "No listener name provided."

        try:
            doc = self.listeners_col.find_one(
                {"name": name},
                {"liked_tracks": 1, "_id": 0}
            )
        except PyMongoError as exc:
            return None, f"Failed to read MongoDB: {exc}"

        if not doc or not doc.get("liked_tracks"):
            return None, f"{name} has no liked songs saved yet."

        # keep only track_ids that still exist in the catalog
        liked_ids = [tid for tid in doc["liked_tracks"] if tid in self.df.index]

        if not liked_ids:
            return None, "No liked tracks found in the current catalog."

        liked_df = self.df.loc[liked_ids, ["Track", "Artist", "playlist_genre"]]
        return liked_df, None

    def get_listener_names(self) -> List[str]:
        if self.listeners_col is None:
            return []

        try:
            names = self.listeners_col.distinct("name")
            names = [n for n in names if n]
            return sorted(names)
        except PyMongoError:
            return []

    def build_user_item_matrix_from_mongo(self) -> Optional[str]:
        if self.listeners_col is None:
            return "MongoDB is not available. People-based mode cannot be used."

        try:
            docs = list(self.listeners_col.find({}, {"name": 1, "liked_tracks": 1}))
        except PyMongoError as exc:
            return f"Failed to read MongoDB listeners: {exc}"

        if not docs:
            self.user_item_matrix = None
            return "No listeners found in the database."

        user_names: List[str] = []
        for d in docs:
            name = d.get("name")
            if isinstance(name, str) and name.strip():
                user_names.append(name.strip())

        user_names = sorted(set(user_names))
        if not user_names:
            self.user_item_matrix = None
            return "No valid listener names were found in the database."

        track_ids: List[str] = list(self.df.index)
        matrix = pd.DataFrame(0, index=user_names, columns=track_ids, dtype=np.int8)

        for d in docs:
            name = d.get("name")
            if not isinstance(name, str):
                continue
            name = name.strip()
            if name not in matrix.index:
                continue

            liked_tracks = d.get("liked_tracks", [])
            if isinstance(liked_tracks, list):
                for tid in liked_tracks:
                    if tid in matrix.columns:
                        matrix.at[name, tid] = 1

        self.user_item_matrix = matrix
        return None

    def get_content_based_recommendations(
        self, seed_track_id: str, k: int = 10
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        if seed_track_id not in self.df.index:
            return None, "Seed track is not in the catalog."

        try:
            row_idx = self.df.index.get_loc(seed_track_id)
        except KeyError:
            return None, "Could not locate the seed track index."

        similarities = self.cbf_similarity_matrix[row_idx]
        indexed_scores = list(enumerate(similarities))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        top_indices = [idx for idx, _ in indexed_scores[1 : k + 1]]
        recs = self.df.iloc[top_indices][["Track", "Artist", "playlist_genre"]]
        return recs, None

    def simulate_collaborative_filtering(
            self, user_id: str, k: int = 10
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        People-based (collaborative) recommender.

        - Find users with similar like patterns.
        - Recommend songs they like but THIS user doesn't.
        """

        # 1) Basic checks
        if self.user_item_matrix is None:
            return None, "User–item matrix is not available. Build it from MongoDB first."

        if user_id not in self.user_item_matrix.index:
            return None, f"No listener profile found for {user_id}."

        # Vector of this user's likes (0/1 per track)
        user_vector = self.user_item_matrix.loc[user_id]
        already_liked: set[str] = set(user_vector[user_vector > 0].index)

        if len(already_liked) == 0:
            return None, f"{user_id} has no liked songs yet."

        # 2) Find similar listeners using cosine similarity
        all_user_matrix = self.user_item_matrix.values
        sims = cosine_similarity(user_vector.values.reshape(1, -1), all_user_matrix)[0]
        sim_series = pd.Series(sims, index=self.user_item_matrix.index)

        # remove self
        sim_series = sim_series.drop(index=user_id, errors="ignore")

        # only users with some positive similarity
        similar_users = sim_series[sim_series > 0]

        candidate_scores: Dict[str, float] = {}

        # 3) If we have similar users, aggregate their liked tracks
        if not similar_users.empty:
            for neighbor_name, score in similar_users.items():
                neighbor_likes = self.user_item_matrix.loc[neighbor_name]
                neighbor_tracks = neighbor_likes[neighbor_likes > 0].index

                for tid in neighbor_tracks:
                    # never recommend songs the user already liked
                    if tid in already_liked:
                        continue
                    candidate_scores[tid] = candidate_scores.get(tid, 0.0) + float(score)

        # 4) Fallback: if we got no candidates (no neighbors or no new tracks),
        #    use "most liked by others" but still exclude this user's liked songs.
        if not candidate_scores:
            likes_per_track = self.user_item_matrix.sum(axis=0)

            # drop tracks this user already likes
            likes_per_track = likes_per_track.drop(
                labels=list(already_liked), errors="ignore"
            )

            # most popular unseen tracks
            candidate_ids = likes_per_track.sort_values(
                ascending=False
            ).index.tolist()[:k]

            if not candidate_ids:
                return None, "Not enough data to suggest new songs yet."

            recs = self.df.loc[candidate_ids, ["Track", "Artist", "playlist_genre"]]
            return recs, None

        # 5) Rank candidate tracks by their accumulated neighbor scores
        sorted_candidates = sorted(
            candidate_scores.items(), key=lambda x: x[1], reverse=True
        )
        top_track_ids = [tid for tid, _ in sorted_candidates[:k]]

        # double-check: never return tracks the user already liked
        top_track_ids = [tid for tid in top_track_ids if tid not in already_liked]

        if not top_track_ids:
            return None, "Not enough unseen songs to recommend."

        recs = self.df.loc[top_track_ids, ["Track", "Artist", "playlist_genre"]]
        return recs, None

    def _average_precision_at_k(
        self, recommended_ids: List[str], relevant_ids: List[str], k: int
    ) -> float:
        hits = 0
        sum_precision = 0.0

        for i, rec_id in enumerate(recommended_ids[:k]):
            if rec_id in relevant_ids:
                hits += 1
                precision_at_i = hits / float(i + 1)
                sum_precision += precision_at_i

        total_relevant = min(len(relevant_ids), k)
        if total_relevant == 0:
            return 0.0

        return sum_precision / float(total_relevant)

    def evaluate_model(
            self,
            model_func,
            test_user_id: Optional[str] = None,
            test_track_id: Optional[str] = None,
            k: int = 10,
    ) -> Tuple[float, Optional[str]]:
        """
        Evaluate either:
        - get_content_based_recommendations (song-based)
        - simulate_collaborative_filtering (people-based)

        Metric: Average Precision@k where "relevant" = songs that fit
        the listener's preferred genres (when we know the listener).
        """

        func_name = getattr(model_func, "__name__", "")

        # ------------------------------------------------------------
        # A) SONG-BASED ENGINE
        # ------------------------------------------------------------
        if func_name == "get_content_based_recommendations":
            if not test_track_id:
                return 0.0, "Missing seed track for content-based evaluation."

            recs, err = model_func(test_track_id, k)
            if err:
                return 0.0, err

            if recs is None or recs.empty:
                return 0.0, "No recommendations were returned by the model."

            if test_track_id not in self.df.index:
                return 0.0, "Seed track not found in catalog for evaluation."

            # ---------- NEW PART: use USER'S favourite genres if possible ----------
            if (
                    test_user_id is not None
                    and self.user_item_matrix is not None
                    and test_user_id in self.user_item_matrix.index
            ):
                user_likes = self.user_item_matrix.loc[test_user_id]
                liked_ids = user_likes[user_likes > 0].index.tolist()

                if liked_ids:
                    liked_genres = set(self.df.loc[liked_ids, "playlist_genre"])
                    relevant_ids = self.df[
                        self.df["playlist_genre"].isin(liked_genres)
                    ].index.tolist()
                else:
                    # fallback: use only the seed genre
                    seed_genre = self.df.loc[test_track_id, "playlist_genre"]
                    relevant_ids = self.df[
                        self.df["playlist_genre"] == seed_genre
                        ].index.tolist()
            else:
                # OLD BEHAVIOUR (for safety, if we don't know the user)
                seed_genre = self.df.loc[test_track_id, "playlist_genre"]
                relevant_ids = self.df[
                    self.df["playlist_genre"] == seed_genre
                    ].index.tolist()

            recommended_ids = recs.index.tolist()
            if not relevant_ids:
                return 0.0, "No relevant songs found for content-based evaluation."

            score = self._average_precision_at_k(recommended_ids, relevant_ids, k)
            return score, None

        # ------------------------------------------------------------
        # B) PEOPLE-BASED ENGINE
        # ------------------------------------------------------------
        elif func_name == "simulate_collaborative_filtering":
            if not test_user_id:
                return 0.0, "Missing listener ID for people-based evaluation."

            if self.user_item_matrix is None:
                return 0.0, "User–item matrix not available."

            if test_user_id not in self.user_item_matrix.index:
                return 0.0, "Listener not found in user–item matrix."
            # --- NEW: detect if the user has any similar listeners ---
            user_vector = self.user_item_matrix.loc[test_user_id].values.reshape(1, -1)
            all_user_matrix = self.user_item_matrix.values
            sims = cosine_similarity(user_vector, all_user_matrix)[0]
            sim_series = pd.Series(sims, index=self.user_item_matrix.index).drop(index=test_user_id)

            # If no similarities > 0 → CF cannot be evaluated
            if (sim_series > 0).sum() == 0:
                return 0.0, "People-based score: N/A (Listener has no similar users for evaluation.)"

            # listener must have at least 2 liked songs
            user_likes = self.user_item_matrix.loc[test_user_id]
            liked_ids = user_likes[user_likes > 0].index.tolist()
            if len(liked_ids) < 2:
                return 0.0, "Listener must have at least 2 liked songs for evaluation."

            recs, err = model_func(test_user_id, k)
            if err:
                return 0.0, err

            if recs is None or recs.empty:
                return 0.0, "No recommendations were returned by the model."

            liked_genres = set(self.df.loc[liked_ids, "playlist_genre"])
            relevant_ids = self.df[
                self.df["playlist_genre"].isin(liked_genres)
            ].index.tolist()

            recommended_ids = recs.index.tolist()

            if not relevant_ids:
                return 0.0, "No relevant songs found for people-based evaluation."

            score = self._average_precision_at_k(recommended_ids, relevant_ids, k)
            return score, None

        # ------------------------------------------------------------
        # Unknown model
        # ------------------------------------------------------------
        else:
            return 0.0, "Unknown model function passed for evaluation."
    def get_hybrid_recommendations(
        self,
        user_id: str,
        seed_track_id: str,
        alpha: float = 0.6,
        k: int = 10,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Hybrid recommender = mix of:
        - content-based (audio similarity from a seed song)
        - people-based (similar listeners)

        alpha = weight for content-based part (0–1).
        (1 - alpha) = weight for people-based part.
        """

        alpha = max(0.0, min(1.0, float(alpha)))

        if seed_track_id not in self.df.index:
            return None, "Seed track is not in the catalog."

        if self.user_item_matrix is None:
            err = self.build_user_item_matrix_from_mongo()
            if err:
                return None, err

        if user_id not in self.user_item_matrix.index:
            return None, f"No listener profile found for {user_id}."

        user_vector = self.user_item_matrix.loc[user_id]
        already_liked = set(user_vector[user_vector > 0].index)

        # 1) Content-based candidates
        cbf_recs, err_cbf = self.get_content_based_recommendations(
            seed_track_id, k=k * 3
        )
        if err_cbf or cbf_recs is None or cbf_recs.empty:
            cbf_recs = pd.DataFrame(columns=["Track", "Artist", "playlist_genre"])

        cbf_scores: dict[str, float] = {}
        n_cbf = len(cbf_recs)
        for rank, tid in enumerate(cbf_recs.index):
            cbf_scores[tid] = (n_cbf - rank) / max(1.0, n_cbf)

        # 2) People-based candidates
        cf_recs, err_cf = self.simulate_collaborative_filtering(user_id, k=k * 3)
        if err_cf or cf_recs is None or cf_recs.empty:
            cf_recs = pd.DataFrame(columns=["Track", "Artist", "playlist_genre"])

        cf_scores: dict[str, float] = {}
        n_cf = len(cf_recs)
        for rank, tid in enumerate(cf_recs.index):
            cf_scores[tid] = (n_cf - rank) / max(1.0, n_cf)

        # 3) Combine
        candidate_ids = set(cbf_scores.keys()) | set(cf_scores.keys())
        if not candidate_ids:
            return None, "Hybrid engine has no candidates to recommend yet."

        hybrid_scores: dict[str, float] = {}
        for tid in candidate_ids:
            score_cbf = cbf_scores.get(tid, 0.0)
            score_cf = cf_scores.get(tid, 0.0)
            hybrid_scores[tid] = alpha * score_cbf + (1.0 - alpha) * score_cf

        filtered_ids = [
            tid for tid in hybrid_scores.keys() if tid not in already_liked
        ]
        if not filtered_ids:
            return None, "No unseen songs left to recommend for this listener."

        filtered_ids.sort(key=lambda tid: hybrid_scores[tid], reverse=True)
        top_ids = filtered_ids[:k]

        recs = self.df.loc[top_ids, ["Track", "Artist", "playlist_genre"]]
        return recs, None