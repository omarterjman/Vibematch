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
        if self.user_item_matrix is None:
            return None, "User–item matrix is not available. Build it from MongoDB first."

        if user_id not in self.user_item_matrix.index:
            return None, f"No listener profile found for {user_id}."

        user_vector = self.user_item_matrix.loc[user_id].values.reshape(1, -1)
        all_user_matrix = self.user_item_matrix.values

        sims = cosine_similarity(user_vector, all_user_matrix)[0]
        sim_series = pd.Series(sims, index=self.user_item_matrix.index)
        sim_series = sim_series.drop(index=user_id, errors="ignore")

        target_likes = self.user_item_matrix.loc[user_id]
        already_liked = set(target_likes[target_likes > 0].index)

        similar_users = sim_series[sim_series > 0]

        if similar_users.empty:
            likes_per_track = self.user_item_matrix.sum(axis=0)
            likes_per_track = likes_per_track.sort_values(ascending=False)
            candidate_ids: List[str] = [
                tid for tid in likes_per_track.index if tid not in already_liked
            ][:k]

            if not candidate_ids:
                return None, "Not enough data to suggest new songs yet."

            recs = self.df.loc[candidate_ids, ["Track", "Artist", "playlist_genre"]]
            return recs, None

        candidate_scores: Dict[str, float] = {}

        for neighbor_name, score in similar_users.items():
            neighbor_likes = self.user_item_matrix.loc[neighbor_name]
            neighbor_tracks = neighbor_likes[neighbor_likes > 0].index

            for tid in neighbor_tracks:
                if tid in already_liked:
                    continue
                candidate_scores[tid] = candidate_scores.get(tid, 0.0) + float(score)

        if not candidate_scores:
            likes_per_track = self.user_item_matrix.sum(axis=0)
            likes_per_track = likes_per_track.sort_values(ascending=False)
            candidate_ids = [
                tid for tid in likes_per_track.index if tid not in already_liked
            ][:k]
            if not candidate_ids:
                return None, "Not enough data to suggest new songs yet."
            recs = self.df.loc[candidate_ids, ["Track", "Artist", "playlist_genre"]]
            return recs, None

        sorted_candidates = sorted(
            candidate_scores.items(), key=lambda x: x[1], reverse=True
        )
        top_track_ids = [tid for tid, _ in sorted_candidates[:k]]

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
        func_name = getattr(model_func, "__name__", "")

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

            seed_genre = self.df.loc[test_track_id, "playlist_genre"]
            relevant_ids = self.df[
                self.df["playlist_genre"] == seed_genre
            ].index.tolist()

            recommended_ids = recs.index.tolist()
            score = self._average_precision_at_k(recommended_ids, relevant_ids, k)
            return score, None

        elif func_name == "simulate_collaborative_filtering":
            if not test_user_id:
                return 0.0, "Missing listener ID for people-based evaluation."

            if self.user_item_matrix is None:
                return 0.0, "User–item matrix not available."

            if test_user_id not in self.user_item_matrix.index:
                return 0.0, "Listener not found in user–item matrix."

            recs, err = model_func(test_user_id, k)
            if err:
                return 0.0, err

            if recs is None or recs.empty:
                return 0.0, "No recommendations were returned by the model."

            user_likes = self.user_item_matrix.loc[test_user_id]
            relevant_ids = user_likes[user_likes > 0].index.tolist()
            recommended_ids = recs.index.tolist()

            score = self._average_precision_at_k(recommended_ids, relevant_ids, k)
            return score, None

        else:
            return 0.0, "Unknown model function passed for evaluation."
