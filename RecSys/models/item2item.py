from typing import Literal, get_args
import numpy as np

SimilarityType = Literal["co-occurrence", "cosine", "jaccard", "probability", "lift"]


class Item2Item:
    def __init__(self, similarity_type: SimilarityType):
        allowed_types = get_args(SimilarityType)

        if similarity_type not in allowed_types:
            raise ValueError(
                f"'similarity_type' must be one of these: {allowed_types}; got {similarity_type!r} instead"
            )

        self.similarity_type = similarity_type

        self.is_fitted = False

    def _check_is_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "The model must be fitted via '.fit()' before prediction"
            )

    def _validate_user(self, user: np.ndarray) -> None:
        if user.ndim != 1:
            raise ValueError("'user' must be a one-dimensional array")
        if user.shape[0] != self.n_items:
            raise ValueError(
                f"'user' must contain {self.n_items} items; " f"got {user.shape[0]}"
            )

    def _validate_users(self, users: np.ndarray) -> None:
        if users.ndim != 2:
            raise ValueError(
                f"'users' must be 2-dimensional; got {users.ndim!r} instead"
            )

        if users.shape[1] != self.n_items:
            raise ValueError(
                f"'user' must contain {self.n_items} items; " f"got {users.shape[1]}"
            )

    @staticmethod
    def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
        if not 1 <= k <= scores.size:
            raise ValueError(
                f"'k' must be between 1 and {scores.size}; got {k} instead"
            )

        topk_indices = np.argpartition(scores, -k)[-k:]
        topk_indices = topk_indices[np.argsort(scores[topk_indices], descending=True)]

        return topk_indices

    def fit(
        self,
        X: np.ndarray,
        use_shrinkage: bool = True,
        use_idf: bool = True,
        *,
        alpha: float = 10,
    ) -> "Item2Item":
        """X: item2user matrix of implicit feedbacks"""
        if use_shrinkage and alpha < 0:
            raise ValueError(f"'alpha' must be non-negative; got {alpha!r} instead")

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"'X' must be two-dimensional; got shape {X.shape}")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("'X' must not be empty")

        if not np.all(np.isfinite(X)):
            raise ValueError("'X' must contain only finite values")

        X = (X > 0).astype(np.float64)

        n_items, n_users = X.shape

        cooccurrence_matrix = X @ X.T
        item_popularity = X.sum(axis=1)

        if self.similarity_type == "co-occurrence":
            score_matrix = cooccurrence_matrix.copy()
        elif self.similarity_type == "cosine":
            denominator = np.sqrt(np.outer(item_popularity, item_popularity))
            score_matrix = np.divide(
                cooccurrence_matrix,
                denominator,
                out=np.zeros_like(cooccurrence_matrix, dtype=float),
                where=denominator > 0,
            )
        elif self.similarity_type == "jaccard":
            union_score = (
                item_popularity[:, None]
                + item_popularity[None, :]
                - cooccurrence_matrix
            )
            score_matrix = np.divide(
                cooccurrence_matrix,
                union_score,
                out=np.zeros_like(cooccurrence_matrix, dtype=float),
                where=union_score > 0,
            )
        elif self.similarity_type == "probability":
            item_counts = np.diag(cooccurrence_matrix)
            score_matrix = np.divide(
                cooccurrence_matrix,
                item_counts[:, None],
                out=np.zeros_like(cooccurrence_matrix, dtype=float),
                where=item_counts[:, None] > 0,
            )
        elif self.similarity_type == "lift":
            item_counts = np.diag(cooccurrence_matrix)
            denominator = item_counts[:, None] * item_counts[None, :]
            score_matrix = n_users * np.divide(
                cooccurrence_matrix,
                denominator,
                out=np.zeros_like(cooccurrence_matrix, dtype=float),
                where=denominator > 0,
            )

        np.fill_diagonal(score_matrix, 0.0)

        if use_shrinkage:
            score_matrix = score_matrix * np.divide(
                cooccurrence_matrix,
                cooccurrence_matrix + alpha,
                out=np.zeros_like(cooccurrence_matrix, dtype=float),
                where=cooccurrence_matrix + alpha != 0,
            )

        if use_idf:
            idf = np.zeros_like(item_popularity, dtype=float)
            observed = item_popularity > 0

            idf[observed] = np.log((n_users + 1) / (item_popularity[observed] + 1))
        else:
            idf = None

        self.idf = idf
        self.score_matrix = score_matrix
        self.item_popularity = item_popularity
        self.n_items = n_items
        self.n_users = n_users

        self.is_fitted = True

        return self

    def similarity_scores(
        self,
        interactions: np.ndarray,
    ) -> np.ndarray:
        if self.idf is not None:
            interactions = interactions * self.idf

        scores = interactions @ self.score_matrix

        return scores

    def predict(
        self,
        user: np.ndarray,
        prediction_size: int = 1,
        return_scores: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()

        user = np.asarray(user)
        user = (user > 0).astype(np.float64)
        self._validate_user(user)

        scores = self.similarity_scores(user)
        scores[user > 0] = -np.inf

        indices = self._topk_indices(
            scores=scores,
            k=prediction_size,
        )

        if return_scores:
            return indices, scores[indices]

        return indices

    def score_users(
        self,
        users: np.ndarray,
    ) -> np.ndarray:
        self._check_is_fitted()

        users = np.asarray(users)
        users = (users > 0).astype(np.float64)

        self._validate_users(users)

        return self.similarity_scores(users)

    def predict_batch(
        self,
        users: np.ndarray,
        prediction_size: int = 1,
        return_scores: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()

        users = np.asarray(users)
        users = (users > 0).astype(np.float64)

        self._validate_users(users)

        user_scores = self.similarity_scores(users)

        predicted_indices = np.empty(
            (users.shape[0], prediction_size),
            dtype=np.int64,
        )
        predicted_scores = np.empty(
            (users.shape[0], prediction_size),
            dtype=np.float64,
        )

        for user_idx in range(users.shape[0]):
            scores = user_scores[user_idx].copy()
            scores[users[user_idx] > 0] = -np.inf

            indices = self._topk_indices(
                scores=scores,
                k=prediction_size,
            )

            predicted_indices[user_idx] = indices
            predicted_scores[user_idx] = scores[indices]

        if return_scores:
            return predicted_indices, predicted_scores

        return predicted_indices

    def similar_items(
        self,
        item: int,
        k: int = 10,
        return_scores: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()

        scores = self.score_matrix[item].copy()
        scores[item] = -np.inf

        indices = self._topk_indices(scores=scores, k=k)

        if return_scores:
            return indices, scores[indices]

        return indices
