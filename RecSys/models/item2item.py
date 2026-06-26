from dataclasses import dataclass
from typing import Literal, get_args
import numpy as np

type SymType = Literal["co-occurrence", "cosine", "jaccard", "probability", "lift"]


class Item2Item:
    def __init__(self, similarity_type: SymType):
        allowed_types = get_args(SymType)

        if similarity_type not in allowed_types:
            raise ValueError(
                f"'similarity_type' must be one of these: {allowed_types}; got {similarity_type!r} instead"
            )

        self.similarity_type = similarity_type

    @staticmethod
    def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
        topk_indices = np.argpartition(scores, -k)[-k:]
        topk_indices = topk_indices[np.argsort(scores[topk_indices], descending=True)]

        return topk_indices

    def fit(self, X: np.ndarray) -> None:
        """X: item2user matrix of implicit feedbacks"""
        X = np.where(X > 0, 1, 0)

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
            item_counts = np.diag(score_matrix)
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

        self.score_matrix = score_matrix
        self.item_popularity = item_popularity
        self.n_items = n_items
        self.n_users = n_users

    def similarity_scores(
        self,
        user: np.ndarray,
    ) -> np.ndarray:
        scores = user @ self.score_matrix

        return scores

    def predict(
        self,
        user: np.ndarray,
        prediction_size: int = 1,
        return_scores: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        scores = self.similarity_scores(user=user)
        scores[user > 0] = -np.inf

        indices = self._topk_indices(
            scores=scores,
            k=prediction_size,
        )

        if return_scores:
            return indices, scores[indices]

        return indices
