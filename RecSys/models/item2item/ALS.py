import numpy as np

from typing import Literal
from tqdm.auto import tqdm


class ImplicitALS:
    def __init__(
        self,
    ) -> None:
        self.is_fitted = False

    def _check_is_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "The model must be fitted via '.fit()' before prediction"
            )

    @staticmethod
    def _validate_2dim_matrix(matrix: np.ndarray, name: str) -> None:
        if matrix.ndim != 2:
            raise ValueError(
                f"'{name}' must be two-dimensional; got shape {matrix.shape}"
            )

    @staticmethod
    def _validate_vector(vector: np.ndarray, name: str) -> None:
        if vector.ndim != 1:
            raise ValueError(f"'{name}' must be a one-dimensional array")

    def _optimize_user_embeddings(
        self,
        P: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
        C: np.ndarray,
        lambda_: float,
        embedding_size: int,
    ) -> None:
        identity = np.eye(embedding_size)

        for u in range(U.shape[0]):
            C_u = np.diag(C[u, :])

            A_u = V.T @ C_u @ V + lambda_ * identity
            b_u = V.T @ C_u @ P[u, :]

            U[u, :] = np.linalg.solve(A_u, b_u)

    def _optimize_item_embeddings(
        self,
        P: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
        C: np.ndarray,
        lambda_: float,
        embedding_size: int,
    ) -> None:
        identity = np.eye(embedding_size)

        for i in range(V.shape[0]):
            C_i = np.diag(C[:, i])

            A_i = U.T @ C_i @ U + lambda_ * identity
            b_i = U.T @ C_i @ P[:, i]

            V[i, :] = np.linalg.solve(A_i, b_i)

    @staticmethod
    def _topk_indices(
        scores: np.ndarray,
        k: int,
    ) -> np.ndarray:
        if not 1 <= k <= scores.size:
            raise ValueError(
                f"'k' must be between 1 and {scores.size}; got {k} instead"
            )

        topk_indices = np.argpartition(scores, -k)[-k:]
        topk_indices = topk_indices[np.argsort(scores[topk_indices], descending=True)]

        return topk_indices

    def fit(
        self,
        interactions_matrix: np.ndarray,
        n_iterations: int = 40,
        embedding_size: int = 16,
        *,
        lambda_: float = 1.0,
        confidence_matrix: np.ndarray | None = None,
        confidence_matrix_type: Literal["linear", "log"] = "log",
        alpha: float = 1.0,
        random_state: int | None = None,
    ) -> "ImplicitALS":
        """interactions_matrix: np.ndarray -> matrix shaped (n_users, n_items) where (u, i) is how many times u interacted with i
        n_iterations: int -> number of iterations done while fitting
        embedding_size: int -> size of item and users embeddings
        lambda_: float -> l2 regularization hyperparameter"""
        if n_iterations <= 0:
            raise ValueError(
                f"'n_iterations' must be positive; got {n_iterations!r} instead"
            )
        if embedding_size <= 0:
            raise ValueError(
                f"'embedding_size' must be positive; got {embedding_size!r} instead"
            )
        if lambda_ < 0:
            raise ValueError(
                f"'lambda_' must be non-negative; got {lambda_!r} instead"
            )

        interactions_matrix = np.asarray(interactions_matrix, dtype=np.float32)
        self._validate_2dim_matrix(interactions_matrix, "interactions_matrix")

        if confidence_matrix is None:
            if confidence_matrix_type == "log":
                confidence_matrix = 1 + alpha * np.log(1 + interactions_matrix)
            elif confidence_matrix_type == "linear":
                confidence_matrix = 1 + alpha * interactions_matrix
            else:
                raise ValueError("'confidence_matrix_type' must be 'log' or 'linear'")
        else:
            confidence_matrix = np.asarray(confidence_matrix, dtype=np.float32)

        self._validate_2dim_matrix(confidence_matrix, "confidence_matrix")
        if confidence_matrix.shape != interactions_matrix.shape:
            raise ValueError(
                f"'confidence_matrix' must have shape {interactions_matrix.shape}; "
                f"got {confidence_matrix.shape} instead"
            )

        P = (interactions_matrix > 0).astype(float).copy()

        n_users, n_items = interactions_matrix.shape
        self.n_users = n_users
        self.n_items = n_items

        rng = np.random.default_rng(random_state)

        U = rng.normal(scale=0.01, size=(n_users, embedding_size))
        V = rng.normal(scale=0.01, size=(n_items, embedding_size))

        iterator = tqdm(range(n_iterations), desc="Optimizing embeddings")

        for _ in iterator:
            self._optimize_item_embeddings(
                P=P,
                U=U,
                V=V,
                C=confidence_matrix,
                lambda_=lambda_,
                embedding_size=embedding_size,
            )
            self._optimize_user_embeddings(
                P=P,
                U=U,
                V=V,
                C=confidence_matrix,
                lambda_=lambda_,
                embedding_size=embedding_size,
            )

        self.U = U
        self.V = V
        self.confidence_matrix = confidence_matrix

        self.embedding_size = embedding_size
        self.lambda_ = lambda_
        self.alpha = alpha

        self.is_fitted = True

        return self

    def predict(
        self,
        interactions_vector: np.ndarray,
        *,
        prediction_size: int = 1,
        return_scores: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()

        interactions_vector = np.asarray(interactions_vector, dtype=np.float32)
        self._validate_vector(interactions_vector, "interactions_vector")
        if interactions_vector.shape[0] != self.n_items:
            raise ValueError(
                f"'interactions_vector' must contain {self.n_items} items; "
                f"got {interactions_vector.shape[0]}"
            )

        p = (interactions_vector > 0).astype(float)
        P = p[None, :]

        confidence_matrix = 1 + self.alpha * interactions_vector
        confidence_matrix = confidence_matrix[None, :]
        user_embeddings = np.empty(shape=(1, self.embedding_size))

        self._optimize_user_embeddings(
            P=P,
            U=user_embeddings,
            V=self.V,
            C=confidence_matrix,
            lambda_=self.lambda_,
            embedding_size=self.embedding_size,
        )

        scores_vector = (user_embeddings @ self.V.T).ravel()
        scores_vector[interactions_vector > 0] = -np.inf

        topk_indexes = self._topk_indices(
            scores=scores_vector,
            k=prediction_size,
        )

        if return_scores:
            topk_scores = scores_vector[topk_indexes]

            return topk_indexes, topk_scores

        return topk_indexes


implicit_ALS = ImplicitALS

