import numpy as np


class EASE:
    def __init__(self, lambda_l2: float = 100.0):
        if lambda_l2 < 0:
            raise ValueError(
                f"'lambda_l2' must be non-negative; got {lambda_l2!r} instead"
            )

        self.lambda_l2 = lambda_l2
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

    @staticmethod
    def _time_decay_weights(
        timedelta: np.ndarray,
        gamma: float,
    ) -> np.ndarray:
        return np.exp(-gamma * timedelta)

    def fit(
        self,
        X: np.ndarray,
        *,
        timedelta_matrix: np.ndarray | None = None,
        time_decay_gamma: float = 0.03,
    ) -> "EASE":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"'X' must be two-dimensional; got shape {X.shape}")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("'X' must not be empty")
        if not np.all(np.isfinite(X)):
            raise ValueError("'X' must contain only finite values")

        X = (X > 0).astype(np.float64)

        if timedelta_matrix is not None:
            timedelta_matrix = np.asarray(timedelta_matrix)
            if timedelta_matrix.shape != X.shape:
                raise ValueError(
                    f"'timedelta_matrix' must have shape {X.shape}; "
                    f"got {timedelta_matrix.shape} instead"
                )
            X = X * self._time_decay_weights(
                timedelta=timedelta_matrix,
                gamma=time_decay_gamma,
            )

        n_users, n_items = X.shape

        G = X.T @ X
        P = np.linalg.inv(G + self.lambda_l2 * np.eye(n_items))
        B = -P / np.diag(P)[None, :]
        np.fill_diagonal(B, 0.0)

        self.n_items = n_items
        self.n_users = n_users
        self.B = B

        self.is_fitted = True

        return self

    def similarity_scores(
        self,
        interactions: np.ndarray,
    ) -> np.ndarray:
        scores = interactions @ self.B

        return scores

    def predict(
        self,
        user: np.ndarray,
        *,
        timedelta_vector: np.ndarray | None = None,
        time_decay_gamma: float = 0.03,
        prediction_size: int = 1,
        return_scores: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()

        user = np.asarray(user)
        self._validate_user(user)

        seen_items = user > 0
        user = seen_items.astype(np.float64)

        if timedelta_vector is not None:
            timedelta_vector = np.asarray(timedelta_vector)
            if timedelta_vector.ndim != 1:
                raise ValueError(
                    f"'timedelta_vector' must be one-dimensional; "
                    f"got {timedelta_vector.ndim!r} dimensions instead"
                )
            if timedelta_vector.shape != user.shape:
                raise ValueError(
                    f"'timedelta_vector' must have shape {user.shape}; "
                    f"got {timedelta_vector.shape} instead"
                )

            user = user * self._time_decay_weights(
                timedelta=timedelta_vector,
                gamma=time_decay_gamma,
            )

        scores = self.similarity_scores(user)
        scores[seen_items] = -np.inf

        indices = self._topk_indices(
            scores=scores,
            k=prediction_size,
        )

        if return_scores:
            return indices, scores[indices]

        return indices
