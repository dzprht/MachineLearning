import numpy as np

from tqdm.auto import tqdm


class implicit_ALS:
    def __init__(
        self,
    ) -> None:
        self.is_fitted = False

    def _check_is_fitted(self) -> None:
        if not self.is_fitted:
            raise ValueError(
                "model must be fitted before using this function via '.fit(...)'"
            )
    
    @staticmethod
    def _validate_2dim_matrix(matrix: np.ndarray, name: str) -> None:
        if matrix.ndim != 2:
            raise ValueError(
                f"'{name}' must be 2-dimensional matrix; got {matrix.ndim}-dimensional instead"
            )

    @staticmethod
    def _validate_vector(vector: np.ndarray, name: str) -> None:
        if vector.ndim != 1:
            raise ValueError(
                f"'{name}' must be 1-dimensional vector; got {vector.ndim}-dimensional instead"
            )

    def _optimize_user_embeddings(
        self,
        P: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
        C: np.ndarray,
        lambda_: float,
        embedding_size: int,
    ) -> None:
        indentify = np.eye(embedding_size)

        for u in range(U.shape[0]):
            C_u = np.diag(C[u, :])

            A_u = V.T @ C_u @ V + lambda_ * indentify
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
        indetify = np.eye(embedding_size)

        for i in range(V.shape[0]):
            C_i = np.diag(C[:, i])

            A_i = U.T @ C_i @ U + lambda_ * indetify
            b_i = U.T @ C_i @ P[:, i]

            V[i, :] = np.linalg.solve(A_i, b_i)

    @staticmethod
    def _get_topk_prediction_indexes(scores: np.ndarray, k: int) -> np.ndarray:
        topk = np.argpartition(scores, -k)[-k:]
        topk = topk[np.argsort(scores[topk], descending=True)]

        return topk

    def fit(
        self,
        interactions_matrix: np.ndarray,
        n_iterations: int = 40,
        embedding_size: int = 16,
        *,
        lambda_: float = 1.0,
        confidence_matrix: np.ndarray | None = None,
        alpha: float = 1.0,
        random_state: int | None = None,
    ) -> "implicit_ALS":
        """interactions_matrix: np.ndarray -> matrix shaped (n_users, n_items) where (u, i) is how many times u interacted with i
        n_iterations: int -> number of iterations done while fitting
        embedding_size: int -> size of item and users embeddings
        lambda_: float -> l2 regularization hyperparameter"""


        interactions_matrix = np.asarray(interactions_matrix, dtype=np.float32)
        self._validate_2dim_matrix(interactions_matrix, "interactions_matrix")

        if confidence_matrix is None:
            confidence_matrix = 1 + alpha * interactions_matrix
        self._validate_2dim_matrix(confidence_matrix, "confidence_matrix")

        P = (interactions_matrix > 0).astype(float).copy()

        n_users, n_items = interactions_matrix.shape
        self.n_users = n_users
        self.n_items = n_items

        rng = np.random.default_rng(random_state)

        U = rng.normal(scale=0.01, size=(n_users, embedding_size))
        V = rng.normal(scale=0.01, size=(n_items, embedding_size))

        for iteration in tqdm(range(n_iterations), desc="Optimizing embeddings"):
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
        k: int = 1,
        return_score: bool = True,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()
        
        interactions_vector = np.asarray(interactions_vector, dtype=np.float32)
        self._validate_vector(interactions_vector, "interactions_vector")
        p = (interactions_vector > 0).astype(float)
        P = p[None, :]

        user_embeddings = np.empty(shape=(1, self.embedding_size))

        confidence_matrix = 1 + self.alpha * interactions_vector

        self._optimize_user_embeddings(
            P=P,
            U=user_embeddings,
            V=self.V,
            C=self.confidence_matrix,
            lambda_=self.lambda_,
            embedding_size=self.embedding_size,
        )

        scores_vector = user_embeddings @ self.V.T
        scores_vector[interactions_vector > 0] = -np.inf

        topk_indexes = self._get_topk_prediction_indexes(scores_vector, k=k)

        if return_score:
            topk_scores = scores_vector[topk_indexes]

            return topk_indexes, topk_scores
        
        return topk_indexes



